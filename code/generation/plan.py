"""Plan text-to-video prompts: one fresh, context-free planning agent per prompt writes a plan (scene, entities, motions,
events, dependencies, timeline) and checks it with the checker until it passes or gives up.

    python generation/plan.py --prompts prompts.json --out plans [--agent api | claude-code] [--jobs 4]
    python generation/plan.py --prompt "A balloon drifts right to left above a statue" --id balloon --out plans

prompts.json is a list of {"id", "prompt"} with optional "category" and "name" (where prepare.py puts the video:
<results>/video/<category>/<name>.mp4; for T2V-CompBench, name = the prompt's 1-based benchmark index NNNN, id = <category>_NNNN). Agents:
  api          (default) the Anthropic Messages API through the official SDK (ANTHROPIC_API_KEY); agents/messages_api.py
  claude-code  an isolated `claude -p` process per prompt (CLAUDE_CLI, default `claude`); agents/claude_code.py
Both get the same task text (prompts/planner.md), the same neutral workspace (task.md and schema.md only) and the same checker
command; the checker is a copy in a neutral directory that the agent can run but not read. Model claude-opus-5-5, effort medium.
A technical failure (API error, a refusal, a run that never used a tool, a CLI error) is retried in a fresh workspace, at most 3
attempts. A run that writes no plan is a planning failure (kept, not retried); a plan the checker rejects is kept as written.
Output per prompt: <out>/<id>/prompt.json, plan.json, plan.lint/ (checker report and preview), task.md, attempts.json (and the
transcripts of API runs); <out>/summary.json.
"""
import argparse, json, secrets, shutil, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CB, PKG, PLANNER_TEMPLATE, jdump  # noqa: E402

ASK = ('Your complete instructions are in the file task.md in the current directory. Read that file first and carry it out '
       'exactly as written. Your final answer is the structured summary the task file asks for.')
TRIES = 3
CHECKER_ENV = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
LOCK = threading.Lock()


def install_checker(tools):
    tools.mkdir(parents=True, exist_ok=True)
    for f in ('scm_lint.py', 'world.py', 'rules_pv2.json'):
        shutil.copy2(CB / f, tools / f); (tools / f).chmod(0o644)


def workspace(d, sid, prompt, tools):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    lint = f'python3 {tools}/scm_lint.py {d}/plan.json --rules {tools}/rules_pv2.json'
    t = (PLANNER_TEMPLATE.read_text().replace('@@SCHEMA@@', str(d / 'schema.md')).replace('@@PROMPT@@', prompt)
         .replace('@@OUT@@', str(d / 'plan.json')).replace('@@SID@@', sid).replace('@@LINT@@', lint))
    assert '@@' not in t
    (d / 'task.md').write_text(t); shutil.copy2(CB / 'scm_schema_pv2.md', d / 'schema.md')


def outcome(d, att):
    """done / planfail from the workspace; the checker's verdict if it ran."""
    try:
        json.loads((d / 'plan.json').read_text()); att['status'] = 'done'
    except (OSError, ValueError):
        att['status'] = 'planfail'
    try:
        att['lint_passed'] = bool(json.loads((d / 'plan.lint' / 'lint.json').read_text()).get('passed'))
    except (OSError, ValueError):
        att['lint_passed'] = None
    return att


def attempt(agent, d, tools, out, k):
    prefix = f'python3 {tools}/scm_lint.py'
    if agent == 'api':
        from agents import messages_api
        try:
            text, info = messages_api.run(ASK.replace('the file task.md in the current directory', f'the file {d}/task.md'), workdir=d,
                                          read_roots=[d], write_roots=[d], bash_prefixes=[prefix + ' '], bash_env=CHECKER_ENV,
                                          transcript=out / f'transcript_{k}.json')
        except Exception as e:   # API errors and refusals: a technical failure, retried in a fresh workspace
            return dict(status='technical', error=f'{type(e).__name__}: {e}'[:400])
        att = dict(turns=info['turns'], tools=info['tools'], usage=info['usage'], seconds=info['seconds'], result=text[-600:])
        if not any(info['tools'].values()):
            return dict(att, status='technical', error='the conversation ended without any tool call')
        return outcome(d, att)
    from agents import claude_code
    deny = [p for p in (tools, PKG, out.parent, Path.home()) if not d.is_relative_to(p)]
    res = claude_code.run(ASK, workdir=d, bash_prefix=prefix, deny_read=deny)
    att = dict(session_id=res.get('session_id'), num_turns=res.get('num_turns'), usage=res.get('usage'), cost_usd=res.get('total_cost_usd'),
               seconds=res['seconds'], result=str(res.get('result', ''))[-600:])
    if res.get('is_error') or not res.get('session_id'):
        return dict(att, status='technical')
    return outcome(d, att)


def plan_one(agent, item, out_root, ws_root, tools):
    out = out_root / item['id']; out.mkdir(parents=True, exist_ok=True); atts = []
    jdump(out / 'prompt.json', item)
    for k in range(TRIES):
        d = ws_root / secrets.token_hex(4)          # neutral name: nothing about the prompt or the project
        workspace(d, f'q{secrets.token_hex(3)}', item['prompt'], tools)
        att = dict(attempt(agent, d, tools, out, k), workspace=str(d), agent=agent); atts.append(att)
        jdump(out / 'attempts.json', atts)
        if att['status'] != 'technical':
            shutil.copy2(d / 'task.md', out / 'task.md')
            if att['status'] == 'done':
                shutil.copy2(d / 'plan.json', out / 'plan.json')
                if (d / 'plan.lint').is_dir():
                    shutil.copytree(d / 'plan.lint', out / 'plan.lint', dirs_exist_ok=True)
            break
    with LOCK:
        print(item['id'], atts[-1]['status'], 'checker passed' if atts[-1].get('lint_passed') else 'checker failed or not run',
              f"({len(atts)} attempt{'s' if len(atts) > 1 else ''})", flush=True)
    return dict(id=item['id'], status=atts[-1]['status'], lint_passed=atts[-1].get('lint_passed'), attempts=len(atts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--prompts', help='JSON list of {"id", "prompt"}'); ap.add_argument('--prompt'); ap.add_argument('--id')
    ap.add_argument('--out', required=True); ap.add_argument('--agent', choices=('api', 'claude-code'), default='api')
    ap.add_argument('--jobs', type=int, default=1); ap.add_argument('--workspace-root', default=str(Path(tempfile.gettempdir()) / 'plan_ws'))
    a = ap.parse_args()
    if a.prompts:
        items = json.loads(Path(a.prompts).read_text())
    elif a.prompt and a.id:
        items = [{'id': a.id, 'prompt': a.prompt}]
    else:
        ap.error('give --prompts, or --prompt with --id')
    out_root = Path(a.out).resolve(); ws_root = Path(a.workspace_root).resolve()
    tools = ws_root / f'tools_{secrets.token_hex(4)}'; install_checker(tools)
    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        rows = list(ex.map(lambda it: plan_one(a.agent, it, out_root, ws_root, tools), items))
    jdump(out_root / 'summary.json', rows)
    print(sum(r['status'] == 'done' for r in rows), 'of', len(rows), 'prompts planned;',
          sum(bool(r['lint_passed']) for r in rows), 'passed the checker')


if __name__ == '__main__':
    main()
