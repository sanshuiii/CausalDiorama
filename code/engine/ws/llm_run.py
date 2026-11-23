"""Agent steps of the editing engine. Every agent task is one fresh, context-free agent whose only message is the spawn message
"Your whole task is written in <task file>. Read that file and follow it exactly." (prompts/agents/spawn_message.txt), with the
tools Read / Write / Edit / Bash. Two ways to run it, chosen by CD_AGENT_MODE:

    api         (default) cc_agent.py: a replica of a Claude Code general-purpose subagent on the Anthropic Messages API (its system
                prompt, context, tool definitions and result formats; adaptive thinking; the stage's effort; prompt cache). Tools run in
                api_agent's Sandbox:
                  read   the run tree, this directory (geomrun, scm_match, cam_helper, cam_track), engine/cb (schema, rules,
                         scm_lint, world) and /tmp
                  write  the task's own directory (the directory of its output file) and /tmp
                  Bash   BASIC_BASH programs; python3 runs geomrun.py / scm_match.py / scm_lint.py / cam_helper.py / cam_track.py,
                         -c or a heredoc
                Each task keeps <task dir>/llm/<stage>.json (full transcript, no credential) and <stage>_calls.jsonl (one line per request).
    claude_cli  one isolated Claude Code process per task: `claude -p "<spawn message>"` started in the task's directory with the
                built-in tools Read / Write / Edit / Bash, no MCP servers, the stage's effort, and an environment without the
                caller's session variables (CLAUDE_CLI names the executable, default `claude`). Its JSON result is kept in
                <task dir>/llm/<stage>.cli.json.
state/llm_usage.jsonl gets one summary line per task. The effort of each stage is in state/llm_params.json (engine/cd/runs.py).

    python3 engine/ws/llm_run.py daemon [IDS] [--auto-every 120]   keep up to LLM_MAX_CONCURRENCY agents busy (deepest pipeline stage
                                                     first) and rerun `flexrun.py auto` periodically; stop by touching state/llm_daemon.stop
    python3 engine/ws/llm_run.py run N [--stages a,b]   run N agents from the newest state/agents_*.jsonl, then exit
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '@CD_CODE@/engine/llm')
import api_llm, api_agent  # noqa: E402
sys.path.insert(0, str(HERE))
import cc_agent  # noqa: E402  (the Claude Code subagent replica)
import flexrun as F  # noqa: E402
sys.path.insert(0, '@CD_CODE@/engine/cb'); import prompt_tpl as PT  # noqa: E402  (prompts/)

FR, ST = F.FR, F.ST
CB = Path('@CD_CODE@/engine/cb')
SCRATCH = Path('/tmp')   # scratch files of the agents (cd /tmp, /tmp/*.json); readable and writable
READ_ROOTS = [FR, HERE, CB, SCRATCH]
PY_SCRIPTS = {'geomrun.py', 'scm_match.py', 'scm_lint.py', 'cam_helper.py', 'cam_track.py'}
TASK = PT.text('agents/spawn_message.txt')   # the spawn message
MODE = os.environ.get('CD_AGENT_MODE', 'api')
MODEL = os.environ.get('LLM_MODEL', 'claude-opus-5-5')
_LOG = threading.Lock()


def run_cli(task, d, stage, effort):
    """one agent task as an isolated Claude Code process (CD_AGENT_MODE=claude_cli) -> (final text, info)"""
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE') and k not in ('PWD', 'OLDPWD', 'PYTHONPATH')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', DISABLE_AUTOUPDATER='1')
    tools = ['Read', 'Write', 'Edit', 'Bash']
    cmd = [os.environ.get('CLAUDE_CLI', 'claude'), '-p', task, '--model', MODEL, '--effort', effort, '--tools', *tools,
           '--allowedTools', *tools, '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json']
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=4 * 3600, env=env)
        res = json.loads(p.stdout) if p.stdout.strip().startswith('{') else {'is_error': True, 'result': (p.stdout + p.stderr)[-600:]}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        res = {'is_error': True, 'result': f'{type(e).__name__}: {e}'}
    res['seconds'] = round(time.time() - t0, 1)
    (d / 'llm').mkdir(parents=True, exist_ok=True); (d / 'llm' / f'{stage}.cli.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if res.get('is_error'):
        raise RuntimeError(str(res.get('result'))[:300])
    return str(res.get('result') or ''), dict(turns=res.get('num_turns'), usage=res.get('usage'), tools={'cli': 1}, finish=res.get('subtype'))


def run_task(stage, n, prompt, out, effort='medium'):
    """one agent task; returns a summary dict (status ok / no_output / truncated / no_tools / error)"""
    prompt, out = Path(prompt), Path(out); d = out.parent; t0 = time.time(); task = TASK.format(prompt)
    rec = dict(stage=stage, id=n, prompt=str(prompt), out=str(out), started=time.strftime('%Y-%m-%dT%H:%M:%S'), effort=effort, mode=MODE)
    try:
        if MODE == 'claude_cli':
            text, info = run_cli(task, d, stage, effort)
        else:
            text, info = cc_agent.run_agent(task, workdir=d, read_roots=READ_ROOTS, write_roots=[d, SCRATCH],
                                            transcript=d / 'llm' / f'{stage}.json', record=d / 'llm' / f'{stage}_calls.jsonl', tag=f'{stage} {n}',
                                            bash_allow=api_agent.BASIC_BASH, py_scripts=PY_SCRIPTS, effort=effort, max_tokens=128000, cache='5m')
        tok = api_llm._TOKEN or ''
        st = 'ok' if out.exists() else ('truncated' if info.get('finish') in ('max_tokens', 'length') else
                                         'no_tools' if not any((info.get('tools') or {}).values()) else 'no_output')
        rec.update(status=st, final=(text.replace(tok, '[REDACTED]') if tok else text)[:300], **{k: info.get(k) for k in ('turns', 'usage', 'tools', 'denied', 'finish')})
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}'[:500])
    rec['seconds'] = round(time.time() - t0, 1)
    with _LOG:
        with open(ST / 'llm_usage.jsonl', 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


def check_agent_mode():
    """checks the agent mode before any run: a known mode, and the Claude Code CLI when it is used."""
    if MODE == 'claude_cli' and not shutil.which(os.environ.get('CLAUDE_CLI', 'claude')):
        raise SystemExit('CD_AGENT_MODE=claude_cli: the Claude Code CLI was not found (set CLAUDE_CLI)')
    if MODE not in ('api', 'claude_cli'):
        raise SystemExit(f'CD_AGENT_MODE={MODE}: use api or claude_cli')


# ---------------------------------------------------------------- take agents from auto's list
def take(k, stages=None):
    """up to k not yet taken agents from the newest agents list, deepest stage first; marks them (state/<n>/agent_<stage>)"""
    files = sorted(ST.glob('agents_*.jsonl'))
    if not files or k <= 0:
        return []
    allow = set(stages) if stages else set(F.DEPTH)
    todo = [json.loads(line) for line in files[-1].read_text().splitlines() if line.endswith('}')]
    todo = sorted((x for x in todo if x['stage'] in allow), key=lambda x: (-F.DEPTH.index(x['stage']), x['id']))
    got, seen = [], set()
    for x in todo:
        if (x['stage'], x['id']) in seen:
            continue
        seen.add((x['stage'], x['id']))
        mk = ST / str(x['id']) / f"agent_{x['stage']}"; fail = mk.with_suffix('.fail'); tech = mk.with_suffix('.tech')
        if Path(x['out']).exists() or mk.exists() or (fail.exists() and int(fail.read_text()) >= 2) or (tech.exists() and int(tech.read_text()) >= 3) \
                or not Path(x['prompt']).exists():
            continue
        mk.parent.mkdir(parents=True, exist_ok=True); mk.write_text(x['out']); got.append(x)
        if len(got) >= k:
            break
    return got


def finished(x, rec):
    """clear the mark of a failed task so that it is taken again. Technical failures (the API gave up, or the conversation ended
    without a single tool call) are retried up to 3 times; an agent that worked but wrote no output gets 2 tries; after that the
    task is left for a look."""
    if rec['status'] == 'ok':   # the agent has ended: its output is final (auto waits for this, not for the file alone)
        (ST / str(x['id']) / f"agent_{x['stage']}.done").write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))
        return
    mk = ST / str(x['id']) / f"agent_{x['stage']}"; tech = rec['status'] in ('error', 'no_tools', 'truncated')
    fail = mk.with_suffix('.tech' if tech else '.fail'); lim = 3 if tech else 2
    k = int(fail.read_text()) + 1 if fail.exists() else 1; fail.write_text(str(k))
    if k < lim:
        mk.unlink(missing_ok=True)
    print(f"  {x['stage']} {x['id']}: {rec['status']} ({k}/{lim}) {rec.get('error', '')[:120]}", flush=True)


def effort_of(stage):
    prm = json.loads((ST / 'llm_params.json').read_text()) if (ST / 'llm_params.json').exists() else {}
    return prm.get(stage, {}).get('effort', 'medium')


def run(a):
    check_agent_mode()
    xs = take(int(a.ids), a.stages.split(',') if a.stages else None)
    with ThreadPoolExecutor(max(1, api_llm.max_concurrency())) as ex:
        for x, rec in zip(xs, ex.map(lambda x: run_task(x['stage'], x['id'], x['prompt'], x['out'], effort_of(x['stage'])), xs)):
            finished(x, rec); print(x['stage'], x['id'], rec['status'], rec['seconds'], flush=True)


def _procs(script, cmd):
    """(pid, argv) of the python processes of this run tree running `<script> <cmd> ...`"""
    for c in Path('/proc').glob('[0-9]*/cmdline'):
        if c.parent.name == str(os.getpid()):
            continue
        try:
            argv = [x.decode(errors='replace') for x in c.read_bytes().split(b'\0') if x]; env = (c.parent / 'environ').read_bytes().split(b'\0')
        except OSError:
            continue
        if len(argv) >= 3 and Path(argv[0]).name.startswith('python') and argv[1].endswith(script) and argv[2] == cmd \
                and (f'CD_TREE={F.SUB}'.encode() in env) == bool(F.SUB):
            yield int(c.parent.name), argv


def daemon(a):
    """agents and auto in one loop: the pool is refilled every 20 s; `flexrun.py auto` reruns every --auto-every seconds"""
    other = next((pid for pid, _ in _procs('llm_run.py', 'daemon')), None)
    if other:
        raise SystemExit(f'another daemon of this tree is running (pid {other}); stop it first (state/llm_daemon.stop, then wait until it exits)')
    check_agent_mode()
    k = 0   # a taken-mark without its output is left over from a daemon stopped mid-task: clear it so the task is taken again
    for mk in ST.glob('*/agent_*'):
        if '.' not in mk.name and not Path(mk.read_text().strip() or '/nonexistent').exists():
            mk.unlink(); k += 1
    print(f'{k} leftover agent marks cleared', flush=True)
    stop = ST / 'llm_daemon.stop'; stop.unlink(missing_ok=True); stages = a.stages.split(',') if a.stages else None
    lanes = max(1, api_llm.max_concurrency()); ids = a.ids or 'all'
    auto_p, auto_t = None, 0.0; running = {}
    with ThreadPoolExecutor(lanes) as ex:
        while True:
            if (auto_p is None or auto_p.poll() is not None) and time.time() - auto_t > a.auto_every and not stop.exists():
                if any(argv[3:4] == [ids] for _, argv in _procs('flexrun.py', 'auto')):   # this tree's auto still running
                    auto_t = time.time(); continue
                log = open(ST / f"auto_{time.strftime('%m%d%H%M%S')}.log", 'w')
                auto_p = subprocess.Popen([sys.executable, str(HERE / 'flexrun.py'), 'auto', ids], stdout=log, stderr=subprocess.STDOUT)
                auto_t = time.time(); print(time.strftime('%H:%M'), 'auto started', flush=True)
            if not stop.exists():
                free = lanes - len(running); ent = sum(1 for x in running.values() if x['stage'] == 'entities')
                picks = take(min(free, 2 - ent), ['entities']) if free > 0 and ent < 2 and (not stages or 'entities' in stages) else []
                for x in picks + take(free - len(picks), stages):   # two lanes keep the shallow end moving; the rest go deepest stage first
                    running[ex.submit(run_task, x['stage'], x['id'], x['prompt'], x['out'], effort_of(x['stage']))] = x
            if stop.exists() and not running:
                print('stopped'); return
            done, _ = wait(list(running), timeout=20, return_when=FIRST_COMPLETED) if running else (set(), None)
            for fu in done:
                x = running.pop(fu); rec = fu.result(); finished(x, rec)
                print(time.strftime('%H:%M'), x['stage'], x['id'], rec['status'], rec['seconds'], 's', flush=True)
            if not running:
                time.sleep(20)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('daemon', 'run')); ap.add_argument('ids', nargs='?', default='')
    ap.add_argument('--stages', default=''); ap.add_argument('--auto-every', type=int, default=120)
    a = ap.parse_args(); {'daemon': daemon, 'run': run}[a.cmd](a)


if __name__ == '__main__':
    main()
