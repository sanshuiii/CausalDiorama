"""Generation: prompt -> Causal Diorama -> video (the generation pipeline of the paper, packed as a Causal Diorama).

  plan      one context-free planning agent (prompts/agents/planner_t2v.md; Claude Opus 5.5, effort medium) in its own workspace
            with the schema and a copy of the checker (scm_lint.py + rules_pv2.json), which it runs itself, at most 3 times; a plan
            that still fails the checker is used as written; no plan file = planning failure (technical failures are retried, at most
            3 attempts). CD_AGENT_MODE=api (default): an API tool loop on the Messages API (max_tokens 32000, thinking kept across
            turns); CD_AGENT_MODE=claude_cli: one isolated `claude -p` process in the workspace
  prepare   pv2.augment (one picture per entity + the empty place), frame-0 boxes, stage-1 job and stage-2 spec
  GPU       one chain of node tasks: Qwen-Image-2.1 pictures -> white model -> stage 1 (free H3 Ref2VA + CWM LoRA, frame 0 kept)
            -> stage-2 case + caches -> stage 2 (locked first frame) -> video
  pack      the Causal Diorama (engine/cd/diorama.py) with its pictures, first frame and video
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

from . import paths, diorama as D

sys.path.insert(0, str(paths.ENGINE / 'cb')); sys.path.insert(0, str(paths.ENGINE / 'llm'))
import pv2  # noqa: E402

P = paths.NODE
PYN = f'{P}/cwm_h3_reproduction_v1/.venv/bin/python'
TORCHRUN = f'{P}/cwm_h3_reproduction_v1/.venv/bin/torchrun'
PYDEPS, QDEPS = f'{P}/compbench_v1/pydeps', f'{P}/qwen_image21/pydeps'
REL = f'{P}/cwm_h3_reproduction_v1/release/models'
MODELS = {'dit': f'{REL}/diffusion_models/minimax_h3_ref2va_bf16.safetensors', 'lora': f'{REL}/lora/minimax-h3-cwm.safetensors',
          'video_vae': f'{REL}/vae/minimax_h3_video_vae_fp16.safetensors', 'processor': f'{REL}/processors/Qwen3-VL-32B-Instruct',
          'text_encoder': f'{REL}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors'}
SYS1 = (paths.ENGINE / 'cb' / 'system_stage1.txt').read_text().strip()
SYS2 = (paths.ENGINE / 'cb' / 'system_stage2.txt').read_text().strip()
SEED = paths.SEED
CWM = f'{paths.SHARED}/tools/cwm'
PRE = (f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat {PYN} {paths.SHARED}/tools/gpu_preclaim.py 15')
RESULTS = paths.SHARED / 'results'
ARM_S1, ARM = 'cd_gen_s1', 'cd_gen'
ASK = (paths.PROMPTS / 'agents' / 'planner_t2v_ask.txt').read_text().strip()   # the planning agent's spawn message
EFFORT, MAX_TOKENS, TRIES = 'medium', 32000, 3


def gdir(run):
    return paths.WORK / 'gen' / run


# ------------------------------------------------------------------ planning agent
def checker(run):
    t = gdir(run) / 'checker'; t.mkdir(parents=True, exist_ok=True)
    for f in ('scm_lint.py', 'world.py', 'rules_pv2.json'):
        shutil.copy2(paths.ENGINE / 'cb' / f, t / f)
    return t


def workspace(d, sid, prompt, tools):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    old = f'{paths.ENGINE}/cb/scm_schema_pv2.md'
    t = (paths.PROMPTS / 'agents' / 'planner_t2v.md').read_text(); assert t.count(old) == 1
    lint = f'python3 {tools}/scm_lint.py {d}/plan.json --rules {tools}/rules_pv2.json'
    t = (t.replace(old, str(d / 'schema.md')).replace('@@PROMPT@@', prompt).replace('@@OUT@@', str(d / 'plan.json'))
         .replace('@@SID@@', sid).replace('@@LINT@@', lint))
    assert '@@' not in t
    (d / 'task.md').write_text(t); shutil.copy2(paths.ENGINE / 'cb' / 'scm_schema_pv2.md', d / 'schema.md')


def plan_cli(d, tools, transcript):
    """the planning agent as an isolated Claude Code process (CD_AGENT_MODE=claude_cli): writes and edits inside its workspace only,
    Bash only for the checker, the checker's code not readable -> (final text, info)"""
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE') and k not in ('PWD', 'OLDPWD', 'PYTHONPATH')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', DISABLE_AUTOUPDATER='1')
    cmd = [os.environ.get('CLAUDE_CLI', 'claude'), '-p', ASK, '--model', os.environ.get('LLM_MODEL', 'claude-opus-5-5'), '--effort', EFFORT,
           '--tools', 'Read', 'Write', 'Edit', 'Bash', '--allowedTools', f'Write(/{d}/**)', f'Edit(/{d}/**)', f'Bash(python3 {tools}/scm_lint.py:*)',
           '--disallowedTools', f'Read(/{tools}/**)', '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json']
    p = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=3600, env=env)
    res = json.loads(p.stdout) if p.stdout.strip().startswith('{') else {'is_error': True, 'result': (p.stdout + p.stderr)[-600:]}
    Path(transcript).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if res.get('is_error'):
        raise RuntimeError(str(res.get('result'))[:300])
    return str(res.get('result') or ''), dict(turns=res.get('num_turns'), usage=res.get('usage'), tools={'cli': 1})


def plan(run, prompt, log=print):
    import site
    import api_agent as AG
    tools = checker(run); g = gdir(run); rec = []
    env = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUSERBASE=site.getuserbase())
    mode = os.environ.get('CD_AGENT_MODE', 'api')
    for k in range(TRIES):
        d = g / 'plan' / (f'a{k}'); workspace(d, 'scene', prompt, tools)
        kw = dict(workdir=d, read_roots=[d], write_roots=[d], transcript=g / 'plan' / f'a{k}.json', record=g / 'plan' / 'calls.jsonl',
                  tag=f'planner {run} a{k}', bash_allow=(), py_scripts=(), py_inline=False, bash_prefixes=[f'python3 {tools}/scm_lint.py '],
                  bash_env=env, effort=EFFORT, max_tokens=MAX_TOKENS, protocol='anthropic')
        try:
            if mode == 'claude_cli':
                text, info = plan_cli(d, tools, g / 'plan' / f'a{k}.cli.json')
            else:
                text, info = AG.run_agent(ASK.replace('the file task.md in the current directory', f'the file {d}/task.md'), **kw)
        except Exception as e:
            rec.append(dict(attempt=k, status='technical', error=f'{type(e).__name__}: {e}'[:300])); log('planner technical failure:', rec[-1]['error']); continue
        if not any((info.get('tools') or {}).values()):
            rec.append(dict(attempt=k, status='technical', error='no tool call')); continue
        try:
            json.loads((d / 'plan.json').read_text())
        except Exception:
            rec.append(dict(attempt=k, status='planfail')); break
        try:
            lp = bool(json.loads((d / 'plan.lint' / 'lint.json').read_text()).get('passed'))
        except Exception:
            lp = None
        rec.append(dict(attempt=k, status='done', lint_passed=lp, turns=info.get('turns'), usage=info.get('usage'))); break
    (g / 'plan' / 'attempts.json').write_text(json.dumps(rec, indent=1))
    if not rec or rec[-1]['status'] != 'done':
        raise SystemExit(f'planning failed: {rec[-1] if rec else "no attempt"} (see {g / "plan"})')
    shutil.copy2(d / 'plan.json', g / 'scm_t2v.json')
    if (d / 'plan.lint').exists():
        shutil.copytree(d / 'plan.lint', g / 'scm_t2v.lint', dirs_exist_ok=True)
    log('plan written:', g / 'scm_t2v.json', '| checker passed' if rec[-1].get('lint_passed') else '| checker did not pass (used as written)')
    return json.loads((g / 'scm_t2v.json').read_text())


# ------------------------------------------------------------------ preparation
def prepare(run, A, pictures_from=None):
    g = gdir(run); pics = g / 'pics'; pics.mkdir(parents=True, exist_ok=True)
    A2, _ = pv2.augment(A)
    (g / 'scm_t2v_v2.json').write_text(json.dumps(A2, ensure_ascii=False, indent=1))
    f0 = json.loads(json.dumps(pv2.frame0(A2)))
    plan = json.loads((g / 'scm_t2v_v2.json').read_text())
    if pictures_from:   # a diorama's own pictures (video from an existing diorama): the worker skips existing outputs
        for k, p in pictures_from.items():
            shutil.copy2(p, pics / f'{k}.png')
    jobs = []
    for key, x in pv2.ref_table(plan).items():
        assert x['source'] == 't2v', (key, x['source'])
        spec = dict(id=f'{run}_{key}', prompt=x['prompt'], width=x['size'][0], height=x['size'][1], seed=SEED, steps=40, out=str(pics / f'{key}.png'))
        jobs.append(write(g / 'jobs' / 'pics' / f'{key}.json', spec))
    nr = f'{paths.NODE_RUNS}/gen/{run}'; render = f'{nr}/render'
    cap1, keys1 = pv2.caption(plan, f0, 1); d1 = f'{nr}/s1'
    s1 = dict(job_id=f'cd_s1_{run}', frames=124, seed=SEED, steps=20, lora=True, models=MODELS, condition_root=f'{render}/conditions',
              system_prompt=SYS1, caption=cap1, images=[{'key': k, 'path': str(pics / f'{k}.png')} for k in keys1],
              cache_dir=f'{d1}/cache', output=f'{d1}/generated.mp4', frame0=f'{d1}/frame0.png', export={'arm': ARM_S1, 'category': 'gen', 'name': run})
    cap2, keys2 = pv2.caption(plan, f0, 2); case_dir = f'{nr}/s2'
    s2 = dict(case_dir=case_dir, plan=str(g / 'scm_t2v_v2.json'), render_dir=render, anchor_from=str(RESULTS / ARM_S1 / 'gen' / f'{run}.png'),
              seed=SEED, images=[{'key': k, 'path': str(pics / f'{k}.png')} for k in keys2], system_prompt=SYS2, caption=cap2)
    return dict(plan=plan, pics=jobs, render_line=f"{g / 'scm_t2v_v2.json'} {render}", s1=write(g / 'jobs' / 's1.json', s1),
                s2=write(g / 'jobs' / 's2.json', s2), cfg=f'{case_dir}/h3_config.json',
                gen=f'{case_dir}/h3_config.json {case_dir}/A/generated.mp4 {ARM} gen {run}', frame0=f0)


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, ensure_ascii=False, indent=1)); return str(p)


# ------------------------------------------------------------------ node tasks
def put(node, tid, task):
    q = paths.QUEUE / 'queue' / node / 'main'; q.mkdir(parents=True, exist_ok=True)
    if (q / f'{tid}.json').exists():
        raise SystemExit(f'task {tid} is already queued on {node}')
    tmp = q.parent / f'.{tid}.tmp'; tmp.write_text(json.dumps(task, ensure_ascii=False, indent=1)); tmp.rename(q / f'{tid}.json')


def state(node, tid):
    ev = paths.QUEUE / 'status' / node / 'events.log'; st = 'pending'
    for line in (ev.read_text().splitlines() if ev.exists() else []):
        w = line.split()
        if len(w) >= 3 and w[2] == tid and w[1] in ('DONE', 'FAILED'):
            st = w[1].lower()
    return st


def queue(run, node, prep, video=True, stamp=None, start='a'):
    """start: 'a' all steps, 'c' from the white model (pictures done), 'h' from stage 2 (first frame done)"""
    g = gdir(run); lst = g / 'jobs'; stamp = stamp or time.strftime('%m%d%H%M')
    pre = lambda s: f'g{s}_{run}_{stamp}'  # noqa: E731
    tasks = [(pre('a') + '_pre', dict(type='shell', retries=0, command=PRE)),
             (pre('b') + '_qwen', dict(type='workers', script='qwen_image/worker.py', pythonpath=QDEPS, configs=prep['pics'], after=[pre('a') + '_pre'])),
             (pre('c') + '_render', dict(type='workers', script='e2/render.py', pythonpath=PYDEPS, configs=[prep['render_line']]))]
    if video:
        (lst / 's1.txt').write_text(prep['s1'] + '\n'); (lst / 'gen.txt').write_text(prep['gen'] + '\n')
        tasks += [(pre('e') + '_pre', dict(type='shell', retries=0, command=PRE)),
                  (pre('f') + '_s1prep', dict(type='workers', script='cwm/ffprep.py', configs=[prep['s1']], after=[pre('e') + '_pre'])),
                  (pre('g') + '_s1gen', dict(type='shell', retries=2, command=(
                      f'{PRE}; cd {CWM} && {TORCHRUN} --standalone --nproc_per_node=8 {CWM}/ffgen_many.py --jobs {lst}/s1.txt '
                      f'--summary {lst}/s1_summary.json --parallel-decode --results-root {RESULTS}'))),
                  (pre('h') + '_pre', dict(type='shell', retries=0, command=PRE)),
                  (pre('i') + '_s2prep', dict(type='workers', script='e2/pv2_case_prep.py', pythonpath=PYDEPS, configs=[prep['s2']], after=[pre('h') + '_pre'])),
                  (pre('j') + '_pre', dict(type='shell', retries=0, command=PRE)),
                  (pre('k') + '_cache', dict(type='workers', script='e2/cwm_prepare.py', configs=[prep['cfg']], after=[pre('j') + '_pre'])),
                  (pre('l') + '_gen', dict(type='shell', retries=2, command=(
                      f'{PRE}; cd {CWM} && {TORCHRUN} --standalone --nproc_per_node=8 {CWM}/generate_many.py --list {lst}/gen.txt '
                      f'--summary {lst}/gen_summary.json --parallel-decode')))]
    tasks = [(tid, dict(t, after=[x for x in t.get('after', []) if x[1] >= start])) for tid, t in tasks if tid[1] >= start]
    for tid, t in tasks:
        put(node, tid, t)
    write(g / 'jobs' / 'queued.json', dict(node=node, tasks=[t for t, _ in tasks]))
    return [t for t, _ in tasks]


def wait(node, tids, outputs, timeout_h=6.0, log=print):
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        sts = {t: state(node, t) for t in tids}
        bad = [t for t, s in sts.items() if s == 'failed']
        if bad:
            raise SystemExit(f'node task failed: {bad} (see {paths.QUEUE / "status" / node / "events.log"})')
        if all(s == 'done' for s in sts.values()):
            miss = [str(o) for o in outputs if not Path(o).exists()]
            if miss:
                raise SystemExit(f'node tasks done but outputs missing: {miss}')
            return True
        time.sleep(30)
    raise SystemExit('timeout waiting for the node tasks')


def outputs(run):
    return dict(frame0=RESULTS / ARM_S1 / 'gen' / f'{run}.png', video=RESULTS / ARM / 'gen' / f'{run}.mp4')


def pack(run, out, prompt, prep, video=True):
    g = gdir(run); o = outputs(run); plan = prep['plan']
    pics = {}
    for k in D.reference_keys(plan):
        node = next(x for x in [plan['background']] + plan['E'] if (x.get('reference') or {}).get('key') == k)
        pics[k] = dict(path=g / 'pics' / f'{k}.png', source=node['reference']['source'], prompt=node['reference'].get('prompt'))
    return D.pack(out, id=Path(out).name, kind='generated', scm=plan, geometry=dict(frames=124, fps=24, width=1344, height=768),
                  pictures=pics, first_frame=o['frame0'] if video else None, first_frame_source='stage 1 (free H3 Ref2VA + CWM LoRA), frame 0',
                  video=o['video'] if video else None, video_source='stage 2 (H3 Ref2VA + CWM LoRA, locked first frame)',
                  text=dict(prompt=prompt), generation=dict(seed=SEED, steps=20, model='MiniMax H3 Ref2VA + CWM LoRA',
                                                            system_prompts=['engine/cb/system_stage1.txt', 'engine/cb/system_stage2.txt']),
                  pipeline='gen')
