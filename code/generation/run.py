"""Queue the generation of prepared prompts (prepare.py) on one GPU node, as one chain of runner tasks.

The node runner (code/shared/tools/runner.py, one per 8-GPU node) reads tasks from <queue>/<node>/main/<task>.json and runs one
task at a time, the first runnable one by name; `after` lists the tasks that must be done first. The chain of one wave:
    a  GPU claim                                     b  reference pictures (Qwen-Image worker, one process per GPU)
    c  white-model renders (CPU)                     e  GPU claim
    f  stage-1 condition caches                      g  stage-1 generation (8 GPUs, frame 0 exported)
    h  GPU claim                                     i  stage-2 preparation (first frame, pictures, caption, conditions)
    j  GPU claim                                     k  stage-2 condition caches
    l  stage-2 generation (8 GPUs): the final videos, exported to <results>/video/<category>/<name>.mp4
A GPU claim holds a little memory on every GPU for a few seconds, so that a node-level GPU keeper that takes idle GPUs adopts
the claim and not the job that follows it.

    python generation/run.py --config config.json --node NODE --wave w1 --ids a,b
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GPU_CLAIM_SECONDS, jdump, load_config  # noqa: E402


def put(cfg, node, tid, task):
    q = Path(cfg['queue_root']) / node / 'main'; q.mkdir(parents=True, exist_ok=True)
    if (q / f'{tid}.json').exists():
        raise SystemExit(f'task {tid} is already queued on {node}; use a new --wave')
    tmp = q.parent / f'.{tid}.tmp'; tmp.write_text(json.dumps(task, ensure_ascii=False, indent=1)); tmp.rename(q / f'{tid}.json')


def wave(cfg, node, wid, ids):
    work = Path(cfg['work_root']); rec = json.loads((work / 'prepared.json').read_text())
    ok = [(pid, rec[pid]) for pid in ids if rec.get(pid, {}).get('state') == 'prepared']
    if not ok:
        raise SystemExit('none of the prompts is prepared')
    claim = f"{cfg['node_python']} {cfg['node_tools']}/gpu_preclaim.py {GPU_CLAIM_SECONDS}"
    cwm = f"{cfg['node_tools']}/cwm"; results = cfg['results_root']
    lst = work / 'jobs' / 'lists'; summ = work / 'jobs' / 'summaries'; summ.mkdir(parents=True, exist_ok=True)
    (lst / f'{wid}_{node}_s1.txt').parent.mkdir(parents=True, exist_ok=True)
    (lst / f'{wid}_{node}_s1.txt').write_text('\n'.join(r['s1'] for _, r in ok) + '\n')
    (lst / f'{wid}_{node}_gen.txt').write_text('\n'.join(r['gen'] for _, r in ok) + '\n')
    name = lambda s: f'g{wid}{s}_{node}'  # noqa: E731
    chain = [
        ('a_claim', dict(type='shell', retries=0, command=claim)),
        ('b_pictures', dict(type='workers', script='qwen_image/worker.py', pythonpath=cfg.get('qwen_pydeps', ''), configs=[p for _, r in ok for p in r['pics']])),
        ('c_render', dict(type='workers', script='e2/render.py', pythonpath=cfg.get('pydeps', ''), configs=[r['render_line'] for _, r in ok])),
        ('e_claim', dict(type='shell', retries=0, command=claim)),
        ('f_s1cache', dict(type='workers', script='cwm/ffprep.py', configs=[r['s1'] for _, r in ok])),
        ('g_s1gen', dict(type='shell', retries=2, command=(
            f"{claim}; cd {cwm} && {cfg['node_torchrun']} --standalone --nproc_per_node=8 {cwm}/ffgen_many.py "
            f"--jobs {lst}/{wid}_{node}_s1.txt --summary {summ}/{wid}_{node}_s1.json --parallel-decode --results-root {results}"))),
        ('h_claim', dict(type='shell', retries=0, command=claim)),
        ('i_s2prep', dict(type='workers', script='e2/pv2_case_prep.py', pythonpath=cfg.get('pydeps', ''), configs=[r['s2'] for _, r in ok])),
        ('j_claim', dict(type='shell', retries=0, command=claim)),
        ('k_s2cache', dict(type='workers', script='e2/cwm_prepare.py', configs=[r['cfg'] for _, r in ok])),
        ('l_s2gen', dict(type='shell', retries=2, command=(
            f"{claim}; cd {cwm} && {cfg['node_torchrun']} --standalone --nproc_per_node=8 {cwm}/generate_many.py "
            f"--list {lst}/{wid}_{node}_gen.txt --summary {summ}/{wid}_{node}_gen.json --parallel-decode --results-root {results}"))),
    ]
    prev = None
    for step, task in chain:
        if prev:
            task['after'] = [prev]
        put(cfg, node, name(step), task); prev = name(step)
    for pid, r in ok:
        r.update(node=node, wave=wid)
    jdump(work / 'prepared.json', rec)
    print('queued wave', wid, 'on', node, ':', len(ok), 'prompts,', len(chain), 'tasks')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', required=True); ap.add_argument('--node', required=True); ap.add_argument('--wave', required=True)
    ap.add_argument('--ids', required=True)
    a = ap.parse_args()
    if not re.fullmatch(r'[0-9a-z]{1,8}', a.wave):
        ap.error('--wave: 1-8 lowercase letters or digits')
    wave(load_config(a.config), a.node, a.wave, [x for x in a.ids.split(',') if x])


if __name__ == '__main__':
    main()
