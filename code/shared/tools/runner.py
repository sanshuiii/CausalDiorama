"""Per-node task scheduler of the Causal Diorama pipelines. One runner per node owns all 8 GPUs and runs the queued tasks one
after another.

Queue (written from the workstation, read-only here):
  RUNNER_SHARED/queue/<node>/{main,idle}/<task_id>.json      (RUNNER_SHARED: the configured queue directory, @CD_QUEUE@)
Local state: @CD_NODE@/compbench_v1/state/{done,failed}/<task_id>.json; logs in @CD_NODE@/compbench_v1/logs
Status mirror: RUNNER_SHARED/status/<node>/{runner.json,events.log}
Node code: @CD_SHARED@/tools is synced to @CD_NODE@/compbench_v1/tools before every task; every command runs after
`source tools/node_env.sh` (the node's Python, CUDA libraries and offline flags).

Priority: a runnable main task always goes first; idle tasks run only while no main task is runnable. A task runs when every task
named in its `after` list is done.
Task types: workers (a script per GPU over a list of job files, shards of t['configs']), shell (one command on the whole node).
Stop: touch RUNNER_SHARED/queue/<node>/STOP (checked between tasks).
"""
import json, os, shutil, socket, subprocess, sys, time, traceback
from pathlib import Path

NODE = sys.argv[1]
SHARED = Path(os.environ.get('RUNNER_SHARED', '@CD_QUEUE@'))   # the queue directory
Q = SHARED / 'queue' / NODE
STATUS = SHARED / 'status' / NODE
P = Path('@CD_NODE@')
WORK = P / 'compbench_v1'
STATE = WORK / 'state'
TOOLS = WORK / 'tools'
ENV = f'source {TOOLS}/node_env.sh'
for d in (STATE / 'done', STATE / 'failed', STATUS, WORK / 'logs'):
    d.mkdir(parents=True, exist_ok=True)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def event(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(STATUS / 'events.log', 'a') as f:
        f.write(line + '\n')


def write_status(**kw):
    counts = {s: len(list((STATE / s).glob('*.json'))) for s in ('done', 'failed')}
    pending = {p: len([t for t in load_tasks(p) if not finished(t['id'])]) for p in ('main', 'idle')}
    doc = dict(node=NODE, host=socket.gethostname(), updated=now(), counts=counts, pending=pending, **kw)
    tmp = STATUS / 'runner.json.tmp'
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, STATUS / 'runner.json')


def load_tasks(priority):
    out = []
    for f in sorted((Q / priority).glob('*.json')):
        try:
            t = json.loads(f.read_text())
            t['id'] = f.stem
            t['priority'] = priority
            out.append(t)
        except Exception:
            pass  # partially written file; picked up next round
    return out


def finished(tid):
    return (STATE / 'done' / f'{tid}.json').exists() or (STATE / 'failed' / f'{tid}.json').exists()


def done(tid):
    return (STATE / 'done' / f'{tid}.json').exists()


def pick():
    for priority in ('main', 'idle'):
        for t in load_tasks(priority):
            if not finished(t['id']) and all(done(d) for d in t.get('after', [])):
                return t
    return None


def sh(cmd, log, cwd=None, env_prefix=ENV):
    with open(log, 'a') as f:
        f.write(f'\n#### {now()} {cmd}\n')
        f.flush()
        return subprocess.call(['bash', '-c', f'{env_prefix} >/dev/null 2>&1; {cmd}'], stdout=f, stderr=subprocess.STDOUT, cwd=cwd)


def run_workers(t, worker_cmd, log):
    """Split t['configs'] over up to 8 single-GPU workers; each worker gets a file with its config paths."""
    cfgs = t['configs']
    n = min(8, len(cfgs))
    shards = [cfgs[i::n] for i in range(n)]
    procs = []
    for i, shard in enumerate(shards):
        lst = WORK / 'logs' / f"{t['id']}.shard{i}.txt"
        lst.write_text('\n'.join(shard) + '\n')
        f = open(f'{log}.w{i}', 'a')
        procs.append((subprocess.Popen(['bash', '-c', f'{ENV} >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES={i}; {worker_cmd} {lst}'],
                                       stdout=f, stderr=subprocess.STDOUT), f))
    codes = []
    for p, f in procs:
        codes.append(p.wait())
        f.close()
    return max(codes) if codes else 0


def execute(t, log):
    typ = t['type']
    if typ == 'workers':  # one line per item, one process per GPU, e.g. qwen_image/worker.py, e2/pv2_case_prep.py
        pp = t.get('pythonpath', '')
        return run_workers(t, f"{'PYTHONPATH=' + pp + ' ' if pp else ''}$PY {TOOLS}/{t['script']}", log)
    if typ == 'shell':
        return sh(t['command'], log, cwd=t.get('cwd'))
    raise ValueError(f'unknown task type {typ}')


def sync_tools():
    subprocess.call(['rsync', '-a', '@CD_SHARED@/tools/', f'{TOOLS}/'])


def main():
    sync_tools()
    event(f'RUNNER_START pid={os.getpid()}')
    while True:
        if (Q / 'STOP').exists() or (Q / 'RESTART').exists():
            event('RUNNER_STOP (STOP/RESTART file)')
            write_status(state='stopped')
            return
        t = pick()
        if t is None:
            write_status(state='idle')
            time.sleep(30)
            continue
        sync_tools()  # worker scripts can be updated between tasks without restarting the runner
        log = WORK / 'logs' / f"{t['id']}.log"
        tries, code, t0 = 0, 1, time.time()
        while tries < 1 + t.get('retries', 1):
            tries += 1
            write_status(state='running', task=t['id'], priority=t['priority'], type=t['type'], started=now(), attempt=tries)
            event(f"START {t['priority']} {t['id']} attempt={tries}")
            try:
                code = execute(t, log)
            except Exception:
                code = 99
                with open(log, 'a') as f:
                    f.write(traceback.format_exc())
            if code == 0:
                break
            event(f"ATTEMPT_FAILED {t['id']} code={code}")
        res = dict(id=t['id'], priority=t['priority'], type=t['type'], code=code, attempts=tries, seconds=round(time.time() - t0, 1), finished=now())
        (STATE / ('done' if code == 0 else 'failed') / f"{t['id']}.json").write_text(json.dumps(res))
        event(f"{'DONE' if code == 0 else 'FAILED'} {t['id']} {res['seconds']}s")
        if code:  # a workers task has per-worker logs (<log>.w<i>) and no <log>; copy whatever exists
            for src in [log] + sorted(log.parent.glob(log.name + '.w*')):
                if src.exists():
                    try:
                        shutil.copy2(src, STATUS / f"{t['id']}.failed{src.name[len(log.name):]}.log")
                    except OSError:
                        pass


if __name__ == '__main__':
    main()
