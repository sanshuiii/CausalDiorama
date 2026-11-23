"""Run trees of the editing engine (engine/ws/flexrun.py) for `cd_edit.py` and `cd_revise.py`.

A run is one data tree WORK/fr/run_<run>/ with its own runs/ (task list) and state/: one clip, id N = 1. The engine runs as is:
`llm_run.py daemon` keeps the agents busy and reruns `flexrun.py auto` every 120 s, which queues the GPU tasks for the node runners.
Every tree uses the same rules: the editor addendum, empty-place rule v2, the first-frame route route_flex, and the agent settings
LLM_PARAMS. A revision tree starts with plan A, its pictures, its first frame and its video already in place (`seed_known_a`), so the
engine begins at the editor; its video B is regenerated with the seed of video A (flexrun.regen_b) instead of the F4 inversion.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

from . import paths, diorama as D

N = 1
LLM_PARAMS = {'entities': {'effort': 'medium'}, 'pcheck': {'effort': 'xhigh'}, 'planner': {'effort': 'medium'}, 'editor': {'effort': 'medium'},
              'bgcheck2': {'effort': 'high'}, 'v4ins': {'effort': 'medium'}, 'bgcheckB': {'effort': 'high'}}   # effort per agent stage


def sub(run):
    return f'run_{run}'   # flat: the engine puts the tree name into file names


def root(run):
    return paths.WORK / 'fr' / sub(run)


def probe(video):
    e = subprocess.run([paths.FFMPEG, '-i', str(video)], capture_output=True, text=True).stderr
    h, m, s = re.search(r'Duration: (\d+):(\d+):([\d.]+)', e).groups(); w, hh = map(int, re.search(r'Video: .*?(\d{2,5})x(\d{2,5})', e).groups())
    fps = float(re.search(r'([\d.]+) fps', e).group(1))
    return int(h) * 3600 + int(m) * 60 + float(s), w, hh, fps


def new_tree(run, video, description, instruction, no_video=False):
    """runs/ and state/ of a one-clip tree"""
    r = root(run)
    if r.exists():
        raise SystemExit(f'{r} exists: choose another --run name or remove it')
    runs, st = r / 'runs', r / 'state'; runs.mkdir(parents=True); st.mkdir(parents=True)
    inp = paths.WORK / 'inputs' / run; inp.mkdir(parents=True, exist_ok=True)
    src = inp / Path(video).name; shutil.copy2(video, src)
    dur, w, h, fps = probe(src)
    (runs / 'checklist.json').write_text(json.dumps([dict(id=N, videoA_path=str(src), original_description=description,
                                                          editing_instruction=instruction)], ensure_ascii=False, indent=1))
    (runs / 'sources.json').write_text(json.dumps([dict(id=N, file=str(src), dur=round(dur, 2), w=w, h=h, fps=fps)], indent=1))
    if no_video:
        (runs / 'no_video').write_text('')   # --outputs diorama: the engine stops before the video of plan B
    (st / 'llm_params.json').write_text(json.dumps(LLM_PARAMS, indent=1))
    return r


def env(run):
    return dict(os.environ, CD_TREE=sub(run))


def engine(run, *args, check=True):
    """python3 engine/ws/<module>.py args... on this run's tree"""
    return subprocess.run([sys.executable, str(paths.ENGINE / 'ws' / args[0])] + list(args[1:]), env=env(run), check=check)


def seed_known_a(run, dio_path):
    """revision: plan A, its pictures, its first frame, its video and the seed of its video come from a Causal Diorama. The A-side
    steps of the editing pipeline (entity list, perception and its check, measured camera, fitting planner, empty-place removal on
    frame 0) are marked done with what the diorama holds; the B side runs as in editing, and video B is regenerated with the seed of
    video A (runs/known_A.json)."""
    p, m = D.load(dio_path); r = root(run); n = str(N)
    seed = (m.get('generation') or {}).get('seed')
    if seed is None:
        raise SystemExit('the diorama records no seed of its video (generation.seed): video B cannot be regenerated from the same noise')
    plan, pics, st = r / 'plan' / n, r / 'pics' / n, r / 'state' / n
    for d in (plan, pics, st, r / 'percept' / n, r / 'percept' / f'{n}_checked', r / 'checks' / 'percept' / n):
        d.mkdir(parents=True, exist_ok=True)
    scm = json.loads(json.dumps(m['scm']))
    for node in [scm.get('background') or {}] + scm['E']:
        node.pop('reference', None)   # flexrun.assets adds the references again (pv2.augment, the same values as at generation)
    (plan / 'scm_A.json').write_text(json.dumps(scm, ensure_ascii=False, indent=1))
    ents = [dict(id=e['id'], label=e.get('label', e['id']), text=e.get('label', e['id'].replace('_', ' ')),
                 alt_text=e['id'].replace('_', ' '), type=e.get('type', 'object'), moving=False, visible=[0, m['geometry']['frames'] - 1],
                 appearance=e.get('appearance', ''), notes='plan A of a Causal Diorama (revision)') for e in scm['E']]
    (plan / 'entities.json').write_text(json.dumps(dict(clip=f'real_{N:04d}', camera=scm['C'].get('controller', 'fixed'),
                                                       scene='outdoor', style=scm.get('style', ''),
                                                       background=(scm.get('background') or {}).get('appearance', ''), entities=ents), indent=1))
    empty = dict(id=f'real_{N:04d}', video='', size=str([m['geometry']['width'], m['geometry']['height']]), entities=[], frames={},
                 note='revision: plan A comes from a Causal Diorama; no detection')
    for f in (r / 'percept' / n / 'perception.json', r / 'percept' / f'{n}_checked' / 'perception.json'):
        f.write_text(json.dumps(empty, indent=1))
    (r / 'checks' / 'percept' / n / 'verdict.json').write_text(json.dumps({'notes': 'revision: no detection to check'}))
    for k, e in m['pictures'].items():
        shutil.copy2(p / e['file'], pics / f'{k}.png')
    bgk = (m['scm'].get('background') or {}).get('reference', {}).get('key', 'bg')
    shutil.copy2(pics / f'{bgk}.png', pics / 'bg_raw.png')   # plan A's empty place is the diorama's (rule v2's removal is not run)
    if bgk != 'bg':
        shutil.copy2(pics / f'{bgk}.png', pics / 'bg.png')
    (st / 'bg_applied').write_text('')
    if m.get('first_frame'):   # frame 0 of video A is the locked first frame it was generated from (reused or edited for plan B)
        f0 = r / 'frames' / n / '000.png'; f0.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p / m['first_frame']['file'], f0)
    (r / 'runs' / 'known_A.json').write_text(json.dumps(dict(diorama=str(p), id=m['id'], scm_sha256=D.scm_sha(m['scm']), seed=int(seed),
                                                             generation=m.get('generation')), indent=1))


def daemon(run, done, timeout_h=12.0, every=120):
    """start `llm_run.py daemon` on the tree and wait until done() is true; then stop it. Returns True when done."""
    r = root(run); st = r / 'state'; log = open(st / 'daemon.out', 'a')
    p = subprocess.Popen([sys.executable, str(paths.ENGINE / 'ws' / 'llm_run.py'), 'daemon', str(N), '--auto-every', str(every)],
                         env=env(run), stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time(); ok = False
    try:
        while time.time() - t0 < timeout_h * 3600:
            if done():
                ok = True; break
            if p.poll() is not None:
                raise SystemExit(f'the agent daemon stopped (exit {p.returncode}); see {st / "daemon.out"}')
            time.sleep(30)
    finally:
        (st / 'llm_daemon.stop').write_text('')
        try:
            p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.terminate()
    return ok


def video_b(run):
    """the video of plan B: the F4 edit (editing) or the same-seed regeneration (revision)"""
    r = root(run)
    if (r / 'runs' / 'known_A.json').exists():
        return r / 'fix' / 'results' / 'revision' / 'real' / f'{N}.mp4'
    return r / 'fix' / str(N) / 'F4_edit.mp4'


def b_ready(r):
    """plan B, its pictures, its first frame and its empty place are complete (the diorama B without its video)"""
    n = str(N); B = r / 'plan' / n / 'B_v2.json'
    if not (B.exists() and (r / 'pics' / n / 'ff_v4.png').exists()):
        return False
    b = json.loads(B.read_text())
    if any(not (r / 'pics' / n / f'{k}.png').exists() for k in D.reference_keys(b)):
        return False
    return (b['background'].get('reference') or {}).get('source') != 'erase_ff' or (r / 'state' / n / 'bgB_applied').exists()


def done(run, outputs):
    return b_ready(root(run)) if outputs == 'diorama' else video_b(run).exists()


def pack_side(run, side, out, *, kind, text, parent=None, video=None, video_source=None, pipeline='edit'):
    """the Causal Diorama of plan A ('A') or plan B ('B') of a finished tree"""
    r = root(run); n = str(N); g = json.loads((r / 'geom' / f'{n}.json').read_text())
    scm = json.loads((r / 'plan' / n / f'{side}_v2.json').read_text()); assets = json.loads((r / 'plan' / n / 'assets.json').read_text())
    known = json.loads((r / 'runs' / 'known_A.json').read_text()) if (r / 'runs' / 'known_A.json').exists() else None
    pics = {}
    for k in D.reference_keys(scm):
        src = r / 'pics' / n / f'{k}.png'; a = assets['pictures'].get(k, {})
        node = next((x for x in [scm.get('background') or {}] + scm['E'] if (x.get('reference') or {}).get('key') == k), {})
        ref = node.get('reference') or {}
        pics[k] = dict(path=src, source=a.get('source') or ref.get('source'), prompt=ref.get('prompt'), **({'from': ref['from']} if ref.get('from') else {}))
    if side == 'A':
        ff, ffs = r / 'frames' / n / '000.png', 'frame 0 of the source video'
    else:
        meta = json.loads((r / 'v4ff' / n / 'meta.json').read_text())
        ff, ffs = r / 'pics' / n / 'ff_v4.png', f"first-frame route {meta['route']}{('-' + meta['kind']) if meta.get('kind') else ''}"
    geometry = dict(frames=g['frames'], fps=24, width=g['width'], height=g['height'])
    method = None if side == 'A' else ('regeneration with the seed of video A' if known else 'F4 inversion')
    return D.pack(out, id=Path(out).name, kind=kind, scm=scm, geometry=geometry, pictures=pics, first_frame=ff, first_frame_source=ffs,
                  video=video, video_source=video_source, text=text, parent=parent,
                  generation=dict(seed=known['seed'] if known else paths.SEED, steps=20, model='MiniMax H3 Ref2VA + CWM LoRA', method=method),
                  pipeline=pipeline)
