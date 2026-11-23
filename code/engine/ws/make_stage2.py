"""Shared helpers of the editing engine: node paths, job files, the node queue, source frames, perception masks, entity cut-outs and
white-model renders. flexrun.py points the data root R at its run tree and sets the clip geometry (GW, GH) and PERCEPT_DIR.
"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '@CD_CODE@/engine/cb')

S = Path('@CD_SHARED@'); INV = S / 'inv'; R = Path('@CD_WORK@/fr')   # R: the run tree (set by flexrun.mods)
P = '@CD_NODE@'
PY = f'{P}/cwm_h3_reproduction_v1/.venv/bin/python'; TORCHRUN = f'{P}/cwm_h3_reproduction_v1/.venv/bin/torchrun'
PYDEPS, QDEPS = f'{P}/compbench_v1/pydeps', f'{P}/qwen_image21/pydeps'
PRE = f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat {PY} {S}/tools/gpu_preclaim.py 15'
SYS2 = (Path('@CD_CODE@/engine/cb') / 'system_stage2.txt').read_text().strip()
CHECK = {}   # the run's checklist (runs/checklist.json), set by flexrun.mods()
GREY = (200, 200, 200)
SEED = 9942001
PERCEPT_DIR = {}   # clip -> its checked perception directory (<n>_checked), set by flexrun.use() after the perception check
GW, GH = 1344, 768   # frame size; flexrun.use() sets the clip's own geometry


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, ensure_ascii=False, indent=1)); return str(path)


def put(tid, task, dry, node):
    q = Path('@CD_QUEUE@') / 'queue' / node / 'main'
    if (q / f'{tid}.json').exists():
        raise SystemExit(f'{tid} already queued')
    print('queue', node, tid, task['type'], len(task.get('configs', [])) or (task.get('command') or '')[:100])
    if not dry:
        q.mkdir(parents=True, exist_ok=True)
        tmp = q.parent / f'.{tid}.tmp'; tmp.write_text(json.dumps(task, ensure_ascii=False, indent=1)); shutil.move(str(tmp), str(q / f'{tid}.json'))


def percept(n):
    return json.loads((R / 'percept' / PERCEPT_DIR.get(n, str(n)) / 'perception.json').read_text())


def best_frame(per, name):
    """Frame for the entity's picture: frame 0 if it is seen there at >= 60 % of its largest area, else its largest view."""
    seen = {int(f): v[name] for f, v in per['frames'].items() if v.get(name)}
    if not seen:
        return None
    big = max(seen, key=lambda f: seen[f]['area'])
    return 0 if 0 in seen and seen[0]['area'] >= 0.6 * seen[big]['area'] else big


def frame_png(n, f):
    p = R / 'frames' / str(n) / f'{f:03d}.png'
    if not p.exists():
        subprocess.run(['@CD_FFMPEG@', '-loglevel', 'error', '-y', '-i', str(R / 'src' / f'{n}.mp4'),
                        '-vf', f'select=eq(n\\,{f})', '-vsync', '0', '-frames:v', '1', str(p)], check=True)
    return p


def mask(n, f, name, grow=0):
    p = R / 'percept' / PERCEPT_DIR.get(n, str(n)) / 'masks' / f'{f:03d}_{name}.png'
    m = Image.open(p).convert('L')
    if grow:
        m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
    return np.asarray(m) > 127


def cutout(n, name, out):
    """The entity alone on a plain light-grey background (the pipeline's picture convention), cut from the real frame."""
    per = percept(n); f = best_frame(per, name)
    if f is None:
        return None
    im = np.asarray(Image.open(frame_png(n, f)).convert('RGB')); m = mask(n, f, name, grow=2)
    ys, xs = np.nonzero(m); H, W = m.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = int(0.15 * (x1 - x0)) + 8, int(0.15 * (y1 - y0)) + 8
    x0, x1, y0, y1 = max(0, x0 - mx), min(W, x1 + mx), max(0, y0 - my), min(H, y1 + my)
    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32)[..., None] / 255
    comp = (im * soft + np.array(GREY, np.float32) * (1 - soft)).astype(np.uint8)[y0:y1, x0:x1]
    Image.fromarray(comp).save(out)
    return dict(frame=f, box_px=[int(x0), int(y0), int(x1), int(y1)], size=[int(x1 - x0), int(y1 - y0)])


def render_local(plan, out):
    sys.path.insert(0, str(S / 'tools' / 'cb'))
    from case_prep import render_conditions
    out = Path(out)
    if not (out / 'render_receipt.json').exists():
        out.mkdir(parents=True, exist_ok=True); render_conditions(plan, out)
