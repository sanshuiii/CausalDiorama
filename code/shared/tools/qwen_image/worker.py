"""Qwen-Image-2.1 worker for the runner ('workers' task, one process per GPU): python worker.py LIST_FILE
Each line of LIST_FILE is one job JSON:
  {id, prompt, images: [{path, max_side?, lowpass?}] (optional), width, height, seed, steps, out,
   plate?: {bg, rects: [{box: [x0, y0, x1, y1] normalised, rgb}], out, lowpass?}}
lowpass f: shrink by f and re-enlarge (removes the fine texture of Qwen-Image edit outputs).
plate: first draws the annotated plate (bg picture centre-cropped to 16:9, resized to 1344x768, one 6-px outline rectangle per
box) to plate.out; the job then uses it like any other input image.
Keeps finished outputs (for a job with 'version': only outputs recorded with that version). Writes OUT and OUT with .json (job + seconds). A job whose inputs are missing or that raises is
recorded in OUT with .failed.json and skipped; the worker still exits 0 so that one bad picture never blocks the node.
"""
import json, sys, time, traceback
from pathlib import Path
import torch
from PIL import Image, ImageDraw

MODEL = '@CD_NODE@/qwen_image21/model'


def reserve():
    marker = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    t = time.monotonic()
    while torch.cuda.mem_get_info()[0] < 60 * 2**30:
        if time.monotonic() - t > 600:
            raise RuntimeError('GPU did not become available')
        time.sleep(3)
    return marker


def lowpass(im, f):
    """Shrink and re-enlarge: removes the fine grid texture that Qwen-Image edit outputs carry (a second edit amplifies it)."""
    w, h = im.size
    return im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS).resize((w, h), Image.LANCZOS)


def plate(spec):
    im = Image.open(spec['bg']).convert('RGB'); w, h = im.size
    if spec.get('lowpass'):
        im = lowpass(im, spec['lowpass'])
    tw = min(w, round(h * 16 / 9)); th = round(tw * 9 / 16)
    im = im.crop(((w - tw) // 2, (h - th) // 2, (w - tw) // 2 + tw, (h - th) // 2 + th)).resize((1344, 768), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for r in spec['rects']:
        X0, Y0, X1, Y1 = r['box'][0] * 1344, r['box'][1] * 768, r['box'][2] * 1344, r['box'][3] * 768
        if X1 - X0 < 16:  # tiny objects (e.g. a cigarette): draw a small square around them instead of failing
            X0, X1 = (X0 + X1) / 2 - 8, (X0 + X1) / 2 + 8
        if Y1 - Y0 < 16:
            Y0, Y1 = (Y0 + Y1) / 2 - 8, (Y0 + Y1) / 2 + 8
        d.rectangle([X0 + 3, Y0 + 3, X1 - 3, Y1 - 3], outline=tuple(r['rgb']), width=6)
    Path(spec['out']).parent.mkdir(parents=True, exist_ok=True)
    im.save(spec['out'])


def load(x):
    im = Image.open(x['path']).convert('RGB')
    m = x.get('max_side')
    if m and max(im.size) > m:
        s = m / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    if x.get('lowpass'):
        im = lowpass(im, x['lowpass'])
    return im


def done(j):
    """Finished output; a job with a 'version' also needs a record of that version (older outputs are redone)."""
    out = Path(j['out'])
    if not out.exists():
        return False
    if 'version' not in j:
        return True
    try:
        return json.loads(out.with_suffix('.json').read_text()).get('version') == j['version']
    except Exception:
        return False


def main():
    jobs = [json.loads(Path(p).read_text()) for p in Path(sys.argv[1]).read_text().split() if p]
    todo = [j for j in jobs if not done(j)]
    print('TODO', len(todo), 'of', len(jobs), flush=True)
    if not todo:
        return
    marker = reserve()
    from diffusers import QwenImage21Pipeline
    t = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('cuda')
    print('LOADED', round(time.time() - t, 1), flush=True)
    for j in todo:
        out = Path(j['out']); out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        try:
            if j.get('plate'):
                plate(j['plate'])
            missing = [x['path'] for x in j.get('images', []) if not Path(x['path']).exists()]
            if missing:
                raise FileNotFoundError(f'missing inputs {missing}')
            kw = dict(prompt=j['prompt'], num_inference_steps=j.get('steps', 40), generator=torch.Generator('cuda').manual_seed(j.get('seed', 42)),
                      width=j['width'], height=j['height'])
            if j.get('images'):
                kw['image'] = [load(x) for x in j['images']]
            img = pipe(**kw).images[0]
            img.save(out)
            rec = dict(j, seconds=round(time.time() - t, 1), size=list(img.size), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            out.with_suffix('.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            print('DONE', j['id'], rec['seconds'], 's', flush=True)
        except Exception:
            out.with_suffix('.failed.json').write_text(json.dumps(dict(j, error=traceback.format_exc()[-2000:]), ensure_ascii=False, indent=1))
            print('JOB_FAILED', j['id'], traceback.format_exc(), flush=True)
    del marker


if __name__ == '__main__':
    main()
