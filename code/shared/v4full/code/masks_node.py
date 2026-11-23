"""Grounding DINO + SAM 2.1 masks on a node GPU, batched over many pictures.
    PYTHONPATH=@CD_NODE@/qwen_image21/pydeps <cwm venv python> masks_node.py LIST
(transformers 5.19 from the node's Qwen-Image deps, used read-only.) LIST: one job JSON per line:
  {id, image, out_dir, entities: [{name, text, alt_text, prior (white-model mask path or null)}]}
Per entity: Grounding DINO (box threshold 0.3, text threshold 0.25) with `text` (plan appearance); no box -> retry with `alt_text`
(the entity's id in words). Boxes are then assigned greedily over the entities of the picture, best score first, one box per
entity and no box (IoU > 0.5) shared by two entities; an entity prefers boxes whose centre lies in its white-model box grown by
10 % of the frame. SAM 2.1 segments each assigned box (single mask). Writes out_dir/<name>.png and out_dir/<id>.json.
"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image

GD = '@CD_SHARED@/models/grounding-dino-base'
SAM = '@CD_SHARED@/models/sam2.1-hiera-large'
BOX_THR, TEXT_THR, GROW = 0.3, 0.25, 0.10


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def bbox(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


class Masker:
    def __init__(self, dev):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.dev = dev
        self.gp = AutoProcessor.from_pretrained(GD); self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(GD).to(dev).eval()
        self.sp = Sam2Processor.from_pretrained(SAM); self.sm = Sam2Model.from_pretrained(SAM).to(dev).eval()

    @torch.no_grad()
    def detect(self, im, text):
        inp = self.gp(images=im, text=text, return_tensors='pt').to(self.dev)
        out = self.gm(**inp)
        r = self.gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=BOX_THR, text_threshold=TEXT_THR,
                                                           target_sizes=[im.size[::-1]])[0]
        return sorted(([float(s), [float(v) for v in b]] for s, b in zip(r['scores'].cpu(), r['boxes'].cpu())), key=lambda x: -x[0])

    @torch.no_grad()
    def segment(self, im, box):
        inp = self.sp(images=im, input_boxes=[[box]], return_tensors='pt').to(self.dev)
        out = self.sm(**inp, multimask_output=False)
        m = self.sp.post_process_masks(out.pred_masks.cpu(), inp['original_sizes'].cpu())[0]
        return np.asarray(m).reshape(-1, *m.shape[-2:])[0] > 0


def run(M, job):
    im = Image.open(job['image']).convert('RGB'); W, H = im.size
    od = Path(job['out_dir']); od.mkdir(parents=True, exist_ok=True)
    rec = {'image': job['image'], 'entities': {}}; cands = []
    for e in job['entities']:
        dets = M.detect(im, e['text']); used = e['text']
        if not dets and e.get('alt_text'):
            dets = M.detect(im, e['alt_text']); used = e['alt_text']
        pb = None
        if e.get('prior') and Path(e['prior']).exists():
            pm = np.asarray(Image.open(e['prior']).convert('L')) > 127
            pb = bbox(pm)
        gx, gy = GROW * W, GROW * H
        for s, b in dets:
            inside = pb is not None and pb[0] - gx <= (b[0] + b[2]) / 2 <= pb[2] + gx and pb[1] - gy <= (b[1] + b[3]) / 2 <= pb[3] + gy
            cands.append((1 if (inside or pb is None) else 0, s, e['name'], b))
        rec['entities'][e['name']] = {'text': used, 'prior_box': pb, 'dets': [[round(s, 3), [round(v) for v in b]] for s, b in dets[:6]]}
    taken = []; done = set()
    for pref, s, name, b in sorted(cands, key=lambda x: (-x[0], -x[1])):
        if name in done or any(iou(b, t) > 0.5 for t in taken):
            continue
        m = M.segment(im, b); done.add(name); taken.append(b)
        Image.fromarray((m * 255).astype(np.uint8)).save(od / f"{name}.png")
        rec['entities'][name].update(pick=[round(s, 3), [round(v) for v in b]], in_prior=bool(pref), area=int(m.sum()))
    for name, r in rec['entities'].items():
        if name not in done:
            r['pick'] = None
    (od / f"{job['id']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    jobs = [json.loads(Path(p).read_text()) for p in Path(sys.argv[1]).read_text().split() if p]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = Masker(dev); t = time.time()
    for j in jobs:
        if (Path(j['out_dir']) / f"{j['id']}.json").exists():
            continue
        try:
            r = run(M, j)
            print('DONE', j['id'], {k: (v.get('pick') or [None])[0] for k, v in r['entities'].items()}, round(time.time() - t, 1), flush=True)
        except Exception:
            print('JOB_FAILED', j['id'], traceback.format_exc(), flush=True)


if __name__ == '__main__':
    main()
