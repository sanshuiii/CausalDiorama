"""Per-clip perception for fitting an SCM to a real video (editing). One GPU per process.
    <cwm venv python> perceive.py LIST      (transformers of the CWM venv; models from the shared disk)
LIST: one job JSON per line: {id, video, frames: [indices], entities: [{name, text, alt_text}], depth_model, out_dir}
Per sampled frame: Grounding DINO (all entity phrases in one prompt, see detect_all) + SAM 2.1 (masks_node.Masker); boxes are assigned
greedily over the entities, best score first, one box per entity, no box shared (IoU > 0.5); from the second frame on a box that
overlaps the entity's previous box gets a bonus of 0.5 * IoU, so identities stay stable. Depth-Anything-V2 (metric) gives a depth
map per sampled frame; each entity gets the median depth inside its mask.
Writes out_dir/perception.json, masks/<frame>_<name>.png, depth/<frame>.npy (336x192 float16, metres), overlay sheet sheet.jpg.
"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]   # the shared root this code is deployed under (<root>/inv/code/perceive.py)
sys.path.insert(0, str(ROOT / 'v4full' / 'code'))
import masks_node  # noqa: E402
from masks_node import Masker, iou, bbox  # noqa: E402
# models on the shared disk; Masker reads these module globals when it is built
masks_node.GD = str(ROOT / 'models' / 'grounding-dino-base')
masks_node.SAM = str(ROOT / 'models' / 'sam2.1-hiera-large')

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def read_frames(path, idx):
    import av
    want, out = set(idx), {}
    with av.open(str(path)) as c:
        for n, f in enumerate(c.decode(video=0)):
            if n in want:
                out[n] = f.to_image().convert('RGB')
            if n >= max(idx):
                break
    return [out[i] for i in idx]


def _words(t):
    return set(''.join(ch.lower() if ch.isalnum() else ' ' for ch in t).split())


@torch.no_grad()
def detect_all(M, im, entities, thr=0.25, text_thr=0.2):
    """Grounding DINO's own multi-phrase mode: one prompt with every entity phrase ('a. b. c.'), so the phrases compete for each
    box; each returned box is given to the entity whose phrase its text label overlaps most. An entity without any box is retried
    alone with its alt_text (as masks_node does). -> [(entity, [(score, box)], query)]"""
    prompt = ' '.join(e['text'].lower().rstrip('.') + '.' for e in entities)
    inp = M.gp(images=im, text=prompt, return_tensors='pt').to(M.dev)
    out = M.gm(**inp)
    r = M.gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=text_thr, target_sizes=[im.size[::-1]])[0]
    labels = r.get('text_labels', r.get('labels'))
    per = {e['name']: [] for e in entities}
    for s, b, lab in zip(r['scores'].cpu(), r['boxes'].cpu(), labels):
        lw = _words(str(lab))
        best = max(entities, key=lambda e: (len(lw & _words(e['text'])) / max(len(_words(e['text'])), 1)))
        if lw & _words(best['text']):
            per[best['name']].append((float(s), [float(v) for v in b]))
    res = []
    for e in entities:
        dets, used = sorted(per[e['name']], key=lambda x: -x[0]), e['text']
        if not dets:
            dets, used = M.detect(im, e['text']), e['text'] + ' (alone)'
        if not dets and e.get('alt_text'):
            dets, used = M.detect(im, e['alt_text']), e['alt_text']
        res.append((e, dets, used))
    return res


class Depth:
    def __init__(self, path, dev):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.dev = dev
        self.p = AutoImageProcessor.from_pretrained(path); self.m = AutoModelForDepthEstimation.from_pretrained(path).to(dev).eval()

    @torch.no_grad()
    def __call__(self, im):
        inp = self.p(images=im, return_tensors='pt').to(self.dev)
        d = self.m(**inp).predicted_depth  # [1,h,w] metres for the metric checkpoints
        d = torch.nn.functional.interpolate(d[:, None], size=im.size[::-1], mode='bicubic', align_corners=False)[0, 0]
        return d.clamp_min(0).float().cpu().numpy()


def run(M, D, job):
    od = Path(job['out_dir']); (od / 'masks').mkdir(parents=True, exist_ok=True); (od / 'depth').mkdir(exist_ok=True)
    frames = read_frames(job['video'], job['frames'])
    W, H = frames[0].size
    prev, rec = {}, {'id': job['id'], 'video': job['video'], 'size': [W, H], 'entities': job['entities'], 'frames': {}}
    for n, im in zip(job['frames'], frames):
        depth = D(im)
        Image.fromarray(np.zeros(1, np.uint8))  # keep PIL imported in workers
        np.save(od / 'depth' / f'{n:03d}.npy', np.asarray(Image.fromarray(depth).resize((336, 192), Image.BILINEAR), dtype=np.float16))
        cands = []
        for e, dets, used in detect_all(M, im, job['entities']):
            for sc_, b in dets:
                bonus = 0.5 * iou(b, prev[e['name']]) if e['name'] in prev else 0.0
                cands.append((sc_ + bonus, sc_, e['name'], b, used))
        cands.sort(key=lambda x: -x[0])
        taken, chosen = [], {}
        for score, s, name, b, used in cands:
            if name in chosen or any(iou(b, t) > 0.5 for t in taken):
                continue
            chosen[name] = (s, b, used); taken.append(b)
        fr = {}
        for e in job['entities']:
            name = e['name']
            if name not in chosen:
                fr[name] = None; continue
            s, b, used = chosen[name]
            m = M.segment(im, b)
            Image.fromarray((m * 255).astype(np.uint8)).save(od / 'masks' / f'{n:03d}_{name}.png')
            bb = bbox(m) or [int(v) for v in b]
            ys, xs = np.nonzero(m)
            dm = float(np.median(depth[m])) if m.any() else None
            fr[name] = dict(score=round(s, 3), query=used, box_px=bb, box=[round(bb[0] / W, 4), round(bb[1] / H, 4), round(bb[2] / W, 4), round(bb[3] / H, 4)],
                            area=round(float(m.mean()), 5), centroid=[round(float(xs.mean()) / W, 4), round(float(ys.mean()) / H, 4)] if m.any() else None,
                            depth_median_m=round(dm, 3) if dm is not None else None,
                            depth_p10_p90_m=[round(float(np.percentile(depth[m], 10)), 3), round(float(np.percentile(depth[m], 90)), 3)] if m.any() else None)
            prev[name] = bb
        fr['_depth'] = dict(p5=round(float(np.percentile(depth, 5)), 3), p50=round(float(np.percentile(depth, 50)), 3),
                            p95=round(float(np.percentile(depth, 95)), 3),
                            bottom_row_median=round(float(np.median(depth[-H // 20:])), 3), centre_median=round(float(np.median(depth[H // 3:2 * H // 3, W // 3:2 * W // 3])), 3))
        rec['frames'][str(n)] = fr
        print('FRAME', job['id'], n, {k: (v['box'] if v else None) for k, v in fr.items() if k != '_depth'}, flush=True)
    (od / 'perception.json').write_text(json.dumps(rec, indent=1))
    # overlay sheet: six frames with boxes, contours and names
    fr = sorted(job['frames']); sel = [fr[round(i * (len(fr) - 1) / 5)] for i in range(6)] if len(fr) > 6 else fr  # 6 frames, evenly spaced
    tw, th = 672, 384; sheet = Image.new('RGB', (tw * 2, th * ((len(sel) + 1) // 2)), 'white')
    for k, n in enumerate(sel):
        im = frames[job['frames'].index(n)].copy(); d = ImageDraw.Draw(im)
        for ci, e in enumerate(job['entities']):
            v = rec['frames'][str(n)][e['name']]
            if v:
                c = COLORS[ci % len(COLORS)]; d.rectangle(v['box_px'], outline=c, width=6)
                d.rectangle([v['box_px'][0], v['box_px'][1], v['box_px'][0] + 14 * len(e['name']) + 120, v['box_px'][1] + 34], fill=c)
                d.text((v['box_px'][0] + 6, v['box_px'][1] + 6), f"{e['name']} {v['depth_median_m']}m", fill='black')
        d.rectangle([0, 0, 220, 30], fill='black'); d.text((6, 6), f'frame {n} ({n / 24:.2f}s)', fill='white')
        sheet.paste(im.resize((tw, th)), ((k % 2) * tw, (k // 2) * th))
    sheet.save(od / 'sheet.jpg', quality=90)
    return rec


def main():
    dev = 'cuda'
    jobs = [json.loads(Path(l).read_text()) for l in Path(sys.argv[1]).read_text().split() if l]
    M = Masker(dev); depth_models = {}
    for job in jobs:
        t = time.time()
        try:
            if job['depth_model'] not in depth_models:
                depth_models[job['depth_model']] = Depth(job['depth_model'], dev)
            run(M, depth_models[job['depth_model']], job)
            print('DONE', job['id'], round(time.time() - t, 1), flush=True)
        except Exception:
            Path(job['out_dir']).mkdir(parents=True, exist_ok=True)
            (Path(job['out_dir']) / 'failed.txt').write_text(traceback.format_exc()); print('FAILED', job['id'], traceback.format_exc(), flush=True)


if __name__ == '__main__':
    main()
