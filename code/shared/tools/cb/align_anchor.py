"""Automatic first-frame alignment: fit the plan's camera pitch and entity placement to the generated anchor image.

Replaces the manual calibration of bg_consistency. Uses OWLv2 (deliberately not the benchmark's GroundingDINO)
to find each foreground entity in the anchor, then
  1) fits the camera pitch so that box heights (with the planned physical sizes) and box bottoms (ground contact)
     agree on each entity's distance,
  2) re-places each detected entity so that its box bottom-centre lands where the anchor shows it,
     shifting its whole track (motion shape unchanged) and rescaling it to the detected box height (0.6-1.6x),
  3) re-renders frame 0 and reports IoU(plan box, detected box) before/after.
Background (backdrops, static props) is left as planned. Undetected entities are left unchanged.

Usage (node venv): python align_anchor.py SCM_JSON ANCHOR_PNG OUT_DIR [--owl DIR] [--device cuda]
Writes OUT_DIR/scm_aligned.json and OUT_DIR/align.json.
"""
import argparse, copy, json, math, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import world

W, H = world.W, world.H          # render resolution (336x192)
AW, AH = 1344, 768               # anchor resolution
MOVABLE = ('human', 'animal', 'vehicle', 'object')


TYPE_NOUN = {'human': 'person', 'animal': 'animal', 'vehicle': 'vehicle', 'object': 'object'}


def noun_of(e):
    """Detector query: the entity's `label` (common noun from the scene description), else a cleaned id, else its type."""
    if e.get('label'):
        return e['label'].strip().lower()
    if e['type'] == 'human':
        return 'person'
    n = re.sub(r'[_\d]+$', '', e['id']).replace('_', ' ').strip()
    return n if len(n) > 2 and n.replace(' ', '').isalpha() else TYPE_NOUN.get(e['type'], 'object')


def cam_basis(cam):
    o = np.array(cam['position'], float)
    fwd = world.norm(np.array(cam['target'], float) - o)
    right = world.norm(np.cross(fwd, [0, 1, 0]))
    up = np.cross(right, fwd)
    return o, fwd, right, up


def pixel_ray(cam, px, py):
    """Ray through render-resolution pixel centre (px, py), matching world.render()."""
    o, fwd, right, up = cam_basis(cam)
    tan = math.tan(math.radians(cam['vertical_fov_deg'] / 2))
    xx = (px + .5) / W * 2 - 1
    yy = 1 - (py + .5) / H * 2
    return o, world.norm(fwd - xx * right * tan * W / H + yy * up * tan)


def ground_hit(cam, px, py, plane_y):
    o, d = pixel_ray(cam, px, py)
    if abs(d[1]) < 1e-6:
        return None
    t = (plane_y - o[1]) / d[1]
    return o + d * t if t > 0 else None


def with_pitch(cam, dpitch_deg):
    """Rotate the view direction about the camera's right axis by dpitch (positive = look up)."""
    o, fwd, right, up = cam_basis(cam)
    a = math.radians(dpitch_deg)
    new_fwd = fwd * math.cos(a) + up * math.sin(a)
    dist = np.linalg.norm(np.array(cam['target'], float) - o)
    c = dict(cam)
    c['target'] = (o + new_fwd * dist).round(4).tolist()
    return c


def boxes_at_frame0(scm):
    st, rgb, depth, sem, ins = world.Scene(scm).render(0)
    out = {}
    for e in scm['E']:
        m = ins == e['instance_id']
        if m.sum() >= 20:
            ys, xs = np.nonzero(m)
            out[e['id']] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return out, st


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.


def detect(anchor_png, queries, owl_dir, device):
    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained(owl_dir)
    model = Owlv2ForObjectDetection.from_pretrained(owl_dir).to(device).eval()
    img = Image.open(anchor_png).convert('RGB')
    inputs = proc(text=[[f'a photo of a {q}' for q in queries]], images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # OWLv2 pads the image to a square; boxes are relative to the padded size (max side).
    side = max(img.size)
    res = proc.post_process_object_detection(outputs, threshold=0.08, target_sizes=torch.tensor([[side, side]]).to(device))[0]
    dets = {q: [] for q in queries}
    for box, score, lab in zip(res['boxes'].tolist(), res['scores'].tolist(), res['labels'].tolist()):
        x0, y0, x1, y1 = box
        x0, x1 = max(0, x0), min(img.size[0], x1); y0, y1 = max(0, y0), min(img.size[1], y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        # to render resolution
        dets[queries[lab]].append(dict(score=score, box=(x0 * W / img.size[0], y0 * H / img.size[1], x1 * W / img.size[0], y1 * H / img.size[1])))
    for q in dets:
        dets[q].sort(key=lambda d: -d['score'])
    return dets


def height_of(e):
    if e['type'] == 'human':
        return e.get('height_m', 1.75)
    return e['size_m'][1]


def scale_entity(e, s):
    if e['type'] == 'human':
        e['height_m'] = round(e.get('height_m', 1.75) * s, 3)
    else:
        e['size_m'] = [round(v * s, 3) for v in e['size_m']]


def base_y(e):
    return float(e['position'][1]) if e['type'] == 'object' and 'position' in e else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scm'); ap.add_argument('anchor'); ap.add_argument('out')
    ap.add_argument('--owl', default='@CD_NODE@/compbench_v1/models/owlv2')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--min_score', type=float, default=0.15)
    a = ap.parse_args()
    run(a.scm, a.anchor, a.out, a.owl, a.device, a.min_score)


_DET_CACHE = {}


def run(scm_path, anchor, out_dir, owl='@CD_NODE@/compbench_v1/models/owlv2', device='cuda', min_score=0.15,
        move=True, scale=True, blend=None):
    """move=False: camera pitch only; scale=False: re-place entities without rescaling them;
    blend=T: moving entities start at the anchor position and ease back onto the planned track by T seconds
    (static entities are re-placed; no rescaling)."""
    class A:  # keep the original body unchanged below
        pass
    a = A(); a.anchor, a.owl, a.device, a.min_score = anchor, owl, device, min_score
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    scm = json.loads(Path(scm_path).read_text())
    ents = {e['id']: e for e in scm['E']}
    plan_boxes, st0 = boxes_at_frame0(scm)
    cand = [eid for eid, e in ents.items() if e['type'] in MOVABLE and e.get('layer', 'foreground') == 'foreground' and eid in plan_boxes]
    report = dict(entities={}, detected=0, candidates=len(cand))
    if not cand:
        Path(out / 'scm_aligned.json').write_text(json.dumps(scm, indent=1)); report['note'] = 'nothing visible at frame 0'
        Path(out / 'align.json').write_text(json.dumps(report, indent=1)); print('ALIGN nothing to align'); return report
    queries = sorted({noun_of(ents[e]) for e in cand})
    key = (str(a.anchor), tuple(queries))
    if key not in _DET_CACHE:
        _DET_CACHE[key] = detect(a.anchor, queries, a.owl, a.device)
    dets = _DET_CACHE[key]
    # greedy assignment per noun by distance between planned and detected box centres
    match = {}
    for q in queries:
        ids = [e for e in cand if noun_of(ents[e]) == q]
        pool = [d for d in dets[q] if d['score'] >= a.min_score and (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]) < 0.9 * W * H][:max(3, 2 * len(ids))]
        pairs = []
        for eid in ids:
            pb = plan_boxes[eid]; pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
            for k, d in enumerate(pool):
                db = d['box']; dc = ((db[0] + db[2]) / 2, (db[1] + db[3]) / 2)
                pairs.append((math.hypot(pc[0] - dc[0], pc[1] - dc[1]) - 30 * d['score'], eid, k))
        used_e, used_d = set(), set()
        for _, eid, k in sorted(pairs):
            if eid in used_e or k in used_d:
                continue
            match[eid] = pool[k]; used_e.add(eid); used_d.add(k)
    report['detected'] = len(match)
    cam0 = scm['camera']
    f_y = (H / 2) / math.tan(math.radians(cam0['vertical_fov_deg'] / 2))

    # 1) pitch: distance from box height vs distance of the ground hit of the box bottom
    ground_ids = [e for e in match if base_y(ents[e]) == 0.]
    def cost(dp):
        cam = with_pitch(cam0, dp)
        o, fwd, _, _ = cam_basis(cam)
        errs = []
        for eid in ground_ids:
            b = match[eid]['box']
            z_size = f_y * height_of(ents[eid]) / max(1., b[3] - b[1])
            p = ground_hit(cam, (b[0] + b[2]) / 2, b[3] - .5, 0.)
            if p is None:
                errs.append(10.); continue
            z_hit = float(np.dot(p - o, fwd))
            errs.append(math.log(max(z_hit, .1) / z_size))
        return float(np.median(np.abs(errs))) if errs else 0.
    dpitch = 0.
    if ground_ids:
        grid = np.arange(-20, 20.01, .25)
        costs = [cost(g) for g in grid]
        dpitch = float(grid[int(np.argmin(costs))])
        # keep the planned camera unless the fit is clearly better
        if cost(dpitch) > .9 * cost(0.):
            dpitch = 0.
    new = copy.deepcopy(scm)
    new['camera'] = with_pitch(cam0, dpitch) if dpitch else dict(cam0)
    report['pitch_change_deg'] = dpitch

    # 2) re-place and rescale each matched entity
    tracks = {tr['entity']: tr for tr in new['S'] if tr.get('kind', 'root_track') == 'root_track'}
    nents = {e['id']: e for e in new['E']}
    for eid, d in (match.items() if move else []):
        e = nents[eid]; b = d['box']
        by = base_y(e)
        p = ground_hit(new['camera'], (b[0] + b[2]) / 2, b[3] - .5, by)
        if p is None:
            continue
        root0 = np.array(st0['roots'].get(eid, [0, 0, 0]), float) if eid in st0['roots'] else np.array(e.get('position', [0, 0, 0]), float)
        dx, dz = float(p[0] - root0[0]), float(p[2] - root0[2])
        if eid in tracks and blend:
            orig = next(tr for tr in scm['S'] if tr.get('entity') == eid and tr.get('kind', 'root_track') == 'root_track')
            T = world.Track(orig, base_y(ents[eid]))
            three = len(orig['keyframes'][0][1]) == 3
            def pt(v):
                return [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)] if three else [round(float(v[0]), 3), round(float(v[2]), 3)]
            p0 = T.pos(0.) + np.array([dx, 0., dz])
            tracks[eid]['keyframes'] = [[0.0, pt(p0)], [float(blend), pt(T.pos(float(blend)))]] + \
                [kf for kf in orig['keyframes'] if kf[0] > float(blend) + 1e-6]
        elif eid in tracks:
            for kf in tracks[eid]['keyframes']:
                pt = kf[1]
                if len(pt) == 2:
                    kf[1] = [round(pt[0] + dx, 3), round(pt[1] + dz, 3)]
                else:
                    kf[1] = [round(pt[0] + dx, 3), pt[1], round(pt[2] + dz, 3)]
        if e['type'] == 'object' and 'position' in e:
            e['position'] = [round(e['position'][0] + dx, 3), e['position'][1], round(e['position'][2] + dz, 3)]
        elif 'root' in e:
            e['root'] = [round(e['root'][0] + dx, 3), e['root'][1], round(e['root'][2] + dz, 3)]
        report['entities'][eid] = dict(noun=noun_of(ents[eid]), score=round(d['score'], 3), det_box=[round(v, 1) for v in b],
                                       plan_box=[int(v) for v in plan_boxes[eid]], moved_m=[round(dx, 2), round(dz, 2)])
    # rescale to detected box height at the new position (one pass)
    boxes_mid, _ = boxes_at_frame0(new)
    for eid, d in (match.items() if (move and scale and not blend) else []):
        if eid not in boxes_mid or eid not in report['entities']:
            continue
        rb = boxes_mid[eid]; b = d['box']
        s = float(np.clip((b[3] - b[1]) / max(1., rb[3] - rb[1]), .6, 1.6))
        if abs(s - 1) > .1:
            scale_entity(nents[eid], s)
        report['entities'][eid]['scale'] = round(s, 3)
    # keep handoff meet points with their participants
    for r in new.get('R', []):
        if r.get('type') == 'handoff':
            mv = [report['entities'].get(r[k], {}).get('moved_m') for k in ('giver', 'receiver')]
            mv = [m for m in mv if m]
            if mv:
                r['meet_point'] = [round(r['meet_point'][0] + np.mean([m[0] for m in mv]), 3), r['meet_point'][1],
                                   round(r['meet_point'][2] + np.mean([m[1] for m in mv]), 3)]
    # 3) validate
    try:
        final_boxes, _ = boxes_at_frame0(new)
        world.Scene(new).state(0)
    except Exception as exc:
        report['error'] = f'aligned plan not executable ({exc}); keeping the planned SCM'
        new = scm; final_boxes = plan_boxes; report['pitch_change_deg'] = 0.
    for eid, d in match.items():
        if eid in report['entities']:
            report['entities'][eid]['iou_before'] = round(iou(plan_boxes[eid], d['box']), 3)
            report['entities'][eid]['iou_after'] = round(iou(final_boxes.get(eid, (0, 0, 0, 0)), d['box']), 3)
    ious = [v['iou_after'] for v in report['entities'].values() if 'iou_after' in v]
    report['mean_iou_before'] = round(float(np.mean([v['iou_before'] for v in report['entities'].values()])), 3) if ious else None
    report['mean_iou_after'] = round(float(np.mean(ious)), 3) if ious else None
    report['undetected'] = sorted(set(cand) - set(match))
    new['alignment'] = dict(method='owlv2-fit-v1', move=move, scale=scale, blend=blend, pitch_change_deg=dpitch, entities=list(report['entities']))
    Path(out / 'scm_aligned.json').write_text(json.dumps(new, indent=1))
    Path(out / 'align.json').write_text(json.dumps(report, indent=1))
    print('ALIGN', json.dumps({k: report[k] for k in ('candidates', 'detected', 'pitch_change_deg', 'mean_iou_before', 'mean_iou_after', 'undetected')}))
    return report


if __name__ == '__main__':
    main()
