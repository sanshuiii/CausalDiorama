"""Fit check of an SCM against a real clip's perception. CPU only.

    python3 scm_match.py PLAN.json PERCEPTION_DIR [--out DIR]

For every frame the detector saw (perception.json) it renders the plan with world.py (the executor) and compares,
per entity, the plan's visible box (instance map) with the detected SAM box, both normalised to the frame:
  iou, centre offset (fraction of the frame), area ratio plan/detected, and the plan's camera distance vs the measured median depth.
Pass: every entity that the detector found in at least 3 frames has mean IoU >= 0.40 and mean centre offset <= 0.08 (or, for thin and
small things, centre offset <= 0.03 with a plan/detected area ratio in [0.5, 2]), and the mean IoU
over all (entity, frame) pairs is >= 0.50; an entity id of the plan that is not in perception.json (or the reverse) is reported.
Depth is reported, not judged (monocular metric depth is only roughly right). Writes DIR/match.json and DIR/match.jpg (detected
boxes solid, plan boxes dashed, on the real frames). Exit 0 = pass, 1 = fail, 2 = the plan cannot be executed.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, '@CD_CODE@/engine/cb')
import world  # noqa: E402

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.


def dashed(d, box, c, w=5, step=18):
    x0, y0, x1, y1 = box
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // step))
        for k in range(0, n, 2):
            p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            q = (a[0] + (b[0] - a[0]) * min(k + 1, n) / n, a[1] + (b[1] - a[1]) * min(k + 1, n) / n)
            d.line([p, q], fill=c, width=w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('plan'); ap.add_argument('perception'); ap.add_argument('--out')
    a = ap.parse_args()
    pdir = Path(a.perception); per = json.loads((pdir / 'perception.json').read_text())
    out = Path(a.out) if a.out else Path(a.plan).with_suffix('.match'); out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(a.plan).read_text())
    try:
        sc = world.Scene(plan)
    except Exception as exc:
        print(f'PLAN CANNOT BE EXECUTED: {type(exc).__name__}: {exc}'); sys.exit(2)
    ents = {e['id']: e for e in plan['E']}
    names = [e['name'] for e in per['entities']]
    report = dict(plan=a.plan, perception=str(pdir), missing_in_plan=[n for n in names if n not in ents],
                  extra_in_plan=[e for e in ents if e not in names], frames={}, entities={})
    rows = {n: [] for n in names if n in ents}
    for fs, det in per['frames'].items():
        f = int(fs)
        try:
            st, rgb, depth, sem, ins = sc.render(f)
        except Exception as exc:
            print(f'PLAN CANNOT BE EXECUTED at frame {f}: {type(exc).__name__}: {exc}'); sys.exit(2)
        cam = sc.camera(f); o = np.array(cam['position'], float); fwd = world.norm(np.array(cam['target'], float) - o)
        fr = {}
        for n in rows:
            e = ents[n]; m = ins == e['instance_id']
            pb = None
            if m.any():
                ys, xs = np.nonzero(m)
                pb = [xs.min() / world.W, ys.min() / world.H, (xs.max() + 1) / world.W, (ys.max() + 1) / world.H]
            d = det.get(n)
            try:
                c = sc.root(n, f / world.FPS); h = float(e.get('height_m') or (e.get('size_m') or [0, 1, 0])[1])
                cz = float((np.asarray(c, float) + np.array([0, h / 2, 0]) - o) @ fwd)
            except Exception:
                cz = None
            r = dict(plan_box=[round(float(v), 4) for v in pb] if pb else None, det_box=d['box'] if d else None,
                     plan_cam_dist_m=round(cz, 2) if cz is not None else None, det_depth_m=d['depth_median_m'] if d else None)
            if pb and d:
                db = d['box']
                r['iou'] = round(iou(pb, db), 3)
                r['centre_offset'] = round(float(np.hypot((pb[0] + pb[2] - db[0] - db[2]) / 2, (pb[1] + pb[3] - db[1] - db[3]) / 2)), 3)
                r['area_ratio'] = round(((pb[2] - pb[0]) * (pb[3] - pb[1])) / max((db[2] - db[0]) * (db[3] - db[1]), 1e-9), 2)
            elif d and not pb:
                r['iou'] = 0.0; r['note'] = 'detected but not visible in the plan'
            elif pb and not d:
                r['note'] = 'visible in the plan but not detected'
            fr[n] = r
            if d:
                rows[n].append(r)
        report['frames'][fs] = fr
    ok, all_iou = True, []
    for n, rs in rows.items():
        ious = [r['iou'] for r in rs]; offs = [r['centre_offset'] for r in rs if 'centre_offset' in r]
        ratios = [r['area_ratio'] for r in rs if 'area_ratio' in r]
        dep = [(r['plan_cam_dist_m'], r['det_depth_m']) for r in rs if r.get('plan_cam_dist_m') is not None and r.get('det_depth_m')]
        s = dict(detected_frames=len(rs), mean_iou=round(float(np.mean(ious)), 3) if ious else None,
                 mean_centre_offset=round(float(np.mean(offs)), 3) if offs else None,
                 median_area_ratio=round(float(np.median(ratios)), 2) if ratios else None,
                 median_plan_dist_over_det_depth=round(float(np.median([p / d for p, d in dep])), 2) if dep else None)
        small_ok = (s['mean_centre_offset'] is not None and s['mean_centre_offset'] <= 0.03 and s['median_area_ratio'] is not None
                    and 0.5 <= s['median_area_ratio'] <= 2.0)  # thin or small things: IoU is too harsh, position + size decide
        s['pass'] = len(rs) < 3 or small_ok or (s['mean_iou'] is not None and s['mean_iou'] >= 0.40 and (s['mean_centre_offset'] or 1) <= 0.08)
        ok &= s['pass']; all_iou += ious
        report['entities'][n] = s
    report['mean_iou'] = round(float(np.mean(all_iou)), 3) if all_iou else None
    report['passed'] = bool(ok and report['mean_iou'] is not None and report['mean_iou'] >= 0.50 and not report['missing_in_plan'])
    (out / 'match.json').write_text(json.dumps(report, indent=1))
    # picture
    fr = sorted(per['frames'], key=int); sel = [fr[round(i * (len(fr) - 1) / 5)] for i in range(6)] if len(fr) > 6 else fr  # 6 frames, evenly spaced
    tw, th = 672, 384; sheet = Image.new('RGB', (tw * 2, th * ((len(sel) + 1) // 2)), 'white')
    src = per['video']
    import subprocess, tempfile
    for k, fs in enumerate(sel):
        tmp = Path(tempfile.mkdtemp()) / 'f.png'
        subprocess.run(['@CD_FFMPEG@', '-loglevel', 'error', '-y', '-i', src, '-vf',
                        f'select=eq(n\\,{fs})', '-vsync', '0', '-frames:v', '1', str(tmp)], check=True)
        im = Image.open(tmp).convert('RGB').resize((tw, th)); d = ImageDraw.Draw(im)
        for ci, n in enumerate(rows):
            c = COLORS[ci % len(COLORS)]; r = report['frames'][fs][n]
            if r['det_box']:
                d.rectangle([v * s for v, s in zip(r['det_box'], (tw, th, tw, th))], outline=c, width=3)
            if r['plan_box']:
                dashed(d, [v * s for v, s in zip(r['plan_box'], (tw, th, tw, th))], c, w=3, step=10)
            lab = r['plan_box'] or r['det_box']
            if lab:
                d.text((lab[0] * tw + 4, lab[1] * th + 4), f"{n} iou={r.get('iou')}", fill=c)
        d.rectangle([0, 0, 300, 18], fill='black'); d.text((4, 3), f'frame {fs}: solid = detected, dashed = plan', fill='white')
        sheet.paste(im, ((k % 2) * tw, (k // 2) * th))
    sheet.save(out / 'match.jpg', quality=88)
    print(f"{'PASS' if report['passed'] else 'FAIL'}  mean IoU {report['mean_iou']}  (preview: {out / 'match.jpg'})")
    if report['missing_in_plan'] or report['extra_in_plan']:
        print('entities missing in the plan:', report['missing_in_plan'], '| plan entities not in the perception:', report['extra_in_plan'])
    for n, s in report['entities'].items():
        print(f"  {'ok  ' if s['pass'] else 'FAIL'} {n}: frames {s['detected_frames']}, mean IoU {s['mean_iou']}, centre offset {s['mean_centre_offset']}, "
              f"area plan/det {s['median_area_ratio']}, plan distance / measured depth {s['median_plan_dist_over_det_depth']}")
    worst = sorted(((r.get('iou', 1), fs, n) for fs, fr in report['frames'].items() for n, r in fr.items() if r.get('det_box')))[:5]
    for v, fs, n in worst:
        r = report['frames'][fs][n]
        print(f"  worst: frame {fs} {n}: iou {v}, plan box {r['plan_box']}, detected box {r['det_box']}")
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
