"""Deterministic self-check for a cb-scm-v1 plan against the planner's own `intent` requirements.

Usage: python3 scm_lint.py SCM_JSON [--rules rules_v1.json] [--out DIR]
Writes DIR/lint.json and DIR/preview.png (6 white-model frames with entity labels); prints a readable report.
Exit code 0 = pass, 1 = violations, 2 = the plan cannot be executed at all.
It never looks at benchmark metadata or evaluators: only the plan and what the plan itself says it must show.
"""
import argparse, fnmatch, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import world

HERE = Path(__file__).parent
W, H, N = world.W, world.H, world.FRAMES
FRAMES = sorted(set(range(0, N, 2)) | {N - 1})  # every 2nd frame is enough for fractions
STOP = {'a', 'an', 'the', 'is', 'are', 'its', 'their', 'his', 'her', 'of', 'to', 'in', 'on', 'at', 'with', 'and', 'while', 'as'}


def words(s):
    return [w for w in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if w not in STOP]


def same_word(a, b):
    """Loose match for inflections (runs/running, climbs/climbing): shared prefix of >= max(3, shorter-1) letters."""
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return a == b or k >= max(3, min(len(a), len(b)) - 1)


def measure(scm):
    """Per sampled frame and entity: visible px, bbox, centroid, median depth, unoccluded silhouette px, touches border."""
    sc = world.Scene(scm)
    ids = {e['id']: e['instance_id'] for e in scm['E']}
    per = {eid: [] for eid in ids}
    for f in FRAMES:
        st, rgb, depth, sem, ins, sil = sc.render(f, with_sil=True)
        for eid, iid in ids.items():
            m = ins == iid
            n = int(m.sum())
            s = sil.get(iid)
            sn = int(s.sum()) if s is not None else 0
            rec = dict(f=f, t=f / world.FPS, px=n, sil=sn, occl=(1 - n / sn) if sn else None)
            if n:
                ys, xs = np.nonzero(m)
                rec.update(x0=int(xs.min()), x1=int(xs.max()), y0=int(ys.min()), y1=int(ys.max()),
                           cx=float(xs.mean()) / W, cy=float(ys.mean()) / H, depth=float(np.median(depth[m])))
            if sn:
                ys, xs = np.nonzero(s)
                rec['border'] = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1)
            per[eid].append(rec)
    return per


def visible(r, min_h):
    return r['px'] >= 60 and r['px'] and (r['y1'] - r['y0'] + 1) / H >= min_h


def in_iv(r, iv):
    return iv[0] - 1e-6 <= r['t'] <= iv[1] + 1e-6


def preview(scm, per, out):
    from PIL import Image, ImageDraw
    sc = world.Scene(scm)
    tiles = []
    for f in (0, 25, 50, 75, 100, 123):
        st, rgb, *_ = sc.render(f)
        im = Image.fromarray(rgb).resize((W * 2, H * 2), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(im)
        d.text((6, 4), f't={f / world.FPS:.2f}s', fill=(255, 255, 0))
        for eid, recs in per.items():
            r = min(recs, key=lambda r: abs(r['f'] - f))
            if r['px']:
                d.rectangle([r['x0'] * 2, r['y0'] * 2, r['x1'] * 2, r['y1'] * 2], outline=(255, 0, 0))
                d.text((r['x0'] * 2 + 2, max(0, r['y0'] * 2 - 12)), eid, fill=(255, 0, 0))
        tiles.append(im)
    grid = Image.new('RGB', (W * 4, H * 6))
    for i, im in enumerate(tiles):
        grid.paste(im, ((i % 2) * W * 2, (i // 2) * H * 2))
    grid.save(out)


HEADS = {'E', 'S', 'R', 'C', 'environment', 'background', 'text', 'intent', 'style', 'edit'}
CAMERA_CONTROLLERS = {'fixed', 'keyframes', 'orbit', 'follow'}
GEOMETRY = ('type', 'size_m', 'height_m', 'position', 'heading_deg', 'shape', 'semantic', 'self_moving', 'root', 'center')


def geometry(scm, eid):
    """Everything about an entity except its look: its own fields, its root track and the holds it takes part in."""
    e = next(x for x in scm['E'] if x['id'] == eid)
    tr = [t.get('keyframes') for t in scm.get('S', []) if t.get('entity') == eid]
    hd = [{k: r.get(k) for k in ('object', 'holder', 'start', 'end', 'mode')} for r in scm.get('R', [])
          if r.get('type') == 'hold' and eid in (r.get('object'), r.get('holder'))]
    return json.dumps([{k: e.get(k) for k in GEOMETRY}, tr, hd], sort_keys=True)


def v2_checks(scm, source, fail, warn):
    """Plan checks: camera node, dependency edges, each thing once, motions with a cause, edit declarations."""
    ents = {e['id']: e for e in scm['E']}
    cam = scm.get('C')
    if not isinstance(cam, dict) or 'camera' in scm:
        fail('the camera must be the SCM node C (and there must be no top-level "camera"); see the schema')
    else:
        ctl = cam.get('controller', 'fixed')
        if ctl not in CAMERA_CONTROLLERS:
            fail(f'C.controller must be one of {sorted(CAMERA_CONTROLLERS)}')
        if ctl == 'orbit' and 'sweep_deg' not in (cam.get('orbit') or {}):
            fail('an orbit camera needs C.orbit.sweep_deg')
        if ctl == 'follow' and (cam.get('follow') or {}).get('entity') not in ents:
            fail('a follow camera needs C.follow.entity, an existing entity id')
        if cam.get('look_at') and cam['look_at'] not in ents:
            fail(f"C.look_at '{cam['look_at']}' is not an entity id")
    ids = {'E': set(ents), 'S': {t.get('id') for t in scm.get('S', [])}, 'R': {r.get('id') for r in scm.get('R', [])}}
    for d in scm.get('D', []):
        for side in ('from', 'to'):
            path = str(d.get(side, '')).strip()
            m = re.match(r'([A-Za-z_]+)(?:\[([^\]]*)\]|\.([^.\[]+))?', path)
            head, key = (m.group(1), (m.group(2) or m.group(3) or '').strip("'\" ")) if m else ('', '')
            if head not in HEADS:
                fail(f'D edge {side} "{path}" must start with one of {sorted(HEADS)} (the camera is C)')
            elif head in ids and key and not any(fnmatch.fnmatch(i, key) for i in ids[head] if i):
                fail(f'D edge {side} "{path}" names {head}.{key}, which matches no existing id')
    bg = ' ' + ((scm.get('background') or {}).get('appearance') or '').lower() + ' '
    for e in scm['E']:
        lab = (e.get('label') or '').strip().lower()
        if len(lab) >= 3 and re.search(r'[^a-z]' + re.escape(lab) + r'(s|es)?[^a-z]', bg):
            fail(f'background.appearance mentions "{lab}", which is the entity {e["id"]}: each thing appears exactly once '
                 f'(the place itself belongs to the background; a thing that is an entity must not be in the background text)')
    sc = world.Scene(scm)
    pos = {eid: [] for eid, e in ents.items() if e['type'] in ('object', 'prop')}
    for f in range(world.FRAMES):
        st = sc.state(f)
        for eid in pos:
            pos[eid].append(np.array(st['roots'].get(eid, st['props'].get(eid, [0, 0, 0])), float))
    for eid, ps in pos.items():
        if ents[eid].get('self_moving'):
            continue
        moving = [f for f in range(1, world.FRAMES) if np.linalg.norm(ps[f] - ps[f - 1]) > 1e-3]
        held = [r for r in sc.holds if r['object'] == eid]
        loose = [f for f in moving if not any(float(r['start']) * world.FPS - 1 <= f <= float(r['end']) * world.FPS + 1 for r in held)]
        if loose:
            fail(f'{eid} moves by itself at {loose[0] / world.FPS:.2f}-{loose[-1] / world.FPS:.2f}s: add an R hold event with the entity that '
                 f'carries or moves it (mode carry or guide), or set "self_moving": true if it genuinely moves on its own '
                 f'(a ball in flight, a bird, a drifting balloon, a boat, a drone)')
    for r in sc.holds:
        h = ents[r['holder']]
        if h['type'] == 'human':
            continue  # a person's hand must reach it: the executor fails otherwise
        f0 = min(round(float(r['start']) * world.FPS), world.FRAMES - 1)
        o = ents[r['object']]; oc = sc.root(r['object'], f0 / world.FPS) + np.array([0, float((o.get('size_m') or [0, 0, 0])[1]) / 2, 0])
        hc = sc.root(r['holder'], f0 / world.FPS) + np.array([0, float((h.get('size_m') or [0, 0, 0])[1]) / 2, 0])
        lim = 0.5 * max(h.get('size_m') or [0.3]) + 0.5 * max(o.get('size_m') or [0.1]) + 0.1
        if np.linalg.norm(oc - hc) > lim:
            fail(f"hold {r.get('id')}: {r['object']} is {np.linalg.norm(oc - hc):.2f} m from {r['holder']} when the hold starts "
                 f"(at most {lim:.2f} m): the holder must touch it first (move the holder there before the hold starts)")
    if source is None:
        return
    ed = scm.get('edit')
    if not isinstance(ed, dict):
        fail('an edited plan needs the "edit" declaration (look_changed, geometry_changed, added, removed, camera_changed, '
             'style_changed, background_changed, summary); see the schema')
        return
    a_ents = {e['id']: e for e in source['E']}
    added, removed = set(ents) - set(a_ents), set(a_ents) - set(ents)
    if added != set(ed.get('added') or []):
        fail(f"edit.added must list exactly the new entities {sorted(added)}")
    if removed != set(ed.get('removed') or []):
        fail(f"edit.removed must list exactly the deleted entities {sorted(removed)}")
    for eid in sorted(set(ents) & set(a_ents)):
        if geometry(scm, eid) != geometry(source, eid) and eid not in (ed.get('geometry_changed') or []):
            fail(f'{eid}: its size, position, motion or holds changed, but it is not in edit.geometry_changed. Change only what the '
                 f'instruction asks (keep existing entities exactly as they are when adding new ones); if the change is intended, declare it')
        if (ents[eid].get('appearance') or '').strip() != (a_ents[eid].get('appearance') or '').strip() \
                and eid not in (ed.get('look_changed') or []) and not ed.get('style_changed'):
            fail(f'{eid}: its appearance text changed, but it is not in edit.look_changed. Appearance describes only how a thing looks; '
                 f'a size or position edit must leave it untouched (its picture is then reused)')
    if json.dumps(scm.get('C'), sort_keys=True) != json.dumps(source.get('C'), sort_keys=True) and not ed.get('camera_changed'):
        fail('the camera node C changed, but edit.camera_changed is not true')
    if (scm.get('style') or '') != (source.get('style') or '') and not ed.get('style_changed'):
        fail('style changed, but edit.style_changed is not true')
    if ((scm.get('background') or {}).get('appearance') or '') != ((source.get('background') or {}).get('appearance') or '') \
            and not ed.get('background_changed'):
        fail('background.appearance changed, but edit.background_changed is not true')


def lint(scm, rules, source=None):
    errors, warnings, checks = [], [], []
    T = world.FRAMES / world.FPS

    def fail(msg, **kw):
        errors.append(dict(msg=msg, **kw))

    # structure
    intent = scm.get('intent')
    if not isinstance(intent, dict) or 'requirements' not in intent:
        fail('missing intent.requirements')
        return errors, warnings, checks, {}
    if intent.get('expressiveness') not in ('full', 'partial', 'text_only'):
        fail('intent.expressiveness must be full | partial | text_only')
    ents = {e['id']: e for e in scm['E']}
    tl = sorted(scm['text']['timeline'], key=lambda x: x[0])
    if abs(tl[0][0]) > 0.05 or abs(tl[-1][1] - T) > 0.05 or any(abs(a[1] - b[0]) > 0.05 for a, b in zip(tl, tl[1:])):
        fail(f'text.timeline must cover 0-{T:.2f}s without gaps')
    for e in scm['E']:
        if e['type'] in world.FOREGROUND_TYPES and not e.get('appearance', '').strip():
            fail(f"entity {e['id']} has no appearance text")
        if e['type'] in world.FOREGROUND_TYPES and e.get('layer', 'foreground') == 'foreground' and not e.get('label', '').strip():
            fail(f"foreground entity {e['id']} has no label (plain common noun from the scene description)")
    reqs = intent['requirements']
    for r in reqs:
        for k in ('entity', 'a', 'b'):
            if k in r and r[k] not in ents:
                fail(f"requirement {r.get('id')} references unknown entity {r[k]}")
        for x in r.get('entities', []):
            if x not in ents:
                fail(f"requirement {r.get('id')} references unknown entity {x}")
    if rules.get('require_v2'):  # reference-picture fields
        if not str(scm.get('style', '')).strip():
            fail('missing style (one sentence: visual style and colour grading of the whole video)')
        if not str((scm.get('background') or {}).get('appearance', '')).strip():
            fail('missing background.appearance (the empty place only, no entity of E)')
    if errors:
        return errors, warnings, checks, {}
    try:
        per = measure(scm)
    except Exception as exc:  # executor constraint (e.g. unreachable handoff) or malformed field
        fail(f'executor error: {exc}')
        return errors, warnings, checks, {}
    if rules.get('require_v2b'):
        try:
            v2_checks(scm, source, fail, lambda m: warnings.append(dict(msg=m)))
        except Exception as exc:
            fail(f'executor error in the V2 checks: {type(exc).__name__}: {exc}')

    th = rules
    contact = set()
    for r in reqs:
        if r['kind'] in ('contact',) or r.get('rel') in ('on_top_of', 'under'):
            contact |= {r.get('a'), r.get('b'), *r.get('entities', [])}

    def frac(xs):
        return sum(xs) / len(xs) if xs else 0.

    for r in reqs:
        kind = r['kind']
        iv = r.get('interval', [0, T])
        rid = r.get('id', kind)
        if kind in ('present', 'count'):
            names = r.get('entities') or [r.get('entity')]
            min_h = th['count_min_height'] if kind == 'count' else th['min_height']
            for eid in names:
                recs = [x for x in per[eid] if in_iv(x, iv)]
                v = frac([visible(x, min_h) for x in recs])
                checks.append(dict(id=rid, kind='visible', entity=eid, value=round(v, 3), need=th['min_visible']))
                if v < th['min_visible']:
                    big = max((x['y1'] - x['y0'] + 1) / H for x in recs if x['px']) if any(x['px'] for x in recs) else 0
                    fail(f"{eid} visible (>= {min_h:.0%} of frame height) in only {v:.0%} of {iv[0]:.2f}-{iv[1]:.2f}s "
                         f"(need {th['min_visible']:.0%}); largest height seen {big:.0%}", req=rid, entity=eid)
                if frac([bool(x.get('border')) for x in recs if x['sil']]) > 1 - th['inside_frame']:
                    warnings.append(dict(msg=f'{eid} touches the frame border in many frames of {rid}', req=rid))
            if kind == 'count':
                nn = int(r['n'])
                good = []
                for f in FRAMES:
                    if not (iv[0] - 1e-6 <= f / world.FPS <= iv[1] + 1e-6):
                        continue
                    k = sum(visible(next(x for x in per[e] if x['f'] == f), th['count_min_height']) for e in names)
                    good.append(k == nn)
                v = frac(good)
                checks.append(dict(id=rid, kind='count', value=round(v, 3), need=th['min_visible']))
                if len(names) != nn:
                    fail(f"count requirement {rid} lists {len(names)} entities but n={nn}", req=rid)
                elif v < th['min_visible']:
                    fail(f"exactly {nn} of {r.get('noun', 'objects')} visible in only {v:.0%} of frames (need {th['min_visible']:.0%})", req=rid)
        elif kind == 'relation':
            a, b, rel = r['a'], r['b'], r['rel']
            pa = {x['f']: x for x in per[a] if in_iv(x, iv)}
            pb = {x['f']: x for x in per[b] if in_iv(x, iv)}
            both = [f for f in pa if pa[f]['px'] >= 60 and pb[f]['px'] >= 60]
            vb = len(both) / max(1, len(pa))
            ok = []
            for f in both:
                A, B = pa[f], pb[f]
                ov = max(0, min(A['x1'], B['x1']) - max(A['x0'], B['x0'])) / max(1, min(A['x1'] - A['x0'], B['x1'] - B['x0']))
                mx, my = th['relation_margin'], th['relation_margin']
                ok.append({'left_of': A['cx'] < B['cx'] - mx, 'right_of': A['cx'] > B['cx'] + mx,
                           'above': A['cy'] < B['cy'] - my, 'below': A['cy'] > B['cy'] + my,
                           'in_front_of': A['depth'] < B['depth'] - th['depth_margin'],
                           'behind': A['depth'] > B['depth'] + th['depth_margin'],
                           'on_top_of': A['cy'] < B['cy'] and ov >= 0.3, 'under': A['cy'] > B['cy'] and ov >= 0.3}[rel])
            v = frac(ok)
            checks.append(dict(id=rid, kind='relation', rel=rel, value=round(v, 3), both_visible=round(vb, 3), need=th['relation_frac']))
            if vb < th['min_visible']:
                fail(f'{a} and {b} are both visible in only {vb:.0%} of {iv[0]:.2f}-{iv[1]:.2f}s', req=rid)
            if v < th['relation_frac']:
                fail(f'relation "{a} {rel} {b}" holds on screen in only {v:.0%} of frames where both are visible '
                     f'(need {th["relation_frac"]:.0%}; margins: {th["relation_margin"]:.0%} of frame, {th["depth_margin"]} m depth)', req=rid)
        elif kind == 'motion':
            eid, dirn = r['entity'], r['direction']
            recs = [x for x in per[eid] if in_iv(x, iv) and x['px'] >= 60]
            if len(recs) < 4:
                fail(f'{eid} is visible in too few frames to show motion in {iv[0]:.2f}-{iv[1]:.2f}s', req=rid)
                continue
            key, sign = {'right': ('cx', 1), 'left': ('cx', -1), 'down': ('cy', 1), 'up': ('cy', -1),
                         'toward_camera': ('depth', -1), 'away_from_camera': ('depth', 1)}[dirn]
            vals = [x[key] for x in recs]
            disp = (vals[-1] - vals[0]) * sign / (abs(vals[0]) if key == 'depth' else 1)
            steps = [(b2 - a2) * sign for a2, b2 in zip(vals, vals[1:]) if abs(b2 - a2) > 1e-3]
            mono = frac([s > 0 for s in steps])
            checks.append(dict(id=rid, kind='motion', value=round(disp, 3), monotonic=round(mono, 3), need=th['motion_min']))
            if disp < th['motion_min']:
                fail(f'{eid} moves {dirn} by only {disp:.0%} of the frame (need {th["motion_min"]:.0%}) during its visible frames', req=rid)
            if mono < 0.8:
                fail(f'{eid} motion {dirn} is not monotonic ({mono:.0%} of steps in the right direction)', req=rid)
        elif kind == 'attribute':
            ap = words(ents[r['entity']].get('appearance', ''))
            miss = [w for w in words(r['phrase']) if w not in ap]
            checks.append(dict(id=rid, kind='attribute', missing=miss))
            if miss:
                fail(f"appearance of {r['entity']} lacks {miss} from attribute '{r['phrase']}'", req=rid)
        elif kind == 'action':
            need = words(r['phrase'])
            best = max((sum(any(same_word(w, x) for x in words(s)) for w in need) / max(1, len(need))
                        for a0, b0, s in tl if a0 < iv[1] and b0 > iv[0]), default=0)
            checks.append(dict(id=rid, kind='action', value=round(best, 2)))
            if best < 0.6:  # wording check only: a warning, never a reason to re-plan geometry
                warnings.append(dict(msg=f"no timeline entry overlapping {iv[0]:.2f}-{iv[1]:.2f}s clearly describes '{r['phrase']}' for {r['entity']}", req=rid))
        elif kind == 'contact':
            pass
        else:
            fail(f'unknown requirement kind {kind}', req=rid)

    # occlusion of every entity named by a requirement
    named = set()
    for r in reqs:
        named |= {r.get('entity'), r.get('a'), r.get('b'), *r.get('entities', [])}
    named.discard(None)
    for eid in sorted(named):
        vis = [x for x in per[eid] if x['px'] >= 60 and x['occl'] is not None]
        if not vis:
            continue
        lim = th['max_occlusion_contact'] if eid in contact else th['max_occlusion']
        v = frac([x['occl'] <= lim for x in vis])
        checks.append(dict(kind='occlusion', entity=eid, value=round(v, 3), limit=lim))
        if v < th['min_visible']:
            worst = max(x['occl'] for x in vis)
            fail(f'{eid} is more than {lim:.0%} hidden behind other things in {1 - v:.0%} of its visible frames (worst {worst:.0%})', entity=eid)
    return errors, warnings, checks, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scm')
    ap.add_argument('--rules', default=str(HERE / 'rules_v1.json'))
    ap.add_argument('--out')
    ap.add_argument('--source', help='plan A, for an edited plan B: checks that only declared things changed (rules with require_v2b)')
    a = ap.parse_args()
    scm = json.loads(Path(a.scm).read_text())
    rules = json.loads(Path(a.rules).read_text())['lint']
    source = json.loads(Path(a.source).read_text()) if a.source else None
    out = Path(a.out) if a.out else Path(a.scm).with_suffix('.lint')
    out.mkdir(parents=True, exist_ok=True)
    errors, warnings, checks, per = lint(scm, rules, source)
    res = dict(passed=not errors, errors=errors, warnings=warnings, checks=checks)
    (out / 'lint.json').write_text(json.dumps(res, indent=1))
    if per:
        preview(scm, per, out / 'preview.png')
    print('LINT', 'PASS' if not errors else f'FAIL ({len(errors)} problems)')
    for e in errors:
        print(' -', e['msg'])
    for w in warnings:
        print(' ~ warning:', w['msg'])
    if per:
        print(f'preview: {out / "preview.png"}')
    sys.exit(0 if not errors else (2 if not per else 1))


if __name__ == '__main__':
    main()
