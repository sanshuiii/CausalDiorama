"""Deterministic executor + analytic renderer for cb-scm-v1 (superset of bgc-scm-v1; see scm_schema.md).

CPU only. Coarse kinematics with ellipsoid/box proxies, not a physics simulator.
Outputs per frame: metric depth (336x192 float32), semantic ID (0-11), instance
ID, and an RGB white-model preview. Optionally, per-instance unoccluded silhouettes (for lint).
New in cb-scm-v1: generic `object` entities (ellipsoid or box, any semantic class) placed and moved in 3D,
so flying / rising / falling / resting-on-top objects only need rough positions. No articulated actions.
Optional camera movement (camera keyframes, see camera_at) and optional top-level `style`; the camera may be the SCM node
`C` (controllers fixed / keyframes / orbit / follow, optional look_at), and R may hold `hold` events (an object carried by, or
guided by the hand of, another entity). Plans without them render and compile as plans with a fixed top-level camera.
"""
from __future__ import annotations
import json, math
import numpy as np

W, H = 336, 192
FPS, FRAMES = 24, 124
SEM = {'void_unknown': 0, 'sky': 1, 'water': 2, 'terrain': 3, 'road_paved': 4, 'vegetation': 5,
       'building_structure': 6, 'infrastructure': 7, 'human': 8, 'animal': 9, 'vehicle': 10, 'prop': 11}
FOREGROUND_TYPES = {'human', 'animal', 'vehicle', 'prop', 'object'}
SCHEMAS = ('bgc-scm-v1', 'cb-scm-v1')
SKIN = np.array([.66, .49, .36])


def norm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def smooth(u):
    u = min(max(u, 0.), 1.)
    return u * u * (3 - 2 * u)


def camera_spec(scm):
    """The camera of a plan: the SCM node C or the top-level camera field."""
    return scm['C'] if 'C' in scm else scm['camera']


def camera_moves(cam):
    return bool(cam.get('keyframes')) or cam.get('controller') in ('orbit', 'follow') or bool(cam.get('look_at'))


def camera_at(cam, t):
    """Camera {position, target, vertical_fov_deg} at time t. The base fields are the camera at t = 0. A moving camera adds
    keyframes [[t, {position?, target?, vertical_fov_deg?}], ...] (missing fields keep the previous value; the camera holds
    its last keyframe), interp 'smooth' (default, eases at every keyframe) or 'linear', and path 'straight' (default) or
    'orbit' (the position swings around the target, interpolating horizontal distance, angle and height)."""
    base = {'position': [float(v) for v in cam['position']], 'target': [float(v) for v in cam['target']],
            'vertical_fov_deg': float(cam.get('vertical_fov_deg', 35))}
    keys = cam.get('keyframes') or []
    if not keys:
        return base
    seq, cur = ([] if float(keys[0][0]) <= 1e-9 else [(0.0, base)]), dict(base)
    for tk, st in keys:
        cur = dict(cur, **{k: ([float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)) for k, v in st.items() if k in base})
        seq.append((float(tk), dict(cur)))
    if t <= seq[0][0]:
        return seq[0][1]
    for (a, x), (b, y) in zip(seq, seq[1:]):
        if t <= b:
            u = (t - a) / (b - a) if b > a else 1.
            u = smooth(u) if cam.get('interp', 'smooth') == 'smooth' else u
            tgt = np.array(x['target']) * (1 - u) + np.array(y['target']) * u
            if cam.get('path') == 'orbit':
                ox, oy = np.array(x['position']) - np.array(x['target']), np.array(y['position']) - np.array(y['target'])
                aa, ab = math.atan2(ox[0], ox[2]), math.atan2(oy[0], oy[2])
                ang = aa + ((ab - aa + math.pi) % (2 * math.pi) - math.pi) * u
                r = math.hypot(ox[0], ox[2]) * (1 - u) + math.hypot(oy[0], oy[2]) * u
                pos = tgt + np.array([r * math.sin(ang), ox[1] * (1 - u) + oy[1] * u, r * math.cos(ang)])
            else:
                pos = np.array(x['position']) * (1 - u) + np.array(y['position']) * u
            return {'position': pos.tolist(), 'target': tgt.tolist(), 'vertical_fov_deg': float(x['vertical_fov_deg'] * (1 - u) + y['vertical_fov_deg'] * u)}
    return seq[-1][1]


def yaw(deg):
    """Rotation whose columns are the local x (right), y (up), z (forward) axes in world coords."""
    r = math.radians(deg)
    f = np.array([math.sin(r), 0., math.cos(r)])
    right = np.array([math.cos(r), 0., -math.sin(r)])
    return np.stack([right, [0., 1., 0.], f], axis=1)


class Track:
    def __init__(self, spec, base_y=0.):
        # [t, [x, z]] (ground) or [t, [x, y, z]] (3D, objects); y is the bottom of the entity above the ground.
        self.keys = [(float(t), np.array([p[0], base_y, p[1]] if len(p) == 2 else p, float)) for t, p in spec['keyframes']]
        assert all(a[0] < b[0] for a, b in zip(self.keys, self.keys[1:])), 'keyframe times must increase'
        self.interp = spec.get('interp', 'linear')
        self.heading = spec.get('heading', 'auto')

    def pos(self, t):
        k = self.keys
        if t <= k[0][0]:
            return k[0][1].copy()
        for (a, x), (b, y) in zip(k, k[1:]):
            if t <= b:
                u = (t - a) / (b - a)
                u = smooth(u) if self.interp == 'smooth' else u
                return x * (1 - u) + y * u
        return k[-1][1].copy()

    def head(self, t):
        if self.heading != 'auto':
            return float(self.heading)
        # Direction of horizontal motion; while stopped (or moving only vertically) keep the most recent direction.
        last = None
        for s in np.arange(0, t + 1e-9, 1 / FPS):
            d = self.pos(s + 1 / FPS) - self.pos(s)
            d[1] = 0.
            if np.linalg.norm(d) > 1e-4:
                last = math.degrees(math.atan2(d[0], d[2]))
        if last is None:
            d = self.keys[-1][1] - self.keys[0][1]
            d[1] = 0.
            last = math.degrees(math.atan2(d[0], d[2])) if np.linalg.norm(d) > 1e-6 else 0.
        return last

    def distance(self, t):
        ts = np.arange(0, t + 1e-9, 1 / FPS)
        return float(sum(np.linalg.norm(self.pos(b) - self.pos(a)) for a, b in zip(ts, ts[1:])))


class Scene:
    def __init__(self, scm):
        assert scm.get('schema') in SCHEMAS, f'schema must be one of {SCHEMAS}'
        self.s = scm
        self.E = {e['id']: e for e in scm['E']}
        assert len(self.E) == len(scm['E']), 'duplicate entity ids'
        ids = [e['instance_id'] for e in scm['E']]
        assert len(ids) == len(set(ids)) and all(1 <= i <= 250 for i in ids), 'instance ids must be unique 1..250'
        self.tracks = {}
        for tr in scm.get('S', []):
            if tr.get('kind', 'root_track') == 'root_track':
                assert tr['entity'] in self.E, f"track for unknown entity {tr['entity']}"
                assert tr['entity'] not in self.tracks, f"two tracks for {tr['entity']}"
                e = self.E[tr['entity']]
                base_y = float(e['position'][1]) if e['type'] == 'object' and 'position' in e else 0.
                self.tracks[tr['entity']] = Track(tr, base_y)
        self.events = [r for r in scm.get('R', []) if r.get('type') == 'handoff']
        for r in self.events:
            assert r['reach_start'] < r['contact'] < r['release'] < r['settle'], f"bad handoff timing {r['id']}"
            for k in ('object', 'giver', 'receiver'):
                assert r[k] in self.E, f"handoff {r['id']} references unknown {k} {r[k]}"
        self.holds = sorted((r for r in scm.get('R', []) if r.get('type') == 'hold'), key=lambda r: float(r['start']))
        for r in self.holds:
            for k in ('object', 'holder'):
                assert r.get(k) in self.E, f"hold {r.get('id')} references unknown {k} {r.get(k)}"
            assert r['object'] != r['holder'], f"hold {r.get('id')}: an entity cannot hold itself"
            assert self.E[r['object']]['type'] in ('object', 'prop'), f"hold {r.get('id')}: only object or prop entities can be held"
            assert float(r['start']) < float(r['end']), f"hold {r.get('id')}: start must be before end"
            assert r.get('mode', 'carry') in ('carry', 'guide'), f"hold {r.get('id')}: mode is carry or guide"
        assert not {r['object'] for r in self.holds} & {r['holder'] for r in self.holds}, 'a held object cannot hold another object'
        # Cache headings: auto heading is a scan, so precompute per frame.
        self._head = {eid: [tr.head(f / FPS) for f in range(FRAMES)] for eid, tr in self.tracks.items()}
        self._held = {}
        self._held = self._hold_positions()
        cam = camera_spec(scm)
        ck = cam.get('keyframes') or []
        assert all(float(a[0]) < float(b[0]) for a, b in zip(ck, ck[1:])), 'camera keyframe times must increase'
        self.cams = [self._camera(cam, f) for f in range(FRAMES)]

    def _own_root(self, eid, t):
        if eid in self.tracks:
            return self.tracks[eid].pos(t)
        e = self.E[eid]
        return np.array(e.get('position', e.get('root', e.get('center', e.get('pole_base', [0, 0, 0])))), float)

    def _anchor(self, holder, frame):
        """Where a holder holds things: a person's right hand hold point, otherwise the holder's root."""
        return self.hold_point(holder, frame) if self.E[holder]['type'] == 'human' else self.root(holder, frame / FPS)

    def _hold_positions(self):
        """Per-frame positions of carried objects. During a carry hold the object keeps its offset to the holder's anchor
        (the offset at the start); after the hold it stays where it was released. Guided objects follow their own track."""
        out = {}
        for obj in {r['object'] for r in self.holds if r.get('mode', 'carry') == 'carry'}:
            hs = [r for r in self.holds if r['object'] == obj and r.get('mode', 'carry') == 'carry']
            pos = []
            for f in range(FRAMES):
                t = f / FPS; p = self._own_root(obj, t)
                for r in hs:
                    f0, f1 = round(float(r['start']) * FPS), round(min(float(r['end']), FRAMES / FPS) * FPS)
                    if f < f0:
                        break
                    base = pos[f0] if f0 < len(pos) else p
                    off = np.asarray(base, float) - self._anchor(r['holder'], min(f0, FRAMES - 1))
                    p = self._anchor(r['holder'], min(f, f1, FRAMES - 1)) + off
                pos.append(np.asarray(p, float))
            out[obj] = pos
        return out

    def _camera(self, cam, frame):
        """Camera pose at one frame. Controllers: fixed / keyframes (camera_at), orbit {center?, sweep_deg (> 0: the camera
        moves to its right), t0, t1, interp}, follow {entity, t0, t1} (position and target move with the entity); any controller
        may add look_at (an entity id: the target follows that entity, keeping the t = 0 offset)."""
        t = frame / FPS
        ctl = cam.get('controller', 'keyframes' if cam.get('keyframes') else 'fixed')
        if ctl == 'orbit':
            o = cam['orbit']; center = np.array(o.get('center', cam['target']), float)
            t0, t1 = float(o.get('t0', 0.)), float(o.get('t1', FRAMES / FPS))
            u = min(max((t - t0) / max(t1 - t0, 1e-6), 0.), 1.)
            u = smooth(u) if o.get('interp', 'linear') == 'smooth' else u
            off = np.array(cam['position'], float) - center
            r, a = math.hypot(off[0], off[2]), math.atan2(off[0], off[2]) - math.radians(float(o['sweep_deg'])) * u
            out = {'position': (center + np.array([r * math.sin(a), off[1], r * math.cos(a)])).tolist(),
                   'target': [float(v) for v in cam['target']], 'vertical_fov_deg': float(cam.get('vertical_fov_deg', 35))}
        elif ctl == 'follow':
            fo = cam['follow']; t0, t1 = float(fo.get('t0', 0.)), float(fo.get('t1', FRAMES / FPS))
            d = self.root(fo['entity'], min(max(t, t0), t1)) - self.root(fo['entity'], t0)
            out = {'position': (np.array(cam['position'], float) + d).tolist(), 'target': (np.array(cam['target'], float) + d).tolist(),
                   'vertical_fov_deg': float(cam.get('vertical_fov_deg', 35))}
        else:
            out = camera_at(cam, t)
        if cam.get('look_at'):
            e = cam['look_at']
            out['target'] = (self.root(e, t) + np.array(cam['target'], float) - self.root(e, 0.)).tolist()
        return out

    def camera(self, frame):
        return self.cams[frame]

    # ---- kinematics -------------------------------------------------------
    def root(self, eid, t):
        if eid in self._held:
            return self._held[eid][min(max(round(t * FPS), 0), FRAMES - 1)].copy()
        return self._own_root(eid, t)

    def heading(self, eid, frame):
        if eid in self._head:
            return self._head[eid][frame]
        e = self.E[eid]
        # Untracked objects default to facing screen right, so their length runs left-right on screen.
        return float(e.get('heading_deg', 90. if e['type'] == 'object' else 180.))

    def shoulder(self, eid, frame, side):
        e = self.E[eid]; s = e.get('height_m', 1.8) / 1.85
        R = yaw(self.heading(eid, frame))
        return self.root(eid, frame / FPS) + R @ (np.array([side * .19, 1.38, 0.]) * s)

    def rest_hand(self, eid, frame, side):
        e = self.E[eid]; s = e.get('height_m', 1.8) / 1.85
        R = yaw(self.heading(eid, frame))
        return self.root(eid, frame / FPS) + R @ (np.array([side * .29, 1.0, .06]) * s)

    def hold_point(self, eid, frame):
        e = self.E[eid]; s = e.get('height_m', 1.8) / 1.85
        R = yaw(self.heading(eid, frame))
        return self.root(eid, frame / FPS) + R @ (np.array([.12, 1.05, .34]) * s)

    def state(self, frame):
        """World state at one frame: roots, headings, hands, prop positions, owners."""
        t = frame / FPS
        st = {'frame': frame, 't': t, 'roots': {}, 'headings': {}, 'hands': {}, 'props': {}, 'owners': {}, 'contacts': {}}
        for eid, e in self.E.items():
            if e['type'] in ('human', 'animal', 'vehicle', 'object'):
                st['roots'][eid] = self.root(eid, t).tolist()
                st['headings'][eid] = self.heading(eid, frame)
        hands = {eid: {-1: self.rest_hand(eid, frame, -1), 1: self.rest_hand(eid, frame, 1)}
                 for eid, e in self.E.items() if e['type'] == 'human'}
        for r in self.events:
            obj, g, rc = r['object'], r['giver'], r['receiver']
            mp = np.array(r['meet_point'], float)
            start = self.hold_point(g, frame)
            end = self.hold_point(rc, frame)
            if t < r['reach_start']:
                p, owner = start, g
            elif t < r['contact']:
                u = smooth((t - r['reach_start']) / (r['contact'] - r['reach_start']))
                p, owner = start * (1 - u) + mp * u, g
            elif t < r['release']:
                p, owner = mp, 'shared'
            elif t < r['settle']:
                u = smooth((t - r['release']) / (r['settle'] - r['release']))
                p, owner = mp * (1 - u) + end * u, rc
            else:
                p, owner = end, rc
            st['props'][obj] = p.tolist(); st['owners'][obj] = owner
            # Giver's near hand carries the prop until release, then returns to rest.
            if t < r['release']:
                hands[g][1] = p.copy()
            elif t < r['settle']:
                u = smooth((t - r['release']) / (r['settle'] - r['release']))
                hands[g][1] = mp * (1 - u) + self.rest_hand(g, frame, 1) * u
            # Receiver reaches during the second half of the approach and keeps the prop after contact.
            reach0 = r['reach_start'] + .4 * (r['contact'] - r['reach_start'])
            if t >= r['contact']:
                hands[rc][-1] = p.copy()
            elif t > reach0:
                u = smooth((t - reach0) / (r['contact'] - reach0))
                hands[rc][-1] = self.rest_hand(rc, frame, -1) * (1 - u) + p * u
            st['contacts'][r['id']] = {'giver': t < r['release'], 'receiver': t >= r['contact']}
        for eid, e in self.E.items():
            if e['type'] == 'prop' and eid not in st['props']:
                st['props'][eid] = (self.root(eid, t) if eid in self._held else np.array(e.get('root', [0, 1, 0]), float)).tolist()
        for r in self.holds:  # a person's right hand stays on what it holds or guides
            if self.E[r['holder']]['type'] == 'human' and float(r['start']) <= t <= float(r['end']):
                o = self.E[r['object']]; c = self.root(r['object'], t)
                if o['type'] == 'object':
                    c = c + np.array([0., float(o['size_m'][1]) / 2, 0.])
                hands[r['holder']][1] = np.asarray(c, float)
            st['contacts'][r.get('id', 'hold')] = {'held': float(r['start']) <= t <= float(r['end'])}
        for eid, hs in hands.items():
            for side, h in hs.items():
                reach = np.linalg.norm(h - self.shoulder(eid, frame, side))
                if reach > .78:
                    raise ValueError(f'frame {frame}: {eid} hand {side} cannot reach ({reach:.2f} m)')
            st['hands'][eid] = [hs[-1].tolist(), hs[1].tolist()]
        return st

    # ---- proxy geometry ---------------------------------------------------
    def primitives(self, st):
        """Ellipsoids: (center, radii, rgb, semantic, instance, rotation)."""
        out = []
        t = st['t']
        for eid, e in self.E.items():
            iid, typ = e['instance_id'], e['type']
            col = np.array(e.get('proxy_color', [.75, .75, .78]), float)
            if typ == 'human':
                out += human(np.array(st['roots'][eid]), st['headings'][eid], e.get('height_m', 1.8),
                             [np.array(h) for h in st['hands'][eid]],
                             self.tracks[eid].distance(t) if eid in self.tracks else 0., iid, col)
            elif typ == 'animal':
                out += animal(np.array(st['roots'][eid]), st['headings'][eid], e['size_m'],
                              self.tracks[eid].distance(t) if eid in self.tracks else 0., iid)
            elif typ == 'vehicle':
                out += vehicle(np.array(st['roots'][eid]), st['headings'][eid], e['size_m'], iid)
            elif typ == 'prop':
                w, h, d = e.get('size_m', [.09, .11, .09])
                out.append((np.array(st['props'][eid]), np.array([w / 2, h / 2, d / 2]), np.array([.95, .95, .95]), 11, iid, np.eye(3)))
            elif typ == 'object':
                L, h, w = e['size_m']  # [length along heading, height, width], same convention as animal/vehicle
                R = yaw(st['headings'][eid])
                c = np.array(st['roots'][eid]) + np.array([0., h / 2, 0.])  # root is the bottom centre
                out.append((c, np.array([w / 2, h / 2, L / 2]), col, SEM[e.get('semantic', 'prop')], iid, R,
                            'b' if e.get('shape') == 'box' else 'e'))
            elif typ == 'flag':
                out += flag(e, t)
            elif typ == 'fountain':
                out += fountain(e, t)
        for p in self.s['environment'].get('static_props', []):
            out.append((np.array(p['center'], float), np.array(p['radii'], float), np.array(p.get('color', [.6, .6, .6]), float),
                        SEM[p.get('semantic', 'prop')], 0, np.eye(3)))
        return out

    def render(self, frame, with_sil=False):
        st = self.state(frame)
        out = render(self.cams[frame], self.s['environment'], self.primitives(st), with_sil=with_sil)
        return (st, *out)


def human(root, heading, height, hands, walked, iid, col):
    out = []; s = height / 1.85; R = yaw(heading)
    dark = np.array([.2, .21, .24])
    phase = walked / .75 * math.pi  # ~1.5 m per stride cycle

    def at(v):
        return root + R @ (np.array(v, float) * s)

    def ball(c, r, color):
        out.append((c, np.array(r, float) * s, color, 8, iid, R))

    def limb(a, b, r, color):
        axis = norm(b - a); side = norm(np.cross(axis, R[:, 2]))
        if np.linalg.norm(np.cross(axis, R[:, 2])) < .1:
            side = norm(np.cross(axis, R[:, 0]))
        rot = np.stack([side, axis, np.cross(side, axis)], axis=1)
        out.append(((a + b) / 2, np.array([r * s, np.linalg.norm(b - a) / 2 + r * s, r * s]), color, 8, iid, rot))

    ball(at([0, 1.16, 0]), [.20, .31, .13], col)
    ball(at([0, .88, 0]), [.18, .13, .12], dark)
    ball(at([0, 1.53, 0]), [.065, .08, .065], SKIN)
    ball(at([0, 1.69, 0]), [.11, .14, .12], SKIN)
    for side in (-1, 1):
        swing = .28 * math.sin(phase + (0 if side < 0 else math.pi)) if walked > 0 else 0.
        hip = at([side * .10, .86, 0]); knee = at([side * .11, .46, swing * .5]); foot = at([side * .12, .07, swing])
        limb(hip, knee, .07, dark); limb(knee, foot, .055, dark)
        out.append((foot + R @ np.array([0, 0, .05 * s]), np.array([.05, .04, .12]) * s, np.array([.15, .15, .15]), 8, iid, R))
        sh = at([side * .19, 1.38, 0]); hand = np.asarray(hands[0 if side < 0 else 1], float)
        v = hand - sh; d = np.linalg.norm(v)
        bend = norm(R @ np.array([0, -1., -.2]))
        bend = norm(bend - norm(v) * np.dot(bend, norm(v)))
        elbow = (hand + sh) / 2 + bend * math.sqrt(max((.38 * s) ** 2 - (d / 2) ** 2, 0))
        limb(sh, elbow, .055, col); limb(elbow, hand, .045, SKIN)
        ball(hand, [.045, .055, .04], SKIN)
    return out


def animal(root, heading, size, walked, iid):
    L, Hh, Wd = size; R = yaw(heading); out = []; col = np.array([.62, .5, .38])
    phase = walked / (.6 * L) * math.pi

    def at(v):
        return root + R @ np.array(v, float)
    leg = .45 * Hh
    out.append((at([0, leg + .2 * Hh, 0]), np.array([Wd / 2, .2 * Hh, .38 * L]), col, 9, iid, R))
    out.append((at([0, leg + .42 * Hh, .45 * L]), np.array([.35 * Wd, .17 * Hh, .14 * L]), col, 9, iid, R))
    out.append((at([0, leg + .3 * Hh, -.48 * L]), np.array([.08 * Wd, .08 * Hh, .14 * L]), col, 9, iid, R))
    for i, (x, z) in enumerate([(-1, 1), (1, 1), (-1, -1), (1, -1)]):
        sw = .08 * L * math.sin(phase + (0 if i in (0, 3) else math.pi)) if walked > 0 else 0.
        out.append((at([x * .3 * Wd, leg / 2, z * .28 * L + sw]), np.array([.12 * Wd, leg / 2, .06 * L]), col, 9, iid, R))
    return out


def vehicle(root, heading, size, iid):
    L, Hh, Wd = size; R = yaw(heading); col = np.array([.8, .8, .82])

    def at(v):
        return root + R @ np.array(v, float)
    out = [(at([0, .42 * Hh, 0]), np.array([Wd / 2, .26 * Hh, L / 2]), col, 10, iid, R),
           (at([0, .72 * Hh, -.05 * L]), np.array([.46 * Wd, .26 * Hh, .28 * L]), col * .9, 10, iid, R)]
    for x in (-1, 1):
        for z in (-1, 1):
            out.append((at([x * .45 * Wd, .2 * Hh, z * .32 * L]), np.array([.08 * Wd, .2 * Hh, .2 * Hh]), np.array([.2, .2, .2]), 10, iid, R))
    return out


def flag(e, t):
    base = np.array(e['pole_base'], float); ph = e['pole_height_m']; L, h = e['flag_size_m']
    iid = e['instance_id']; per = e.get('wave_period_s', 1.2); amp = e.get('wave_amplitude_m', .25)
    d = math.radians(e.get('direction_deg', 0.)); along = np.array([math.cos(d), 0, math.sin(d)])
    across = np.array([-math.sin(d), 0, math.cos(d)])
    out = [(base + [0, ph / 2, 0], np.array([.05, ph / 2, .05]), np.array([.95, .95, .95]), 7, iid, np.eye(3))]
    n = 8
    for i in range(n):
        u = (i + .5) / n
        off = amp * u * math.sin(2 * math.pi * (t / per) - 2.2 * math.pi * u)
        droop = .08 * h * u * (1 + .5 * math.sin(2 * math.pi * t / per * .5))
        c = base + [0, ph - h / 2 - droop, 0] + along * (u * L) + across * off
        rot = np.stack([along, [0, 1, 0], across], axis=1)
        out.append((c, np.array([L / n * .62, h / 2, .03]), np.array([.9, .2, .2]), 11, iid, rot))
    return out


def fountain(e, t):
    c = np.array(e['center'], float); r = e['basin_radius_m']; jh = e['jet_height_m']; iid = e['instance_id']
    pulse = 1 + .06 * math.sin(2 * math.pi * t * 1.3)
    return [(c + [0, .3, 0], np.array([r, .3, r]), np.array([.75, .73, .7]), 6, iid, np.eye(3)),
            (c + [0, .58, 0], np.array([r * .9, .04, r * .9]), np.array([.5, .7, .9]), 2, iid, np.eye(3)),
            (c + [0, .6 + jh * pulse / 2, 0], np.array([.18, jh * pulse / 2, .18]), np.array([.7, .85, 1.]), 2, iid, np.eye(3)),
            (c + [0, .6 + jh * pulse * .92, 0], np.array([.55, .18, .55]), np.array([.7, .85, 1.]), 2, iid, np.eye(3))]


def render(cam, env, prims, width=W, height=H, with_sil=False):
    origin = np.array(cam['position'], float); fwd = norm(np.array(cam['target'], float) - origin)
    right = norm(np.cross(fwd, [0, 1, 0])); up = np.cross(right, fwd)
    xx, yy = np.meshgrid((np.arange(width) + .5) / width * 2 - 1, 1 - (np.arange(height) + .5) / height * 2)
    tan = math.tan(math.radians(cam['vertical_fov_deg'] / 2))
    # Screen right is -right here, i.e. +X for a camera looking toward +Z (same convention as the earlier studies).
    ray = norm(fwd - xx[..., None] * right * tan * width / height + yy[..., None] * up * tan)
    ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    dist = np.full((height, width), np.inf); rgb = np.zeros((height, width, 3))
    sem = np.full((height, width), SEM['sky'] if env.get('sky', True) else 0, np.uint8); ins = np.zeros((height, width), np.uint8)
    rgb[:] = [.72, .84, .95] if env.get('sky', True) else [.3, .3, .3]
    g = env['ground']
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = -origin[1] / ray[..., 1]
    m = (tg > 0) & np.isfinite(tg)
    dist[m] = tg[m]; rgb[m] = g.get('color', [.6, .6, .6]); sem[m] = SEM[g.get('semantic', 'road_paved')]
    for b in env.get('backdrop', []):
        with np.errstate(divide='ignore', invalid='ignore'):
            tb = (b['z'] - origin[2]) / ray[..., 2]
        p = origin + ray * np.where(np.isfinite(tb), tb, 0)[..., None]
        m = (tb > 0) & (tb < dist) & (p[..., 0] >= b['x_range'][0]) & (p[..., 0] <= b['x_range'][1]) & (p[..., 1] >= 0) & (p[..., 1] <= b['height'])
        dist[m] = tb[m]; rgb[m] = b.get('color', [.8, .8, .8]); sem[m] = SEM[b.get('semantic', 'building_structure')]
    light = norm([-2, 5, -4])
    sil = {}
    for prim in prims:
        c, r, col, label, iid, rot = prim[:6]
        shape = prim[6] if len(prim) > 6 else 'e'
        if shape == 'b':
            # Ray / oriented-box slab test in the box frame.
            o = (origin - c) @ rot; d = ray @ rot
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (-r - o) / d; t2 = (r - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), -1); tmax = np.nanmin(np.maximum(t1, t2), -1)
            hit = (tmax >= np.maximum(tmin, 0)) & np.isfinite(tmin) & np.isfinite(tmax)
            tt = np.where(tmin > 0, tmin, tmax)
            face = np.argmax(np.minimum(t1, t2), -1)
        else:
            o = ((origin - c) @ rot) / r; d = (ray @ rot) / r
            aa = np.sum(d * d, -1); bb = np.sum(d * o, -1); cc = np.dot(o, o) - 1
            disc = bb * bb - aa * cc; tt = (-bb - np.sqrt(np.maximum(disc, 0))) / aa
            hit = disc >= 0
        if with_sil and iid:
            s_hit = hit & (tt > 0)
            sil[iid] = sil[iid] | s_hit if iid in sil else s_hit
        m = hit & (tt > 0) & (tt < dist)
        if not m.any():
            continue
        pos = origin + ray[m] * tt[m, None]
        if shape == 'b':
            n = np.zeros((int(m.sum()), 3)); fm = face[m]
            loc = (pos - c) @ rot
            n[np.arange(len(fm)), fm] = np.sign(loc[np.arange(len(fm)), fm])
            n = n @ rot.T
        else:
            n = (((pos - c) @ rot) / (r * r)) @ rot.T
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        rgb[m] = np.asarray(col) * (.54 + .46 * np.maximum(n @ light, 0))[:, None]
        dist[m] = tt[m]; sem[m] = label; ins[m] = iid
    depth = dist * np.sum(ray * fwd, -1)
    depth[~np.isfinite(depth)] = 0.
    out = (np.clip(rgb * 255, 0, 255).astype(np.uint8), depth.astype('<f4'), sem, ins)
    return out + (sil,) if with_sil else out


def foreground_ids(scm):
    return {e['instance_id'] for e in scm['E'] if e['type'] in FOREGROUND_TYPES and e.get('layer', 'foreground') == 'foreground'}


def compile_prompt(scm):
    """CWM-style caption from the SCM text fields (deterministic)."""
    tx = scm['text']; env = scm['environment']; cam = camera_spec(scm)
    ents = [e for e in scm['E']]
    lines = ['Appearance and scene:', '', tx['scene'].strip(), env.get('description', '').strip(), '']
    if scm.get('style', '').strip():
        lines[-1:-1] = [f"Visual style of the whole video: {scm['style'].strip()}"]
    for e in ents:
        lines.append(f"{e['id']}: {e['appearance'].strip()}")
    if camera_moves(cam):
        motion = (cam.get('motion') or '').strip() or 'the camera moves as in Video 1.'
        cam_line = (f'Camera movement: {motion} One continuous shot, no cuts. The first frame matches Picture 1; '
                    'preserve its scene appearance and lighting.')
    else:
        cam_line = 'The camera is fixed for the entire shot: no pan, zoom, orbit or cut. Preserve the exact scene appearance, framing and lighting of Picture 1.'
    lines += ['', cam_line, '', 'Timeline:', '']
    for a, b, s in tx['timeline']:
        lines.append(f'{float(a):.2f}–{float(b):.2f}s: {s.strip()}')
    lines += ['', 'Constraints:', '']
    lines += [f'- {c.strip()}' for c in tx.get('constraints', [])]
    lines.append('- Silent, photographic, no on-screen text, labels or captions.')
    return '\n'.join(lines).strip() + '\n'


def t2v_prompt(scm):
    """Text-only prompt for the native H3 first-frame generation (same content, explicit framing)."""
    cam = camera_spec(scm)
    how = 'a camera that starts' if camera_moves(cam) else 'a completely static camera'
    return (f"One continuous 5.1667-second photographic video, 124 frames at 24 fps, landscape 1344 by 768, captured by {how} "
            f"at {cam['position']} metres looking at {cam['target']} with a {cam['vertical_fov_deg']} degree vertical field of view (Y up, ground at Y=0). "
            + compile_prompt(scm).replace('Preserve the exact scene appearance, framing and lighting of Picture 1.', ''))


def load(path):
    with open(path) as f:
        return json.load(f)
