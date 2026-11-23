"""First-frame routing helpers for an edit (plan A -> plan B): the camera pose and the entity placements at t=0, and a tolerant
comparison. The route itself (reuse / edit / regen) is engine/ws/flexrun.route_flex.
"""


def cam0(plan):
    """the camera's pose at t=0 (position, target, vertical field of view): the first camera keyframe; fields that keyframe does
    not set fall back to C's own pose, as world.py does when rendering"""
    c = plan['C']
    if c.get('controller') == 'keyframes' and c.get('keyframes'):
        k = sorted(c['keyframes'], key=lambda x: x[0])[0][1]
        return k.get('position', c.get('position')), k.get('target', c.get('target')), k.get('vertical_fov_deg', c.get('vertical_fov_deg'))
    return c.get('position'), c.get('target'), c.get('vertical_fov_deg')


def place0(plan):
    """entity -> (x, z, heading, height) at t=0 from its root track (else its E position)."""
    out = {}
    for e in plan.get('E', []):
        p = e.get('position') or [0, 0, 0]
        out[e['id']] = [p[0], p[2] if len(p) > 2 else 0, e.get('heading_deg', 0), e.get('height_m') or e.get('size_m')]
    for s in plan.get('S', []):
        if s.get('kind') == 'root_track' and s.get('keyframes') and s.get('entity') in out:
            k = sorted(s['keyframes'], key=lambda x: x[0])[0][1]
            out[s['entity']][0:2] = [k[0], k[2]] if len(k) >= 3 else [k[0], k[1]]  # free paths are 3-D (x, y, z), ground tracks (x, z)
            if s.get('heading') is not None:
                out[s['entity']][2] = s['heading']
    return out


def near(a, b, tol):
    if a is None or b is None:
        return a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(near(x, y, tol) for x, y in zip(a, b))
    return abs(a - b) <= tol
