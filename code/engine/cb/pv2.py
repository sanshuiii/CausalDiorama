"""Reference pictures, captions and job specs from plans A / B.

Plans follow scm_schema_pv2.md (camera node C, hold events, self_moving, edit declaration in plan B). Design:
 - one single-view picture per entity, the side the camera sees at t = 0 (no three-view sheets);
 - plan B reuses plan A's picture unless the entity is in edit.look_changed or the style changed; a size, position or
   camera change keeps every picture, including the empty place (pictures fix appearance only; the white model fixes the
   view); a changed place (edit.background_changed) or style is an edit of A's picture; new entities are drawn anew;
 - the first frame comes from H3 Ref2VA + CWM LoRA (free first frame, full 124-frame window, frame 0), is checked by an
   independent agent and regenerated at most twice; plan B never uses plan A's generated first frame or video.
augment() adds `reference` fields: {key, source t2v | inherit | edit | same_as, prompt, size[, from]}.
"""
import copy, math
import numpy as np

import world

DEFAULT_STYLE = 'photorealistic live-action footage, natural colours'
BG = (1344, 768)
MAX_PICS_S1, MAX_PICS_S2 = 9, 8          # H3 Ref2VA: at most 9 pictures (stage 2: the first frame + 8)
import prompt_tpl as PT  # noqa: E402  (prompts/)
KEEP = tuple(PT.text('video/caption/stage1_removed.txt').split('\n'))   # removed from the stage-1 caption body


def style(plan):
    return (plan.get('style') or '').strip().rstrip('.') or DEFAULT_STYLE


def bg_text(plan):
    return ((plan.get('background') or {}).get('appearance') or plan['environment'].get('description', '')).strip()


def label(e):
    return e.get('label') or e['id'].replace('_', ' ')


def height(e):
    return float(e.get('height_m') or (e.get('size_m') or [0, 1, 0])[1])


def frame0(plan):
    """Per entity at t = 0: unoccluded silhouette box (normalised), screen area, how the camera sees it."""
    sc = world.Scene(plan)
    st, rgb, depth, sem, ins, sil = sc.render(0, with_sil=True)
    cam = sc.camera(0); o = np.array(cam['position']); t = np.array(cam['target'])
    fwd = t - o; fwd[1] = 0; fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    right = -world.norm(np.cross(fwd, [0, 1, 0]))
    out = {}
    for e in plan['E']:
        if e['type'] not in world.FOREGROUND_TYPES:
            continue
        m = sil.get(e['instance_id'])
        box = None
        if m is not None and m.any():
            ys, xs = np.nonzero(m)
            box = [xs.min() / world.W, ys.min() / world.H, (xs.max() + 1) / world.W, (ys.max() + 1) / world.H]
        c = sc.root(e['id'], 0.) + np.array([0., height(e) / 2, 0.])
        d = o - c; elev = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        vert = 'seen from directly above' if elev > 60 else 'seen from above at an angle' if elev > 25 else 'seen from below' if elev < -25 else 'seen at eye level'
        view = vert
        if e['type'] in ('human', 'animal', 'vehicle'):
            h = math.radians(sc.heading(e['id'], 0)); f = np.array([math.sin(h), 0., math.cos(h)])
            to_cam = np.array([d[0], 0., d[2]]); to_cam /= max(np.linalg.norm(to_cam), 1e-9)
            k = float(f @ to_cam); side = 'right' if f @ right > 0 else 'left'
            facing = ('from the front, facing the viewer' if k > 0.75 else f'in three-quarter view from the front, turned to the {side}' if k > 0.3
                      else f'from the side, facing {side}' if k > -0.3 else f'in three-quarter view from behind, turned to the {side}' if k > -0.75
                      else 'from behind')
            view = f'seen {facing}, ' + vert.replace('seen ', '')
        out[e['id']] = {'box': [round(float(v), 4) for v in box] if box else None, 'view': view,
                        'area': float((box[2] - box[0]) * (box[3] - box[1])) if box else 0.0}
    return out


def pic_size(e, f0):
    b = f0.get(e['id'], {}).get('box')
    if b:
        aspect = (b[2] - b[0]) * world.W / max((b[3] - b[1]) * world.H, 1e-6)
    elif e['type'] == 'human':
        aspect = 0.4
    else:
        s = e.get('size_m') or [1, 1, 1]; aspect = float(s[0]) / max(float(s[1]), 1e-6)
    return [832, 1216] if aspect < 0.75 else [1216, 832] if aspect > 1.33 else [1024, 1024]


def ref_prompt(e, view, st):
    app = e['appearance'].strip().rstrip('.')
    if e['type'] == 'human' or e.get('semantic') == 'human':
        return PT.fill('image/entity_person.txt', APP=app, VIEW=view, STYLE=st)
    what = 'animal' if e['type'] == 'animal' or e.get('semantic') == 'animal' else label(e)
    return PT.fill('image/entity.txt', APP=app, LABEL=label(e), VIEW=view, WHAT=what, STYLE=st)


def camera_phrase(plan):
    cam = world.Scene(plan).camera(0); p, t = np.array(cam['position']), np.array(cam['target'])
    d = float(np.linalg.norm(t - p)); pitch = math.atan2(t[1] - p[1], math.hypot(t[0] - p[0], t[2] - p[2]))
    where = 'a low, near-ground' if p[1] < 0.8 else 'an eye-level' if p[1] <= 2.2 else 'an elevated'
    look = ' looking steeply down' if pitch < math.radians(-35) else ' looking down' if pitch < math.radians(-12) else ' looking up' if pitch > math.radians(12) else ''
    width = 2 * d * math.tan(math.radians(cam['vertical_fov_deg']) / 2) * 16 / 9
    return PT.fill('image/camera_phrase.txt', WHERE=where, LOOK=look, WIDTH=f'{width:.1f}')


def bg_prompt(plan):
    return PT.fill('image/empty_place.txt', BG=bg_text(plan).rstrip('.'), CAMERA=camera_phrase(plan), STYLE=style(plan))


def edit_size(size, max_side=1024, area=1024 * 1024):
    """Output size of a picture edit: the size the pipeline gives the condition picture (Qwen-Image-2.1 `calculate_dimensions` at
    output_resolution 1024 on the input after the worker's max_side resize). An edit drawn at any other size comes out covered in
    grain (e.g. 1216x832 noisy, 1248x832 clean for the same 1216x832 input)."""
    w, h = size
    if max(w, h) > max_side:
        s = max_side / max(w, h); w, h = round(w * s), round(h * s)
    width = math.sqrt(area * w / h)
    return [round(width / 32) * 32, round(width / (w / h) / 32) * 32]


def edit_prompt(a, b, sa, sb):
    parts = []
    if a['appearance'].strip() != b['appearance'].strip() or label(a) != label(b) or a['type'] != b['type']:
        parts.append(PT.fill('image/edit_entity_look.txt', APP=b['appearance'].strip().rstrip('.'), LABEL=label(b)))
    if sa != sb:
        parts.append(PT.fill('image/edit_style.txt', STYLE=sb))
    return PT.fill('image/edit_entity.txt', APP=a['appearance'].strip().rstrip('.'), CHANGES='; '.join(parts) or PT.text('image/edit_nothing.txt'))


def bg_edit_prompt(A, B):
    parts = []
    if bg_text(A) != bg_text(B):
        parts.append(PT.fill('image/edit_place_look.txt', BG=bg_text(B).rstrip('.')))
    if style(A) != style(B):
        parts.append(PT.fill('image/edit_style.txt', STYLE=style(B)))
    return PT.fill('image/edit_empty_place.txt', CHANGES='; '.join(parts))


def _sig(e):
    return (e['type'], label(e), e['appearance'].strip())


def augment(plan_a, plan_b=None):
    """Plans A and B with `reference` fields (B's pictures are inherited, edited or new according to its edit declaration)."""
    A = copy.deepcopy(plan_a); fa = frame0(A)
    A['background'] = dict(A.get('background') or {}, appearance=bg_text(A))
    A['background']['reference'] = {'key': 'bg', 'source': 't2v', 'prompt': bg_prompt(A), 'size': list(BG)}
    seen = {}
    for e in A['E']:
        if _sig(e) in seen:
            e['reference'] = {'key': seen[_sig(e)], 'source': 'same_as', 'same_as': seen[_sig(e)]}; continue
        e['reference'] = {'key': e['id'], 'source': 't2v', 'prompt': ref_prompt(e, fa.get(e['id'], {}).get('view', 'seen at eye level'), style(A)),
                          'size': pic_size(e, fa)}
        seen[_sig(e)] = e['id']
    if plan_b is None:
        return A, None
    B = copy.deepcopy(plan_b); fb = frame0(B); ed = B.get('edit') or {}
    restyle = bool(ed.get('style_changed')) or style(A) != style(B)
    B['background'] = dict(B.get('background') or {}, appearance=bg_text(B))
    if (ed.get('background_changed') or bg_text(A) != bg_text(B)) or restyle:
        # drawn anew from B's text: an edited empty place comes out heavily textured at any size
        B['background']['reference'] = {'key': 'bg_B', 'source': 't2v', 'prompt': bg_prompt(B), 'size': list(BG)}
    else:
        B['background']['reference'] = {'key': 'bg', 'source': 'inherit'}
    ea = {e['id']: e for e in A['E']}
    look = set(ed.get('look_changed') or [])
    seen = {}
    for e in B['E']:
        if _sig(e) in seen:
            e['reference'] = {'key': seen[_sig(e)], 'source': 'same_as', 'same_as': seen[_sig(e)]}; continue
        a = ea.get(e['id'])
        if a is not None and e['id'] not in look and not restyle:
            e['reference'] = {'key': a['reference']['key'], 'source': 'inherit'}
        elif a is not None:
            e['reference'] = {'key': e['id'] + '_B', 'source': 'edit', 'from': a['reference']['key'],
                              'prompt': edit_prompt(a, e, style(A), style(B)), 'size': edit_size(a['reference'].get('size') or pic_size(a, fa))}
        else:
            e['reference'] = {'key': e['id'] if e['id'] not in ea else e['id'] + '_B', 'source': 't2v',
                              'prompt': ref_prompt(e, fb.get(e['id'], {}).get('view', 'seen at eye level'), style(B)), 'size': pic_size(e, fb)}
        seen[_sig(e)] = e['reference']['key']
    return A, B


def ref_table(A, B=None):
    out = {}
    for plan in [p for p in (A, B) if p]:
        for node in [plan['background']] + plan['E']:
            r = node['reference']
            if r['source'] in ('t2v', 'edit') and r['key'] not in out:
                out[r['key']] = {k: r[k] for k in ('source', 'prompt', 'from', 'size') if k in r}
    return out


def pictures(plan, f0, cap):
    """Picture keys: the empty place first, then at most cap-1 entity pictures, largest at t = 0 first."""
    keys = []
    for e in sorted(plan['E'], key=lambda e: -f0.get(e['id'], {}).get('area', 0.0)):
        k = e['reference']['key']
        if k not in keys and len(keys) < cap - 1:
            keys.append(k)
    keys = [plan['background']['reference']['key']] + keys
    return keys, {e['id']: (keys.index(e['reference']['key']) if e['reference']['key'] in keys else None) for e in plan['E']}


def caption(plan, f0, stage, additions=()):
    """Stage 1 (free first frame): Picture 1 = the empty place, then the entity pictures. Stage 2 (CWM, locked first frame):
    Picture 1 = the first frame, Picture 2 = the empty place, then the entity pictures."""
    off = 1 if stage == 1 else 2
    keys, idx = pictures(plan, f0, MAX_PICS_S1 if stage == 1 else MAX_PICS_S2)
    lines = [PT.text('video/caption/header.txt'), PT.fill('video/caption/environment.txt', BG=bg_text(plan), PIC=off)]
    for i, e in enumerate(plan['E'], 2):
        app = e['appearance'].strip().rstrip('.')
        if idx[e['id']] is not None:
            lines.append(PT.fill('video/caption/subject_with_picture.txt', I=i, ID=e['id'], APP=app, PIC=idx[e['id']] + off))
        else:
            lines.append(PT.fill('video/caption/subject.txt', I=i, ID=e['id'], APP=app))
    body = world.compile_prompt(plan)
    if stage == 1:
        lines.append(PT.text('video/caption/pictures_stage1.txt'))
        for k in KEEP:
            body = body.replace(k, '')
    else:
        lines.append(PT.text('video/caption/pictures_stage2.txt'))
    extra = ('\n' + PT.text('video/caption/first_frame_requirements.txt') + ' '.join(a.strip() for a in additions if a.strip()) + '\n') if additions else ''
    return '\n'.join(lines) + PT.fill('video/caption/tail.txt', BODY=body, EXTRA=extra), keys
