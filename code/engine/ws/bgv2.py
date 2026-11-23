"""Empty-place rule v2: how the empty place (the environment picture, Picture 2 of the video model) of plan A and plan B is made.

A picture of the place drawn from text does not share the first frame's layout, and the video model then cuts to it mid-clip, so
the empty place is always made by removing the entities from a real frame, never drawn.

Plan A's empty place: up to three removal passes in a row on the real frame 0. Pass 1 is bg_raw (pictures, wave 1). A check agent
looks at the passes made so far; when the last one is clean, or after pass 3, its choice stands (the pass with the fewest entities
left among those that still look like a plain photograph; a tie goes to the earlier pass), otherwise one more pass removes what is
left of the last one. The agent lists what is still recognisable. Removing parts of the place itself is accepted.

Plan B's empty place, by first-frame route (flexrun.route_flex):
    regen           unchanged: bg_B drawn from plan B's text when the background or style changes, else plan A's empty place
                    (the stage-1 first frame is generated from it, so the two agree anyway)
    reuse           plan A's empty place
    edit            the background or style changes (the editor's background_changed / style_changed): the same three passes on
                    plan B's first frame (ff_v4) and the same check; otherwise plan A's empty place
When plan B uses plan A's empty place and an entity the edit removes, moves or restyles is still recognisable in it, group B gets
no environment picture at all (the leftover would pull the old state back).

Files: checks/bg2/<n>/ (plan A: e1..e3.png, the pass jobs, agent_input.md, choice.json) and checks/bgB/<n>/ (plan B: f1..f3.png ...).
"""
import json, os, re, shutil, sys
from pathlib import Path

FR = Path('@CD_WORK@/fr') / os.environ.get('CD_TREE', '')   # the same run tree as flexrun
SIDES = {'A': ('bg2', 'e'), 'B': ('bgB', 'f')}
sys.path.insert(0, '@CD_CODE@/engine/cb'); import prompt_tpl as PT  # noqa: E402  (prompts/)


def d(n, side):
    return FR / 'checks' / SIDES[side][0] / str(n)


def img(n, side, k):
    return d(n, side) / f'{SIDES[side][1]}{k}.png'


def b_mode(A2, B2, route):
    """'erase_ff' | 'inherit' | 'keep' (regen: whatever pv2.augment chose)"""
    ed = B2.get('edit') or {}
    if route == 'regen':
        return 'keep'
    if route == 'edit' and (ed.get('background_changed') or ed.get('style_changed')):
        return 'erase_ff'
    return 'inherit'


def set_b_reference(A2, B2, route):
    """plan B's background reference under the v2 rule (called in flexrun.assets after pv2.augment)"""
    mode = b_mode(A2, B2, route)
    if mode == 'erase_ff':
        B2['background']['reference'] = {'key': 'bg_B', 'source': 'erase_ff', 'note': 'plan B first frame, entities removed (bgv2)'}
    elif mode == 'inherit':
        B2['background']['reference'] = {'key': 'bg', 'source': 'inherit'}
    return mode


def pass_job(M2, n, side, k, src, labels, texts):
    """one Qwen removal pass k on picture src (k = 1: the first pass, same prompt as bg_raw; later passes: what is left)"""
    W, H = M2.GW, M2.GH; out = img(n, side, k); d(n, side).mkdir(parents=True, exist_ok=True)
    if k == 1:
        prompt = PT.fill('image/remove_entities.txt', LABELS=labels)
    else:
        prompt = PT.fill('image/remove_remaining.txt', TEXTS=texts)
    return M2.write(d(n, side) / f'pass{k}_job.json', dict(id=f'bgv2_{side}{k}_{n}', images=[{'path': str(src), 'max_side': max(W, H)}],
                                                         width=W, height=H, seed=M2.SEED, steps=40, out=str(out), prompt=prompt))


def visible_A(n):
    """plan A's entities seen in the real frame 0: those with a frame-0 mask in the (checked) perception. Only these are named in
    the removal prompt: naming an entity that is not in the frame can make the image editor draw it."""
    p = FR / 'percept' / f'{n}_checked'
    base = (p if p.exists() else FR / 'percept' / str(n)) / 'masks'
    return {q.name[4:-4] for q in base.glob('000_*.png')}


def visible_B(pv2, B2):
    """plan B's entities in view at t = 0 (a box in its white model's frame 0); the clip geometry must be set (flexrun.use)"""
    return {k for k, v in pv2.frame0(B2).items() if v['box']}


def first_from(n, side, how):
    d(n, side).mkdir(parents=True, exist_ok=True); (d(n, side) / 'pass1_from.txt').write_text(how)


def entity_list(n, side, B2=None):
    """(id, short text, appearance) of the entities that must be gone: plan A's entity list, or plan B's entities"""
    if side == 'A':
        ents = json.loads((FR / 'plan' / str(n) / 'entities.json').read_text())['entities']
        return [(e['id'], e.get('text', ''), e.get('appearance', '')) for e in ents]
    return [(e['id'], e.get('label') or e['id'].replace('_', ' '), e.get('appearance', '')) for e in B2['E']]


def passes(n, side):
    """the number of removal passes made so far (they are made in order)"""
    return max([k for k in (1, 2, 3) if img(n, side, k).exists()], default=0)


def check_input(n, side, first, B2=None, k=3):
    """the check agent's task file for the first k passes (one context-free agent; it reads the pictures and writes choice.json)"""
    dd = d(n, side); out = dd / 'choice.json'; ents = entity_list(n, side, B2)
    what = PT.text('agents/bgcheck/what_A.txt') if side == 'A' else PT.text('agents/bgcheck/what_B.txt')
    nth = ['first', 'second', 'third']
    cands = [PT.fill('agents/bgcheck/candidate.txt', J=j, IMG=img(n, side, j), NTH=nth[j - 1],
                     ON=PT.text('agents/bgcheck/candidate_on_image0.txt') if j == 1 else PT.fill('agents/bgcheck/candidate_on_previous.txt', PREV=j - 1))
             for j in range(1, k + 1)]
    src = d(n, side) / 'pass1_from.txt'
    if src.exists() and src.read_text().startswith('copy'):
        cands[0] = PT.fill('agents/bgcheck/candidate_copy.txt', IMG=img(n, side, 1))
    md = PT.lines('agents/bgcheck/task.md', dict(CANDIDATES=cands, ENTITIES=[PT.fill('agents/bgcheck/entity_line.txt', ID=i, TEXT=t, APP=a) for i, t, a in ents]),
                  FIRST=first, WHAT=what, TIMES=PT.text('agents/bgcheck/times_once.txt') if k == 1 else PT.fill('agents/bgcheck/times_k.txt', K=k),
                  CHOICES=' | '.join(str(j) for j in range(1, k + 1)), OUT=out, N4=f'{n:04d}')
    (dd / 'agent_input.md').write_text(md + '\n'); return dd / 'agent_input.md', out


def apply(n, side):
    """copy the chosen pass to the picture the stage-2 specs read: plan A -> pics/<n>/bg.png, plan B -> pics/<n>/bg_B.png"""
    c = json.loads((d(n, side) / 'choice.json').read_text()); k = int(c.get('choice', 1))
    dst = FR / 'pics' / str(n) / ('bg.png' if side == 'A' else 'bg_B.png')
    shutil.copy2(img(n, side, k), dst); return k, c


def no_env_picture(n, B2, route_rec):
    """group B gets no environment picture: plan B uses plan A's empty place and an entity the edit removes, moves or restyles is
    still recognisable in it"""
    if B2['background']['reference'].get('key') != 'bg':
        return False
    p = d(n, 'A') / 'choice.json'
    if not p.exists():
        return False
    left = set(json.loads(p.read_text()).get('left') or [])
    ed = B2.get('edit') or {}
    hit = set(ed.get('removed') or []) | set(route_rec.get('moved') or []) | set(ed.get('look_changed') or [])
    return bool(left & hit)


def drop_env(cap, keys, stage):
    """caption and picture keys without the environment picture: the environment line loses its picture, later pictures move up"""
    off = 1 if stage == 1 else 2
    cap = cap.replace(f' Its look follows <Picture {off}>; it contains none of the other subjects.', ' It contains none of the other subjects.')
    cap = re.sub(r'<Picture (\d+)>', lambda m: f'<Picture {int(m.group(1)) - 1}>' if int(m.group(1)) > off else m.group(0), cap)
    return cap, keys[1:]
