"""Editing engine: one run tree, from a source video and an edit instruction to the fitted plan A, the edited plan B, plan B's first
frame and the edited video. A revision tree (runs/known_A.json) starts with plan A, its pictures and its video taken from a Causal
Diorama, and its video B is regenerated with the seed of video A instead of the F4 inversion.

Run tree FR = WORK/fr/<CD_TREE> (bin/cd_edit.py and bin/cd_revise.py create it, engine/cd/runs.py):
  runs/        checklist.json (id, source description, edit instruction), sources.json (source probe), known_A.json (revision only),
               no_video (only the dioramas: stop before the video of B)
  src/<n>.mp4  frames/<n>/NNN.png (every 12th frame + the last)  sheets/<n>.jpg  geom/<n>.json  schema/<n>.md (the schema with this
  clip's duration)  plan/<n>/  percept/<n>[_checked]/  pics/<n>/  render/<n>_{A,B}/  checks/{percept,bg2,bgB}/<n>/  v4ff/<n>/  fix/<n>/
  jobs/  state/
Clip geometry: frames = 17n+5 <= duration x 24 at 24 fps, up to 362 (15.08 s); W x H keeps the source aspect at about 1344x768 pixels,
multiples of 64 (shared/inv/code/flex_geom.py).
Steps (`auto` advances every clip as far as the finished work allows; agents run in llm_run.py, node tasks on the GPU runners):
  prep           geometry, source clip (centre crop, 24 fps), frames, contact sheet, schema copy
  entities       entity-analyst agent (prompts/agents/entities_flex.md)
  percept        Grounding DINO + SAM 2.1 + metric depth on the sampled frames (node), then the perception-check agent (pcheck)
  camtrack       the measured camera: background feature tracks (node, CPU), turned into the camera node C by cam_track.py
  planner        fitting planner agent -> plan A (prompts/agents/planner_flex.md)
  editor         editor agent -> plan B (prompts/agents/editor_B.md followed by prompts/agents/editor_addendum.md)
  assets         reference pictures: cut-outs of plan A's entities, plan B's pictures (Qwen-Image-2.1)
  empty place    rule v2 (bgv2.py): removal passes on frame 0 and a check agent (bgcheck2); plan B's by route (bgcheckB)
  first frame    route_flex: reuse frame 0, edit it (first-frame instruction agent v4ins + Qwen-Image-2.1), or regenerate it (stage 1)
  finish         editing: stage-2 conditions A and B, CWM caches, F4 inversion (shared/inv/code/inv_fix_v4.py);
                 revision: stage-2 conditions B, CWM caches, generation with the seed of video A (shared/tools/cwm/generate_many.py)

    python3 engine/ws/flexrun.py prep IDS          (IDS: comma list of clip ids)
    python3 engine/ws/flexrun.py auto IDS|all [--dry]
"""
import argparse, os, json, math, shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; CODE = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODE.parent / 'shared' / 'inv' / 'code'))
import flex_geom as G  # noqa: E402  (geometry_for, derived)
import bgv2  # noqa: E402  (empty-place rule v2)
sys.path.insert(0, '@CD_CODE@/engine/cb'); import prompt_tpl as PT  # noqa: E402  (prompts/)

SUB = os.environ.get('CD_TREE', '')   # the run tree under WORK/fr
FR = Path('@CD_WORK@/fr') / SUB
META = FR / 'runs'
NS = Path('@CD_SHARED@')   # the shared directory of the nodes: node code, models, results
FF = '@CD_FFMPEG@'
SCHEMA = Path('@CD_CODE@/engine/cb/scm_schema_pv2.md')
FONT = '@CD_CODE@/assets/fonts/NotoSansSC-Regular.otf'
ADDENDUM = Path('@CD_CODE@/prompts/agents/editor_addendum.md')   # appended to every editor task


def ids(arg):
    return [int(x) for x in arg.split(',') if x]


def known_a():
    """revision tree: the Causal Diorama plan A came from (bin/cd_revise.py writes runs/known_A.json, with the seed of video A)"""
    f = META / 'known_A.json'
    return json.loads(f.read_text()) if f.exists() else None


def seed():
    """the H3 seed of every stage-1 and stage-2 generation of this tree: the seed video A was generated with (revision), else SEED"""
    k = known_a()
    return int(k['seed']) if k else mods()['M2'].SEED


def probe():
    return {t['id']: t for t in json.loads((META / 'sources.json').read_text())}


def checklist():
    return {t['id']: t for t in json.loads((META / 'checklist.json').read_text())}


def geom(n):
    return json.loads((FR / 'geom' / f'{n}.json').read_text())


def gstr(n):
    g = geom(n); return f"{g['frames']},{g['width']},{g['height']}"


def frame_list(F):
    return sorted(set(range(0, F, 12)) | {F - 1})


def sheet_frames(F):
    fr = frame_list(F); return [fr[round(i * (len(fr) - 1) / 5)] for i in range(6)]


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, ensure_ascii=False, indent=1)); return str(path)


# ---------------------------------------------------------------- prep
def prep(a):
    P = probe()
    for n in ids(a.ids):
        t = P[n]; F, W, H = G.geometry_for(t['dur'], t['w'], t['h']); D = round(F / 24, 2)
        g = dict(frames=F, width=W, height=H, duration=D, source=t['file'], source_duration=t['dur'], source_size=[t['w'], t['h']], source_fps=t['fps'])
        out = FR / 'src' / f'{n}.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            vf = f'fps=24,scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,tpad=stop_mode=clone:stop_duration=2'
            subprocess.run([FF, '-loglevel', 'error', '-y', '-i', str(t['file']), '-vf', vf, '-frames:v', str(F), '-an', '-c:v', 'libx264',
                            '-crf', '12', '-pix_fmt', 'yuv420p', str(out)], check=True)
        fd = FR / 'frames' / str(n); fd.mkdir(parents=True, exist_ok=True); want = frame_list(F)
        if not all((fd / f'{f:03d}.png').exists() for f in want):
            tmp = fd / '_tmp'; tmp.mkdir(exist_ok=True)
            sel = '+'.join(f'eq(n\\,{f})' for f in want)
            subprocess.run([FF, '-loglevel', 'error', '-y', '-i', str(out), '-vf', f'select={sel}', '-vsync', '0', str(tmp / '%04d.png')], check=True)
            got = sorted(tmp.glob('*.png')); assert len(got) == len(want), (n, len(got), len(want))
            for f, p in zip(want, got):
                p.replace(fd / f'{f:03d}.png')
            tmp.rmdir()
        sh = FR / 'sheets' / f'{n}.jpg'; sh.parent.mkdir(parents=True, exist_ok=True)
        if not sh.exists():
            tw = 448 if W >= H else 256; th = round(tw * H / W); font = ImageFont.truetype(FONT, 18)
            im = Image.new('RGB', (3 * tw, 2 * (th + 26)), 'white'); dr = ImageDraw.Draw(im)
            for k, f in enumerate(sheet_frames(F)):
                x, y = (k % 3) * tw, (k // 3) * (th + 26)
                im.paste(Image.open(fd / f'{f:03d}.png').convert('RGB').resize((tw, th), Image.LANCZOS), (x, y + 26))
                dr.text((x + 4, y + 2), f'frame {f} ({f / 24:.2f}s)', fill='black', font=font)
            im.save(sh, quality=88)
        s = SCHEMA.read_text()
        for x, y in (('a 5.17-second, single-shot scene: 124 frames at 24 fps', f'a {D:.2f}-second, single-shot scene: {F} frames at 24 fps'),
                     ('"t1": 5.17', f'"t1": {D:.2f}'), ('0 to 5.17 s', f'0 to {D:.2f} s'), ('[0, 5.17]', f'[0, {D:.2f}]')):
            assert x in s, x
            s = s.replace(x, y)
        assert F == 124 or ('5.17' not in s and '124 frames' not in s), [l for l in s.splitlines() if '5.17' in l or '124 frames' in l]
        (FR / 'schema').mkdir(parents=True, exist_ok=True); (FR / 'schema' / f'{n}.md').write_text(s)
        write(FR / 'geom' / f'{n}.json', g)
        print(n, F, W, H, D, 's', t['file'])


# ---------------------------------------------------------------- prompts
def clip_vars(n):
    g = geom(n); F, W, H, D = g['frames'], g['width'], g['height'], g['duration']; fr = frame_list(F); ck = checklist()[n]
    return {'ID': f'real_{n:04d}', 'SHEET': str(FR / 'sheets' / f'{n}.jpg'), 'FRAMES': str(FR / 'frames' / str(n)),
            'DESC': ck['original_description'], 'INSTR': ck['editing_instruction'],
            'CLIP': f'{D:.2f} s, {F} frames at 24 fps, {W}x{H}' + (', portrait' if H > W else ''),
            'SHEETF': ', '.join(str(f) for f in sheet_frames(F)), 'FRAMELIST': f'000.png, 012.png, 024.png, ... {fr[-2]:03d}.png, {fr[-1]:03d}.png',
            'LASTF': str(F - 1), 'DUR': f'{D:.2f}', 'GEOM': gstr(n), 'SCHEMA': str(FR / 'schema' / f'{n}.md')}


def fill(template, n, out, **kv):
    v = clip_vars(n); v.update({k: str(x) for k, x in kv.items()})
    s = Path(template).read_text()
    for k, x in v.items():
        s = s.replace(f'@@{k}@@', x)
    assert '@@' not in s, [l for l in s.splitlines() if '@@' in l]
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(s); return out


def entities(a):
    for n in ids(a.ids):
        d = FR / 'plan' / str(n)
        print(fill(Path('@CD_CODE@/prompts/agents') / 'entities_flex.md', n, d / 'entities_prompt.md', OUT=d / 'entities.json'))


# ---------------------------------------------------------------- helper modules and the first-frame route
_M = {}


def mods():
    """make_stage2 / checkers / make_v4ff pointed at this run tree (they read their module globals at call time)"""
    if not _M:
        import make_stage2 as M2, checkers as C, make_v4ff as V4F
        import world, pv2, ff_prep, ff_route
        M2.R = FR; C.R = FR; C.CK = FR / 'checks'; V4F.R = FR; V4F.V = FR / 'v4ff'
        M2.CHECK = checklist()
        _M.update(M2=M2, C=C, V4F=V4F, world=world, pv2=pv2, ff_prep=ff_prep, ffr=ff_route)
    return _M


def cam_changed(ca, cb):
    """t=0 camera change that a new first frame is needed for: the position moves > 5 cm, the view direction turns > 1 degree or
    the vertical field of view changes > 1 degree"""
    (pa, ta, fa), (pb, tb, fb) = ca, cb
    if None in (pa, ta, pb, tb):
        return (pa, ta, fa) != (pb, tb, fb)
    if math.dist(pa, pb) > 0.05:
        return True
    da = [t - p for t, p in zip(ta, pa)]; db = [t - p for t, p in zip(tb, pb)]
    cos = sum(x * y for x, y in zip(da, db)) / max(1e-9, math.hypot(*da) * math.hypot(*db))
    if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > 1.0:
        return True
    return fa is not None and fb is not None and abs(fa - fb) > 1.0


def route_flex(A, B):
    """first-frame route of plan B: regenerate only when the edit asks for another camera, i.e. the editor declares a camera change
    (edit.camera_changed) and the t=0 view differs (cam_changed); otherwise the real frame 0 is edited (look / added / removed / style /
    background) or reused. A common entity placed elsewhere at t=0 (ground > 5 cm and its box shifted by more than 10% of the frame)
    is moved by the image edit (`moved`); smaller shifts are left to the video model. Headings are not compared."""
    m = mods(); pv2, ffr = m['pv2'], m['ffr']
    fa, fb = pv2.frame0(A), pv2.frame0(B); ed = B.get('edit', {})
    a0 = {k for k, v in fa.items() if v['box']}; b0 = {k for k, v in fb.items() if v['box']}
    look = sorted(set(ed.get('look_changed', [])) & (a0 | b0))
    added = sorted((set(ed.get('added', [])) & b0) | (b0 - a0)); removed = sorted((set(ed.get('removed', [])) & a0) | (a0 - b0))
    glob = [k for k in ('style_changed', 'background_changed') if ed.get(k)]
    if ed.get('camera_changed') and cam_changed(ffr.cam0(A), ffr.cam0(B)):
        return dict(route='regen', why=['the edit changes the camera and the t=0 view differs'], look=look, added=added, removed=removed, glob=glob)
    pa, pb = ffr.place0(A), ffr.place0(B)

    def shift(e):
        ba, bb = fa[e]['box'], fb[e]['box']
        return max(abs((ba[0] + ba[2]) - (bb[0] + bb[2])) / 2, abs((ba[1] + ba[3]) - (bb[1] + bb[3])) / 2)
    moved = sorted(e for e in (a0 & b0) - set(added) - set(removed) if e in pa and e in pb
                   and math.dist([float(x) for x in pa[e][0:2]], [float(x) for x in pb[e][0:2]]) > 0.05 and shift(e) > 0.10)
    if look or added or removed or glob or moved:
        return dict(route='edit', kind='global' if glob else 'local', look=look, added=added, removed=removed, glob=glob, moved=moved)
    return dict(route='reuse', look=look, added=added, removed=removed, glob=glob)


def g_region(d, mt):
    """the G region of a local first-frame edit, a moved entity counted on both sides: its old place (A, frame 0) and its new one (B)"""
    mv = mt.get('moved', [])
    return mods()['V4F'].g_region(d, dict(mt, added=mt['added'] + mv, removed=mt['removed'] + mv))


def use(n):
    """set every module's frame geometry to clip n (renders at 1/4 size, masks and plates at W x H)"""
    m = mods(); g = geom(n); F, W, H = g['frames'], g['width'], g['height']
    m['world'].FRAMES, m['world'].W, m['world'].H = F, W // 4, H // 4
    m['world'].render.__defaults__ = (W // 4, H // 4, False)  # render()'s width / height defaults are bound at import
    m['M2'].GW, m['M2'].GH = W, H; m['V4F'].W, m['V4F'].H = W, H
    m['ff_prep'].W, m['ff_prep'].H = W, H; m['pv2'].BG = (W, H)
    if (FR / 'percept' / f'{n}_checked').exists():
        m['M2'].PERCEPT_DIR[n] = f'{n}_checked'
    return m


def put(tid, task, a, node=None):
    mods()['M2'].put(tid, task, a.dry, node=node or a.node)


# ---------------------------------------------------------------- perception
OUTDOOR = ('sky', 'outdoor', 'street', 'field', 'lake', 'sea', 'beach', 'park', 'forest', 'garden', 'road', 'mountain', 'desert', 'river',
           'city', 'grass', 'snow', 'ruins', 'harbour', 'harbor', 'village', 'countryside', 'court', 'track')


def no_entities(*plans):
    """suffix for a drawn (t2v) empty place: the clip's entities named as absent (the generic "no people, no vehicles..." of the
    prompt alone can bring look-alikes of the entities)"""
    labels = sorted({pv2_label(e) for p in plans for e in p['E']})
    return PT.fill('image/empty_place_absent.txt', LABELS=', '.join(labels)) if labels else ''


def pv2_label(e):
    return mods()['pv2'].label(e)


def scene_of(ent):
    if ent.get('scene') in ('indoor', 'outdoor'):
        return ent['scene']
    b = (ent.get('background') or '').lower()
    return 'outdoor' if any(w in b for w in OUTDOOR) else 'indoor'


def percept(a):
    jobs = []
    for n in ids(a.ids):
        if (FR / 'percept' / str(n) / 'perception.json').exists():
            continue
        ent = json.loads((FR / 'plan' / str(n) / 'entities.json').read_text()); F = geom(n)['frames']
        model = 'Depth-Anything-V2-Metric-Outdoor-Small-hf' if scene_of(ent) == 'outdoor' else 'Depth-Anything-V2-Metric-Indoor-Small-hf'
        jobs.append(write(FR / 'jobs' / 'percept' / f'{n}.json', dict(
            id=f'real_{n:04d}', video=str(FR / 'src' / f'{n}.mp4'), frames=frame_list(F),
            entities=[dict(name=e['id'], text=e['text'], alt_text=e['alt_text']) for e in ent['entities']],
            depth_model=f'{NS}/models/{model}', out_dir=str(FR / 'percept' / str(n)))))
    if not jobs:
        return print('nothing to perceive')
    lst = FR / 'jobs' / f'percept{a.tag}.txt'; lst.write_text('\n'.join(jobs) + '\n')
    M2 = mods()['M2']
    put(f'f10_percept{a.tag}', dict(type='shell', retries=0, command=(  # one list per GPU; a clip that fails is queued again by auto
        f'set -o pipefail; {M2.PRE}; split -n r/8 -d {lst} {lst}.g && for i in 0 1 2 3 4 5 6 7; do [ -s {lst}.g0$i ] && '
        f'CUDA_VISIBLE_DEVICES=$i LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat HF_HUB_OFFLINE=1 {M2.PY} {M2.INV}/code/perceive.py {lst}.g0$i '
        f'> {M2.INV}/logs/f10_percept{a.tag}.g0$i.log 2>&1 & done; wait')), a)


def camtrack(a):
    """measured camera (plan A): background feature tracks between the detector frames, CPU only on a node (shared/inv/code/camtrack.py,
    OpenCV); engine/ws/cam_track.py turns them into the planner's camera node C"""
    jobs = []
    for n in ids(a.ids):
        out = FR / 'percept' / str(n) / 'camtrack.npz'
        if out.exists():
            continue
        pdir = FR / 'percept' / f'{n}_checked'; ent = json.loads((FR / 'plan' / str(n) / 'entities.json').read_text())
        jobs.append(write(FR / 'jobs' / 'camtrack' / f'{n}.json', dict(
            id=n, video=str(FR / 'src' / f'{n}.mp4'), frames=frame_list(geom(n)['frames']), depth_dir=str(pdir / 'depth'), mask_dir=str(pdir / 'masks'),
            types={e['id']: e['type'] for e in ent['entities']}, out=str(out))))
    if not jobs:
        return print('nothing to track')
    lst = FR / 'jobs' / f'camtrack{a.tag}.txt'; lst.write_text('\n'.join(jobs) + '\n')
    M2 = mods()['M2']
    put(f'f12_camtrack{a.tag}', dict(type='shell', retries=0, command=(
        f'set -o pipefail; {M2.PY} {INVC}/camtrack.py {lst} --procs 32 2>&1 | tee {M2.INV}/logs/f12_camtrack{a.tag}.log')), a)


def pcheck(a):
    m = mods()
    for n in ids(a.ids):
        use(n); m['C'].percept_inputs(argparse.Namespace(clips=str(n)))


def pcheck_apply(a):
    m = mods()
    for n in ids(a.ids):
        use(n); m['C'].percept_apply(argparse.Namespace(clips=str(n)))


# ---------------------------------------------------------------- planner / editor prompts
def planner(a):
    for n in ids(a.ids):
        d = FR / 'plan' / str(n); pdir = FR / 'percept' / f'{n}_checked'
        assert (pdir / 'perception.json').exists(), f'{n}: perception check not applied'
        print(fill(Path('@CD_CODE@/prompts/agents') / 'planner_flex.md', n, d / 'planner_prompt.md', PERCEPT=pdir, ENTITIES=d / 'entities.json',
                   OUT=d / 'scm_A.json', OUTDIR=d / 'checks_A', CAMTRACK=FR / 'percept' / str(n) / 'camtrack.npz'))


def _layout(x, ind=0):
    """JSON text of x with every line short where it can be: a value that fits in 160 characters stays on one line, longer
    objects and lists put one member per line (a long string stays whole: JSON cannot split it)"""
    one = json.dumps(x, ensure_ascii=False)
    if not isinstance(x, (dict, list)) or len(one) + ind <= 160:
        return one
    sp = ' ' * (ind + 1)
    if isinstance(x, dict):
        body = ',\n'.join(f'{sp}{json.dumps(k, ensure_ascii=False)}: {_layout(v, ind + 1)}' for k, v in x.items())
        return '{\n' + body + '\n' + ' ' * ind + '}'
    return '[\n' + ',\n'.join(sp + _layout(v, ind + 1) for v in x) + '\n' + ' ' * ind + ']'


def tidy_plan(path):
    """re-lay a plan file so that the editor agent can read all of it: its Read tool cuts lines after 2000 characters, and the measured
    camera written on one line (several thousand characters) would reach plan B cut short. The JSON content is unchanged; only line breaks move. Returns True when the file was rewritten."""
    text = Path(path).read_text()
    if max((len(l) for l in text.splitlines()), default=0) <= 1800:
        return False
    obj = json.loads(text); new = _layout(obj) + '\n'
    assert json.loads(new) == obj
    tmp = Path(path).with_suffix('.tidy.tmp'); tmp.write_text(new); os.replace(tmp, path)
    return True


def editor(a):
    tpl = (Path('@CD_CODE@/prompts/agents') / 'editor_B.md').read_text()
    old = '@CD_CODE@/engine/cb/scm_schema_pv2.md'; assert old in tpl
    for n in ids(a.ids):
        d = FR / 'plan' / str(n); v = clip_vars(n)
        lint = (f"OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 {HERE}/geomrun.py {v['GEOM']} @CD_CODE@/engine/cb/scm_lint.py "
                f"{d / 'scm_B.json'} --rules @CD_CODE@/engine/cb/rules_pv2.json --source {d / 'scm_A.json'} --out {d / 'checks_B' / 'lint'}")
        s = (tpl.replace(old, v['SCHEMA']).replace('@@SRC@@', str(d / 'scm_A.json')).replace('@@INSTR@@', v['INSTR'])
             .replace('@@OUT@@', str(d / 'scm_B.json')).replace('@@SID@@', v['ID']).replace('@@LINT@@', lint))
        assert '@@' not in s
        s = s.rstrip('\n') + '\n' + ADDENDUM.read_text()   # the editor addendum (environment motion, added-object motion, background replacement)
        (d / 'editor_prompt.md').write_text(s); print(d / 'editor_prompt.md')


# ---------------------------------------------------------------- reference pictures
def pic_job(n, key, prompt, size, out, images=None, seed=None):
    M2 = mods()['M2']; spec = dict(id=f'cd{n}_{key}', prompt=prompt, width=size[0], height=size[1], seed=seed or M2.SEED, steps=40, out=str(out))
    if images:
        spec['images'] = images
    return write(FR / 'jobs' / 'pics' / f'{n}_{key}.json', spec)


def assets(a):
    """wave 1: cut-outs of plan A's entities from the real frames (drawn from the description when never detected), the empty place
    pass 1 (Qwen removal on the real frame 0), plan B's pictures except those edited from the empty place (wave 2, after the check).
    Revision: plan A's pictures are the diorama's (already in pics/<n>/), and so is plan A's empty place (bg_raw.png)."""
    m = mods(); M2, pv2 = m['M2'], m['pv2']; jobs = []
    for n in ids(a.ids):
        use(n); d = FR / 'plan' / str(n); pd = FR / 'pics' / str(n); pd.mkdir(parents=True, exist_ok=True); W, H = M2.GW, M2.GH
        A = json.loads((d / 'scm_A.json').read_text()); B = json.loads((d / 'scm_B.json').read_text()); A2, B2 = pv2.augment(A, B)
        rec = dict(clip=n, pictures={}, wave2=[])
        mode = bgv2.set_b_reference(A2, B2, route_flex(A2, B2)['route'])   # rule v2: plan B's empty place by route
        for e in A2['E']:
            k = e['reference']['key']
            if e['reference']['source'] == 'same_as' or k in rec['pictures']:
                continue
            if known_a() and (pd / f'{k}.png').exists():
                rec['pictures'][k] = dict(source='diorama (plan A)'); continue
            info = M2.cutout(n, e['id'], pd / f'{k}.png')
            if info is None:
                jobs.append(pic_job(n, k, e['reference']['prompt'], e['reference']['size'], pd / f'{k}.png')); rec['pictures'][k] = dict(source='t2v (not detected)')
            else:
                rec['pictures'][k] = dict(source='cutout', **info)
        rm = ', '.join(sorted({pv2.label(e) for e in A2['E']}))
        if not (known_a() and (pd / 'bg_raw.png').exists()):
            jobs.append(pic_job(n, 'bg_raw', PT.fill('image/remove_entities.txt', LABELS=rm), (W, H), pd / 'bg_raw.png',
                                images=[{'path': str(M2.frame_png(n, 0)), 'max_side': max(W, H)}]))
        rec['pictures']['bg'] = dict(source='frame 0, entities removed (empty-place check)' if not known_a() else 'diorama (plan A)')
        if mode == 'erase_ff':
            rec['pictures']['bg_B'] = dict(source='plan B first frame, entities removed (bgv2)')
        for node in [B2['background']] + B2['E']:
            r = node['reference']
            if r['source'] == 'edit' and r['key'] not in rec['pictures']:
                if r['from'] == 'bg' or r['from'] not in rec['pictures'] or rec['pictures'][r['from']].get('source', '').startswith('t2v'):
                    rec['wave2'].append(r['key']); rec['pictures'][r['key']] = dict(source='edit (wave 2)', of=r['from']); continue
                src = rec['pictures'][r['from']]; size = pv2.edit_size(src['size']) if src.get('size') else r['size']
                jobs.append(pic_job(n, r['key'], r['prompt'], size, pd / f"{r['key']}.png", images=[{'path': str(pd / f"{r['from']}.png"), 'max_side': 1024}]))
                rec['pictures'][r['key']] = dict(source='edit', of=r['from'])
            elif r['source'] == 't2v' and r['key'] not in rec['pictures']:
                pr = r['prompt'] + (no_entities(A2, B2) if node is B2['background'] else '')   # a drawn empty place: name the entities as absent
                jobs.append(pic_job(n, r['key'], pr, r['size'], pd / f"{r['key']}.png")); rec['pictures'][r['key']] = dict(source='t2v')
        rec['wave1'] = ['bg_raw'] + [k for k in rec['pictures'] if k not in rec['wave2'] and k != 'bg' and not (k == 'bg_B' and mode == 'erase_ff')]
        write(d / 'A_v2.json', A2); write(d / 'B_v2.json', B2); write(d / 'assets.json', rec)
        print(n, len(rec['pictures']), 'pictures, wave 2:', rec['wave2'])
    put(f'f20_pics{a.tag}', dict(type='workers', script='qwen_image/worker.py', pythonpath=mods()['M2'].QDEPS, configs=jobs, retries=0), a)


def assets2(a):
    """wave 2 (after the empty-place check): plan B's pictures edited from the empty place or from a drawn picture"""
    M2 = mods()['M2']; jobs = []
    for n in ids(a.ids):
        use(n); d = FR / 'plan' / str(n); pd = FR / 'pics' / str(n); rec = json.loads((d / 'assets.json').read_text())
        B2 = json.loads((d / 'B_v2.json').read_text()); want = set(rec['wave2'])
        for node in [B2['background']] + B2['E']:
            r = node['reference']
            if r.get('key') in want and not (pd / f"{r['key']}.png").exists():
                src = pd / f"{r['from']}.png"; assert src.exists(), (n, r['key'], src)
                jobs.append(pic_job(n, r['key'], r['prompt'], r['size'], pd / f"{r['key']}.png", images=[{'path': str(src), 'max_side': 1024}]))
    if jobs:
        put(f'f30_pics2{a.tag}', dict(type='workers', script='qwen_image/worker.py', pythonpath=M2.QDEPS, configs=jobs, retries=0), a)
    else:
        print('no wave-2 pictures')


# ---------------------------------------------------------------- renders, first frame, stage 2
NR = '@CD_NODE_RUNS@/edit/' + SUB   # node-local case directories
INVC = f'{NS}/inv/code'


def plate(f0, rects, out, W, H):
    """the red-box plate (a 6-px red outline per added or moved entity) on the clip's own frame size"""
    im = Image.open(f0).convert('RGB').resize((W, H), Image.LANCZOS); d = ImageDraw.Draw(im)
    for r in rects:
        X0, Y0, X1, Y1 = r['box'][0] * W, r['box'][1] * H, r['box'][2] * W, r['box'][3] * H
        if X1 - X0 < 16:
            X0, X1 = (X0 + X1) / 2 - 8, (X0 + X1) / 2 + 8
        if Y1 - Y0 < 16:
            Y0, Y1 = (Y0 + Y1) / 2 - 8, (Y0 + Y1) / 2 + 8
        d.rectangle([X0 + 3, Y0 + 3, X1 - 3, Y1 - 3], outline=tuple(r['rgb']), width=6)
    Path(out).parent.mkdir(parents=True, exist_ok=True); im.save(out); return out


def renders(n):
    m = use(n); M2 = m['M2']
    for ver in ('A', 'B'):
        M2.render_local(json.loads((FR / 'plan' / str(n) / f'{ver}_v2.json').read_text()), FR / 'render' / f'{n}_{ver}')


def v4ins_lines(n, A, B, ed, look, added, removed, moved, glob, rects, pics, Ea, Eb, out):
    """the first-frame instruction agent's task file (prompts/agents/v4ins/), as lines. The agent gets the list of first-frame changes
    alone, not the edit instruction of the whole video (with it in view, agents wrote changes the first frame does not have)."""
    lines = PT.lines('agents/v4ins/head.md').split('\n') + ['']
    for e in look:
        lines.append(PT.fill('agents/v4ins/look.txt', E=e, A=Ea.get(e, {}).get('appearance', ''), B=Eb.get(e, {}).get('appearance', '')))
    for e in added:
        box = next((x for x in rects if x['entity'] == e), None)
        lines.append(PT.fill('agents/v4ins/added.txt', E=e, APP=Eb.get(e, {}).get('appearance', ''),
                             BOX=PT.text('agents/v4ins/added_box.txt') if box else PT.text('agents/v4ins/added_nobox.txt')))
    for e in removed:
        lines.append(PT.fill('agents/v4ins/removed.txt', E=e, APP=Ea.get(e, {}).get('appearance', '')))
    for e in moved:
        lines.append(PT.fill('agents/v4ins/moved.txt', E=e, APP=Eb.get(e, {}).get('appearance', '')))
    if 'style_changed' in glob:
        lines.append(PT.fill('agents/v4ins/style.txt', A=A.get('style', ''), B=B.get('style', '')))
    if 'background_changed' in glob:
        lines.append(PT.fill('agents/v4ins/background.txt', A=(A.get('background') or {}).get('appearance', ''), B=(B.get('background') or {}).get('appearance', '')))
    for k, p in enumerate(pics, 2):
        lines.append(PT.fill('agents/v4ins/picture.txt', K=k, E=p['entity']))
    return lines + PT.lines('agents/v4ins/tail.md', OUT=out, N4=f'{n:04d}').split('\n')


def v4(a):
    """renders A / B; first-frame route (route_flex); edit: agent input + plate + white-model masks + mask job specs; regen: stage-1 job"""
    m = mods(); V4F, pv2, ff_prep = m['V4F'], m['pv2'], m['ff_prep']; M2 = m['M2']
    for n in ids(a.ids):
        renders(n); use(n); W, H = M2.GW, M2.GH; g = geom(n)
        A, B = V4F.plans(n); r = route_flex(A, B); ed = B.get('edit', {}); d = FR / 'v4ff' / str(n); d.mkdir(parents=True, exist_ok=True)
        pd = FR / 'pics' / str(n); f0 = M2.frame_png(n, 0)
        if r['route'] == 'regen':
            f0b = pv2.frame0(B); cap1, keys1 = pv2.caption(B, f0b, 1)
            if bgv2.no_env_picture(n, B, r):   # rule v2: plan A's empty place still shows a changed entity
                cap1, keys1 = bgv2.drop_env(cap1, keys1, 1)
            d1 = f"{NR}/v4s1/{n}_B_{inputs_digest(json.dumps(B, sort_keys=True), cap1, *[pd / f'{k}.png' for k in keys1])}"
            job = dict(job_id=f'cd_s1_{n}_B', frames=g['frames'], seed=seed(), steps=20, lora=True, models=V4F.MODELS,
                       condition_root=str(FR / 'render' / f'{n}_B' / 'conditions'), system_prompt=V4F.SYS1, caption=cap1,
                       images=[{'key': k, 'path': str(pd / f'{k}.png')} for k in keys1], cache_dir=f'{d1}/cache', output=f'{d1}/generated.mp4',
                       frame0=f'{d1}/frame0.png', export={'arm': 'v4s1', 'category': 'real', 'name': str(n)})
            assert all(Path(x['path']).exists() for x in job['images']), n
            r['s1_job'] = write(d / 's1_job.json', job)
        elif r['route'] == 'edit':
            look, added, removed, glob, moved = r['look'], r['added'], r['removed'], r['glob'], r.get('moved', [])
            ia, ida = ff_prep.inst0(A); ib, idb = ff_prep.inst0(B); rects = []
            for e in added + moved:   # a red box where an added or moved entity goes
                mk = ff_prep.mask_of(ib, idb, {e})
                if mk.any():
                    import numpy as np
                    ys, xs = np.nonzero(mk)
                    rects.append({'entity': e, 'box': [xs.min() / ib.shape[1], ys.min() / ib.shape[0], (xs.max() + 1) / ib.shape[1], (ys.max() + 1) / ib.shape[0]], 'rgb': [255, 0, 0]})
            Eb = {e['id']: e for e in B['E']}; Ea = {e['id']: e for e in A['E']}
            area = {e: int(ff_prep.mask_of(ib, idb, {e}).sum()) for e in look}; pics = []
            for e in added + moved + sorted(look, key=lambda e: -area[e]):
                k = (Eb.get(e, {}).get('reference') or {}).get('key'); p = pd / f'{k}.png'
                if k and p.exists() and len(pics) < 3:
                    pics.append({'entity': e, 'path': str(p)})
            out = d / 'agent_output.json'
            lines = v4ins_lines(n, A, B, ed, look, added, removed, moved, glob, rects, pics, Ea, Eb, out)
            (d / 'agent_input.md').write_text('\n'.join(lines) + '\n')
            first = plate(f0, rects, d / 'plate.png', W, H) if rects else f0
            md = d / 'masks'
            import numpy as np
            for tag, inst, idx in (('A', ia, ida), ('B', ib, idb)):
                for i, e in idx.items():
                    if e in set(look) | set(removed) | set(added) | set(moved):
                        mk = np.asarray(ff_prep.up(inst == i)) > 0
                        if mk.any():
                            md.mkdir(parents=True, exist_ok=True); Image.fromarray((mk * 255).astype(np.uint8)).save(md / f'w{tag}_{e}.png')
            r.update(rects=rects, pics=pics, first=str(first))
            if r['kind'] == 'local':
                ents_a = [dict(name=e, text=V4F.entity_text(A, e), alt_text=' '.join(e.split('_')) + '.', prior=str(md / f'wA_{e}.png')) for e in look + removed + moved if e in Ea]
                ents_b = [dict(name=e, text=V4F.entity_text(B, e), alt_text=' '.join(e.split('_')) + '.', prior=str(md / f'wB_{e}.png')) for e in look + added + moved if e in Eb]
                r['mask_jobs'] = {}
                if ents_a:
                    r['mask_jobs']['src'] = write(d / 'mask_src.json', dict(id='src', image=str(f0), out_dir=str(md / 'src'), entities=ents_a))
                if ents_b:
                    r['mask_jobs']['ed'] = write(d / 'mask_ed.json', dict(id='ed', image=str(d / 'edited.png'), out_dir=str(md / 'ed'), entities=ents_b))
        write(d / 'meta.json', r); print(n, r['route'], r.get('kind', ''), r.get('why', ''))


def s1(a):
    """stage-1 free generations of the regen route, one task per geometry (flex_exec + the flex copies of ffprep / ffgen_many)"""
    M2 = mods()['M2']; groups = {}
    for n in ids(a.ids):
        mt = json.loads((FR / 'v4ff' / str(n) / 'meta.json').read_text())
        if mt['route'] == 'regen' and not (FR / 'v4ff' / 'results' / 'v4s1' / 'real' / f'{n}.png').exists():
            groups.setdefault(gstr(n), []).append(mt['s1_job'])
    for k, (gs, jobs) in enumerate(sorted(groups.items())):
        lst = FR / 'jobs' / f's1_{gs.replace(",", "x")}{a.tag}.txt'; lst.write_text('\n'.join(jobs) + '\n'); T = f'{M2.P}/compbench_v1/tools/cwm'
        put(f'f40_s1_{k}{a.tag}', dict(type='shell', retries=0, command=(
            f'set -o pipefail; {M2.PRE}; CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat {M2.PY} {INVC}/flex_exec.py {gs} {T}/ffprep.py {lst} && '
            f'cd {T} && {M2.TORCHRUN} --standalone --nproc_per_node=8 {INVC}/flex_exec.py {gs} {T}/ffgen_many.py --jobs {lst} --summary {lst}.summary.json '
            f'--parallel-decode --results-root {FR}/v4ff/results 2>&1 | tee {M2.INV}/logs/f40_s1_{k}{a.tag}.log')), a)


def v4_edits(a):
    """Qwen-Image edits of the edit route (plate drawn here, passed as image 1) + the mask task (A state on frame 0, B state on the edit)"""
    M2 = mods()['M2']; jobs, masks = [], []
    for n in ids(a.ids):
        d = FR / 'v4ff' / str(n); mt = json.loads((d / 'meta.json').read_text())
        todo_masks = [p for s, p in mt.get('mask_jobs', {}).items() if not (d / 'masks' / s / f'{s}.json').exists()]
        if mt['route'] != 'edit' or ((d / 'edited.png').exists() and not todo_masks):
            continue
        if not (d / 'edited.png').exists():
            ins = json.loads((d / 'agent_output.json').read_text())['edit_instruction'].strip(); g = geom(n)
            jobs.append(write(d / 'edit_job.json', dict(id=f'cd_ff_{n}', prompt=ins, width=g['width'], height=g['height'], seed=42, steps=40, out=str(d / 'edited.png'),
                                                        images=[{'path': mt['first']}] + [{'path': p['path'], 'max_side': 1024} for p in mt['pics']])))
        masks += todo_masks   # also when only the masks are missing (the edit exists)
    put(f'f45_v4edit{a.tag}', dict(type='workers', script='qwen_image/worker.py', pythonpath=M2.QDEPS, configs=jobs, retries=0), a)
    if masks:
        lst = FR / 'jobs' / f'v4masks{a.tag}.txt'; lst.parent.mkdir(parents=True, exist_ok=True); lst.write_text('\n'.join(masks) + '\n')
        put(f'f46_v4masks{a.tag}', dict(type='shell', retries=0, after=[f'f45_v4edit{a.tag}'],
                                        command=f'{M2.PRE}; bash {M2.S}/v4full/code/masks_par.sh {lst} {M2.INV}/logs/f46_v4masks{a.tag}'), a)


def v4_paste(a):
    m = mods(); V4F, M2 = m['V4F'], m['M2']
    import numpy as np
    for n in ids(a.ids):
        use(n); d = FR / 'v4ff' / str(n); mt = json.loads((d / 'meta.json').read_text()); W, H = M2.GW, M2.GH
        f0 = M2.frame_png(n, 0); dst = FR / 'pics' / str(n) / 'ff_v4.png'
        if mt['route'] == 'reuse':
            shutil.copy2(f0, dst); how = 'reuse'
        elif mt['route'] == 'regen':
            Image.open(FR / 'v4ff' / 'results' / 'v4s1' / 'real' / f'{n}.png').convert('RGB').resize((W, H), Image.LANCZOS).save(dst); how = 'regen'
        else:
            ed = Image.open(d / 'edited.png').convert('RGB').resize((W, H), Image.LANCZOS)
            if mt['kind'] == 'global':
                ed.save(dst); how = 'edit-global'
            else:
                u, used = g_region(d, mt); soft = V4F.soften(u); soft.save(d / 'g_region.png')
                k = np.asarray(soft).astype(np.float32)[..., None] / 255
                sa = np.asarray(Image.open(f0).convert('RGB'), np.float32); fa = np.asarray(ed, np.float32)
                Image.fromarray(np.clip(np.round(fa * k + sa * (1 - k)), 0, 255).astype(np.uint8)).save(dst); how = f'edit-local {used}'
        print(n, how[:200])


# ---------------------------------------------------------------- empty-place rule v2 (bgv2.py)
ERASE = {'eraseA1': ('A', 1, 'f24'), 'eraseA2': ('A', 2, 'f26'), 'eraseA3': ('A', 3, 'f27'), 'eraseB1': ('B', 1, 'f47'), 'eraseB2': ('B', 2, 'f48'), 'eraseB3': ('B', 3, 'f49')}


def erase(a, side, k):
    """removal pass k of the empty place (rule v2) for plan A (on the real frame 0) or plan B (on its first frame). Only the entities
    in view in that frame are named (bgv2.visible_A / visible_B). Pass 1: no entity in view -> the frame itself; plan A with every
    entity in view -> bg_raw (the same prompt); else a Qwen-Image job. Later passes remove what is left of the previous one."""
    m = mods(); M2, pv2 = m['M2'], m['pv2']; jobs = []
    for n in ids(a.ids):
        use(n); d = FR / 'plan' / str(n); pd = FR / 'pics' / str(n)
        if side == 'A':
            plan = json.loads((d / 'A_v2.json').read_text()); vis = bgv2.visible_A(n); frame = FR / 'frames' / str(n) / '000.png'
            ents = [(i, t or i) for i, t, _ in bgv2.entity_list(n, 'A')]
        else:
            plan = json.loads((d / 'B_v2.json').read_text()); vis = bgv2.visible_B(pv2, plan); frame = pd / 'ff_v4.png'
            ents = [(i, t) for i, t, _ in bgv2.entity_list(n, 'B', plan)]
        seen = [e for e in plan['E'] if e['id'] in vis]
        if k == 1 and not seen:
            bgv2.d(n, side).mkdir(parents=True, exist_ok=True); shutil.copy2(frame, bgv2.img(n, side, 1)); bgv2.first_from(n, side, 'copy of the frame (no entity in view)')
            print(f'  {n}: plan {side} pass 1 = the frame itself (no entity in view)'); continue
        if side == 'A' and k == 1 and {e['id'] for e in plan['E']} <= vis:
            bgv2.d(n, side).mkdir(parents=True, exist_ok=True); shutil.copy2(pd / 'bg_raw.png', bgv2.img(n, side, 1)); bgv2.first_from(n, side, 'bg_raw (every entity in view)')
            print(f'  {n}: plan A pass 1 = bg_raw (every entity in view)'); continue
        labels = ', '.join(sorted({pv2.label(e) for e in (seen or plan['E'])}))
        texts = '; '.join(t for i, t in ents if i in vis) or '; '.join(t for _, t in ents)
        src = frame if k == 1 else bgv2.img(n, side, k - 1)
        if k == 1:
            bgv2.first_from(n, side, 'removal pass on the frame (entities in view: ' + ', '.join(sorted(vis)) + ')')
        jobs.append(str(bgv2.pass_job(M2, n, side, k, src, labels, texts)))
    if jobs:
        put(f'{ERASE[f"erase{side}{k}"][2]}_erase{side}{k}{a.tag}', dict(type='workers', script='qwen_image/worker.py', pythonpath=M2.QDEPS, configs=jobs, retries=0), a)


def bgv2_side(n, side, todo, emit, first, B2=None):
    """rule v2 removal passes and checks for one side: check the passes made so far; a clean last pass (or pass 3) settles it,
    otherwise one more pass is made and checked. True once the chosen pass is applied."""
    dd = bgv2.d(n, side); k = bgv2.passes(n, side); stage = 'bgcheck2' if side == 'A' else 'bgcheckB'
    if k == 0:   # pass 1 (erase: the frame itself, bg_raw or a removal job)
        if not queued_ok(n, f'erase{side}1'):
            todo[f'erase{side}1'].append(n)
        return False
    if (dd / f'choice_{k}.json').exists():   # pass k was checked and is not clean: pass k + 1 is on its way
        if not queued_ok(n, f'erase{side}{k + 1}'):
            todo[f'erase{side}{k + 1}'].append(n)
        return False
    if not final(dd / 'choice.json', n, stage):
        if not (dd / 'agent_input.md').exists():
            bgv2.check_input(n, side, first, B2, k)
        emit(agent_list(stage, dd / 'agent_input.md', n, dd / 'choice.json')); return False
    c = json.loads((dd / 'choice.json').read_text())
    if not c.get('clean') and k < 3:   # keep this check, make one more pass and check again
        for f, g in (('choice.json', f'choice_{k}.json'), ('agent_input.md', f'agent_input_{k}.md')):
            (dd / f).rename(dd / g)
        for x in (f'agent_{stage}', f'agent_{stage}.done', f'agent_{stage}.fail', f'agent_{stage}.tech'):
            (ST / str(n) / x).unlink(missing_ok=True)
        todo[f'erase{side}{k + 1}'].append(n); return False
    bgv2.apply(n, side); return True


def bgv2_a(n, todo, emit):
    """plan A's empty place under rule v2; True once pics/<n>/bg.png holds the chosen pass"""
    if marked(n, 'bg_applied'):
        return True
    if not bgv2_side(n, 'A', todo, emit, FR / 'frames' / str(n) / '000.png'):
        return False
    mark(n, 'bg_applied'); return True


def bgv2_b(n, todo, emit):
    """plan B's empty place under rule v2: removal passes on plan B's first frame when the edit changes the background or style;
    True when it is applied or not needed"""
    if marked(n, 'bgB_applied'):
        return True
    B2 = json.loads((FR / 'plan' / str(n) / 'B_v2.json').read_text())
    if B2['background']['reference'].get('source') != 'erase_ff':
        return True
    if not bgv2_side(n, 'B', todo, emit, FR / 'pics' / str(n) / 'ff_v4.png', B2):
        return False
    mark(n, 'bgB_applied'); return True


def inputs_digest(*items):
    """short digest of a node job's inputs (text, or files by path, size and mtime). Node-local case and cache folders carry it in their
    names: the node scripts keep a folder that once passed (cwm_prepare: prepare_receipt.json; ffprep: manifest.json; ffgen_many: an
    existing output), so without it a clip redone with a new plan or first frame on the same node would get its old caches"""
    import hashlib
    h = hashlib.sha1()
    for x in items:
        if isinstance(x, Path):
            st = x.stat(); h.update(f'{x}:{st.st_size}:{st.st_mtime_ns}'.encode())
        else:
            h.update(str(x).encode())
    return h.hexdigest()[:10]


def stage2_spec(n, name, ver, anchor):
    """the stage-2 case spec of plan A ('A', anchored on frame 0) or plan B ('B', anchored on its first frame ff_v4) of clip n"""
    m = mods(); M2, pv2 = m['M2'], m['pv2']; pd = FR / 'pics' / str(n)
    plan = json.loads((FR / 'plan' / str(n) / f'{ver}_v2.json').read_text())
    cap, keys = pv2.caption(plan, pv2.frame0(plan), 2)
    if ver == 'B' and bgv2.no_env_picture(n, plan, json.loads((FR / 'v4ff' / str(n) / 'meta.json').read_text())):
        cap, keys = bgv2.drop_env(cap, keys, 2)   # rule v2: group B gets no environment picture
    miss = [k for k in keys if not (pd / f'{k}.png').exists()]; assert not miss, (n, name, miss)
    assert Path(anchor).exists(), (n, anchor)
    dg = inputs_digest(json.dumps(plan, sort_keys=True), cap, Path(anchor), *[pd / f'{k}.png' for k in keys], seed())
    return dict(case_dir=f'{NR}/{n}_{name}_{dg}', plan=write(FR / 'jobs' / 'plans' / f'{n}_{ver}.json', plan), render_dir=str(FR / 'render' / f'{n}_{ver}'),
                anchor_from=str(anchor), seed=seed(), images=[{'key': k, 'path': str(pd / f'{k}.png')} for k in keys], system_prompt=M2.SYS2, caption=cap)


def finish(a):
    """editing: stage-2 specs (A: real frame 0; B: plan B's first frame) and, per geometry, case prep + CWM caches + F4 inversion
    (without its reconstruction pass) under flex_exec"""
    m = mods(); M2 = m['M2']; groups = {}
    for n in ids(a.ids):
        use(n); pd = FR / 'pics' / str(n)
        specs = {name: stage2_spec(n, name, ver, anchor) for name, ver, anchor in (('A', 'A', M2.frame_png(n, 0)), ('B_v4', 'B', pd / 'ff_v4.png'))}
        sp = [write(FR / 'jobs' / 's2' / f'{n}_{k}.json', v) for k, v in specs.items()]
        case = dict(id=f'cd_{n}', cond_a=f"{specs['A']['case_dir']}/h3_config.json", cond_b=f"{specs['B_v4']['case_dir']}/h3_config.json",
                    source=str(FR / 'src' / f'{n}.mp4'), out_dir=str(FR / 'fix' / f'{n}'), compare={})
        gr = groups.setdefault(gstr(n), dict(specs=[], cfgs=[], cases=[])); gr['specs'] += sp; gr['cfgs'] += [case['cond_a'], case['cond_b']]; gr['cases'].append(case)
    for k, (gs, gr) in enumerate(sorted(groups.items(), key=lambda x: -int(x[0].split(',')[0]))):
        tag = f'{gs.replace(",", "x")}{a.tag}'; ls = FR / 'jobs' / f's2_{tag}.txt'; lc = FR / 'jobs' / f'cfg_{tag}.txt'
        ls.write_text('\n'.join(gr['specs']) + '\n'); lc.write_text('\n'.join(gr['cfgs']) + '\n'); cf = write(FR / 'jobs' / f'cases_{tag}.json', gr['cases'])
        T, PY, X = f'{M2.P}/compbench_v1/tools', M2.PY, f'{INVC}/flex_exec.py'; env = 'LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat'
        put(f'f60_{k}_{tag}', dict(type='shell', retries=0, command=(
            f'set -o pipefail; {M2.PRE}; cd {T}/e2 && PYTHONPATH={M2.PYDEPS} CUDA_VISIBLE_DEVICES=0 {env} {PY} {X} {gs} {T}/e2/pv2_case_prep.py {ls} && '
            f'split -n r/8 -d {lc} {lc}.g && for i in 0 1 2 3 4 5 6 7; do [ -s {lc}.g0$i ] && CUDA_VISIBLE_DEVICES=$i {env} {PY} {X} {gs} {T}/e2/cwm_prepare.py {lc}.g0$i '
            f'> {lc}.g0$i.log 2>&1 & done; wait; '
            f'cd {T}/cwm && CD_GEOM={gs} {M2.TORCHRUN} --standalone --nproc_per_node=8 {INVC}/inv_fix_v4.py --cases {cf} --f4-skip-recon '
            f'2>&1 | tee {M2.INV}/logs/f60_{tag}.log')), a)


REGEN = 'revision'   # export arm of the regenerated videos B: fix/results/revision/real/<n>.mp4


def regen_out(n):
    return FR / 'fix' / 'results' / REGEN / 'real' / f'{n}.mp4'


def regen_b(a):
    """revision: video B regenerated as a whole under conditions B (plan B, its pictures, its white model, its first frame locked) with
    the seed recorded when video A was generated (runs/known_A.json): stage-2 case prep, CWM caches, generate_many (124 frames, 20 steps)"""
    m = mods(); M2 = m['M2']; specs, cfgs, gens = [], [], []
    for n in ids(a.ids):
        use(n); g = geom(n)
        assert (g['frames'], g['width'], g['height']) == G.DEFAULT, f'{n}: a revision regenerates at 124 frames, 1344x768 (the geometry of a generated video)'
        sp = stage2_spec(n, 'B_regen', 'B', FR / 'pics' / str(n) / 'ff_v4.png')
        specs.append(write(FR / 'jobs' / 's2' / f'{n}_B_regen.json', sp)); cfgs.append(f"{sp['case_dir']}/h3_config.json")
        gens.append(f"{sp['case_dir']}/h3_config.json {sp['case_dir']}/B/generated.mp4 {REGEN} real {n}")
    tag = a.tag; ls, lc, lg = (FR / 'jobs' / f'{x}_regen{tag}.txt' for x in ('s2', 'cfg', 'gen'))
    ls.write_text('\n'.join(specs) + '\n'); lc.write_text('\n'.join(cfgs) + '\n'); lg.write_text('\n'.join(gens) + '\n')
    T, PY = f'{M2.P}/compbench_v1/tools', M2.PY; env = 'LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat'
    put(f'f61_regen{tag}', dict(type='shell', retries=0, command=(
        f'set -o pipefail; {M2.PRE}; cd {T}/e2 && PYTHONPATH={M2.PYDEPS} CUDA_VISIBLE_DEVICES=0 {env} {PY} {T}/e2/pv2_case_prep.py {ls} && '
        f'CUDA_VISIBLE_DEVICES=0 {env} {PY} {T}/e2/cwm_prepare.py {lc} && '
        f'cd {T}/cwm && {M2.TORCHRUN} --standalone --nproc_per_node=8 {T}/cwm/generate_many.py --list {lg} --summary {lg}.summary.json '
        f'--parallel-decode --results-root {FR}/fix/results 2>&1 | tee {M2.INV}/logs/f61_regen{tag}.log')), a)


# ---------------------------------------------------------------- automatic advance
ST = FR / 'state'
NODES = tuple('@CD_NODES@'.split(','))   # the GPU nodes whose runners take the node tasks (bin/configure.py --nodes)
QS = Path('@CD_QUEUE@')   # node queues and status (the runners' RUNNER_SHARED)


def mark(n, stage, val=''):
    p = ST / str(n) / stage; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(val); return p


def marked(n, stage):
    return (ST / str(n) / stage).exists()


def task_state(node, tid):
    """'done' / 'failed' / 'pending' from the node's runner event log (lines: time VERB name ...)"""
    ev = QS / 'status' / node / 'events.log'; st = 'pending'
    for line in (ev.read_text().splitlines() if ev.exists() else []):
        w = line.split()
        if len(w) >= 3 and w[2] == tid and w[1] in ('DONE', 'FAILED'):
            st = 'done' if w[1] == 'DONE' else 'failed'
    return st


def queued_ok(n, stage):
    """a node stage was queued for n and its task has not failed (a failed task clears the mark so the stage is queued again)"""
    p = ST / str(n) / stage
    if not p.exists():
        return False
    node, tid = p.read_text().split(); st = task_state(node, tid)
    if st == 'done' and stage == 'v4edit':   # the mask task of the same batch runs after the edit task: wait for it as well
        mt = tid.replace('f45_v4edit', 'f46_v4masks', 1)
        if (QS / 'queue' / node / 'main' / f'{mt}.json').exists():
            st = task_state(node, mt)
    if st == 'pending':
        return True
    if st == 'failed':
        p.unlink(); print(f'  {n}: {stage} task {tid} on {node} failed, will queue again'); return False
    if st == 'done':  # called only while the stage's output is missing: the task ran but this clip failed inside it
        r = ST / str(n) / f'{stage}.retries'; k = int(r.read_text()) if r.exists() else 0
        if k < 2:
            r.write_text(str(k + 1)); p.unlink(); print(f'  {n}: {stage} task {tid} on {node} done without output, will queue again ({k + 1}/2)'); return False
        print(f'  {n}: {stage} still has no output after 2 retries, needs a look')
    return True


def node_load():
    """pending main tasks per node (the next batch goes to the least loaded one)"""
    load = {}
    for node in NODES:
        fin = set()
        ev = QS / 'status' / node / 'events.log'
        for line in (ev.read_text().splitlines() if ev.exists() else []):
            w = line.split()
            if len(w) >= 3 and w[1] in ('DONE', 'FAILED'):
                fin.add(w[2])
        load[node] = sum(1 for q in (QS / 'queue' / node / 'main').glob('f*.json') if q.stem not in fin)
    return load


def waiting_tasks():
    """(prefix, geometry or None) of node tasks queued but not started: not DONE / FAILED / START in the node's events log"""
    import re
    out = set()
    for node in NODES:
        ev = QS / 'status' / node / 'events.log'; seen = set()
        for line in (ev.read_text().splitlines() if ev.exists() else []):
            w = line.split()
            if len(w) >= 4 and w[1] == 'START':
                seen.add(w[3])
            elif len(w) >= 3 and w[1] in ('DONE', 'FAILED'):
                seen.add(w[2])
        for q in (QS / 'queue' / node / 'main').glob('f*.json'):
            if q.stem in seen:
                continue
            m = re.match(r'^(f\d\d)_(?:\d+_(\d+x\d+x\d+)_)?', q.stem)
            if m:
                out.add((m.group(1), m.group(2)))
    return out


_DRY = [False]


def final(path, n, stage):
    """an agent's output is final once the file exists and the agent has ended: an agent may write its file early and go on editing
    it. Running = taken-mark without a .done mark, younger than 3 h; an output without a mark (e.g. plan A of a revision) is final."""
    if not Path(path).exists():
        return False
    mk = ST / str(n) / f'agent_{stage}'
    if mk.exists() and not mk.with_name(mk.name + '.done').exists() and (time.time() - mk.stat().st_mtime) < 3 * 3600:
        return False
    return True


def agent_list(stage, path, n, out):
    """a context-free agent whose whole task is the file at `path`; listed once (re-listed after 3 h without output)"""
    p = ST / str(n) / f'agent_{stage}'
    if p.exists() and (time.time() - p.stat().st_mtime) < 3 * 3600:
        return None
    return dict(stage=stage, id=n, prompt=str(path), out=str(out))


DEPTH = ['entities', 'pcheck', 'planner', 'editor', 'bgcheck2', 'v4ins', 'bgcheckB']  # agent stages in pipeline order


def step(n, todo, emit):
    """advance clip n as far as the finished work allows; node work goes to `todo`, agents to emit()"""
    if not (FR / 'geom' / f'{n}.json').exists():
        return
    d = FR / 'plan' / str(n)
    if not final(d / 'entities.json', n, 'entities'):
        if not (d / 'entities_prompt.md').exists():
            entities(argparse.Namespace(ids=str(n)))
        return emit(agent_list('entities', d / 'entities_prompt.md', n, d / 'entities.json'))
    if not (FR / 'percept' / str(n) / 'perception.json').exists():
        if not queued_ok(n, 'percept'):
            todo['percept'].append(n)
        return
    vd = FR / 'checks' / 'percept' / str(n)
    if not final(vd / 'verdict.json', n, 'pcheck'):
        if not (vd / 'agent_input.md').exists():
            pcheck(argparse.Namespace(ids=str(n)))
        return emit(agent_list('pcheck', vd / 'agent_input.md', n, vd / 'verdict.json'))
    if not (FR / 'percept' / f'{n}_checked' / 'perception.json').exists():
        pcheck_apply(argparse.Namespace(ids=str(n)))
    if not final(d / 'scm_A.json', n, 'planner'):
        if not (FR / 'percept' / str(n) / 'camtrack.npz').exists():   # the measured camera comes before the plan
            if not queued_ok(n, 'camtrack'):
                todo['camtrack'].append(n)
            return
        if not (d / 'planner_prompt.md').exists():
            planner(argparse.Namespace(ids=str(n)))
        return emit(agent_list('planner', d / 'planner_prompt.md', n, d / 'scm_A.json'))
    if not final(d / 'scm_B.json', n, 'editor'):
        tidy_plan(d / 'scm_A.json')   # before the editor reads it (long lines are cut by its Read tool)
        if not (d / 'editor_prompt.md').exists():
            editor(argparse.Namespace(ids=str(n)))
        return emit(agent_list('editor', d / 'editor_prompt.md', n, d / 'scm_B.json'))
    pd = FR / 'pics' / str(n)
    if not (d / 'assets.json').exists():
        todo['pics'].append(n); return
    rec = json.loads((d / 'assets.json').read_text())
    if any(not (pd / f'{k}.png').exists() for k in rec['wave1']):
        if not queued_ok(n, 'pics'):
            todo['pics'].append(n)
        return
    if not bgv2_a(n, todo, emit):   # empty-place rule v2: removal passes on frame 0 and a check, never a drawing
        return
    if any(not (pd / f'{k}.png').exists() for k in rec.get('wave2', [])):
        if not queued_ok(n, 'pics2'):
            todo['pics2'].append(n)
        return
    vd4 = FR / 'v4ff' / str(n)
    if not (vd4 / 'meta.json').exists():
        v4(argparse.Namespace(ids=str(n)))
    mt = json.loads((vd4 / 'meta.json').read_text())
    if not (pd / 'ff_v4.png').exists():
        if mt['route'] == 'regen' and not (FR / 'v4ff' / 'results' / 'v4s1' / 'real' / f'{n}.png').exists():
            if not queued_ok(n, 's1'):
                todo['s1'].append(n)
            return
        if mt['route'] == 'edit':
            if not final(vd4 / 'agent_output.json', n, 'v4ins'):
                return emit(agent_list('v4ins', vd4 / 'agent_input.md', n, vd4 / 'agent_output.json'))
            need = [vd4 / 'edited.png'] + ([vd4 / 'masks' / s / f'{s}.json' for s in mt.get('mask_jobs', {})] if mt.get('kind') == 'local' else [])
            if not all(p.exists() for p in need):
                if not queued_ok(n, 'v4edit'):
                    todo['v4edit'].append(n)
                return
        v4_paste(argparse.Namespace(ids=str(n)))
    if not bgv2_b(n, todo, emit):   # rule v2: plan B's empty place from its first frame
        return
    if (META / 'no_video').exists():   # --outputs diorama: no video of plan B
        return
    if known_a():   # revision: video B regenerated with the seed of video A
        if not regen_out(n).exists() and not queued_ok(n, 'regen'):
            todo['regen'].append(n)
    elif not (FR / 'fix' / str(n) / 'metrics.json').exists():   # editing: F4 inversion
        if not queued_ok(n, 'finish'):
            todo['finish'].append(n)


KINDS = {'percept': ('f10', percept), 'camtrack': ('f12', camtrack), 'pics': ('f20', assets), 'pics2': ('f30', assets2), 'v4edit': ('f45', v4_edits),
         **{k: (v[2], (lambda side, j: lambda ns: erase(ns, side, j))(v[0], v[1])) for k, v in ERASE.items()}}


def dispatch_nodes(todo, a, stamp):
    """node batches: each kind of node task has at most one task waiting at a time (F4: one per geometry); clips that become ready
    while it waits join the next task, so a busy node gets large batches (one model load each) and an idle one gets work at once.
    A batch goes to the least loaded node."""
    waiting = waiting_tasks()

    def node_for():
        load = node_load(); return min(NODES, key=lambda k: (load[k], NODES.index(k)))

    def mark_all(kind, xs, node, tid):
        if not a.dry:
            for n in xs:
                mark(n, kind, f'{node} {tid}')
    for kind, (prefix, fn) in KINDS.items():
        xs = sorted(todo.get(kind, []))
        if not xs or any(k == prefix for k, _ in waiting):
            if xs:
                print(f'  {kind}: {len(xs)} ready, joining the next batch (one {prefix} task is still waiting)')
            continue
        node = node_for(); ns = argparse.Namespace(ids=','.join(map(str, xs)), node=node, tag=f'_{kind}{stamp}', dry=a.dry)
        fn(ns)
        tid = f'{prefix}_{kind}{ns.tag}'
        mark_all(kind, xs, node, tid); print(f'  queued {kind} for {len(xs)} on {node}')
    if todo['s1'] and not any(k == 'f40' for k, _ in waiting):   # one task per geometry
        node = node_for(); xs = sorted(todo['s1'])
        ns = argparse.Namespace(ids=','.join(map(str, xs)), node=node, tag=f'_s1{stamp}', dry=a.dry); s1(ns)
        groups = sorted({gstr(n) for n in xs})
        for n in xs:
            mark_all('s1', [n], node, f"f40_s1_{groups.index(gstr(n))}_s1{stamp}")
    elif todo['s1']:
        print(f"  s1: {len(todo['s1'])} ready, joining the next batch")
    bygeom = {}
    for n in todo['finish']:
        bygeom.setdefault(gstr(n), []).append(n)
    for gs, xs in sorted(bygeom.items(), key=lambda kv: -int(kv[0].split(',')[0]) * len(kv[1])):
        gk = gs.replace(',', 'x')
        if ('f60', gk) in waiting:
            print(f'  finish {gs}: {len(xs)} ready, joining the next batch'); continue
        node = node_for(); tag = f'_fin{stamp}'
        finish(argparse.Namespace(ids=','.join(map(str, sorted(xs))), node=node, tag=tag, dry=a.dry))
        mark_all('finish', xs, node, f'f60_0_{gk}{tag}'); waiting.add(('f60', gk))
    if todo['regen'] and not any(k == 'f61' for k, _ in waiting):
        node = node_for(); xs = sorted(todo['regen']); tag = f'_regen{stamp}'
        regen_b(argparse.Namespace(ids=','.join(map(str, xs)), node=node, tag=tag, dry=a.dry))
        mark_all('regen', xs, node, f'f61_regen{tag}')


def auto(a):
    """advance every clip as far as the finished work allows; the agents to start go to state/agents_<stamp>.jsonl (read by llm_run.py)"""
    t0 = time.time()
    want = ids(a.ids) if a.ids != 'all' else sorted(checklist())
    todo = {k: [] for k in ('s1', 'finish', 'regen', *KINDS)}; agents = []

    def emit(x):
        if x:
            agents.append(x)
    for n in want:
        try:
            step(n, todo, emit)
        except Exception as e:
            print(f'  {n}: ERROR {type(e).__name__}: {str(e)[:300]}')
    stamp = SUB + time.strftime('%m%d%H%M%S')
    out = ST / f"agents_{time.strftime('%Y%m%d%H%M%S')}.jsonl"; ST.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(x) + '\n' for x in agents)); tmp.replace(out)
    dispatch_nodes(todo, a, stamp)
    print(f'AUTO {len(want)} ids, {len(agents)} agents to start -> {out}, {round(time.time() - t0)} s')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('prep', 'auto')); ap.add_argument('ids')
    ap.add_argument('--node', default=NODES[0], choices=NODES); ap.add_argument('--tag', default=''); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args(); {'prep': prep, 'auto': auto}[a.cmd](a)


if __name__ == '__main__':
    main()
