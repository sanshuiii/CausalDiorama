"""Portable Causal Diorama: one directory (or the same directory zipped) that carries everything needed to render, inspect,
edit or move a diorama, independent of the pipeline's work trees.

    <name>/
      diorama.json        the manifest (below)
      pictures/<key>.png  one reference picture per reference key of the plan (entities and the empty place)
      first_frame.png     the frame the video model locks as frame 0 (optional)
      video.mp4           the video made from this diorama (optional)

diorama.json:
  format, version         "causal-diorama", 1
  id, kind                a name; "generated" (from a prompt), "fitted" (from a real video) or "edited" (an edit of a parent)
  geometry                {"frames", "fps", "width", "height"}
  scm                     the plan (schema cb-scm-v1, engine/cb/scm_schema_pv2.md) with its `reference` fields: the
                          variables C (camera), style, background, environment, E (entities), S (state tracks), R (shared
                          events), D (dependencies), text, intent, and for an edited diorama its `edit` declaration
  pictures                {key: {"file", "sha256", "source", "prompt"?, "from"?}}; source: t2v (drawn from text), edit (an
                          image edit of another picture), cutout (cut from a real frame), frame (a real frame), erase (a frame
                          with its entities removed)
  first_frame / video     {"file", "sha256", "source"} or null
  text                    {"prompt": the scene description, "instruction": the edit instruction of an edited diorama}
  parent                  {"id", "scm_sha256"} of the diorama an edited one was made from, else null
  generation              {"seed", "steps", "model", ...} of the video
  provenance              {"pipeline", "created", "code"}
Everything is plain JSON and PNG/MP4 with sha256 sums, so a diorama can be edited by hand (the plan is ordinary JSON),
validated (`cd_diorama.py validate`), rendered as a white model (`cd_diorama.py render`) and moved as one zip file.
"""
import hashlib, json, shutil, time, zipfile
from pathlib import Path

FORMAT, VERSION = 'causal-diorama', 1


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def scm_sha(scm):
    return hashlib.sha256(json.dumps(scm, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _put(src, out, name, extra):
    if not src:
        return None
    src = Path(src); dst = out / name; dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dict(file=name, sha256=sha(dst), **{k: v for k, v in extra.items() if v is not None})


def pack(out, *, id, kind, scm, geometry, pictures, first_frame=None, video=None, text=None, parent=None, generation=None,
         pipeline='', first_frame_source=None, video_source=None):
    """pictures: {key: {'path': file, 'source': ..., 'prompt'?: ..., 'from'?: ...}}; writes <out>/ and returns the manifest"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    refs = {k for k in reference_keys(scm)}
    miss = sorted(refs - set(pictures))
    assert not miss, f'pictures missing for reference keys {miss}'
    pics = {k: _put(v['path'], out, f'pictures/{k}.png', {x: v.get(x) for x in ('source', 'prompt', 'from')}) for k, v in sorted(pictures.items())}
    m = dict(format=FORMAT, version=VERSION, id=id, kind=kind, geometry=geometry, scm=scm, pictures=pics,
             first_frame=_put(first_frame, out, 'first_frame.png', {'source': first_frame_source}),
             video=_put(video, out, 'video.mp4', {'source': video_source}),
             text=text or {}, parent=parent, generation=generation or {},
             provenance=dict(pipeline=pipeline, created=time.strftime('%Y-%m-%dT%H:%M:%S'), code='causal_diorama release 1'))
    (out / 'diorama.json').write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return m


def reference_keys(scm):
    keys = []
    for node in [scm.get('background') or {}] + list(scm.get('E') or []):
        r = node.get('reference') or {}
        k = r.get('same_as') or r.get('key')
        if k and k not in keys:
            keys.append(k)
    return keys


def load(path):
    path = Path(path)
    if path.suffix == '.zip':
        path = unzip(path, path.with_suffix(''))
    m = json.loads((path / 'diorama.json').read_text())
    assert m.get('format') == FORMAT, f'{path} is not a causal diorama'
    return path, m


def file(path, entry):
    return (Path(path) / entry['file']) if entry else None


def validate(path, render=False):
    """hash and completeness checks; render=True also renders the plan with the white-model executor (CPU)"""
    path, m = load(path); errs = []
    for k in reference_keys(m['scm']):
        if k not in m['pictures']:
            errs.append(f'no picture for reference key {k}')
    for name, e in [(f'picture {k}', v) for k, v in m['pictures'].items()] + [('first_frame', m.get('first_frame')), ('video', m.get('video'))]:
        if e and (not (path / e['file']).exists() or sha(path / e['file']) != e['sha256']):
            errs.append(f'{name}: missing file or wrong sha256')
    if render:
        import world   # engine/cb on sys.path
        g = m['geometry']; world.FRAMES, world.W, world.H = g['frames'], g['width'] // 4, g['height'] // 4
        world.render.__defaults__ = (g['width'] // 4, g['height'] // 4, False)
        try:
            sc = world.Scene(m['scm'])
            for f in (0, g['frames'] // 2, g['frames'] - 1):
                sc.render(f)
        except Exception as e:
            errs.append(f'white model does not render: {type(e).__name__}: {e}')
    return errs


def zip_dir(path, out=None):
    path = Path(path); out = Path(out or path.with_suffix('.zip'))
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(path.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(path))
    return out


def unzip(zp, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(out)
    return out


def render_preview(path, out, frames=None):
    """white-model depth and semantic images of a few frames (same normalisation and palette as the paper figures)"""
    import numpy as np
    from PIL import Image
    import world
    path, m = load(path); g = m['geometry']; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    world.FRAMES, world.W, world.H = g['frames'], g['width'] // 4, g['height'] // 4
    world.render.__defaults__ = (g['width'] // 4, g['height'] // 4, False)
    sc = world.Scene(m['scm']); frames = frames or [0, g['frames'] // 2, g['frames'] - 1]
    pal = {0: (0, 0, 0), 1: (135, 206, 235), 2: (30, 90, 200), 3: (150, 120, 80), 4: (128, 128, 128), 5: (60, 160, 60), 6: (180, 90, 70),
           7: (230, 200, 40), 8: (220, 40, 40), 9: (240, 140, 30), 10: (40, 40, 140), 11: (200, 60, 200)}
    up = lambda a: Image.fromarray(a).resize((a.shape[1] * 4, a.shape[0] * 4), Image.NEAREST)  # noqa: E731
    for f in frames:
        st, rgb, depth, sem, ins = sc.render(f)
        sky = sem == world.SEM['sky']; d = depth[~sky]
        near, far = (float(d.min()), float(np.percentile(d, 99))) if d.size else (1.0, 2.0)
        v = (np.log(far) - np.log(np.clip(depth, near, far))) / max(np.log(far) - np.log(near), 1e-6)
        img = (np.clip(v, 0, 1) * 255).astype(np.uint8); img[sky] = 0
        col = np.zeros(sem.shape + (3,), np.uint8)
        for k, c in pal.items():
            col[sem == k] = c
        up(img).save(out / f'depth_f{f:03d}.png'); up(col).save(out / f'semantic_f{f:03d}.png')
    return out
