"""Flexible clip geometry for the H3 Ref2VA + CWM path: real clips of any length up to 15.08 s and portrait clips at their own
length and aspect instead of the fixed 124 frames at 1344x768.

H3 itself accepts 17n+5 frames and latent sizes divisible by the 2x2 patch; the released CWM inference package freezes one geometry
in cwm_h3_inference.constants (124-frame window, 1344x768 target, 336x192 white-model proxy, the latent shapes derived from them)
and its modules bind those names at import time. apply() rewrites the constants for one geometry BEFORE any other cwm_h3_inference
module is imported, so one process serves one geometry. The default geometry (124, 1344, 768) leaves every value unchanged.
Also patched: the proxy-video reference padding (the package pads the latent width by one column, right for 16:9 only; here height
and width are padded to even, which is the same for 16:9), the white-model renderer (world.FRAMES and its 1/4-size render) and
multiref's reference-video duration.

    geometry_for(duration_s, width, height) -> (frames, W, H)   frames = 17n+5 <= duration x 24 (n <= 21, 15.08 s); W x H keeps the
                                                                 aspect with about 1344x768 pixels, multiples of 64 (proxy = 1/4)
    apply(frames, W, H)        patch the constants (call first);  apply_from_env() reads CD_GEOM="frames,W,H"
"""
import math, os, sys

FPS = 24
DEFAULT = (124, 1344, 768)
PIXELS = 1344 * 768
MAX_N = 21
ACTIVE = None


def frames_for(duration_s):
    n = min(MAX_N, int((duration_s * FPS - 5) // 17))
    if n < 0:
        raise ValueError(f'clip too short: {duration_s} s')
    return 17 * n + 5


def size_for(width, height):
    a = width / height
    w = max(64, round(math.sqrt(PIXELS * a) / 64) * 64); h = max(64, round(math.sqrt(PIXELS / a) / 64) * 64)
    return w, h


def geometry_for(duration_s, width, height):
    return (frames_for(duration_s),) + size_for(width, height)


def derived(frames, w, h):
    assert frames >= 5 and (frames - 5) % 17 == 0, frames
    assert w % 64 == 0 and h % 64 == 0, (w, h)
    lf = 5 * ((frames - 5) // 17) + 2; cw, ch = w // 4, h // 4; rh, rw = ch // 16, cw // 16
    return dict(WINDOW_FRAMES=frames, STRIDE_FRAMES=frames - 34, CONDITION_WIDTH=cw, CONDITION_HEIGHT=ch, TARGET_WIDTH=w, TARGET_HEIGHT=h,
                VIDEO_LATENT_SHAPE=(1, 24, lf, h // 16, w // 16), AUDIO_LATENT_SHAPE=(1, 32, 2, (10 * frames + 3) // 6),
                REFERENCE_LATENT_SHAPE=(24, lf, rh, rw), REFERENCE_RUNTIME_SHAPE=(1, 24, lf, rh + rh % 2, rw + rw % 2),
                SINGLE_FRAME_LATENT_SHAPE=(24, 1, h // 16, w // 16), WINDOW_AUDIO_SAMPLES=round(frames * 32000 / FPS),
                QWEN_OFFSETS=tuple(range(0, frames, 12)))


def apply(frames, w, h):
    global ACTIVE
    for m in ('engine', 'cache', 'duv', 'config', 'runtime', 'cli'):
        if f'cwm_h3_inference.{m}' in sys.modules and (frames, w, h) != DEFAULT:
            raise RuntimeError(f'flex_geom.apply after cwm_h3_inference.{m} was imported: its constants are already bound')
    sys.path.insert(0, '@CD_SHARED@/tools/cwm')  # resident.py puts the package on sys.path
    import resident  # noqa: F401
    import cwm_h3_inference.constants as K
    d = derived(frames, w, h)
    for k, v in d.items():
        setattr(K, k, v)
    import cwm_h3_inference.engine as EN

    def _load_window_cache(self, index):
        import torch
        from safetensors import safe_open
        paths = EN.cache_paths(self.config, index)
        with safe_open(str(paths['anchor']), framework='pt', device='cpu') as f:
            anchor = f.get_tensor(EN._unique_key(f, prefix='latents_ref_'))
        with safe_open(str(paths['reference']), framework='pt', device='cpu') as f:
            reference = f.get_tensor(EN._unique_key(f, prefix='latents_ref_'))
        with safe_open(str(paths['qwen']), framework='pt', device='cpu') as f:
            hidden = f.get_tensor(EN._unique_key(f, exact='varlen_mmh3_hidden_states_bfloat16'))
            tags = f.get_tensor(EN._unique_key(f, exact='varlen_mmh3_token_tags_int64'))
        if tuple(reference.shape) != K.REFERENCE_LATENT_SHAPE or reference.dtype != torch.float32:
            raise ValueError(f'window {index} DUV reference cache shape/dtype changed: {tuple(reference.shape)}')
        rh, rw = reference.shape[-2:]
        reference = torch.nn.functional.pad(reference.unsqueeze(0), (0, rw % 2, 0, rh % 2), value=0.0).contiguous()
        if tuple(reference.shape) != K.REFERENCE_RUNTIME_SHAPE:
            raise ValueError(f'window {index} padded DUV reference shape changed')
        if anchor.ndim != 4 or tuple(anchor.shape[:2]) != (24, 1) or anchor.dtype != torch.float32:
            raise ValueError('anchor cache shape/dtype changed')
        if hidden.ndim != 2 or hidden.shape[1] != 5120 or tags.shape != (hidden.shape[0],):
            raise ValueError(f'window {index} Qwen cache changed')
        return anchor.unsqueeze(0), reference, hidden.unsqueeze(0), tags.unsqueeze(0)

    EN.Ref2VAEngine._load_window_cache = _load_window_cache
    sys.path.insert(1, '@CD_SHARED@/tools/cb')  # node copy of the white-model renderer (world.py)
    try:
        import world
        world.FRAMES, world.W, world.H = frames, w // 4, h // 4
        world.render.__defaults__ = (w // 4, h // 4, False)  # its width / height defaults were bound at import
    except ImportError:
        pass
    flex = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flex')   # geometry-aware copies of multiref / ffgen / ffgen_many
    sys.path.insert(0, flex)
    import importlib.util
    spec = importlib.util.spec_from_file_location('multiref', os.path.join(flex, 'multiref.py'))
    mr = importlib.util.module_from_spec(spec); spec.loader.exec_module(mr); sys.modules['multiref'] = mr
    os.environ['CD_GEOM'] = f'{frames},{w},{h}'
    ACTIVE = (frames, w, h)
    return d


def apply_from_env():
    g = os.environ.get('CD_GEOM')
    if g:
        return apply(*(int(x) for x in g.split(',')))
    return None


def active():
    return ACTIVE or DEFAULT
