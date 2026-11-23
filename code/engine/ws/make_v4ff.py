"""First-frame helpers of the editing engine (used by flexrun.v4 / v4_paste):
plans A / B with their pictures, entity texts for the mask jobs, the G region of a local edit (Grounding DINO + SAM 2.1 masks of
the changed entities on frame 0 (A state) and on the edited picture (B state), the white-model mask where nothing was detected, the
planned boxes of added entities), its soft version (dilated 24 px, feathered 8 px), and the stage-1 settings of the regen route.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_stage2 as M2  # noqa: E402  (paths)

R = M2.R; V = R / 'v4ff'; W, H = 1344, 768   # R, V, W, H: set by flexrun for the run tree and the clip
SYS1 = (Path('@CD_CODE@/engine/cb') / 'system_stage1.txt').read_text().strip()
_REL = '@CD_NODE@/cwm_h3_reproduction_v1/release/models'   # the models of every H3 job
MODELS = {'dit': f'{_REL}/diffusion_models/minimax_h3_ref2va_bf16.safetensors', 'lora': f'{_REL}/lora/minimax-h3-cwm.safetensors',
          'video_vae': f'{_REL}/vae/minimax_h3_video_vae_fp16.safetensors', 'processor': f'{_REL}/processors/Qwen3-VL-32B-Instruct',
          'text_encoder': f'{_REL}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors'}
DILATE, FEATHER = 49, 8   # MaxFilter(49) = 24 px dilation, GaussianBlur(8)


def plans(n):
    d = R / 'plan' / str(n)
    return json.loads((d / 'A_v2.json').read_text()), json.loads((d / 'B_v2.json').read_text())


def entity_text(plan, e):
    x = next((x for x in plan['E'] if x['id'] == e), None)
    t = ((x or {}).get('appearance') or '').strip().rstrip('.').lower()
    return t + '.' if t else ' '.join(w for w in e.split('_') if not w.isdigit()) + '.'


def lmask(p):
    return np.asarray(Image.open(p).convert('L')) > 127


def g_region(d, m):
    """the G region of a local first-frame edit, with the masks under d/masks"""
    md = d / 'masks'; u = np.zeros((H, W), bool); used = {}
    for tag, sub, ents in (('A', 'src', m['look'] + m['removed']), ('B', 'ed', m['look'] + m['added'])):
        if sub not in m.get('mask_jobs', {}):
            continue
        has = (md / sub / f'{sub}.json').exists(); rec = json.loads((md / sub / f'{sub}.json').read_text())['entities'] if has else {}
        for e in ents:
            g, w = md / sub / f'{e}.png', md / f'w{tag}_{e}.png'
            if g.exists() and has:
                u |= lmask(g); used[f'{tag}:{e}'] = ['G', rec.get(e, {}).get('pick')]
            elif w.exists():
                u |= lmask(w); used[f'{tag}:{e}'] = ['W fallback', None]
            else:
                used[f'{tag}:{e}'] = ['none', None]
    for x in m['rects']:
        b = x['box']; u[round(b[1] * H):round(b[3] * H), round(b[0] * W):round(b[2] * W)] = True; used[f"box:{x['entity']}"] = ['planned box', None]
    return u, used


def soften(mask_bool):
    """Dilate 24 px and feather 8 px."""
    im = Image.fromarray((mask_bool * 255).astype(np.uint8))
    return im.filter(ImageFilter.MaxFilter(DILATE)).filter(ImageFilter.GaussianBlur(FEATHER))
