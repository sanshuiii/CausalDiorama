"""First-frame helpers on a plan's white model (the first-frame route and its edit): the frame-0 instance map of a plan, the
mask of a set of entities in it, and its upscaling to the frame size. W, H: the frame size (set per clip by the editing engine).
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import world  # noqa: E402
W, H, UP = 1344, 768, 4


def inst0(plan):
    out = world.Scene(plan).render(0)
    return out[-1], {e['instance_id']: e['id'] for e in plan['E']}


def mask_of(inst, ids, ents):
    m = np.zeros(inst.shape, bool)
    for i, e in ids.items():
        if e in ents:
            m |= inst == i
    return m


def up(m):
    return Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
