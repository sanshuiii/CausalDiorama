"""Package paths, fixed generation settings and small helpers shared by plan.py, prepare.py and run.py."""
import json, os
from pathlib import Path

GEN = Path(__file__).resolve().parent     # code/generation
PKG = GEN.parents[1]                      # package root
CB = GEN.parent / 'engine' / 'cb'          # executor world.py, checker scm_lint.py + rules_pv2.json, scm_schema_pv2.md, pv2.py, stage prompts
PLANNER_TEMPLATE = GEN / 'prompts' / 'planner.md'

SEED = 9942001
FRAMES = 124
STAGE_STEPS = 20
PICTURE_STEPS = 40
ARM_STAGE1, ARM_VIDEO = 'stage1', 'video'   # result trees: <results>/<arm>/<category>/<name>.{png,mp4}
GPU_CLAIM_SECONDS = 15


def load_config(path):
    """config.json (see config.example.json): where the shared work tree, the results, the task queue, the node-local work tree,
    the node tools and environments and the model weights are."""
    return json.loads(Path(path).read_text())


def jdump(path, obj):
    """Write JSON atomically (a reader never sees a partial file); returns the path as a string."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1)); os.replace(tmp, path)
    return str(path)
