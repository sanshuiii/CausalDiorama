"""Run a workstation tool under one clip geometry: python3 geomrun.py FRAMES,W,H SCRIPT [ARGS...]
Sets world.FRAMES = FRAMES and the white-model render size world.W x world.H = W/4 x H/4 (the CWM proxy size) before SCRIPT imports
anything (scm_lint.py binds them at import), exports CD_GEOM for cam_helper.py, then runs SCRIPT as __main__. Without it the tools
keep the pipeline's 124 frames at 1344x768."""
import os, runpy, sys
sys.path.insert(0, '@CD_CODE@/engine/cb')
import world  # noqa: E402

F, W, H = (int(x) for x in sys.argv[1].split(','))
world.FRAMES, world.W, world.H = F, W // 4, H // 4
world.render.__defaults__ = (W // 4, H // 4, False)  # render(cam, env, prims, width=W, height=H, ...) bound the defaults at import
os.environ['CD_GEOM'] = sys.argv[1]
# scm_lint.preview renders the fixed frames 0, 25, 50, 75, 100, 123: on a clip shorter than 124 frames it would fail after lint.json
# is written, and agents would read the traceback as a failed check. Frames past the clip's last one are drawn as its last frame; nothing else asks for them.
_render = world.Scene.render
world.Scene.render = lambda self, f, *a, **k: _render(self, min(int(f), world.FRAMES - 1), *a, **k)
script = sys.argv[2]; sys.argv = [script] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name='__main__')
