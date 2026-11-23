"""Run a node script under one clip geometry: python flex_exec.py FRAMES,W,H SCRIPT [ARGS...]
Applies flex_geom (cwm_h3_inference constants, proxy padding, renderer, multiref) before the script imports anything, then runs SCRIPT
as __main__ with its own argv (also under torchrun: torchrun ... flex_exec.py FRAMES,W,H SCRIPT ...). Scripts named ffgen.py /
ffgen_many.py / ffprep.py are taken from the flex/ copies (their decode and latent sizes follow the geometry)."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import flex_geom  # noqa: E402

flex_geom.apply(*(int(x) for x in sys.argv[1].split(',')))
script = sys.argv[2]
if os.path.basename(script) in ('ffgen.py', 'ffgen_many.py', 'ffprep.py'):
    script = os.path.join(HERE, 'flex', os.path.basename(script))
sys.argv = [script] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name='__main__')
