"""Install locations (filled in by bin/configure.py; see README "Configure")."""
from pathlib import Path

CODE = Path('@CD_CODE@')              # this repository (configured copy)
WORK = Path('@CD_WORK@')              # run data: editing / revision trees WORK/fr/run_<run>, generation WORK/gen/<run>, inputs WORK/inputs
SHARED = Path('@CD_SHARED@')          # code the GPU nodes run (tools/, inv/code/, v4full/code/), models/, results/, generation/
QUEUE = Path('@CD_QUEUE@')            # node queues and runner status: QUEUE/queue/<node>/main, QUEUE/status/<node>
NODE = '@CD_NODE@'                    # root of the GPU environments on every node (see README "GPU environment")
NODE_RUNS = '@CD_NODE_RUNS@'          # node-local scratch (white-model renders, condition caches, case directories)
FFMPEG = '@CD_FFMPEG@'
NODES = tuple('@CD_NODES@'.split(','))
PROMPTS = CODE / 'prompts'
ENGINE = CODE / 'engine'
SEED = 9942001
