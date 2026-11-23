# Checks of this package and reproduction tests

## 1. Install and static checks

| Check | Result |
|---|---|
| `python -m compileall` over the package | passes |
| pyflakes | no undefined names (remaining messages: unused imports kept for re-export) |
| Fresh install: `bin/configure.py` with a test site config, then `--help` of every entry script, the engine and generation imports | passes; `generation/config.json` written |
| Scan for machine paths, host and user names, internal names, dates and quotes (`scrub.py`, rules in the build notes) | no hits |

## 2. Prompts: byte-for-byte comparison with the runs

The package's code, reading every prompt from `prompts/`, regenerated the texts from the runs' stored inputs; they were compared
byte for byte with what the runs sent or wrote (paths of the run trees and code mapped back to the run's).

| Text | Run | Count | Identical |
|---|---|---|---|
| Reference-picture prompts (entities, empty place, picture edits; `pv2.augment` of plans A / B) | OpenVE (431 clips) | 2,669 | 2,669 |
| Stage-2 captions, plan A and plan B | OpenVE | 431 + 431 | all |
| Entity-analyst task files | OpenVE | 431 | 431 |
| Fitting-planner task files | OpenVE | 431 | 431 |
| Editor task files, the template part | OpenVE | 431 | 431 |
| Editor task files, whole (template + addendum) | OpenVE | 431 | 351; the other 80 clips ran without the addendum (by design, `CHANGES_FROM_RUN.md` 3.1) |
| First-frame instruction task files (edit route) | OpenVE | 377 | 377 |
| Empty-place check task files, plan A / plan B | OpenVE | 431 / 131 | all |
| Entity-removal prompts, first pass / later passes | OpenVE | 705 / 556 | all |
| Perception-check task files (strips and frames drawn anew from the perception results) | OpenVE, 3 clips | 3 | 3 |
| CoVEBench editing, every 5th clip (121 clips): reference-picture prompts / stage-2 captions A and B / entity tasks / removal prompts (first, later) / empty-place check tasks A, B | CoVEBench | 990 / 121 + 121 / 121 / 687, 114 / 13, 9 | all |
| CoVEBench editing: fitting-planner task files | CoVEBench | 121 | 99; 22 clips were planned with the earlier planner template, before the measured camera (`CHANGES_FROM_RUN.md` 4.4) |
| CoVEBench editing: editor task files, template part / whole | CoVEBench | 121 / 121 | 121 / 0 (the run had no addendum, 4.1) |
| CoVEBench editing: first-frame instruction task files | CoVEBench | 99 | 12; in the other 87 the run's agent also got the edit instruction and the editor's summary, which the package leaves out (as in the OpenVE run) |

## 3. Revision: offline run up to the video step (CoVEBench 0282)

Without GPUs (node tasks only written, `flexrun.py auto --dry`), a revision tree was built from the run's diorama of task 0282
(plan A, pictures, first frame, video A, seed 9942001), with the run's plan B in place of a new editor output:
- the engine started at the editor (one editor task, addendum appended); after plan B it planned the pictures of plan B, routed
  the first frame (`route_flex`: edit, global; the same as the offline route table) and queued the empty-place passes of plan B;
- the stage-2 job of video B has the same caption, stage-2 system prompt, seed (9942001) and picture keys as the run's stage-2 job
  of plan B; its first frame is the routed one; the generation task is `generate_many.py` with the seed of video A.

## 4. Reproduction tests

Each test has two forms. Replay gives the run's stored plans, pictures and first frame and runs only the video step. Fresh runs
the package end to end from the inputs. The fresh test was run on 10 items of each editing benchmark with this package, with
`--agent claude-code` (one isolated `claude -p` process per agent task), on two nodes (8 H20 each), with the 30 runs started
together. All 30 finished, with no failed node task and no failed agent task. The wall time was 9.5 h for all 30, set by the two
nodes; F4 inversion and same-seed regeneration queue behind each other.
The outputs were scored the same way as the paper's runs. CoVEBench used the official judge (Qwen3.5-122B-A10B, official code
and settings). Its subjective metrics were aggregated per task by the official `aggregate_subjective.py`, and VQR comes from the
official objective metrics. Revision was scored on the evaluation clip of video B (81 frames, 832x480, 16 fps) against video A,
as in the run. Editing was scored on the edited video at the clip's own length and size. OpenVE-Bench used the judge of the run
(Gemini 2.5 Pro, the paper's per-category prompt). Agent outputs differ between runs, so a fresh run gives other plans and other
videos. With one judge pass per video, per-item differences are large (a single answer moves UAS of one task by 33 to 100 points).

| Set | Clips | Inputs |
|---|---|---|
| CoVEBench revision | 10 tasks: 7 whose earlier first-frame route equals `route_flex` (edit-local 2, edit-global 2, reuse 1, regen 2) and 3 where it differs | plan A, its pictures, its stage-1 first frame, video A, seed 9942001, the instruction |
| CoVEBench editing | 10 clips: 4 made with empty-place rule v2, 3 with the old rule, 3 whose camera was written by eye; none had the editor addendum in the run | source video, description, instruction |
| OpenVE-Bench | 10 clips (3 from the clips edited without the addendum) | source video, description, instruction |

Mean over the 10 items (CoVEBench pooled as in the official aggregation, paper run / package):

| Set | UAS | IFS | VRS | SEM | VQR | OpenVE overall (IC / CF / VQ) |
|---|---|---|---|---|---|---|
| CoVEBench revision | 70.4 / 70.4 | 92.8 / 91.3 | 82.2 / 80.0 | 98.2 / 98.8 | 4.53 / 4.55 | |
| CoVEBench editing | 81.2 / 81.2 | 89.9 / 82.6 | 75.9 / 86.2 | 95.7 / 95.5 | 4.33 / 4.29 | |
| OpenVE-Bench | | | | | | 3.37 / 3.73 (3.90 / 4.10, 3.00 / 3.40, 3.20 / 3.70) |

### 4.1 CoVEBench revision (the run: V4 plans B and videos B, `ff_route`; package: `route_flex`)

| Task | Route: run / package | UAS run / package | IFS | VRS | SEM | VQR |
|---|---|---|---|---|---|---|
| 0361 | edit-local / edit-local | 50.0 / 0.0 | 88.9 / 88.9 | 85.7 / 57.1 | 100.0 / 100.0 | 4.71 / 4.67 |
| 0178 | edit-local / edit-local | 0.0 / 66.7 | 81.8 / 81.8 | 40.0 / 80.0 | 100.0 / 100.0 | 4.68 / 4.67 |
| 0296 | edit-global / edit-global | 100.0 / 66.7 | 100.0 / 100.0 | 100.0 / 83.3 | 100.0 / 100.0 | 4.74 / 4.79 |
| 0557 | edit-global / edit-global | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 92.5 / 100.0 | 4.34 / 4.40 |
| 0019 | reuse / reuse | 0.0 / 0.0 | 100.0 / 75.0 | 20.0 / 20.0 | 100.0 / 100.0 | 4.53 / 4.48 |
| 0211 | regen / regen | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 4.56 / 4.54 |
| 0215 | regen / regen | 66.7 / 66.7 | 85.7 / 85.7 | 100.0 / 100.0 | 100.0 / 100.0 | 4.66 / 4.62 |
| 0168 | regen / edit-local (differs) | 66.7 / 66.7 | 88.9 / 88.9 | 100.0 / 100.0 | 96.0 / 98.0 | 4.33 / 4.61 |
| 0331 | regen / edit-local (differs) | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 90.0 | 4.50 / 4.52 |
| 0026 | regen / edit-local (differs) | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 93.3 / 100.0 | 4.25 / 4.18 |
| mean of tasks | | 68.3 / 66.7 | 94.5 / 92.0 | 84.6 / 84.0 | 98.2 / 98.8 | 4.53 / 4.55 |
| mean difference | | -1.7 | -2.5 | -0.5 | +0.6 | +0.02 |

### 4.2 CoVEBench editing

| Task | Addendum in the run | UAS run / package | IFS | VRS | SEM | VQR |
|---|---|---|---|---|---|---|
| 0438 | no | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 80.0 | 4.54 / 4.37 |
| 0282 | no | 66.7 / 100.0 | 100.0 / 100.0 | 50.0 / 100.0 | 85.0 / 95.0 | 4.50 / 4.58 |
| 0063 | no | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 90.0 / 100.0 | 3.61 / 3.65 |
| 0113 | no | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 95.0 / 97.5 | 4.60 / 4.68 |
| 0423 | no | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 90.0 / 90.0 | 4.48 / 4.51 |
| 0411 | no | 75.0 / 100.0 | 87.5 / 100.0 | 75.0 / 100.0 | 100.0 / 100.0 | 3.86 / 3.88 |
| 0175 | no | 100.0 / 50.0 | 100.0 / 42.9 | 100.0 / 50.0 | 96.7 / 96.7 | 4.69 / 4.60 |
| 0083 | no | 66.7 / 66.7 | 66.7 / 66.7 | 50.0 / 50.0 | 100.0 / 97.5 | 4.32 / 4.04 |
| 0034 | no | 50.0 / 0.0 | 75.0 / 25.0 | 0.0 / 0.0 | 100.0 / 100.0 | 4.25 / 4.25 |
| 0218 | no | 33.3 / 66.7 | 57.1 / 57.1 | 50.0 / 83.3 | 100.0 / 98.3 | 4.50 / 4.30 |
| mean of tasks | | 79.2 / 78.3 | 88.6 / 79.2 | 72.5 / 78.3 | 95.7 / 95.5 | 4.33 / 4.29 |
| mean difference | | -0.8 | -9.5 | +5.8 | -0.2 | -0.05 |

### 4.3 OpenVE-Bench

| Clip | Addendum in the run | Overall run / package | IC | CF | VQ |
|---|---|---|---|---|---|
| 90031 | no | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 |
| 90032 | no | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 |
| 90064 | no | 3.00 / 4.33 | 5.00 / 5.00 | 2.00 / 4.00 | 2.00 / 4.00 |
| 90115 | yes | 3.00 / 5.00 | 3.00 / 5.00 | 3.00 / 5.00 | 3.00 / 5.00 |
| 90218 | yes | 2.33 / 4.00 | 3.00 / 4.00 | 2.00 / 4.00 | 2.00 / 4.00 |
| 90290 | yes | 2.33 / 2.33 | 3.00 / 3.00 | 2.00 / 1.00 | 2.00 / 3.00 |
| 90299 | yes | 4.00 / 2.33 | 4.00 / 3.00 | 4.00 / 2.00 | 4.00 / 2.00 |
| 90322 | yes | 2.00 / 2.00 | 3.00 / 3.00 | 1.00 / 1.00 | 2.00 / 2.00 |
| 90323 | yes | 2.00 / 2.33 | 3.00 / 3.00 | 1.00 / 2.00 | 2.00 / 2.00 |
| 90424 | yes | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 |
| mean | | 3.37 / 3.73 | 3.90 / 4.10 | 3.00 / 3.40 | 3.20 / 3.70 |
| mean difference | | +0.37 | +0.20 | +0.40 | +0.50 |

### 4.4 Fixed after the tests

In a revision tree, the first-frame edit queued its Qwen-Image edit but stopped before it queued the mask task. The list file of
the mask task went to `jobs/`, which no earlier step of a revision tree had created. When the edit was done, the mask task was
never queued again, so a local first-frame edit could wait forever (1 of 10 revision tasks). Also, the engine counted the edit
task as finished before its mask task had run and queued both again. `engine/ws/flexrun.py` now creates the directory, queues
missing masks even when the edited frame exists, and waits for the mask task of the batch. The affected task was continued with
the fixed code.
