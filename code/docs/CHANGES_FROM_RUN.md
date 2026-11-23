# Differences between this package and the code that produced the paper's numbers

This package was built from the code of the runs: the files were copied, cluster paths were replaced by placeholders, and the
changes below were made. Everything not listed here is the code that ran.

## 1. Sources

| Part | Taken from | Results |
|---|---|---|
| Editing engine (`engine/ws/`, `shared/inv/`, `shared/v4full/`) | the editing code as it ran for OpenVE-Bench | OpenVE-Bench (431 clips); the same engine with earlier rules produced the CoVEBench editing numbers (section 4) |
| Revision | the editing engine (plan B, pictures, first frame) + stage-2 generation with the seed of video A | CoVEBench revision (615 tasks) was produced by an earlier pipeline (section 5) |
| Generation (`generation/`, `bin/cd_gen.py`) | the T2V-CompBench generation code (planning agent, preparation, node task chain) | T2V-CompBench (1,400 prompts) |
| Shared modules (`engine/cb/`) | the executor, checker, schema, picture and caption code used by all three | all |
| Node code (`shared/tools/`, `node_env/`) | the runner and node scripts the GPU nodes ran | all |

## 2. Changes common to all paths

1. **Locations.** Cluster paths, host and user names became the placeholders `@CD_CODE@`, `@CD_WORK@`, `@CD_SHARED@`,
   `@CD_QUEUE@`, `@CD_NODE@`, `@CD_NODE_RUNS@`, `@CD_FFMPEG@`, `@CD_NODES@`, filled in by `bin/configure.py`; node names are
   configured (`gpu-0`, ...). Environment variables were renamed (`CD_TREE`, `CD_GEOM`).
2. **Prompts in files.** Prompts written inline in the code (agent tasks, picture prompts, caption and scene-text pieces) were
   moved to `prompts/` and are read by `engine/cb/prompt_tpl.py`. The regenerated texts are byte-identical to what the runs
   sent (`docs/TESTS.md`), except where this document names a change.
3. **Comments.** Dates, internal version and study names, user quotes and cluster notes were removed from comments and
   docstrings. In `engine/cb/world.py` (and its node copy) one unused local variable was removed. No other statement changed in
   `engine/cb/`.
4. **LLM transport.** The original runs used a different API transport; this package calls the public Anthropic Messages API directly, with the same prompts, rules and effort settings.
5. **Agent execution.** In the original runs, part of the agent tasks were executed by subagents of an interactive session rather than by isolated processes; this package uses only the two isolated modes (Messages API, or one `claude -p` process per task). The task texts are the same in every mode.
6. **Scope.** Removed: baselines, ablations, evaluation and scoring code, inversion variants F1 to F3 and F5 and the
   no-LoRA ("native") model path, the old empty-place rule (three candidates), per-clip lists and switches (rule lists,
   priority lists, hold switches, node roles, nodes-down lists), calibration and audit tools, research-only test cases, and
   dead code. The node environment of the no-LoRA path is no longer required.
7. **Merged duplicates.** One `flex_geom.py` (`shared/inv/code/`, also imported by the workstation code); one `masks_node.py`
   (`shared/v4full/code/`); one `ff_route.py` (`engine/cb/`, the version with the camera-keyframe fallback that the editing runs
   used; the first-frame study entry points were removed); `ff_prep.py` keeps only its white-model helpers. Two `world.py` files
   remain on purpose: `engine/cb/world.py` (workstation; scene text read from `prompts/`) and `shared/tools/cb/world.py` (node
   copy, used by the renderer, which does not need the prompt files); their executor and renderer code is identical.
   `shared/inv/code/flex/` holds geometry-aware copies of four node scripts of `shared/tools/cwm/` (used for clips other than
   124 x 1344 x 768); the originals stay for generation and revision.
8. **Node environment.** The runner sourced an environment script of the node that was not part of the code; it is now
   `shared/tools/node_env.sh`. Detection and segmentation models are read from the shared directory only (the runs read the
   same weights from a node-local copy for the first-frame masks).

## 3. Editing: OpenVE-Bench (431 clips)

The OpenVE run used the rules of this package: empty-place rule v2 for every clip, the first-frame route
`route_flex`, the measured camera, and the stage efforts of `LLM_PARAMS` (medium; xhigh for the perception check). Differences:

1. **Editor addendum.** In the OpenVE run, 351 of the 431 clips were edited with the editor addendum (the run appended its tree's
   `runs/editor_addendum.md`, the same text as `prompts/agents/editor_addendum.md`). The other 80 clips (90001 to 90080) were
   carried over from an earlier 80-clip run of the same code and were edited without it. The package appends the addendum to
   every editor task.
2. **Empty-place check agents (`bgcheck2`, `bgcheckB`).** The run sent them over that transport with a
   plain tool loop (no system prompt, the model's default effort). The package runs them like every other stage (`api` mode: the
   Claude Code replica, effort high).
3. **Agent execution.** In the OpenVE run, 1,356 agent tasks ran as subagents of an interactive session (entities 351, perception check 333,
   planner 198, editor 174, first-frame instruction 116, empty-place checks 184) and 1,716 through the API (entities 80,
   perception check 98, planner 230, editor 256, first-frame instruction 261, empty-place checks 791).
4. **Scheduling.** The run advanced 431 clips in one tree with priorities, node roles and parallel preparation. The package
   runs one clip per tree and advances it step by step; the steps, their inputs and their order per clip are the same. Whether the
   wave-1 pictures are complete is now checked picture by picture (the run checked the empty-place removal job, which was queued
   with them).
5. **F4.** `inv_fix_v4.py` keeps only the F4 path; its arithmetic is unchanged. The reconstruction pass is skipped, as in the run.
6. **Perception.** The perception directory of a clip is `percept/<n>_checked` after the perception check; hand-corrected
   perception directories of the earlier eight-clip study are gone (they did not apply to OpenVE).

## 4. Editing: CoVEBench (601 clips)

The CoVEBench editing numbers come from the same editing engine at earlier stages of its rules:

1. **(a) No editor addendum.** The CoVEBench editing numbers (601 clips) were produced without the editor addendum; this package
   always appends it.
2. **Empty-place rule.** Rule v2 was applied to 72 clips; the others used the old rule (three candidates: the removal pasted back
   inside the entity masks, a second pass, and a picture drawn from text, chosen by a check agent). The package uses rule v2 for
   every clip.
3. **First-frame route.** 32 finished clips were routed by the earlier rule (regenerate when the camera or a common entity's
   placement changes at t=0), the others by `route_flex`.
4. **Measured camera.** In 154 clips the planner wrote the camera keyframes by eye (before the measured camera was added);
   the package always gives the planner the measured camera.
5. **Agent execution.** Agents ran as subagents of an interactive session, as the Claude Code replica over that transport, or as a plain API
   loop, in different shares per stage.
6. **First-frame instruction.** For clips outside the empty-place v2 list the run's first-frame instruction agent also got the
   edit instruction of the whole video and the editor's summary; the package (as the OpenVE run) gives it the list of first-frame
   changes only.
7. **Planner template.** Clips planned before the measured camera used an earlier planner task text (one camera helper instead of
   two); the package uses the current template.

## 5. Revision: CoVEBench (615 tasks)

The CoVEBench revision numbers come from an earlier pipeline built on the generation code (plans A and B, pictures, first-frame
route, stage 2):

1. **(c) Video B.** As in the experiments, video B is regenerated as a whole under conditions B with the same seed as video A
   (9942001): the stage-2 job of plan B (its pictures, its white model, its caption and stage-2 system prompt), with the first
   frame of the route locked as frame 0, 124 frames, 20 steps. The package's stage-2 job for plan B has the same caption, system
   prompt, seed and picture keys as the run's (checked on task 0282, `docs/TESTS.md`). No inversion is used.
2. **(b) First-frame route.** The CoVEBench revision numbers used the earlier first-frame route `ff_route` (regenerate when the
   camera's t=0 pose or any common entity's t=0 placement differs by more than 0.05; 25 tasks whose route raised an error were
   regenerated). This package uses `route_flex`, the route of editing (regenerate only when the edit declares a camera change and
   the t=0 view differs; a moved entity is moved by the first-frame edit). On the 615 tasks, 249 routes differ (mostly regenerate
   in the run, edit in the package).
3. **Plan B.** The run's plans B were written by the editor agent without the editor addendum and without the re-layout of long
   plan lines, as subagents of an interactive session. The package runs the editing engine's editor (with the addendum).
4. **Pictures of plan B.** The run drew a new empty place from plan B's text when the background or style changed. The package
   uses empty-place rule v2 (removal passes on plan B's first frame, checked by an agent) and leaves out the environment picture
   when the old place still shows a changed entity.
5. **First-frame edit.** The run's first-frame instruction agent also saw the edit instruction and the editor's summary; the
   package's (as in editing) sees only the list of first-frame changes. Regenerated first frames: the run reused the stage-1
   frame of plan B from the generation run (same seed); the package generates it with the seed of video A.

## 6. Generation: T2V-CompBench (1,400 prompts)

`generation/` is the generation side's release of the run code; `bin/cd_gen.py` runs the same steps for one prompt and packs a
Causal Diorama.

1. **Planning agent.** The run used three ways: 210 prompts (including the 10 of `generation/examples/compbench_check10.json`)
   as subagents of an interactive session, 683 as isolated `claude -p` processes, 507 as an API tool loop over that transport. The package offers the API mode (Messages API, adaptive thinking carried across turns, images in tool
   results, prompt cache, SDK retries) and the `claude -p` mode.
2. **CLI version.** The run used a fixed snapshot of the Claude Code CLI that no longer exists; the package uses the CLI on the
   path (`CLAUDE_CLI`).
3. **Task text.** Unchanged; the schema path is a placeholder filled per workspace (`generation/prompts/planner.md`), so the
   rendered task file is identical for the same workspace. Model, effort (medium), max_tokens (32,000 in API mode), the
   checker and the retry rules (at most 3 attempts on technical failures; a plan that still fails the checker is used as written)
   are unchanged. A refusal now counts as a technical failure.
4. **Workspace and isolation.** Neutral random workspace names instead of the run's shuffled ids; the CLI mode refuses reads of
   the checker copy, the package, the plan output and the home directory instead of the run's machine-specific directories. The
   run's after-the-fact transcript audit is not included (the restrictions are enforced by the sandbox and the CLI flags).
5. **Preparation and node chain.** The same functions and fixed settings (pictures 40 steps, seed 9942001, 124 frames, 20 steps
   per stage, the two stage system prompts); checked on 10 prompts: augmented plans, frame-0 records, picture jobs, stage jobs and
   white-model renders are identical. Step d of the run's chain (a diagnostic render report) is left out; the GPU claim runs
   without the node-specific environment prefix. Job names and result directories are neutral (`stage1/`, `video/`).
