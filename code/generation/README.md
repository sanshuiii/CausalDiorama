# Text-to-video generation (T2V-CompBench, Table 2)

This directory generates videos from text prompts with the paper's main method: a planning agent writes a structured plan of
the scene (entities, motions, events, camera, dependencies, timeline), a deterministic executor turns the plan into a white
model (per-frame depth and semantic maps) and reference pictures, and a two-stage video model renders the video conditioned on
them. Only the generation path is here; the benchmark's scoring is the official T2V-CompBench code.

```
code/generation/
  plan.py              prompts -> plans (one context-free agent per prompt)
  prepare.py           plans -> job files (CPU)
  run.py               job files -> one chain of tasks for a GPU node's runner
  common.py            paths and fixed settings
  agents/              messages_api.py (Anthropic Messages API), claude_code.py (`claude -p`), sandbox.py (the API agent's tools)
  prompts/planner.md   the planning task text
  examples/compbench_check10.json, compbench_check10_scores.csv, paper_plans/<id>/{plan.json, prompt.json}
  config.example.json  requirements.txt
code/engine/cb/        executor world.py, checker scm_lint.py + rules_pv2.json, plan schema scm_schema_pv2.md,
                       conditions pv2.py, stage-1/2 system prompts (shared with revision and editing)
code/shared/tools/     node side: runner.py, qwen_image/worker.py, e2/render.py, cwm/ffprep.py, cwm/ffgen_many.py,
                       e2/pv2_case_prep.py, e2/cwm_prepare.py, cwm/generate_many.py, gpu_preclaim.py
```

## 1. Plan

Input: a JSON list of `{"id", "prompt"}`, optionally with `"category"` and `"name"`, which set where the video goes:
`<results>/video/<category>/<name>.mp4`. For T2V-CompBench, use `id = <category>_NNNN` and `name = NNNN`, the prompt's 1-based
index in the benchmark's prompt list, as in `examples/compbench_check10.json`.

The agent is selected with `--agent`; both modes give the agent the same task text, the same workspace (`task.md` and
`schema.md` only) and the same checker command.

Anthropic Messages API (default; official SDK, credentials from `ANTHROPIC_API_KEY`):

```
pip install -r code/generation/requirements.txt
export ANTHROPIC_API_KEY=...
python code/generation/plan.py --prompts code/generation/examples/compbench_check10.json --out plans --agent api --jobs 4
```

Status: the API mode has been run through an Anthropic-compatible endpoint (`ANTHROPIC_BASE_URL` with `ANTHROPIC_AUTH_TOKEN`).

Claude Code, one isolated `claude -p` process per prompt (CLI from `CLAUDE_CLI`, default `claude` on `PATH`):

```
python code/generation/plan.py --prompts code/generation/examples/compbench_check10.json --out plans --agent claude-code --jobs 4
```

A single prompt: `--prompt "..." --id NAME` instead of `--prompts`.

Each agent starts without any context and works in a fresh directory with a random name under `--workspace-root` (default
`$TMPDIR/plan_ws`). It may read and write only that directory and run only the checker,
`python3 <tools>/scm_lint.py <dir>/plan.json --rules <tools>/rules_pv2.json`. The checker is a copy in a separate directory
that the agent can run but not read. The API mode enforces this in `agents/sandbox.py`; the CLI mode enforces it with
`--tools`, `--allowedTools` and `--disallowedTools`. Settings: `claude-opus-5-5`, effort `medium`, adaptive thinking, at most
32000 output tokens per turn. A technical failure is retried in a fresh workspace, at most 3 attempts in all; this covers an
API error, a refusal, a run without any tool call, and a CLI error. A plan that the checker still rejects at the end is kept
as written. The executor runs it when it can; otherwise prepare.py records the prompt as a planning failure.

Output: `plans/<id>/{prompt.json, plan.json, plan.lint/, task.md, attempts.json}`, the API transcripts and `plans/summary.json`.

## 2. Prepare (CPU)

Copy `config.example.json` to `config.json` and replace the placeholders:

| key | meaning |
|---|---|
| `work_root` | shared by the planning machine and the nodes: plans, job files, reference pictures |
| `results_root` | shared: stage-1 first frames `stage1/<category>/<name>.png` and videos `video/<category>/<name>.mp4` |
| `queue_root` | the runners' queue root (`<queue_root>/<node>/main`) |
| `node_root` | node-local work tree: white-model renders, condition caches, stage-2 case directories |
| `node_tools` | the runner's tools directory on the node (a copy of `code/shared/tools`) |
| `node_python`, `node_torchrun` | the video model's environment on the node |
| `pydeps`, `qwen_pydeps` | extra `PYTHONPATH` for the render / stage-2 preparation workers and the picture worker (may be empty) |
| `models` | H3 Ref2VA weights, CWM LoRA, video VAE, Qwen3-VL processor and text encoder |

```
python code/generation/prepare.py --plans plans --config config.json
```

For each plan this runs the executor (`pv2.augment`, `pv2.frame0`) and writes the job files listed below. A plan the
executor cannot run is recorded as failed in `<work_root>/prepared.json` and is not re-planned.
- reference pictures: Qwen-Image-2.1, 40 steps, seed 9942001;
- stage 1: H3 Ref2VA + CWM LoRA conditioned on the pictures, the white model and the stage-1 caption; 124 frames, 20 steps; frame 0 kept;
- stage 2: the same model, with the stage-1 frame 0 locked as the first frame; 124 frames, 20 steps, 1344x768, 24 fps.

## 3. Generate (GPU node, 8 GPUs)

Start the runner on the node (`code/shared/tools`), then queue a wave of prepared prompts:

```
python code/generation/run.py --config config.json --node NODE --wave w1 --ids consistent_attr_0028,dynamic_attr_0003
```

The chain runs these steps in order:
1. reference pictures;
2. white-model renders;
3. stage-1 condition caches;
4. stage-1 generation;
5. stage-2 preparation;
6. stage-2 condition caches;
7. stage-2 generation.

Before each GPU step, a short GPU claim runs. A node-level GPU keeper adopts the claim instead of the job that follows. The
videos land in `<results_root>/video/<category>/<name>.mp4`, ready for the official T2V-CompBench evaluation scripts.

## Reproducibility

The 10 test prompts in `examples/compbench_check10.json` were rerun with this code, starting from the plans of the paper's run.
- The reference pictures (41/41) and the stage-1 first frames (10/10) were pixel-identical to the paper's run.
- 6 of the 10 final videos were pixel-identical. The other 4 differ slightly (PSNR 40-47 dB), by the same amount as two runs of the
  same job on the same node, i.e. from floating-point nondeterminism in multi-GPU inference.
- The official scores of these videos average 0.543 against 0.553 for the paper's videos.
- Replanning the same prompts with this code (`--agent claude-code`) gives 0.610. The plans differ, and the difference is not
  significant on 10 prompts.

Per-prompt scores are in `examples/compbench_check10_scores.csv`.

To reproduce the `this_code_paper_plans` column, run prepare.py and run.py on `examples/paper_plans`, the paper's plans for
these prompts (only their `scene_id` was renamed to the prompt id; no step reads it):

```
python code/generation/prepare.py --plans code/generation/examples/paper_plans --config config.json
python code/generation/run.py --config config.json --node NODE --wave p1 --ids consistent_attr_0028,dynamic_attr_0003,dynamic_attr_0021,spatial_0006,spatial_0086,motion_0053,motion_0120,action_binding_0058,interaction_0100,numeracy_0107
```
