# Causal Diorama: generation, revision and editing

This repository holds the code of the three video tasks of the paper. All three work through the same intermediate
representation, the Causal Diorama: a structured plan of the scene (camera, entities, state tracks, events, dependencies,
timeline), executed by a deterministic white-model renderer, plus one reference picture per entity and one of the empty place.

| Task | Entry point | Input | Output | Paper results |
|---|---|---|---|---|
| Generation | `generation/` (plan.py, prepare.py, run.py) | prompts | plans and videos | T2V-CompBench |
| Generation (as a diorama) | `bin/cd_gen.py` | one prompt | a Causal Diorama and its video (the input of revision) | same pipeline |
| Revision | `bin/cd_revise.py` | a Causal Diorama, an edit instruction | the edited diorama and its video | CoVEBench revision |
| Editing | `bin/cd_edit.py` | a real video, an edit instruction, an optional description | the fitted diorama, the edited diorama, the edited video | CoVEBench editing, OpenVE-Bench |

- Revision and editing share one editing engine (`engine/ws/`): the same editor agent and rules, reference pictures,
  first-frame route and empty-place rule. Editing renders video B by F4 inversion of the real video; revision regenerates
  video B under the edited plan from the same initial noise (seed) that video A was generated with.
- Generation, revision and editing share the plan executor, checker, schema, picture prompts and captions (`engine/cb/`) and
  the GPU node code (`shared/tools/`).
- `docs/CHANGES_FROM_RUN.md` lists the differences between this code and the code that produced the paper's numbers.
  `docs/TESTS.md` lists the checks made on this package and the reproduction tests with their per-item scores.

## Layout

| Path | Contents |
|---|---|
| `bin/` | `configure.py` (install), `cd_init.py` (checks, downloads, runner start), `cd_gen.py`, `cd_revise.py`, `cd_edit.py`, `cd_diorama.py` (diorama tools) |
| `generation/` | the generation path of the paper: planning agent (`plan.py`, `agents/`, `prompts/planner.md`), job preparation (`prepare.py`), node task chain (`run.py`); see `generation/README.md` |
| `engine/cb/` | shared modules: executor `world.py`, checker `scm_lint.py` + `rules_pv2.json`, plan schema `scm_schema_pv2.md`, pictures and captions `pv2.py`, first-frame helpers `ff_route.py` / `ff_prep.py`, the stage system prompts `system_stage1.txt` / `system_stage2.txt`, the prompt reader `prompt_tpl.py` |
| `engine/ws/` | the editing engine: `flexrun.py` (steps and run tree), `llm_run.py` (agent runner, both agent modes), `cc_agent.py` (Claude Code subagent replica), `bgv2.py` (empty-place rule), `checkers.py` (perception check), `make_stage2.py` / `make_v4ff.py` (helpers), the agents' tools `geomrun.py`, `scm_match.py`, `cam_helper.py`, `cam_track.py` |
| `engine/cd/` | front-end logic: run trees (`runs.py`), generation as a diorama (`gen.py`), the diorama format (`diorama.py`), source description (`describe.py`), install paths (`paths.py`) |
| `engine/llm/` | Messages API client (`api_llm.py`) and the sandboxed tool loop (`api_agent.py`) |
| `prompts/` | every text sent to a model by the revision / editing / `cd_gen.py` code (section 4) |
| `shared/` | code the GPU nodes run, copied to the shared directory by `configure.py`: `tools/` (runner, node environment, GPU claim, Qwen-Image worker, white-model render, stage-1 / stage-2 preparation and generation), `inv/code/` (perception, camera tracks, F4 inversion, clip-geometry support), `v4full/code/` (detection + segmentation masks of the first-frame edit) |
| `node_env/` | our code inside the node environments (8-GPU H3 sampler and weight loading, GPU reservation) and the CWM environment's version lock |
| `config/` | `site.example.json` (install locations), `cd.env.example` (API settings) |
| `assets/fonts/` | Noto Sans SC (SIL Open Font License), used for the check sheets the agents read |
| `docs/` | differences from the runs, tests, the node environment's package list |

Directory names kept from the runs: `compbench_v1/` on the nodes is the runner's working directory (tool copy, state, logs)
and holds `pydeps/`; `shared/inv/` and `shared/v4full/` hold the editing node code; in a run tree, `v4ff/` holds the first frame
of plan B and `fix/` the video B.

## 1. Requirements

### 1.1 Machines
- GPU nodes, each with 8 GPUs of at least 96 GB (tested: NVIDIA H20, CUDA 12.8). H3 sampling runs sequence-parallel on 8 GPUs.
  One runner per node takes the queued tasks one after another and owns all 8 GPUs.
- A workstation (CPU only) for the agents and the scheduling: Python 3.10+, numpy, pillow, scipy, requests; ffmpeg 6.x (static
  build, needs `-vsync`). `generation/` also needs the `anthropic` package (`generation/requirements.txt`).
- A shared directory that the workstation and every node can read and write (node code, models used by the nodes, queues, results).
- One machine can play both roles.

### 1.2 Language model and agents
- All agents use Claude Opus 5.5 (`claude-opus-5-5`) through the Anthropic Messages API
  (<https://platform.claude.com/docs/en/api/messages/create>).
- Two agent modes, selected by `--agent api|claude-code` on every entry point (`generation/plan.py`, `bin/cd_gen.py`,
  `bin/cd_revise.py`, `bin/cd_edit.py`; the engine reads it as `CD_AGENT_MODE=api|claude_cli`):

| `--agent` | How an agent task runs | Needs |
|---|---|---|
| `api` (default) | revision / editing: a replica of a Claude Code general-purpose subagent on the Messages API (`engine/ws/cc_agent.py`: Claude Code's system prompt, context, tool definitions and result formats; tools sandboxed by `engine/llm/api_agent.py`); `cd_gen.py`: the tool loop of `engine/llm/api_agent.py`; `generation/`: `generation/agents/messages_api.py` | `ANTHROPIC_API_KEY` |
| `claude-code` | one isolated Claude Code process per task, `claude -p "<task>"` in the task's directory with the tools Read / Write / Edit / Bash, no MCP servers, without the caller's session variables | the Claude Code CLI (`CLAUDE_CLI`, default `claude` on `PATH`) and its credential |

- In both modes every agent task starts without context; its only message points to its task file (for example
  "Your whole task is written in <file>. Read that file and follow it exactly.").
- Environment variables (template `config/cd.env.example`):

| Variable | Use | Default |
|---|---|---|
| `CD_AGENT_MODE` | default of `--agent`: `api` or `claude_cli` | `api` |
| `CLAUDE_CLI` | the Claude Code executable (`--agent claude-code`) | `claude` |
| `ANTHROPIC_API_KEY` | API key for the public endpoint (all three paths); read from the environment only, never written to a file or transcript | required |
| `ANTHROPIC_BASE_URL` | optional base URL of an Anthropic-compatible endpoint (all three paths; revision and editing append `/v1/messages`) | unset = public endpoint |
| `ANTHROPIC_AUTH_TOKEN` | optional bearer credential for such an endpoint, used when `ANTHROPIC_API_KEY` is unset | unset |
| `CD_LLM_MESSAGES_URL` | full Messages URL, overrides `ANTHROPIC_BASE_URL` (revision, editing, `cd_gen.py`) | `https://api.anthropic.com/v1/messages` |
| `CD_LLM_AUTH` | `x-api-key`, or `bearer` for a compatible endpoint | `x-api-key` (`bearer` when only `ANTHROPIC_AUTH_TOKEN` is set) |
| `LLM_MODEL` | model | `claude-opus-5-5` |
| `LLM_MAX_CONCURRENCY` | agents running at the same time per run | 8 |

- Agent settings (adaptive thinking; effort per stage): editing and revision stages in `engine/cd/runs.py` (`LLM_PARAMS`);
  generation planning effort medium (`engine/cd/gen.py`, `generation/`).

### 1.3 Models

`python3 bin/cd_init.py --downloads` prints the download commands for the configured directories.

| Model | Use | Source | Files / revision | License |
|---|---|---|---|---|
| MiniMax H3 Ref2VA (ComfyUI pack) | video model, video VAE, audio VAE, Qwen3-VL text encoder | <https://huggingface.co/Comfy-Org/MiniMax-H3> | revision `0543966fbdce5ba05709a8f2031c94bdba629b4a`: `diffusion_models/minimax_h3_ref2va_bf16.safetensors`, `vae/minimax_h3_video_vae_fp16.safetensors`, `vae/minimax_h3_audio_vae_fp32.safetensors`, `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | MiniMax H3 Community License |
| CWM LoRA | white-model control | <https://huggingface.co/NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints> | revision `a3df90f6742748896457cec84b13dbedbce56fff`, `q4_system_chat/rigid_3ep_2e5_to_1e6/minimax-h3-new1344-q4-system-chat-fresh-official-3ep-warmup100-cosine2e5to1e6-fa3-v1.safetensors`, renamed `lora/minimax-h3-cwm.safetensors`, sha256 `0e14ad22ae203f684ba6521e0b6fbf58f64668ed454fbce346acafd7f448fdb2` | MiniMax H3 Community License |
| CWM inference code | `cwm_h3_inference` (H3 Ref2VA inference and caches), with Musubi Tuner (`musubi_tuner`) | <https://github.com/buaacyw/code-world-model> | commit `17fa82ba2fb8168aa0d30fc51a1d1d0ebbc9180b` | Apache-2.0 |
| Qwen3-VL-32B-Instruct | text processor files of H3 | <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct> | revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb` (tokenizer and processor files) | Apache-2.0 |
| Qwen-Image-2.1 | reference pictures, empty place, first-frame edit, entity removal | <https://huggingface.co/Qwen/Qwen-Image-2.1> | diffusers layout | Qwen Research License (non-commercial) |
| Grounding DINO base | entity detection (editing) | <https://huggingface.co/IDEA-Research/grounding-dino-base> | transformers layout | Apache-2.0 |
| SAM 2.1 hiera large | entity masks (editing) | <https://huggingface.co/facebook/sam2.1-hiera-large> | transformers layout | Apache-2.0 |
| Depth-Anything-V2 Metric Small (outdoor, indoor) | metric depth (editing) | <https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf>, <https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf> | config.json, model.safetensors, preprocessor_config.json | Apache-2.0 |

License notes: the MiniMax H3 license restricts some regions and large companies and requires its NOTICE on redistribution;
Qwen-Image-2.1 is for research and non-commercial use. No weights are included. Model file names are the upstream names.

### 1.4 GPU node environment

Under the configured node root (`node` in the site config) on every node:

| Path | Contents | How to prepare |
|---|---|---|
| `cwm_h3_reproduction_v1/.venv` | Python 3.10, torch 2.9.1+cu128, transformers 4.57.6, diffusers 0.32.1, FlashAttention-3 (commit `c68c592f`), Musubi Tuner (commit `c4dc6b80` with the two CWM patches), `cwm_h3_inference`, OpenCV | `ENVIRONMENT.md` of the CWM repository at commit `17fa82b`; versions in `node_env/cwm_h3_reproduction_v1/release_lock.json`, full list in `docs/node_env_freeze.txt` |
| `cwm_h3_reproduction_v1/code/verify_h20_runtime.py` | GPU reservation of the single-GPU workers | copy from `node_env/cwm_h3_reproduction_v1/code/` |
| `cwm_h3_reproduction_v1/release/models/` | `diffusion_models/`, `lora/`, `vae/`, `text_encoders/`, `processors/Qwen3-VL-32B-Instruct` | section 1.3 |
| `cwm_h3_acceleration_v1/code/` | our 8-GPU H3 sampler (sequence parallel, AdaLN precompute, fused kernels) and weight loading | copy `node_env/cwm_h3_acceleration_v1/code/` |
| `qwen_image21/model`, `qwen_image21/pydeps` | Qwen-Image-2.1 and its packages (transformers 5.19.0, diffusers 0.41.0.dev0, `pip install --target`) | kept apart from the CWM environment (incompatible transformers versions) |
| `compbench_v1/pydeps` | scipy 1.15.3 for the white-model render and case preparation (`pip install --target`) | `docs/node_env_freeze.txt` |
| `compbench_v1/tools/` | the runner's copy of `shared/tools` (synced before every task), with `node_env.sh` (Python, CUDA libraries, offline flags) | created by the runner |

In the shared directory: `models/grounding-dino-base`, `models/sam2.1-hiera-large`,
`models/Depth-Anything-V2-Metric-{Outdoor,Indoor}-Small-hf`.

## 2. Configure

```bash
cp config/site.example.json config/my_site.json        # edit the locations
python3 bin/configure.py --config config/my_site.json  # writes the configured copy to <code>, the node code to <shared>
cd <code>                                              # everything below runs in the configured copy
cp config/cd.env.example config/cd.env && vi config/cd.env && source config/cd.env
python3 bin/cd_init.py --start-runner gpu-0            # prints the command that starts the runner; run it on that node
python3 bin/cd_init.py --check                         # API, workstation modules, ffmpeg, runner heartbeats
python3 bin/cd_init.py --check-node gpu-0              # models, environments, imports and LoRA hash on the node
```

The repository keeps every location as a placeholder that `configure.py` fills in:

| Placeholder | Site config key | Meaning |
|---|---|---|
| `@CD_CODE@` | `code` | the configured copy of this repository |
| `@CD_WORK@` | `work` | run data on the workstation side (one tree per run) |
| `@CD_SHARED@` | `shared` | shared directory: node code (`tools/`, `inv/code/`, `v4full/code/`), `models/`, `results/`, `generation/` |
| `@CD_QUEUE@` | `queue` | node queues and runner status (`queue/<node>/main`, `status/<node>`); often the same as `shared` |
| `@CD_NODE@` | `node` | root of the node environments (section 1.4) |
| `@CD_NODE_RUNS@` | `node_runs` | node-local scratch: white-model renders, condition caches, case directories |
| `@CD_FFMPEG@` | `ffmpeg` | ffmpeg on the workstation |
| `@CD_NODES@` | `nodes` | GPU node names (`gpu-0`, `gpu-1`, ...), one runner each |

`configure.py` also writes `generation/config.json` from `generation/config.example.json`, whose placeholders it maps as
follows: `{{SHARED}}` -> `shared` (work and results under `<shared>/generation/`), `{{NODE_WORK}}` -> `node_runs`, `{{H3_ENV}}`
-> `<node>/cwm_h3_reproduction_v1/.venv`, `{{MODELS}}` -> `<node>/cwm_h3_reproduction_v1/release/models`; `queue_root` is set
to `<queue>/queue`, `node_tools` to `<node>/compbench_v1/tools` (the runner's copy of `shared/tools`), `pydeps` and
`qwen_pydeps` to the node paths of section 1.4. The commands in `generation/README.md` are written from the package root
(`code/generation/...`); in the configured copy they are `generation/...`, as below.

## 3. Run

`--run` names the run's data tree and must be new (or use `--resume`). `--outputs diorama|video|both` selects the outputs
(`diorama`: no video of plan B, so no GPU video step for B). Each command runs in either agent mode; the mode only changes how
agent tasks are executed.

### 3.1 Generation

The paper's generation path (`generation/README.md` for details):

```bash
python3 generation/plan.py --prompts generation/examples/compbench_check10.json --out plans --jobs 4 --agent api
python3 generation/plan.py --prompts generation/examples/compbench_check10.json --out plans --jobs 4 --agent claude-code
python3 generation/prepare.py --plans plans --config generation/config.json
python3 generation/run.py --config generation/config.json --node gpu-0 --wave w1 --ids <id>,<id>
```

The same pipeline for one prompt, packed as a Causal Diorama (the input of revision):

```bash
python3 bin/cd_gen.py --prompt "A hot air balloon floating under a sheep" --run balloon --out out/balloon --agent api
python3 bin/cd_gen.py --prompt "A hot air balloon floating under a sheep" --run balloon --out out/balloon --agent claude-code
```

Output: `out/balloon/diorama/` (plan, pictures, first frame, video, seed) and `out/balloon/video.mp4` (1344x768, 124 frames,
24 fps). Steps: planning agent with the checker (at most 3 runs of it); Qwen-Image-2.1 pictures; white model; stage 1 (H3 Ref2VA +
CWM LoRA, free first frame, frame 0 kept); stage 2 (that frame locked). `--plan FILE` skips the planning agent.

### 3.2 Revision

```bash
python3 bin/cd_revise.py --diorama out/balloon/diorama --instruction "Make the balloon blue and let it rise slowly" --run balloon_blue --out out/balloon_blue --agent api
python3 bin/cd_revise.py --diorama out/balloon/diorama --instruction "Make the balloon blue and let it rise slowly" --run balloon_blue --out out/balloon_blue --agent claude-code
```

Output: `out/balloon_blue/diorama_B/` (the edited diorama; `parent` points to the input) and `video.mp4`. The input diorama must
be 124 frames at 1344x768 and record the seed of its video (`generation.seed`); a diorama without a video gets its video A first.
Steps: plan A, its pictures, its first frame and its video come from the diorama; then the editing engine runs the editor agent
(with the editor addendum), the reference pictures of plan B, the first-frame route (`route_flex`: regenerate only when the edit
changes the camera; otherwise edit or reuse frame 0 of video A) and the empty-place rule; finally video B is generated as a whole
under conditions B (plan B, its pictures, its white model, its first frame locked) from the seed of video A, 124 frames, 20 steps
(`flexrun.regen_b`, `shared/tools/cwm/generate_many.py`).

### 3.3 Editing

```bash
python3 bin/cd_edit.py --video in.mp4 --instruction "Replace the background with a snowy forest" --run snow1 --out out/snow1 [--prompt "..."] --agent api
python3 bin/cd_edit.py --video in.mp4 --instruction "Replace the background with a snowy forest" --run snow1 --out out/snow1 [--prompt "..."] --agent claude-code
```

Output: `diorama_A/` (fitted to the source, with the source video), `diorama_B/` (edited, with the edited video), `video.mp4`.
Length and aspect follow the source: 17n+5 frames at 24 fps, at most 362 (15.08 s; longer sources are cut there), about
1344x768 pixels in multiples of 64. Without `--prompt`, Claude Opus describes 6 frames of the source in about 40 words.
Steps: entity-analyst agent; Grounding DINO + SAM 2.1 + metric depth, checked by the perception-check agent; measured camera;
fitting planner agent (plan A); editor agent (plan B, with the editor addendum); reference pictures (cut-outs from the real
frames for plan A); empty-place rule (removal passes on frame 0 and a check agent); first-frame route and edit; F4 inversion
(conditions A along the real video, conditions B with the recorded offsets, 20 steps).

### 3.4 Causal Diorama files

```bash
python3 bin/cd_diorama.py info     out/balloon/diorama
python3 bin/cd_diorama.py validate out/balloon/diorama --render
python3 bin/cd_diorama.py render   out/balloon/diorama prev/
python3 bin/cd_diorama.py zip      out/balloon/diorama
```

A diorama is a directory (or its zip): `diorama.json` (plan and metadata, with sha256 of every file), `pictures/<key>.png`,
optional `first_frame.png` and `video.mp4`. The format is described at the top of `engine/cd/diorama.py`; the plan's fields in
`engine/cb/scm_schema_pv2.md`. A plan edited by hand can be revised with `cd_revise.py` or regenerated with `cd_gen.py --plan`.

## 4. Prompts

Every text that revision, editing and `cd_gen.py` send to a model is a file in `prompts/` (code reads it through
`engine/cb/prompt_tpl.py`; `{NAME}` is filled in, literal braces are doubled):

| Directory | Contents |
|---|---|
| `prompts/agents/` | agent task templates: `planner_t2v.md` (+ `planner_t2v_ask.txt`), `entities_flex.md`, `planner_flex.md`, `editor_B.md`, `editor_addendum.md` (appended to every editor task), `describe_source.md`, `pcheck/`, `bgcheck/`, `v4ins/` (perception check, empty-place check, first-frame instruction), `spawn_message.txt` |
| `prompts/image/` | Qwen-Image-2.1 prompts: entity pictures, empty place, picture edits, entity removal |
| `prompts/video/` | caption and scene-text pieces (`caption/`, `scene/`); the two stage system prompts are `engine/cb/system_stage{1,2}.txt` |
| `prompts/cc/` | the Claude Code subagent context and tool definitions replicated by `cc_agent.py` |

`generation/prompts/planner.md` is the generation path's planning task (the text of `planner_t2v.md` with the schema path as a
placeholder).

## 5. Known limits
- Agent outputs vary between runs, so a rerun gives other plans and videos; given the plans and agent outputs, the GPU steps
  are deterministic on the same hardware up to video encoding.
- A runner runs one task at a time with all 8 GPUs; runs queue behind each other.
- Editing handles at most 15.08 s of a source; revision handles generated dioramas (124 frames, 1344x768).

## 6. Not included
- Model weights, benchmark data and the run data of the paper.
- Evaluation code (CoVEBench, OpenVE-Bench and T2V-CompBench scoring); use the benchmarks' official code.
- Baselines, ablations, inversion variants other than F4, and research-only switches of the runs.
- Execution of agent tasks by subagents of an interactive session, used for part of the original runs;
  `--agent claude-code` runs the same task text in a separate Claude Code process.
- The API client of the original runs; this package calls the public Messages API.
