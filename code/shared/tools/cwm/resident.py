"""Resident H3 Ref2VA DiT shared by ffgen_many.py and generate_many.py (many generations per torchrun process).

What a single-job script (ffgen.py) does once per process, and how it is kept or reset between jobs here:
  * loading.load_broadcast: DiT + merged CWM LoRA on every rank                                  -> once per process
  * optimizations.install(model, state): a forward pre-hook (state.begin) and wrapped block/attention forwards,
    all bound to ONE RuntimeState. Installing again would add a second hook and wrap the wrappers  -> once per process
  * optimizations.precompute_adaln: replaces every block's adaln_proj (and the final layer's) by a CachedAdaLN
    lookup table and drops the original projection weights (24.3 GiB per rank). The tables are a pure function of
    the per-step unique timesteps (build_timestep_rows(...).unique_timesteps -> time_embedder -> adaln_proj), which
    depend only on the step count, the fixed shifts and which row kinds exist (e.g. a fixed first frame adds t=1.0);
    not on text length, references, frames or seed. Each job recomputes that signature exactly as precompute_adaln
    does; equal signature -> the tables are reused (bit-identical to rebuilding them), otherwise they are rebuilt
    from the original projections, which are kept on the CPU for that purpose after the first build.
  * RuntimeState (step cursor, expected times, cache decisions/residuals, row-id cache)            -> reset per job
  * model._rotary_cache (keyed by layout, at most 2 entries)                                        -> cleared per job
  * video VAE: loaded once on rank 0 and kept (the DiT stays resident during decoding)
Collective safety: every step that can fail is run on all ranks and followed by an all-reduce of its status
(`guarded`), and rank-0-only steps broadcast their status (`on_rank0`), so all ranks abandon a failing job together
and the collectives stay matched. A failure that hits only some ranks INSIDE a collective section (e.g. an OOM on one
rank in the middle of a denoising step) cannot be recovered; NCCL then times out (30 min) as in the single-job scripts.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import traceback

sys.path.insert(0, '@CD_NODE@/cwm_h3_acceleration_v1/code')
import torch
import torch.distributed as dist
from common import setup, sync, finish  # noqa: F401  (re-exported for the entry scripts)
from optimizations import RuntimeState, install, precompute_adaln, rank, world


class JobFailed(Exception):
    """Raised on every rank when a step of the current job failed on at least one rank."""


def agree(ok: bool, device) -> bool:
    """True only if ok is True on every rank."""
    if world() == 1:
        return bool(ok)
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def broadcast(obj, device):
    """Rank 0's (picklable) object on every rank."""
    if world() == 1:
        return obj
    box = [obj if rank() == 0 else None]
    dist.broadcast_object_list(box, src=0, device=device)
    return box[0]


def guarded(device, fn, what):
    """Run fn on every rank; if it raised on any rank, raise JobFailed on all ranks (after the same collectives)."""
    error = None
    try:
        value = fn()
    except Exception:
        value, error = None, traceback.format_exc()
    if not agree(error is None, device):
        raise JobFailed(f'{what} failed' + (f' on rank {rank()}:\n{error}' if error else ' on another rank'))
    return value


def on_rank0(device, fn, what):
    """Run fn on rank 0 only (the other ranks wait); all ranks raise JobFailed if it raised."""
    error = None
    if rank() == 0:
        try:
            fn()
        except Exception:
            error = traceback.format_exc()
    error = broadcast(error, device)
    if error is not None:
        raise JobFailed(f'{what} failed on rank 0:\n{error}')


def quiet(fn, label):
    """Rank-0 bookkeeping (metrics, export copies) must never desynchronise the ranks: log and continue."""
    try:
        fn()
    except Exception:
        print(f'WARNING {label} failed:\n{traceback.format_exc()}', flush=True)


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Resident:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.key = None
        self.state = None
        self.gate = None
        self.details = None
        self.loaded_by = None
        self.originals = None      # original adaln_proj modules (on the CPU once the first tables exist)
        self.signature = None      # signature of the CachedAdaLN tables currently installed
        self.table_info = None
        self.builds = 0
        self.load_failed = False
        self.vae = None            # rank 0 only
        self.vae_path = None

    # ---- DiT ------------------------------------------------------------------------------------------------
    def check_key(self, key):
        if self.model is not None and key != self.key:
            raise ValueError(f'this job needs model {key} but the resident model is {self.key}; run it in a separate list')

    def load(self, key, loader, job_id):
        """loader() -> (model, gate, details) exactly as load_broadcast; then install() once."""
        model, gate, details = loader()
        model.eval().requires_grad_(False)
        self.model, self.key, self.gate, self.details, self.loaded_by = model, key, gate, details, job_id
        self.state = install(model, RuntimeState(20))
        return gate, details

    def _blocks(self):
        return [*self.model.blocks, self.model.final_layer]

    @torch.inference_mode()
    def adaln_signature(self, layout, tags, video_mask, video_shape, steps):
        """Per-step (model times, unique timesteps), computed exactly as optimizations.precompute_adaln does."""
        from musubi_tuner.minimax_h3.packing import build_timestep_rows, pack_video_rows
        from musubi_tuner.minimax_h3.sampling import build_shifted_schedule
        device = self.model.device
        schedule = build_shifted_schedule(steps, video_shift=12.0, audio_shift=3.0, device=device)
        vrows = self.model._target_denoise_rows(video_mask, torch.empty(video_shape, device=device), pack_video_rows, label='video')
        times, signature = [], []
        for i in range(steps):
            tv, ta = 1.0 - schedule.video[i].float(), 1.0 - schedule.audio[i].float()
            times.append((float(tv), float(ta)))
            rows = build_timestep_rows(layout, text_token_tags=tags, model_t_video=tv, model_t_audio=ta,
                                       visual_condition_clean=0.999, audio_condition_clean=1.0,
                                       target_video_denoise_rows=vrows, target_audio_denoise_rows=None)
            signature.append([float(tv), float(ta), [float(x) for x in rows.unique_timesteps.tolist()]])
        return [steps, signature], times

    def prepare_adaln(self, job_id, layout, tags, video_mask, video_shape, steps, fused, cache_threshold):
        """Make the CachedAdaLN tables valid for this job and reset the per-job RuntimeState."""
        signature, times = self.adaln_signature(layout, tags, video_mask, video_shape, steps)
        st = self.state
        if self.signature is not None and signature == self.signature:
            info = dict(self.table_info, tables='reused')
        else:
            self.signature = None  # only a completed build makes the installed tables valid again
            t = time.monotonic()
            blocks = self._blocks()
            if self.originals is None:
                self.originals = [b.adaln_proj for b in blocks]
            else:  # rebuild: restore the original projections (bit-identical weights) on the GPU first
                for block, original in zip(blocks, self.originals):
                    original.to(self.device)
                    block.adaln_proj = original
            st.steps = steps
            info = precompute_adaln(self.model, st, layout, tags, video_mask, video_shape)
            info['precompute_seconds'] = time.monotonic() - t
            t = time.monotonic()
            for original in self.originals:   # free the 24.3 GiB of projections on the GPU, keep them for a rebuild
                original.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
            info['originals_to_cpu_seconds'] = time.monotonic() - t
            if st.expected_times != times:
                raise RuntimeError('AdaLN schedule differs from the signature computation')
            self.builds += 1
            self.signature = signature
            self.table_info = dict(info, signature_sha256=sha256_json(signature), built_by_job=job_id)
            info = dict(self.table_info, tables='built', build_index=self.builds)
        # per-job runtime state (what a fresh RuntimeState + precompute_adaln would hold)
        st.steps = steps
        st.reset()
        st.row_cache.clear()
        st.expected_times = list(times)
        st.precomputed = True
        st.fused = fused
        st.cache_threshold = cache_threshold
        self.model._rotary_cache.clear()
        return info

    # ---- video VAE (rank 0) ---------------------------------------------------------------------------------
    def video_vae(self, path):
        """Returns (vae, loaded_now)."""
        path = str(path)
        if self.vae is not None and self.vae_path == path:
            return self.vae, False
        from musubi_tuner.minimax_h3.video_vae import load_video_vae
        self.vae = None
        gc.collect()
        self.vae = load_video_vae(path, device=self.device, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)
        self.vae_path = path
        return self.vae, True


def free_gib(device):
    """Smallest free GPU memory over all ranks (GiB), e.g. to see whether the GPU occupier came back between jobs."""
    free = torch.tensor([torch.cuda.mem_get_info(device)[0] / 2**30], device=device)
    if world() > 1:
        dist.all_reduce(free, op=dist.ReduceOp.MIN)
    return round(free.item(), 2)


def peak_memory(device):
    mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30], device=device)
    if world() > 1:
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    return mem[0].item(), mem[1].item()


def read_list(path):
    lines = []
    for line in open(path).read().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line)
    return lines


# ---- optional parallel VAE decode ----------------------------------------------------------------------------------
# MiniMaxH3VideoVAE.decode(latents) = affine de-normalisation -> _decode_video: fixed temporal clips, each decoded by
# _decode_clip = independent decoder calls on fixed spatial tiles + _stitch_tiles -> clips blended in order. The tile
# decoder calls are stateless, so they are spread over all ranks here; rank 0 stitches every clip with the original
# _stitch_tiles and then runs the ORIGINAL vae.decode with _decode_clip replaced by a replay of those clips (the replay
# checks that each call receives exactly the planned latent slice). Same kernels and shapes -> bit-identical pixels
# (verified with --verify-decode, which also runs the serial decode and requires torch.equal).

@torch.inference_mode()
def parallel_clips(vae, latents, device):
    """All ranks, same fp16 latents [1,24,T,H,W] as passed to vae.decode. Returns (slices, clips) on rank 0, (None, None) elsewhere."""
    if not vae.use_tiling:
        raise RuntimeError('parallel decode expects the tiled VAE')
    x = latents * vae.latents_std.view(1, -1, 1, 1, 1).to(latents) + vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
    if x.shape[2] == 1:
        slices = [x]
    else:  # clip plan of MiniMaxH3VideoVAE._decode_video
        chunk_tokens = vae.tokens_chunk_size
        padded_token_count = x.shape[2] + vae.token_drop
        pad_tokens = (-padded_token_count) % chunk_tokens
        num_chunks = (padded_token_count + pad_tokens) // chunk_tokens - int(vae.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += chunk_tokens
            num_chunks = 1
        if pad_tokens:
            x = torch.cat((x, x[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)
        slices = [x[:, :, i * chunk_tokens: i * chunk_tokens + chunk_tokens + vae.token_overlap] for i in range(num_chunks)]
    hs, hl, ho = vae._split_tiles(x.shape[-2] * vae.vae_ratio)   # tile plan of _decode_clip (same for every clip)
    ws, wl, wo = vae._split_tiles(x.shape[-1] * vae.vae_ratio)
    r = vae.vae_ratio
    work = [(c, i, j) for c in range(len(slices)) for i in range(len(hs)) for j in range(len(ws))]
    mine = {}
    for k in range(rank(), len(work), world()):
        c, i, j = work[k]
        mine[k] = vae.decoder(vae.post_quant_conv(slices[c][..., hs[i] // r:(hs[i] + hl[i]) // r, ws[j] // r:(ws[j] + wl[j]) // r]))
    meta = [None] * world()
    dist.all_gather_object(meta, {k: (list(v.shape), str(v.dtype).replace('torch.', '')) for k, v in mine.items()})
    if rank() != 0:
        for k in sorted(mine):
            dist.send(mine[k].contiguous(), dst=0)
        torch.cuda.synchronize()
        return None, None
    tiles = dict(mine)
    for src in range(1, world()):
        for k in sorted(meta[src]):
            shape, dtype = meta[src][k]
            buf = torch.empty(shape, dtype=getattr(torch, dtype), device=device)
            dist.recv(buf, src=src)
            tiles[k] = buf
    clips = [vae._stitch_tiles([[tiles[(c * len(hs) + i) * len(ws) + j] for j in range(len(ws))] for i in range(len(hs))], ho, wo)
             for c in range(len(slices))]
    return slices, clips


class ClipReplay:
    """Rank 0: inside the block, vae._decode_clip returns the precomputed clips in call order (inputs are checked)."""

    def __init__(self, vae, slices, clips):
        self.vae, self.items, self.used = vae, list(zip(slices, clips)), 0

    def __enter__(self):
        def replay(arg):
            if self.used >= len(self.items):
                raise RuntimeError('parallel decode: more clip calls than planned')
            planned, clip = self.items[self.used]
            if arg.shape != planned.shape or not torch.equal(arg, planned):
                raise RuntimeError(f'parallel decode: clip {self.used} input differs from the plan')
            self.used += 1
            return clip
        self.vae._decode_clip = replay  # instance attribute shadows the method
        return self

    def __exit__(self, kind, value, tb):
        del self.vae._decode_clip
        if kind is None and self.used != len(self.items):
            raise RuntimeError(f'parallel decode: {self.used} of {len(self.items)} clips used')
        return False
