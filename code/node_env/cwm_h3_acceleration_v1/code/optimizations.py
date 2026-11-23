"""Inference-only optimizations for the pinned CWM/Musubi H3 model.

The original forward packing, reference conditions, retake masks, sampler and
weights stay in use. Ulysses shards the DiT residual stream, not the videos in a
batch. The approximate cache follows Sol-Engine H100's head-residual criterion.
"""
from __future__ import annotations

import math
import types
import time
import torch
import torch.distributed as dist
from torch import nn


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def error_metrics(actual, expected):
    a, b = actual.float(), expected.float()
    d = a - b
    return {"max_abs": d.abs().max().item(), "rmse": d.square().mean().sqrt().item(),
            "relative_l2": (d.norm() / b.norm().clamp_min(1e-12)).item(),
            "finite": bool(torch.isfinite(a).all())}


def shard_sequence(x, start, count):
    local = x[:, start:start + count].contiguous()
    if local.shape[1] < count:
        pad = local.new_zeros((local.shape[0], count - local.shape[1], *local.shape[2:]))
        local = torch.cat((local, pad), dim=1)
    return local


def local_segments(segments, start, count, valid):
    result = []
    for batch in segments:
        rows = [(max(a, start)-start, min(b, start+valid)-start, row)
                for a, b, row in batch if max(a, start) < min(b, start+valid)]
        if valid < count:
            rows.append((valid, count, 0))
        result.append(tuple(rows))
    return tuple(result)


def exchange_qkv(q, k, v):
    """[B, local-S, H, D] -> [B, global-S, H/world, D], BF16 only."""
    w = world()
    if w == 1:
        return q, k, v
    b, s, heads, d = q.shape
    if heads % w or q.dtype != torch.bfloat16:
        raise ValueError("Ulysses requires divisible heads and BF16 communication")
    h = heads // w
    send = torch.stack((q, k, v)).reshape(3, b, s, w, h, d).permute(3, 0, 1, 2, 4, 5).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send)
    merged = receive.permute(1, 2, 0, 3, 4, 5).reshape(3, b, w*s, h, d)
    return tuple(merged.unbind(0))


def exchange_output(out, local_count):
    w = world()
    if w == 1:
        return out
    b, length, heads, d = out.shape
    if out.dtype != torch.bfloat16:
        raise ValueError("attention output communication must remain BF16")
    if length < local_count*w:
        out = torch.cat((out, out.new_zeros((b, local_count*w-length, heads, d))), dim=1)
    send = out.reshape(b, w, local_count, heads, d).permute(1, 0, 2, 3, 4).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send)
    return receive.permute(1, 2, 0, 3, 4).reshape(b, local_count, w*heads, d)


def attention_forward(module, hidden, rotation, sequence_length):
    from musubi_tuner.minimax_h3.model import _apply_rope_split_half
    from musubi_tuner.wan.modules.attention import flash_attention
    b, s, _ = hidden.shape
    q, k, v = module.qkv_proj(hidden).split(module.inner_dim, dim=-1)
    q = module.q_norm(q.reshape(b, s, module.num_heads, module.head_dim))
    k = module.k_norm(k.reshape(b, s, module.num_heads, module.head_dim))
    v = v.reshape(b, s, module.num_heads, module.head_dim)
    if rotation is not None:
        q, k = _apply_rope_split_half(q, rotation), _apply_rope_split_half(k, rotation)
    q, k, v = exchange_qkv(q, k, v)
    # Remove padding from BOTH queries and keys; it must never change softmax.
    q, k, v = [x[:, :sequence_length].contiguous() for x in (q, k, v)]
    out = flash_attention([q, k, v], attn_mode="flash3", split_attn=False, dtype=torch.bfloat16)
    out = exchange_output(out, s)
    return module.out_proj(out.reshape(b, s, module.inner_dim))


class CachedAdaLN(nn.Module):
    def __init__(self, tables, state):
        super().__init__()
        self.state = state
        self.names = []
        for step, values in enumerate(tables):
            names = []
            for chunk, value in enumerate(values):
                name = f"step_{step}_chunk_{chunk}"
                self.register_buffer(name, value.contiguous(), persistent=False)
                names.append(name)
            self.names.append(names)

    def forward(self, timestep_embeddings):
        if self.state.step < 0 or self.state.step >= len(self.names):
            raise RuntimeError("AdaLN cursor is outside the precomputed schedule")
        return tuple(getattr(self, name) for name in self.names[self.state.step])


class RuntimeState:
    def __init__(self, steps=20):
        self.steps = steps
        self.fused = False
        self.cache_threshold = 0.0
        self.precomputed = False
        self.expected_times = None
        self.row_cache = {}
        self.reset()

    def reset(self):
        self.step = -1
        self.skip = False
        self.previous_head_residual = None
        self.tail_residual = None
        self.pending_head = None
        self.decisions = []

    def begin(self, module, args, kwargs):
        self.step += 1
        self.skip = False
        if self.step >= self.steps:
            raise RuntimeError("reset the runtime before another sampling request")
        if self.expected_times is not None:
            observed = (float(kwargs["model_t_video"]), float(kwargs["model_t_audio"]))
            if observed != self.expected_times[self.step]:
                raise RuntimeError("sampling schedule changed after AdaLN precompute")

    def gather(self, hidden):
        if world() == 1:
            return hidden
        if hidden.shape[0] != 1:
            raise ValueError("CWM runtime supports batch size one")
        local = hidden.squeeze(0).contiguous()
        gathered = local.new_empty((local.shape[0]*world(), local.shape[1]))
        dist.all_gather_into_tensor(gathered, local)
        return gathered[:self.sequence_length].unsqueeze(0)

    def after_head(self, before, after):
        if not self.cache_threshold:
            return
        residual = (after-before).detach()
        ratio = None
        if self.previous_head_residual is not None and self.tail_residual is not None:
            # Exclude artificial padding from the collective criterion.
            a, b = residual[:, :self.valid], self.previous_head_residual[:, :self.valid]
            totals = torch.stack(((a-b).abs().sum(dtype=torch.float32), b.abs().sum(dtype=torch.float32)))
            if world() > 1:
                dist.all_reduce(totals)
            ratio = (totals[0] / totals[1].clamp_min(1e-8)).item()
            self.skip = ratio <= self.cache_threshold
        if not self.skip:
            self.previous_head_residual = residual.clone()
            self.pending_head = after.detach().clone()
        self.decisions.append({"step": self.step+1, "reuse": self.skip, "relative_l1": ratio})

    def rows(self, hidden, segments):
        key = (segments, hidden.shape[1], hidden.device)
        if key not in self.row_cache:
            ids = torch.empty(hidden.shape[:2], dtype=torch.int32, device=hidden.device)
            for bi, batch in enumerate(segments):
                for a, b, row in batch:
                    ids[bi, a:b] = row
            self.row_cache[key] = ids
        return self.row_cache[key]


@torch.inference_mode()
def precompute_adaln(model, state, layout, tags, video_mask, video_shape, audio_mask=None, audio_shape=None):
    from musubi_tuner.minimax_h3.packing import build_timestep_rows, pack_video_rows, pack_audio_rows
    from musubi_tuner.minimax_h3.sampling import build_shifted_schedule
    device = model.device
    schedule = build_shifted_schedule(state.steps, video_shift=12.0, audio_shift=3.0, device=device)
    vrows = model._target_denoise_rows(video_mask, torch.empty(video_shape, device=device), pack_video_rows, label="video")
    arows = None if audio_mask is None else model._target_denoise_rows(
        audio_mask, torch.empty(audio_shape, device=device), pack_audio_rows, label="audio")
    embeddings, times = [], []
    for i in range(state.steps):
        tv, ta = 1.0-schedule.video[i].float(), 1.0-schedule.audio[i].float()
        times.append((float(tv), float(ta)))
        rows = build_timestep_rows(layout, text_token_tags=tags, model_t_video=tv, model_t_audio=ta,
                                   visual_condition_clean=0.999, audio_condition_clean=1.0,
                                   target_video_denoise_rows=vrows, target_audio_denoise_rows=arows)
        embeddings.append(model.time_embedder(rows.unique_timesteps.to(device)).to(model.dtype))
    released, cached = 0, 0
    checks = []
    for i, block in enumerate([*model.blocks, model.final_layer]):
        original = block.adaln_proj
        tables = [tuple(x.clone() for x in original(embedding)) for embedding in embeddings]
        released += sum(p.numel()*p.element_size() for p in original.parameters())
        replacement = CachedAdaLN(tables, state)
        # Repeat a live projection independently at both ends of the schedule.
        for step in (0, state.steps-1):
            state.step = step
            live, saved = original(embeddings[step]), replacement(embeddings[step])
            err = max((a-b).abs().max().item() for a, b in zip(live, saved))
            if err != 0.0:
                raise RuntimeError(f"AdaLN lookup differs from live projection: block {i}, error {err}")
        checks.append(i)
        cached += sum(b.numel()*b.element_size() for b in replacement.buffers())
        block.adaln_proj = replacement
        del original, tables, replacement, live, saved
    state.expected_times = times
    state.precomputed = True
    state.reset()
    torch.cuda.empty_cache()
    return {"released_weight_gib": released/2**30, "table_gib": cached/2**30,
            "checked_modules": len(checks), "lookup_max_abs_error": 0.0}


def install(model, state):
    """Instance-scoped wrappers. Original module classes/files are untouched."""
    model.register_forward_pre_hook(state.begin, with_kwargs=True)
    n = len(model.blocks)
    for index, block in enumerate(model.blocks):
        original = block.forward

        def wrapped(self, hidden, temb, segments, rotation, _index=index, _original=original):
            if _index == 0:
                state.sequence_length = hidden.shape[1]
                state.local_count = math.ceil(state.sequence_length/world())
                state.start = rank()*state.local_count
                state.valid = max(0, min(state.local_count, state.sequence_length-state.start))
                if world() > 1:
                    hidden = shard_sequence(hidden, state.start, state.local_count)
            if world() > 1:
                rotation = shard_sequence(rotation, state.start, state.local_count)
                segments = local_segments(segments, state.start, state.local_count, state.valid)
            if _index > 0 and state.skip:
                output = hidden
                if _index == n-1:
                    output = output + state.tail_residual
            else:
                if state.fused:
                    from fused_ops import scale_shift, gate_add, swiglu
                    shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln_proj(temb)
                    ids = state.rows(hidden, segments)
                    normalized = scale_shift(self.norm1(hidden), shift1, scale1, ids)
                    output = gate_add(hidden, self.attn(normalized, rotation), gate1, ids)
                    normalized = scale_shift(self.norm2(output), shift2, scale2, ids)
                    fc = self.mlp.fc1(normalized)
                    update = self.mlp.fc2(swiglu(fc))
                    output = gate_add(output, update, gate2, ids)
                else:
                    output = _original(hidden, temb, segments, rotation)
                if _index == 0:
                    state.after_head(hidden, output)
                if _index == n-1 and state.cache_threshold:
                    state.tail_residual = (output-state.pending_head).detach()
                    state.pending_head = None
            if _index == n-1:
                output = state.gather(output)
            return output

        block.forward = types.MethodType(wrapped, block)
        if world() > 1:
            def parallel_attention(self, hidden, rotation=None):
                return attention_forward(self, hidden, rotation, state.sequence_length)
            block.attn.forward = types.MethodType(parallel_attention, block.attn)
    return state
