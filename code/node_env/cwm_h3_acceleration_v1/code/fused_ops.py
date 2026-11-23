"""SM90-compatible Triton pointwise fusions preserving BF16 stage rounding."""
import torch
import triton
import triton.language as tl


@triton.jit
def _modulate(X, Shift, Scale, Rows, Y, C: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = p < N
    r, c = p//C, p % C
    row = tl.load(Rows+r, valid, 0)
    dtype = X.dtype.element_ty
    x = tl.load(X+p, valid, 0).to(tl.float32)
    scale = tl.load(Scale+row*S0+c*S1, valid, 0).to(tl.float32)
    shift = tl.load(Shift+row*S0+c*S1, valid, 0).to(tl.float32)
    factor = (1.0+scale).to(dtype).to(tl.float32)
    product = (x*factor).to(dtype).to(tl.float32)
    tl.store(Y+p, product+shift, valid)


@triton.jit
def _gate(X, Update, Gate, Rows, Y, C: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = p < N
    row = tl.load(Rows+p//C, valid, 0)
    dtype = X.dtype.element_ty
    x = tl.load(X+p, valid, 0).to(tl.float32)
    update = tl.load(Update+p, valid, 0).to(tl.float32)
    gate = tl.load(Gate+row*S0+p % C*S1, valid, 0).to(tl.float32)
    product = (update*gate).to(dtype).to(tl.float32)
    tl.store(Y+p, x+product, valid)


@triton.jit
def _swiglu(X, Y, C: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = p < N
    offset = p//C*(2*C)+p % C
    dtype = X.dtype.element_ty
    x = tl.load(X+offset, valid, 0).to(tl.float32)
    value = tl.load(X+offset+C, valid, 0).to(tl.float32)
    activated = (x/(1.0+tl.exp(-x))).to(dtype).to(tl.float32)
    tl.store(Y+p, activated*value, valid)


def scale_shift(x, shift, scale, rows):
    assert x.is_contiguous() and rows.is_contiguous() and shift.stride() == scale.stride()
    out = torch.empty_like(x)
    _modulate[(triton.cdiv(x.numel(), 1024),)](x, shift, scale, rows, out, x.shape[-1], x.numel(), *scale.stride(), 1024, enable_fp_fusion=False)
    return out


def gate_add(x, update, gate, rows):
    assert x.is_contiguous() and update.is_contiguous() and rows.is_contiguous()
    out = torch.empty_like(x)
    _gate[(triton.cdiv(x.numel(), 1024),)](x, update, gate, rows, out, x.shape[-1], x.numel(), *gate.stride(), 1024, enable_fp_fusion=False)
    return out


def swiglu(x):
    assert x.is_contiguous()
    out = x.new_empty((*x.shape[:-1], x.shape[-1]//2))
    _swiglu[(triton.cdiv(out.numel(), 1024),)](x, out, out.shape[-1], out.numel(), 1024, enable_fp_fusion=False)
    return out
