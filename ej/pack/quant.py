"""EXACT int8 / fp16 RECOVERY of already-quantised float32 tensors for ejpack v1.

The runtime's student_lb.compact() stores the int8 heads as float32 values v = round(w / s).clamp(-127, 127) * s with an
fp16-rounded scale s per row (matrices, rows = dim 0) or per block of 64 in flat order (the cross weights, zero-padded last
block). This module recovers the int8 integers q and the fp16 scales s from the stored floats, and accepts them only when the
dequantisation q.float() * s.float() -- the same fp32 multiply compact() did -- reproduces EVERY stored float bit-for-bit (int32
views equal; stronger than torch.equal, which treats -0.0 == 0.0). A -0.0 (round of a small negative value, times s) cannot be
held by an int8 0, so its flat positions are returned and restored after dequantisation. fp16 storage is accepted only for a
tensor whose float32 -> float16 -> float32 round trip is bit-identical. Nothing is ever approximated: a tensor that fails every
exact encoding stays float32."""
import torch

QB = 64  # the cross-weight block (student_lb.QB)


def bits_equal(a, b):
    """True when two float32 tensors have identical shapes and identical bit patterns."""
    return a.shape == b.shape and a.dtype == b.dtype == torch.float32 and torch.equal(a.contiguous().view(torch.int32),
                                                                                   b.contiguous().view(torch.int32))


def fp16_exact(t):
    """True for a float32 tensor whose float16 round trip is bit-identical."""
    return t.dtype == torch.float32 and bits_equal(t.half().float(), t)


def _neighbours(s):
    """The fp16 scales one ulp below and above s (bit pattern -1 / +1; s > 0)."""
    b = s.view(torch.int16)
    return [(b - 1).view(torch.float16), (b + 1).view(torch.float16)]


def _try(g, s):
    """(q float, dequantised) of groups g (rows, k) at fp16 scales s (rows,)."""
    sf = s.float()
    q = torch.round(g / torch.where(sf > 0, sf, torch.ones_like(sf))[:, None]).clamp(-127, 127)
    return q, q * sf[:, None]


def _row_ok(deq, g):
    return (deq.view(torch.int32) == g.view(torch.int32)) | ((deq == 0) & (g == 0))  # zeros: sign fixed separately


def recover_groups(g):
    """(q int8 (rows, k), s fp16 (rows,), negative-zero flat positions) with q * s == g bitwise, or None."""
    g = g.contiguous().float()
    s = (g.abs().amax(1) / 127).half()
    q, deq = _try(g, s)
    ok = _row_ok(deq, g).all(1)
    for cand in _neighbours(s):  # a stored row whose scale is one fp16 ulp off the naive estimate
        if bool(ok.all()):
            break
        q2, d2 = _try(g, cand)
        fix = ~ok & _row_ok(d2, g).all(1)
        q[fix], deq[fix], s[fix] = q2[fix], d2[fix], cand[fix]
        ok |= fix
    if not bool(ok.all()):
        return None
    negz = torch.nonzero((g == 0) & torch.signbit(g)).tolist()
    flat = [r * g.shape[1] + c for r, c in negz]
    return q.to(torch.int8), s, flat


def _set_negzero(t, flat):
    if flat:
        t.view(-1)[torch.as_tensor(flat, dtype=torch.long)] = -0.0
    return t


def encode_rows(w):
    """Per-row int8 of a float32 tensor with dim >= 2 (rows = dim 0): (q int8 flat, s fp16 (rows,), negzero) or None."""
    if w.dim() < 2 or w.dtype != torch.float32 or w.numel() == 0:
        return None
    g = w.reshape(w.shape[0], -1)
    r = recover_groups(g)
    if r is None:
        return None
    q, s, negz = r
    out = (q.reshape(-1), s, negz)
    return out if bits_equal(decode_rows(*out, tuple(w.shape)), w.contiguous()) else None


def decode_rows(q, s, negz, shape):
    """float32 tensor of shape from per-row int8 q (flat) and fp16 scales s (one per row)."""
    rows = shape[0]
    t = (q.reshape(rows, -1).float() * s.float()[:, None]).reshape(shape)
    return _set_negzero(t, negz)


def encode_blocks(w, block=QB):
    """int8 per block of `block` (flat order, zero-padded): (q int8 (n,), s fp16 (ceil(n/block),), negzero) or None."""
    if w.dtype != torch.float32 or w.numel() == 0:
        return None
    n = w.numel()
    pad = (-n) % block
    g = torch.cat([w.reshape(-1), torch.zeros(pad)]).reshape(-1, block)
    r = recover_groups(g)
    if r is None:
        return None
    q, s, negz = r
    out = (q.reshape(-1)[:n].clone(), s, [i for i in negz if i < n])
    return out if bits_equal(decode_blocks(*out, tuple(w.shape), block), w.contiguous()) else None


def decode_blocks(q, s, negz, shape, block=QB):
    """float32 tensor of shape from block int8 q (n,) and fp16 block scales s."""
    n = q.numel()
    qp = torch.cat([q.float(), torch.zeros((-n) % block)]).reshape(-1, block)
    t = (qp * s.float()[:, None]).reshape(-1)[:n].reshape(shape)
    return _set_negzero(t, negz)
