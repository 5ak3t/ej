"""INT8 HEADS ON USE: the head matrices of a compacted state (student_lb.compact: rich / attn / dd / dn, every nn.Linear weight
= round(w / s).clamp(-127, 127) * s with an fp16 row scale) are decoded by the package as fp32. Here each such nn.Linear is
replaced by Int8Linear, which keeps the int8 codes + fp16 row scales (+ the flat positions of stored -0.0 values, which an int8
0 cannot hold) and rebuilds the fp32 weight inside forward with the dequantisation quant.py verified bit-for-bit at pack time
(q.float() * s.float()[:, None], then the -0.0 positions). A module is swapped ONLY when quant.encode_rows recovers an exact
form AND the rebuilt weight is bit-identical to the one it replaces (checked at load); anything else stays fp32. The heads call
their Linears through forward, so F.linear sees the same fp32 values."""
import torch
import torch.nn.functional as F

from . import quant as QN

KEYS = ('rich', 'attn', 'dd', 'dn')


class Int8Linear(torch.nn.Module):
    """nn.Linear with an int8 weight (per-row fp16 scale) dequantised on use; bias kept as is."""

    def __init__(self, lin, q, s, negz):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.register_buffer('q', q.reshape(lin.out_features, lin.in_features).contiguous(), persistent=False)
        self.register_buffer('s', s.contiguous(), persistent=False)
        self.register_buffer('negz', torch.as_tensor(negz, dtype=torch.long), persistent=False)
        self.bias = lin.bias

    def deq(self):
        """The fp32 weight (= quant.decode_rows)."""
        t = self.q.float() * self.s.float()[:, None]
        if self.negz.numel():
            t.view(-1)[self.negz] = -0.0
        return t

    def forward(self, x):
        return F.linear(x, self.deq(), self.bias)


def _swap(parent):
    """Replace every exact nn.Linear below parent; returns (swapped, kept fp32) counts."""
    done = kept = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, torch.nn.Linear):
            r = QN.encode_rows(child.weight.detach())
            new = Int8Linear(child, *r) if r is not None else None
            if new is not None and QN.bits_equal(new.deq(), child.weight.detach().contiguous()):
                setattr(parent, name, new)
                done += 1
            else:
                kept += 1
        else:
            d, k = _swap(child)
            done, kept = done + d, kept + k
    return done, kept


def install(state, keys=KEYS):
    """Swap the exact int8 head Linears of state in place; returns {key: (swapped, kept fp32)}."""
    out = {}
    for k in keys:
        m = state.get(k)
        if isinstance(m, torch.nn.Module):
            out[k] = _swap(m)
    return out
