"""STREAMING DEQUANTISATION of the low-bit encoder (ej.load(..., low_memory=True)): "low-bit storage, fp32 compute" without
custom kernels. Each quantised encoder weight (72 Linears + word / position embeddings) is held ONLY as its b-bit codes + its fp16
per-group steps s and offsets z. Its fp32 weight is rebuilt just in time inside forward with EXACTLY the arithmetic of
student_lbq.dequant, ((q.float() - z.float()) * s.float()).reshape(shape): q - z is an exact small integer in fp32 and the one
rounding is the multiply by s, so the weight -- and therefore F.linear / the embedding rows -- is bit-identical to the
resident-fp32 path; it is dropped when forward returns.
  LAYOUT  the package stream (bits.py: code i at bits [i*b, (i+1)*b), little-endian) is re-laid at load as PER = 32 // b codes
          per int32 word (code k of a word at bits [k*b, (k+1)*b)), so one broadcast shift + mask unpacks a whole weight: 2-bit =
          16 codes / word = the package bytes as they are (2 bits / weight); 3-bit = 10 codes / word (3.2 bits / weight)
          instead of codes straddling bytes (about 3x slower to unpack).
  StreamLinear     F.linear(x, deq(), bias); bias stays an fp32 Parameter.
  StreamEmbedding  dequantises only the looked-up rows (a row = embedding_dim / GROUP whole groups and whole words, so its
                   values are the same elementwise results as in the full table; F.embedding is a row copy); else the table.
  LRU              optional cache of dequantised Linear weights (default 0 = off): trades memory for speed.
unpack_codes() is the torch twin of bits.unpack (checked against it in tests/test_pack.py)."""
import collections

import numpy as np
import torch
import torch.nn.functional as F

from . import bits as B


def unpack_codes(raw, bits, n):
    """uint8 tensor of the n b-bit codes in the packed uint8 tensor raw (inverse of bits.pack)."""
    if n % 8:
        return torch.from_numpy(B.unpack(raw.numpy().tobytes(), bits, n))
    w = raw.reshape(-1, bits).to(torch.int64)
    word = w[:, 0].clone()
    for k in range(1, bits):
        word |= w[:, k] << (8 * k)
    shifts = torch.arange(0, 8 * bits, bits, dtype=torch.int64)
    return ((word[:, None] >> shifts) & ((1 << bits) - 1)).to(torch.uint8).reshape(-1)[:n]


def to_words(q, bits):
    """int32 words holding the codes q (flat uint8) PER = 32 // bits to a word (zero-padded at the end)."""
    per = 32 // bits
    q = q.reshape(-1).to(torch.int64)
    q = torch.cat([q, q.new_zeros((-q.numel()) % per)]).reshape(-1, per)
    w = (q << torch.arange(0, per * bits, bits, dtype=torch.int64)).sum(1)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


def raw_to_words(raw, bits, n, chunk=40960):
    """to_words(unpack_codes(raw, bits, n), bits) without materialising all codes: b | 32 (2-bit) -> the stream bytes ARE the
    word layout (a copy); else chunk-wise (chunk = a multiple of 8 and of PER codes) to bound the load-time transient."""
    per = 32 // bits
    if per * bits == 32 and n % per == 0 and raw.numel() * 8 == n * bits:
        w = torch.empty(n // per, dtype=torch.int32)
        w.view(torch.uint8).copy_(raw.reshape(-1))
        return w
    step = chunk * bits // 8
    out = [to_words(unpack_codes(raw[i:i + step], bits, min(chunk, n - i * 8 // bits)), bits) for i in range(0, raw.numel(), step)]
    return torch.cat(out)


def from_words(words, bits, n):
    """int32 tensor of the first n codes held in words (inverse of to_words)."""
    sh = torch.arange(0, (32 // bits) * bits, bits, dtype=torch.int32)
    return (words[:, None] >> sh).bitwise_and_((1 << bits) - 1).reshape(-1)[:n]


def dequant(q, s, z, shape):
    """student_lbq.dequant on unpacked codes q (groups, GROUP) and fp16 s, z (groups, 1): same ops, same order (in place on the
    fresh float copy of q, which rounds exactly as the out-of-place expression)."""
    x = q.float()
    x -= z.float()
    x *= s.float()
    return x.reshape(shape)


class LRU:
    """Cache of dequantised Linear weights keyed by module id; size = max entries (0 = off). policy 'pin' (default) keeps the
    first `size` weights it sees and never evicts; 'lru' evicts the least recently used. The encoder reads its 72 Linears in the
    same cyclic order every batch, which defeats LRU (0 hits whenever size < 72); pinning gives size / 72 of the hits."""

    def __init__(self, size=0, policy='pin'):
        if policy not in ('pin', 'lru'):
            raise ValueError(f'policy must be pin or lru, got {policy!r}')
        self.size, self.policy, self.d, self.hits, self.misses = int(size), policy, collections.OrderedDict(), 0, 0

    def get(self, key, make):
        if key in self.d:
            self.hits += 1
            if self.policy == 'lru':
                self.d.move_to_end(key)
            return self.d[key]
        self.misses += 1
        w = make()
        if self.size > 0 and (len(self.d) < self.size or self.policy == 'lru'):
            self.d[key] = w
            while len(self.d) > self.size:
                self.d.popitem(last=False)
        return w

    def clear(self):
        self.d.clear()


class _Codes(torch.nn.Module):
    """Codes (int32 words) + fp16 steps / offsets of one weight, held as non-persistent buffers (not parameters: the model's
    first floating parameter, which transformers reads as the model dtype, stays an fp32 one)."""

    def __init__(self, raw, s, z, bits, shape, group):
        super().__init__()
        self.bits, self.wshape, self.group = int(bits), tuple(shape), int(group)
        self.n = int(np.prod(self.wshape))
        words = raw_to_words(raw, self.bits, self.n)
        for k, t in (('words', words), ('s', s.detach().clone()), ('z', z.detach().clone())):
            self.register_buffer(k, t, persistent=False)

    def nbytes(self):
        """Resident bytes of the stored form (codes + steps + offsets)."""
        return sum(t.untyped_storage().nbytes() for t in (self.words, self.s, self.z))

    def deq(self):
        """The full fp32 weight (bit-identical to student_lbq.dequant of the checkpoint entry)."""
        q = from_words(self.words, self.bits, self.n).reshape(-1, self.group)
        return dequant(q, self.s, self.z, self.wshape)


class StreamLinear(_Codes):
    """nn.Linear whose fp32 weight exists only during forward."""

    def __init__(self, raw, s, z, bits, shape, group, bias, lru=None):
        super().__init__(raw, s, z, bits, shape, group)
        self.out_features, self.in_features = self.wshape
        self.bias = bias
        self.lru = lru

    def forward(self, x):
        w = self.lru.get(id(self), self.deq) if self.lru is not None else self.deq()
        return F.linear(x, w, self.bias)


class StreamEmbedding(_Codes):
    """nn.Embedding (no padding_idx, as student_lbd._shell builds it) that dequantises only the rows it looks up."""

    def __init__(self, raw, s, z, bits, shape, group):
        super().__init__(raw, s, z, bits, shape, group)
        self.num_embeddings, self.embedding_dim = self.wshape
        per = 32 // self.bits
        self.by_row = self.embedding_dim % group == 0 and self.embedding_dim % per == 0
        self.wpr, self.gpr = self.embedding_dim // per, self.embedding_dim // group  # words / groups per row

    def rows(self, idx):
        """fp32 rows idx (1-D long) of the table."""
        if not self.by_row:
            return self.deq()[idx]
        words = self.words.reshape(self.num_embeddings, self.wpr)[idx].reshape(-1)
        q = from_words(words, self.bits, len(idx) * self.embedding_dim).reshape(-1, self.group)
        s = self.s.reshape(self.num_embeddings, self.gpr)[idx].reshape(-1, 1)
        z = self.z.reshape(self.num_embeddings, self.gpr)[idx].reshape(-1, 1)
        return dequant(q, s, z, (len(idx), self.embedding_dim))

    def forward(self, ids):
        uniq, inv = torch.unique(ids.reshape(-1), return_inverse=True)
        return self.rows(uniq)[inv].reshape(*ids.shape, self.embedding_dim)


def from_entry(c, raw, group, kind, bias=None, lru=None):
    """A stream module of checkpoint code entry c (its 's', 'z', 'bits', 'shape') over packed codes raw (uint8 tensor)."""
    args = (raw, c['s'], c['z'], c['bits'], c['shape'], group)
    return StreamLinear(*args, bias=bias, lru=lru) if kind == 'linear' else StreamEmbedding(*args)


def pack_q(q, bits):
    """Packed uint8 tensor of unpacked codes q (any shape, uint8) -- the enc.q<b> bytes of the weight."""
    return torch.frombuffer(bytearray(B.pack(q.numpy(), bits)), dtype=torch.uint8)

