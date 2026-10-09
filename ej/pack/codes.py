"""LOW-BIT ENCODER CODES for ejpack v1: ck['codes'] of the low-bit encoder checkpoint as bit-packed sections.

Per quantised weight (74 in w23: 2-bit embeddings / feed-forward, 3-bit attention) the checkpoint holds q uint8 (groups, GROUP)
codes, s fp16 (groups, 1) steps and z fp16 (groups, 1) rounded offsets, dequantised by student_lbq.dequant as
((q - z) * s).reshape(shape). Sections: 'enc.q<b>' = the codes of every b-bit weight, each weight's stream bit-packed (bits.py)
and byte-aligned, in checkpoint order; 'enc.s' / 'enc.z' = all steps / offsets as raw fp16 in the same order. The header
descriptor per weight gives its name, bits, shape, group count and offsets. decode() rebuilds the exact checkpoint entries
(same dtypes, shapes, values, requires_grad flags and dict order)."""
import numpy as np
import torch

from . import bits as B


def encode(codes):
    """(sections {name: bytes}, descriptors [..]) of an ordered ck['codes'] dict."""
    streams, s_parts, z_parts, desc = {}, [], [], []
    n_groups = 0
    for name, c in codes.items():
        q, s, z, b = c['q'], c['s'], c['z'], int(c['bits'])
        if q.dtype != torch.uint8 or s.dtype != torch.float16 or z.dtype != torch.float16:
            raise ValueError(f'{name}: unexpected dtypes {q.dtype} {s.dtype} {z.dtype}')
        g = q.shape[0]
        if s.shape != (g, 1) or z.shape != (g, 1):
            raise ValueError(f'{name}: scale / offset shapes {tuple(s.shape)} {tuple(z.shape)} for {g} groups')
        key = f'enc.q{b}'
        buf = streams.setdefault(key, bytearray())
        desc.append({'name': name, 'bits': b, 'shape': list(c['shape']), 'qshape': list(q.shape), 'q_off': len(buf),
                     'g_off': n_groups, 'groups': g, 'grad': [bool(s.requires_grad), bool(z.requires_grad)]})
        buf += B.pack(q.numpy(), b)
        s_parts.append(s.detach().contiguous().view(torch.int16).numpy().tobytes())
        z_parts.append(z.detach().contiguous().view(torch.int16).numpy().tobytes())
        n_groups += g
        extra = set(c) - {'q', 's', 'z', 'bits', 'shape'}
        if extra:
            raise ValueError(f'{name}: unexpected code fields {sorted(extra)}')
    sections = {k: bytes(v) for k, v in sorted(streams.items())}
    sections['enc.s'], sections['enc.z'] = b''.join(s_parts), b''.join(z_parts)
    return sections, desc


def _fp16(buf, off, n):
    a = np.frombuffer(buf, dtype=np.int16, count=n, offset=2 * off).copy()
    return torch.from_numpy(a).view(torch.float16).reshape(n, 1)


def decode(sections, desc, unpack=True):
    """The ordered ck['codes'] dict rebuilt from sections (name -> bytes-like) and the descriptors. unpack=False (the
    runtime): each entry holds 'raw' = its packed codes (a uint8 tensor copy of its stream) instead of 'q'."""
    out = {}
    for d in desc:
        n = int(np.prod(d['qshape']))
        stream = sections[f"enc.q{d['bits']}"]
        buf = stream[d['q_off']:d['q_off'] + B.nbytes(n, d['bits'])]
        s, z = (_fp16(sections[k], d['g_off'], d['groups']).requires_grad_(g) for k, g in zip(('enc.s', 'enc.z'), d['grad']))
        e = {'s': s, 'z': z, 'bits': d['bits'], 'shape': tuple(d['shape'])}
        if unpack:
            e = {'q': torch.from_numpy(B.unpack(buf, d['bits'], n)).reshape(d['qshape']), **e}
        else:
            e['raw'] = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        out[d['name']] = e
    return out
