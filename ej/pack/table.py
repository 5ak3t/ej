"""STATE TABLES for ejpack v1: the object graph (fitted state minus its vocabularies, encoder checkpoint minus its codes) through
ej's pickle-free codec (ej.safe_codec / ej.safe_decode), with the codec's flat tensor table re-encoded tensor by tensor into the
cheapest EXACT storage:
  'rows'   float32 matrix (dim >= 2) = per-row int8 + fp16 row scale (the int8 heads)        -> sections heads.q8 / heads.s16
  'blocks' float32 vector = int8 per block of 64 + fp16 block scale (the cross weights)         -> sections cross.q8 / cross.s16
           (+ heads.negz / cross.negz: a 1-bit plane over each tensor's zero codes marking the stored -0.0 values)
  'f16'    float32 tensor whose fp16 round trip is bit-identical                               -> rest.safetensors as F16
  'f32'    any other float32 tensor (kept as is: NOT approximated)                            -> rest.safetensors as F32
  'raw'    non-float32 tensors (bool / int / fp16)                                             -> rest.safetensors unchanged
Every int8 candidate is verified bit-for-bit (quant.py) before it is chosen. The skeleton (JSON, gzip with mtime 0) goes to
section rest.json.gz. Decoding instantiates only the allowlisted classes (ej.safe.ALLOWED_CLASSES; cls.__new__ + __dict__),
never pickle / eval / exec."""
import gzip
import json
import sys

import numpy as np
import torch

from .. import safe_codec as sc
from ..safe import ALLOWED_CLASSES
from ..safe_decode import Decoder
from . import bits as B
from . import quant as QN

MIN_Q8 = 64  # smaller float32 tensors are not tried as int8


def tensor_paths(node, path='', out=None):
    """{table name: first skeleton path that references it} (for reports)."""
    out = {} if out is None else out
    if isinstance(node, list):
        for i, v in enumerate(node):
            tensor_paths(v, f'{path}[{i}]', out)
    elif isinstance(node, dict):
        if 'T' in node:
            out.setdefault(node['T'], path)
        for k, v in node.items():
            if k == 'd' and isinstance(v, dict):
                for kk, vv in v.items():
                    tensor_paths(vv, f'{path}.{kk}', out)
            elif k == 'a' and isinstance(v, dict):
                for kk, vv in v.items():
                    tensor_paths(vv, f'{path}.{kk}', out)
            elif k in ('p', 'od') and isinstance(v, list):
                for kv in v:
                    tensor_paths(kv[1], f'{path}[{json.dumps(kv[0])[:40]}]', out)
            elif k in ('l', 's', 'fs'):
                tensor_paths(v, path, out)
    return out


def choose(t):
    """(kind, payload) of the cheapest exact storage of tensor t (see the module docstring)."""
    if t.dtype != torch.float32:
        return 'raw', t
    t = t.contiguous()
    if t.numel() >= MIN_Q8:
        r = QN.encode_rows(t) if t.dim() >= 2 else QN.encode_blocks(t)
        if r is not None:
            return ('rows' if t.dim() >= 2 else 'blocks'), r
    return ('f16', t.half()) if QN.fp16_exact(t) else ('f32', t)


def negz_plane(q, negz):
    """Packed 1-bit plane over the zero codes of q (flat order): 1 = the stored float is -0.0; b'' when there is none."""
    if not negz:
        return b''
    zero = torch.nonzero(q.reshape(-1) == 0).reshape(-1).numpy()
    return B.pack(np.isin(zero, np.asarray(negz)).astype(np.uint8), 1)


def negz_positions(q, section, off):
    """Flat positions of the -0.0 values from the plane of negz_plane at byte offset off."""
    zero = torch.nonzero(q.reshape(-1) == 0).reshape(-1).numpy()
    plane = B.unpack(section[off:off + B.nbytes(zero.size, 1)], 1, zero.size)
    return zero[plane == 1].tolist()


def _i16(t):
    return t.contiguous().view(torch.int16).numpy().tobytes()


def encode(obj):
    """(sections {name: bytes}, header descriptor, report rows) of obj."""
    from safetensors.torch import save
    enc = sc.Encoder()
    root = enc.encode(obj)
    bad = sorted(enc.classes - ALLOWED_CLASSES)
    if bad:
        raise sc.CodecError(f'classes outside the allowlist: {bad}')
    paths = tensor_paths(root)
    secs = {k: bytearray() for k in ('heads.q8', 'heads.s16', 'heads.negz', 'cross.q8', 'cross.s16', 'cross.negz')}
    rest, q8, cast, report = {}, {}, [], []
    for name, t in enc.tensors.items():
        kind, pl = choose(t)
        if kind in ('rows', 'blocks'):
            q, s, negz = pl
            pre = 'heads' if kind == 'rows' else 'cross'
            q8[name] = {'kind': kind, 'shape': list(t.shape), 'q_off': len(secs[pre + '.q8']), 'n': q.numel(),
                        's_off': len(secs[pre + '.s16']) // 2, 'ns': s.numel(), 'nz_off': len(secs[pre + '.negz']),
                        'n_negz': len(negz)}
            plane = negz_plane(q, negz)
            secs[pre + '.q8'] += q.numpy().tobytes()
            secs[pre + '.s16'] += _i16(s)
            secs[pre + '.negz'] += plane
            stored = q.numel() + 2 * s.numel() + len(plane)
        else:
            rest[name] = pl.contiguous()
            if kind == 'f16':
                cast.append(name)
            stored = pl.numel() * pl.element_size()
        report.append({'name': name, 'path': paths.get(name, '?'), 'kind': kind, 'dtype': str(t.dtype).split('.')[-1],
                       'shape': list(t.shape), 'orig_bytes': t.numel() * t.element_size(), 'stored_bytes': stored})
    doc = {'format': sc.FORMAT, 'classes': sorted(enc.classes), 'root': root}
    out = {k: bytes(v) for k, v in secs.items()}
    out['rest.safetensors'] = save(rest, metadata={'format': 'ejpack-v1-rest'}) if rest else b''
    out['rest.json.gz'] = gzip.compress(json.dumps(doc, allow_nan=False, separators=(',', ':'), ensure_ascii=False).encode(),
                                        9, mtime=0)
    return out, {'q8': q8, 'cast32': sorted(cast)}, report


def decode(sections, desc, allow=ALLOWED_CLASSES):
    """The object encoded by encode(), from sections (name -> bytes-like) and the header descriptor."""
    from safetensors.torch import load
    doc = json.loads(gzip.decompress(bytes(sections['rest.json.gz'])))
    if doc.get('format') != sc.FORMAT:
        raise sc.CodecError(f'skeleton format {doc.get("format")!r}')
    bad = sorted(set(doc.get('classes', [])) - set(allow))
    if bad:
        raise sc.CodecError(f'skeleton names classes outside the allowlist: {bad}')
    raw = bytes(sections['rest.safetensors'])
    tensors = dict(load(raw)) if raw else {}
    for name in desc['cast32']:
        tensors[name] = tensors[name].float()
    for name, d in desc['q8'].items():
        pre = 'heads' if d['kind'] == 'rows' else 'cross'
        q = torch.from_numpy(np.frombuffer(sections[pre + '.q8'], np.int8, d['n'], d['q_off']).copy())
        s = torch.from_numpy(np.frombuffer(sections[pre + '.s16'], np.int16, d['ns'], 2 * d['s_off']).copy()).view(torch.float16)
        negz = negz_positions(q, sections[pre + '.negz'], d['nz_off']) if d['n_negz'] else []
        if len(negz) != d['n_negz']:
            raise sc.CodecError(f'{name}: {len(negz)} signed zeros decoded, header says {d["n_negz"]}')
        fn = QN.decode_rows if d['kind'] == 'rows' else QN.decode_blocks
        tensors[name] = fn(q, s, negz, tuple(d['shape']))
    dec = Decoder(tensors, allow)
    out = dec.decode(doc['root'])
    if dec.filled:
        print(f'ejpack: nn.Module defaults filled: {sorted(set(dec.filled))}', file=sys.stderr)
    return out
