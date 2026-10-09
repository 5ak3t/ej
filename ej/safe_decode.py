"""ej -- decoder half of the pickle-free codec (format: ej/safe_codec.py docstring).

No pickle, eval or exec: JSON values are mapped back by tag; the only names resolved are classes in the caller's allowlist
(module imported only after the allowlist check, instantiated with cls.__new__(cls), then __dict__ restored; __init__ and
__setstate__ are never called) and defaultdict factories from safe_codec.CALLABLES."""
import base64
import collections
import copy
import importlib
import struct

import numpy as np
import torch
from torch.torch_version import TorchVersion

from .safe_codec import CALLABLES, SAFE_DTYPES, CodecError, class_name

_MODULE_DEFAULTS = {}  # attribute defaults of a bare nn.Module (filled once; see Decoder._obj)


def resolve(name, allow):
    """The class named 'module.QualName', only when name is in allow; refuses anything else."""
    if name not in allow:
        raise CodecError(f'class {name!r} is not in the allowlist; refusing to instantiate it')
    parts = name.split('.')
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p, None)
        if isinstance(obj, type) and class_name(obj) == name:
            return obj
    raise CodecError(f'allowlisted class {name!r} cannot be resolved (is the ej runtime on sys.path?)')


def _np_dtype(s):
    dt = np.dtype(str(s))
    if dt.hasobject or dt.fields:
        raise CodecError(f'numpy dtype {s!r} refused')
    return dt


class Decoder:
    """Rebuilds an object from its skeleton and the loaded tensor table {name: tensor}."""

    def __init__(self, tensors, allow):
        self.tensors, self.allow, self.labels, self.filled = tensors, frozenset(allow), {}, []
        self.tags = {'@': self._ref, 'f': self._float, 'b': self._bytes, 'fs': self._frozenset, 'dtype': self._dtype,
                     'dev': lambda j: torch.device(str(j['dev'])), 'tv': lambda j: TorchVersion(str(j['tv'])),
                     'ng': self._np_scalar, 'N': self._np_array, 'T': self._tensor, 'd': self._dict, 'p': self._dict,
                     'od': self._dict, 'dd': self._dict, 'l': self._list, 's': self._set, 'o': self._obj}

    def decode(self, j):
        """The Python value of skeleton node j."""
        t = type(j)
        if j is None or t in (bool, int, float, str):
            return j
        if t is list:
            return tuple(self.decode(v) for v in j)
        if t is not dict:
            raise CodecError(f'unexpected JSON node {t.__name__}')
        tag = next((k for k in ('dd', 'ng') if k in j), None) or next((k for k in j if k in self.tags), None)
        if tag is None:
            raise CodecError(f'unknown skeleton node with keys {sorted(j)[:5]}')
        return self.tags[tag](j)

    def _keep(self, j, out):
        if '#' in j:
            self.labels[int(j['#'])] = out
        return out

    def _ref(self, j):
        if j['@'] not in self.labels:
            raise CodecError(f'reference to unknown label {j["@"]}')
        return self.labels[j['@']]

    def _float(self, j):
        return struct.unpack('>d', bytes.fromhex(j['f']))[0]

    def _bytes(self, j):
        return base64.b64decode(j['b'], validate=True)

    def _frozenset(self, j):
        return frozenset(self.decode(v) for v in j['fs'])

    def _dtype(self, j):
        dt = getattr(torch, str(j['dtype']), None)
        if not isinstance(dt, torch.dtype):
            raise CodecError(f'unknown torch dtype {j["dtype"]!r}')
        return dt

    def _np_scalar(self, j):
        return np.frombuffer(self._bytes(j), dtype=_np_dtype(j['ng']))[0]

    def _np_array(self, j):
        flat = self.tensors[j['N']].numpy()
        return self._keep(j, flat.view(_np_dtype(j['dt'])).reshape([int(n) for n in j['sh']]).copy())

    def _tensor(self, j):
        t = self.tensors[j['T']]
        if t.dtype not in SAFE_DTYPES:
            raise CodecError(f'tensor dtype {t.dtype} refused')
        if 'v' in j:
            off, size, stride = j['v']
            t = torch.as_strided(t, [int(n) for n in size], [int(n) for n in stride], int(off))
        if j.get('P'):
            t = torch.nn.Parameter(t, requires_grad=bool(j.get('g')))
        elif j.get('g'):
            t = t.requires_grad_(True)
        return self._keep(j, t)

    def _dict(self, j):
        kind = next(k for k in ('dd', 'od', 'd', 'p') if k in j)
        if kind == 'dd':
            if j['dd'] is not None and j['dd'] not in CALLABLES:
                raise CodecError(f'defaultdict factory {j["dd"]!r} is not in CALLABLES')
            out = collections.defaultdict(CALLABLES[j['dd']] if j['dd'] is not None else None)
        else:
            out = collections.OrderedDict() if kind == 'od' else {}
        self._keep(j, out)
        if kind == 'd':
            for k, v in j['d'].items():
                out[k] = self.decode(v)
        else:
            for k, v in j['p' if kind == 'dd' else kind]:
                out[self.decode(k)] = self.decode(v)
        return out

    def _list(self, j):
        out = self._keep(j, [])
        out.extend(self.decode(v) for v in j['l'])
        return out

    def _set(self, j):
        out = self._keep(j, set())
        out.update(self.decode(v) for v in j['s'])
        return out

    def _obj(self, j):
        cls = resolve(j['o'], self.allow)
        out = self._keep(j, cls.__new__(cls))
        attrs = {str(k): self.decode(v) for k, v in j['a'].items()}
        out.__dict__.update(attrs)
        if isinstance(out, torch.nn.Module):  # what Module.__setstate__ adds to old checkpoints, from torch's own defaults
            if not _MODULE_DEFAULTS:
                _MODULE_DEFAULTS.update(vars(torch.nn.Module()))
            for k, v in _MODULE_DEFAULTS.items():
                if k not in out.__dict__:
                    out.__dict__[k] = copy.deepcopy(v)
                    self.filled.append(f'{j["o"]}.{k}')
        return out
