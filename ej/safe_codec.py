"""ej -- pickle-free codec behind ej.safe: any object graph <-> (JSON skeleton, flat tensor table).

The tensor table is written with safetensors; the skeleton is plain JSON. Decoding runs no pickle, eval or exec: the only names it
resolves are classes from an explicit allowlist (instantiated with cls.__new__(cls) + __dict__ restore, never __init__ or
__setstate__) and defaultdict factories from the fixed CALLABLES table.

Skeleton encoding (a JSON value; objects carry exactly one tag key, plus '#' = label when the object is shared or cyclic):
  None / bool / int / str / finite float -> itself         tuple -> JSON array            {'@': n} -> the object labelled n
  {'f': hex}   non-finite float (IEEE-754 big-endian bits) {'b': base64} bytes            {'l': [..]} list
  {'d': {str: v}} dict with str keys  {'p': [[k, v]..]} dict with other keys  {'od': pairs} OrderedDict
  {'dd': factory name, 'p': pairs} defaultdict              {'s': [..]} set               {'fs': [..]} frozenset
  {'T': name, 'v'?: [offset, size, stride], 'P'?: 1, 'g'?: 1} tensor ('v': view of a shared storage; 'P': Parameter;
      'g': requires_grad)                                   {'N': name, 'dt': dtype, 'sh': shape} numpy array (raw bytes)
  {'ng': dtype, 'b': base64} numpy scalar                   {'dtype': name} torch.dtype    {'dev': str} torch.device
  {'tv': str} torch TorchVersion                            {'o': 'module.QualName', 'a': {attr: v}} object / nn.Module"""
import base64
import collections
import functools
import math
import struct
import types

import numpy as np
import torch
from torch.torch_version import TorchVersion

FORMAT = 'rev-safe-v1'
CALLABLES = {'builtins.list': list, 'builtins.dict': dict, 'builtins.set': set, 'builtins.int': int,
             'builtins.float': float, 'builtins.str': str}  # the only callables ever stored (defaultdict factories), by name
SAFE_DTYPES = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16,
               torch.float32, torch.float64}
_NOT_DATA = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.GeneratorType, types.ModuleType,
             types.CodeType, types.FrameType, functools.partial, type)
_BUILTIN_DATA = (str, bytes, bytearray, int, float, complex, tuple, list, dict, set, frozenset)
_IDENTITY = (dict, list, set, torch.Tensor, np.ndarray)  # mutable objects whose sharing / cycles are preserved


class CodecError(ValueError):
    """An object with no pickle-free encoding (export) or a skeleton the decoder refuses (load)."""


def class_name(t):
    """'module.QualName' of a class."""
    return f'{t.__module__}.{t.__qualname__}'


def _b64(b):
    return base64.b64encode(b).decode('ascii')


def _plain(o):
    """True for an nn.Module or an object rebuilt by default pickling (own __dict__, no slots, no custom reduce/getstate)."""
    t = type(o)
    if isinstance(o, _NOT_DATA + _BUILTIN_DATA) or t.__module__ in ('builtins', '__main__'):
        return False  # subclasses of builtin data types (str, tuple, dict, ...) would lose their value: not "plain"
    if isinstance(o, torch.nn.Module):
        return True
    return (hasattr(o, '__dict__') and not hasattr(t, '__slots__') and t.__reduce_ex__ is object.__reduce_ex__
            and t.__reduce__ is object.__reduce__ and getattr(t, '__getstate__', object.__getstate__) is object.__getstate__)


def _identity(o):
    return isinstance(o, _IDENTITY) or _plain(o)


def _storage_key(t):
    s = t.untyped_storage()
    return (str(t.device), s.data_ptr()) if s.nbytes() else None


class Encoder:
    """Two passes over one object graph: count shared objects / storages, then emit the skeleton and fill self.tensors."""

    def __init__(self):
        self.tensors, self.classes, self.problems = {}, set(), []
        self.seen, self.shared, self.labels, self.groups, self.bases = {}, set(), {}, {}, {}

    def encode(self, obj):
        """The JSON skeleton of obj; raises CodecError listing every unsupported part."""
        self._count(obj)
        root = self._enc(obj, 'root')
        if self.problems:
            raise CodecError(f'{len(self.problems)} unsupported part(s), no pickle-free encoding:\n  '
                             + '\n  '.join(self.problems[:50]))
        return root

    def _count(self, o):
        stack = [o]
        while stack:
            o = stack.pop()
            if type(o) in (tuple, frozenset):
                stack.extend(o)
                continue
            if not _identity(o):
                continue
            if id(o) in self.seen:
                self.shared.add(id(o))
                continue
            self.seen[id(o)] = o  # keeps o alive, so ids stay unique during the walk
            if isinstance(o, torch.Tensor):
                k = _storage_key(o)
                if k is not None:
                    self.groups.setdefault(k, set()).add(id(o))
            elif isinstance(o, dict):
                for kv in o.items():
                    stack.extend(kv)
            elif isinstance(o, (list, set)):
                stack.extend(o)
            elif not isinstance(o, np.ndarray):
                stack.extend(vars(o).values())

    def _bad(self, o, where, why='unsupported type'):
        self.problems.append(f'{where}: {class_name(type(o))} ({why})')

    def _enc(self, o, where):
        t = type(o)
        if o is None or t in (bool, int, str):
            return o
        if t is float:
            return o if math.isfinite(o) else {'f': struct.pack('>d', o).hex()}
        if t is tuple:
            return [self._enc(v, f'{where}[{i}]') for i, v in enumerate(o)]
        if t is bytes:
            return {'b': _b64(o)}
        if t is frozenset:
            return {'fs': [self._enc(v, where + '{}') for v in _ordered(o)]}
        if t is torch.dtype:
            return {'dtype': str(o).split('.')[-1]}
        if t is torch.device:
            return {'dev': str(o)}
        if t is TorchVersion:
            return {'tv': str(o)}
        if isinstance(o, np.generic):
            if o.dtype.hasobject or o.dtype.fields:
                return self._bad(o, where, f'numpy dtype {o.dtype}')
            return {'ng': o.dtype.str, 'b': _b64(o.tobytes())}
        if not _identity(o):
            return self._bad(o, where)
        if id(o) in self.labels:
            return {'@': self.labels[id(o)]}
        if id(o) in self.shared:
            self.labels[id(o)] = len(self.labels)  # labelled before recursing: cycles resolve to the same object
        out = self._container(o, t, where)
        if out is not None and id(o) in self.shared:
            out['#'] = self.labels[id(o)]
        return out

    def _pairs(self, o, where):
        return [[self._enc(k, f'{where}.key'), self._enc(v, f'{where}[{k!r}]')] for k, v in o.items()]

    def _container(self, o, t, where):
        if isinstance(o, torch.Tensor):
            return self._tensor(o, where)
        if t is np.ndarray:
            if o.dtype.hasobject or o.dtype.fields:
                return self._bad(o, where, f'numpy dtype {o.dtype}')
            name = f'n{len(self.tensors)}'
            self.tensors[name] = torch.from_numpy(np.frombuffer(np.ascontiguousarray(o).tobytes(), np.uint8).copy())
            return {'N': name, 'dt': o.dtype.str, 'sh': list(o.shape)}
        if t is dict:
            if all(type(k) is str for k in o):
                return {'d': {k: self._enc(v, f'{where}[{k!r}]') for k, v in o.items()}}
            return {'p': self._pairs(o, where)}
        if t is collections.OrderedDict:
            return {'od': self._pairs(o, where)}
        if t is collections.defaultdict:
            fac = o.default_factory
            name = None if fac is None else next((n for n, f in CALLABLES.items() if f is fac), False)
            if name is False:
                return self._bad(fac, f'{where}.default_factory', 'callable outside CALLABLES')
            return {'dd': name, 'p': self._pairs(o, where)}
        if t is list:
            return {'l': [self._enc(v, f'{where}[{i}]') for i, v in enumerate(o)]}
        if t is set:
            return {'s': [self._enc(v, where + '{}') for v in _ordered(o)]}
        if _plain(o):
            attrs = vars(o)
            if not all(type(k) is str for k in attrs):
                return self._bad(o, where, 'non-str attribute name')
            self.classes.add(class_name(t))
            return {'o': class_name(t), 'a': {k: self._enc(v, f'{where}.{k}') for k, v in attrs.items()}}
        return self._bad(o, where)

    def _tensor(self, o, where):
        if o.layout != torch.strided or o.is_quantized or o.dtype not in SAFE_DTYPES:
            return self._bad(o, where, f'tensor {o.dtype} {o.layout}')
        t = o.detach()
        k = _storage_key(o)
        out = {'P': 1} if isinstance(o, torch.nn.Parameter) else {}
        if o.requires_grad and o.is_leaf:
            out['g'] = 1
        if k is not None and len(self.groups.get(k, ())) > 1:  # several tensor objects view one storage: keep the aliasing
            if k not in self.bases:
                self.bases[k] = (f'b{len(self.tensors)}', t.dtype)
                n = t.untyped_storage().nbytes() // t.element_size()
                self.tensors[self.bases[k][0]] = torch.empty(0, dtype=t.dtype, device=t.device).set_(
                    t.untyped_storage(), 0, (n,), (1,)).cpu()
            name, dt = self.bases[k]
            if dt != t.dtype:
                return self._bad(o, where, 'storage shared by tensors of different dtypes')
            out.update(T=name, v=[t.storage_offset(), list(t.shape), list(t.stride())])
            return out
        t = t.cpu()
        whole = t.is_contiguous() and t.storage_offset() == 0 and t.untyped_storage().nbytes() == t.numel() * t.element_size()
        name = f't{len(self.tensors)}'
        self.tensors[name] = t if whole else t.contiguous().clone()
        out['T'] = name
        return out


def _ordered(s):
    """Set elements in a deterministic order when they are mutually comparable, else in iteration order."""
    try:
        return sorted(s)
    except TypeError:
        return list(s)
