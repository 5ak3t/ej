"""Fitted-state persistence: a portable encoding of the fitted state (tensors detached to CPU, numpy arrays as raw-bytes
records, unencodable parts replaced by markers), save/load of state files keyed by a sha256 over the code and the training
pool, and the fit-time lookup wrapper. ej uses decode() to finish a state read from safetensors."""
import collections
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import types

import numpy as np
import torch

VERSION = 'edge-state-v1'
HERE = os.path.dirname(os.path.abspath(__file__))
_ATOMS = (type(None), bool, int, float, complex, str, bytes, range, slice, type(Ellipsis))


def ckpt_root():
    return os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))


def pool_path():
    return os.environ.get('EDGE_DATA', os.path.expanduser('~/.cache/ej/data')) + '/train/pool.jsonl'


def sources(here=HERE):
    """Basenames of the hashed source files, in key order."""
    return sorted(f for f in os.listdir(here) if f.endswith('.py') and (f.startswith('student') or f.startswith('train_')))


def state_key(pool=None, here=HERE):
    """64-hex key over VERSION, the source files and the pool bytes (see the module docstring)."""
    h = hashlib.sha256(VERSION.encode() + b'\x00')
    for f in sources(here):
        h.update(f.encode() + b'\x00')
        with open(os.path.join(here, f), 'rb') as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    h.update(b'pool\x00')
    with open(pool or pool_path(), 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def path_for(key, root=None):
    return f'{root or ckpt_root()}/state-{key[:16]}.pt'


class NP:
    """A numpy array or scalar as raw bytes (dtype string, shape, buffer); object arrays keep their encoded elements."""

    def __init__(self, a, items=None):
        self.scalar = isinstance(a, np.generic)
        a = np.asarray(a)
        self.dtype, self.shape = a.dtype.str, a.shape
        self.items, self.data = items, None if items is not None else np.ascontiguousarray(a).tobytes()

    def value(self, items=None):
        if self.items is not None:
            out = np.empty(len(self.items), dtype=object)
            out[:] = items
            a = out.reshape(self.shape)
        else:
            a = np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()
        return a[()] if self.scalar else a


class Dropped:
    """Placeholder for a state part that could not be serialised (path, type, repr)."""

    def __init__(self, where, obj):
        self.where, self.type, self.repr = where, f'{type(obj).__module__}.{type(obj).__qualname__}', repr(obj)[:200]

    def __repr__(self):
        return f'<Dropped {self.where}: {self.type}>'


def _picklable(o):
    try:
        pickle.dumps(o, protocol=4)
        return True
    except Exception:  # noqa: BLE001 -- any pickling failure means "not serialisable"
        return False


def _plain(o):
    """A Python object rebuilt from its __dict__ (default pickling, no slots) -- also nn.Module."""
    t = type(o)
    if isinstance(o, torch.nn.Module):
        return True
    return (hasattr(o, '__dict__') and not hasattr(t, '__slots__') and t.__reduce_ex__ is object.__reduce_ex__
            and t.__reduce__ is object.__reduce__ and getattr(t, '__getstate__', object.__getstate__) is object.__getstate__ and t.__module__ not in ('builtins',)
            and not isinstance(o, (types.ModuleType, types.FunctionType, type)))


class _Enc:
    def __init__(self):
        self.memo, self.dropped, self.notes = {}, [], []

    def __call__(self, o, where='state'):
        if isinstance(o, _ATOMS):
            return o
        k = id(o)
        if k in self.memo:
            return self.memo[k][1]
        out = self._enc(o, where)
        self.memo[k] = (o, out)
        return out

    def _keep(self, o, out):  # register before recursing: shared / cyclic references resolve to the same copy
        self.memo[id(o)] = (o, out)
        return out

    def _enc(self, o, where):
        if isinstance(o, torch.Tensor):
            t = o
            if t.requires_grad and not t.is_leaf:
                t = t.detach(); self.notes.append(f'{where}: non-leaf tensor detached')
            if t.device.type != 'cpu':
                t = torch.nn.Parameter(t.detach().cpu(), t.requires_grad) if isinstance(t, torch.nn.Parameter) else t.detach().cpu()
            return t
        if isinstance(o, (np.ndarray, np.generic)):
            if o.dtype.hasobject:
                return NP(o, [self(x, f'{where}[{i}]') for i, x in enumerate(np.asarray(o).ravel().tolist())])
            return NP(o)
        t = type(o)
        if t in (dict, collections.OrderedDict, collections.defaultdict):
            fac = getattr(o, 'default_factory', None)
            out = t() if t is not collections.defaultdict else collections.defaultdict(fac) if _picklable(fac) else {}
            if type(out) is not t:
                self.notes.append(f'{where}: defaultdict with unpicklable factory -> dict')
            self._keep(o, out)
            for kk, v in o.items():
                out[self(kk, f'{where}.key')] = self(v, f'{where}[{kk!r}]')
            return out
        if t is list:
            out = self._keep(o, [])
            out.extend(self(v, f'{where}[{i}]') for i, v in enumerate(o))
            return out
        if t in (tuple, set, frozenset):
            return t(self(v, f'{where}[{i}]') for i, v in enumerate(o))
        if isinstance(o, tuple) and hasattr(t, '_fields'):  # namedtuple
            return t(*(self(v, f'{where}.{f}') for f, v in zip(t._fields, o)))
        if _plain(o):
            out = self._keep(o, t.__new__(t))
            out.__dict__.update({kk: self(v, f'{where}.{kk}') for kk, v in vars(o).items()})
            return out
        if _picklable(o):
            return o
        d = Dropped(where, o)
        self.dropped.append(repr(d))
        return d


def encode(state):
    """(portable copy of state, dropped parts, conversion notes); the input object is not modified."""
    e = _Enc()
    out = e(state)
    return out, e.dropped, e.notes


def decode(o, memo=None):
    """Inverse of encode on a freshly loaded object (numpy records restored in place; containers mutated in place)."""
    memo = {} if memo is None else memo
    if isinstance(o, _ATOMS) or isinstance(o, torch.Tensor):
        return o
    if id(o) in memo:
        return memo[id(o)]
    if isinstance(o, NP):
        memo[id(o)] = out = o.value([decode(x, memo) for x in o.items] if o.items is not None else None)
        return out
    memo[id(o)] = o
    if isinstance(o, dict):
        items = [(decode(kk, memo), decode(v, memo)) for kk, v in o.items()]
        o.clear(); o.update(items)
    elif type(o) is list:
        o[:] = [decode(v, memo) for v in o]
    elif type(o) in (tuple, set, frozenset):
        memo[id(o)] = out = type(o)(decode(v, memo) for v in o)
        return out
    elif isinstance(o, tuple) and hasattr(type(o), '_fields'):
        memo[id(o)] = out = type(o)(*(decode(v, memo) for v in o))
        return out
    elif _plain(o):
        for kk, v in list(vars(o).items()):
            o.__dict__[kk] = decode(v, memo)
    return o


def save(state, key, meta=None, root=None):
    """Write state-<key[:16]>.pt atomically; returns (path, dropped parts, notes)."""
    enc, dropped, notes = encode(state)
    payload = {'version': VERSION, 'key': key, 'dropped': dropped, 'notes': notes, 'state': enc,
               'meta': dict(meta or {}, torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0],
                            saved=time.strftime('%Y-%m-%dT%H:%M:%S'), sources=sources())}
    path = path_for(key, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path + '.tmp'); os.replace(path + '.tmp', path)  # noqa: E702
    return path, dropped, notes


def load(path, key, verbose=True):
    """The fitted state stored at path; asserts it records the same key (and version)."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload.get('version') == VERSION and payload.get('key') == key, \
        f'{path}: records key {str(payload.get("key"))[:16]} / {payload.get("version")}, expected {key[:16]} / {VERSION}'
    if verbose:
        m = payload.get('meta', {})
        print(f'student_state: loaded {os.path.basename(path)} (fitted {m.get("saved")} on {m.get("host", "?")}, torch {m.get("torch")}, '
              f'fit {m.get("fit_seconds")} s); dropped parts: {payload.get("dropped") or "none"}', file=sys.stderr, flush=True)
    return decode(payload['state'])


def _same_records(train, pool):
    with open(pool) as f:
        recs = [json.loads(line) for line in f]
    return recs == list(train)


def lookup(train):
    """(state, path) when a matching whole-state checkpoint exists for these records, else (None, None)."""
    if os.environ.get('EDGE_STATE', '1') == '0' or not os.path.exists(pool_path()):
        return None, None
    key = state_key()
    path = path_for(key)
    if not os.path.exists(path):
        return None, None
    if not _same_records(train, pool_path()):
        print(f'student_state: {os.path.basename(path)} exists but fit() got other records than {pool_path()}; fitting',
              file=sys.stderr, flush=True)
        return None, None
    return load(path, key), path


def checkpointed(fit):
    """Decorator for student.fit: return the saved whole state when one matches (see lookup), else fit as before."""
    def run(train):
        t0 = time.time()
        state, path = lookup(train)
        if state is not None:
            print(f'student_state: fit() skipped, state from {path} in {time.time() - t0:.1f} s', file=sys.stderr, flush=True)
            return state
        state = fit(train)
        if os.environ.get('EDGE_STATE_SAVE') == '1' and os.path.exists(pool_path()) and _same_records(train, pool_path()):
            print('student_state: saved', save(state, state_key(), {'fit_seconds': round(time.time() - t0, 1)})[0], file=sys.stderr)
        return state
    run.__doc__, run.__wrapped__ = fit.__doc__, fit
    return run


class _Probe:
    """Plain object for the self-test."""


def _test():
    """Round trip of an encoded state with tensors, numpy, modules, objects, sharing and an unpicklable part."""
    import tempfile
    lin = torch.nn.Linear(3, 2)
    shared = torch.arange(4.0)
    obj = _Probe(); obj.a, obj.s = np.arange(3, dtype=np.int16), np.float32(1.5)
    st = {'m': lin, 'x': shared, 'y': [shared, (1, 'a')], 'np': np.ones((2, 2)), 'g': np.str_('uci/x'), 'o': obj,
          'nl': (torch.ones(2, requires_grad=True) * 2), 'f': lambda v: v, 'dd': collections.defaultdict(list, {'k': [1]}),
          'obj': np.array(['a', 1], dtype=object), 'nk': {np.str_('g1'): 1}}
    with tempfile.TemporaryDirectory() as d:
        key = 'ab' * 32
        p, dropped, notes = save(st, key, root=d)
        got = load(p, key, verbose=False)
    assert got['y'][0] is got['x'] and torch.equal(got['x'], shared) and got['y'][1] == (1, 'a')
    assert isinstance(got['m'], torch.nn.Linear) and torch.equal(got['m'].weight, lin.weight)
    assert isinstance(got['np'], np.ndarray) and got['np'].shape == (2, 2) and got['g'] == 'uci/x' and isinstance(got['g'], np.str_)
    assert got['o'].a.dtype == np.int16 and got['o'].s == np.float32(1.5) and type(got['o'].s) is np.float32
    assert isinstance(got['f'], Dropped) and dropped == ["<Dropped state['f']: builtins.function>"], dropped
    assert got['dd']['k'] == [1] and type(got['dd']) is collections.defaultdict and not got['nl'].requires_grad
    assert list(got['obj']) == ['a', 1] and len(notes) == 1 and type(next(iter(got['nk']))) is np.str_
    assert copy.deepcopy(got['x']).sum() == 6
    k1 = state_key(pool=__file__)
    assert k1 == state_key(pool=__file__) and len(k1) == 64
    print('student_state ok', notes, dropped)


if __name__ == '__main__':
    _test()
