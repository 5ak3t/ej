"""Process hygiene for the ej runtime: nothing ej does may change the caller's process state outside an ej call.

- Threads and RNG: the runtime sets `torch.set_num_threads(2)` and `torch.manual_seed(0)` when it is first
  imported. `isolated_import()` snapshots the thread count and the CPU RNG state around that import and restores both;
  `predict_scope(threads)` sets the thread count only for the duration of one predict call and restores it afterwards. The
  RNG is never consumed by a prediction (tested), so predictions do not depend on the caller's seed.
- Base-model revision: `e5_pinned()` gives every `from_pretrained(E5_MODEL)` call without a revision the pinned revision,
  only inside an ej call (no process-wide monkeypatch of transformers).
- No unpickling: `forbid_unpickling()` replaces the module global `torch` (and `pickle`) of every ej runtime
  module by a copy whose `load` / `loads` / `Unpickler` refuse, so no runtime code path can unpickle a file. Only ej's own
  runtime modules are changed; the real `torch` and `pickle` modules are untouched.
- Bounded encoder memo: `LRUMemo` replaces the runtime's process-wide, never-evicted text -> encoder-state memo;
  `trim_memo()` evicts the least recently used entries down to `MEMO_MAX` texts after each predict chunk."""
import contextlib
import os
import sys
import types
from collections import OrderedDict

MEMO_MAX = int(os.environ.get('EJ_MEMO_MAX', '20000'))  # texts kept in the encoder memo between calls (~21 KB per record)
DEFAULT_THREADS = 2  # the thread count the reference predictions and the development numbers were made with


class UnpicklingRefused(RuntimeError):
    """Raised when ej runtime code tries to unpickle anything."""


def _refuse(*_a, **_k):
    raise UnpicklingRefused('ej runtime: unpickling is disabled (weights load from safetensors + JSON only)')


class _Refusing:
    """Stands in for pickle.Unpickler: constructing it refuses."""

    def __init__(self, *_a, **_k):
        _refuse()


def guarded_module(real, refused):
    """A module object with real's namespace, except the names in `refused`, which raise UnpicklingRefused."""
    m = types.ModuleType(real.__name__, getattr(real, '__doc__', None))
    m.__dict__.update({k: v for k, v in vars(real).items() if k not in ('__name__', '__doc__')})
    for name in refused:
        m.__dict__[name] = _Refusing if name == 'Unpickler' else _refuse
    m.__getattr__ = lambda name: getattr(real, name)  # lazily created attributes of the real module
    m._ej_guarded = True
    return m


def runtime_modules(runtime_dir):
    """The imported modules whose file lies in the ej runtime directory."""
    root = os.path.realpath(runtime_dir)
    out = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and os.path.dirname(os.path.realpath(f)) == root:
            out.append(mod)
    return out


def forbid_unpickling(runtime_dir):
    """Install the refusing torch / pickle globals in every imported runtime module (idempotent). Returns the count."""
    import pickle
    import torch
    gt = guarded_module(torch, ('load',))
    gp = guarded_module(pickle, ('load', 'loads', 'Unpickler'))
    n = 0
    for mod in runtime_modules(runtime_dir):
        for name, real, guard in (('torch', torch, gt), ('pickle', pickle, gp)):
            if mod.__dict__.get(name) is real:
                mod.__dict__[name] = guard
                n += 1
    return n


@contextlib.contextmanager
def e5_pinned(model, revision):
    """Inside the block, transformers Auto*.from_pretrained(model) without a revision uses `revision`; restored on exit."""
    import transformers
    saved = {}
    for name in ('AutoModel', 'AutoTokenizer', 'AutoConfig'):
        cls = getattr(transformers, name)
        orig = cls.__dict__.get('from_pretrained')
        fn = cls.from_pretrained

        def wrapped(*args, _orig=fn, **kw):
            name_or_path = args[0] if args else kw.get('pretrained_model_name_or_path')
            if name_or_path == model and kw.get('revision') is None:
                kw['revision'] = revision
            return _orig(*args, **kw)
        saved[cls] = orig
        cls.from_pretrained = staticmethod(wrapped)
    try:
        yield
    finally:
        for cls, orig in saved.items():
            if orig is None:
                delattr(cls, 'from_pretrained')
            else:
                cls.from_pretrained = orig


@contextlib.contextmanager
def isolated_import():
    """Restore torch's thread count and CPU RNG state after the block (the runtime's import-time side effects)."""
    import torch
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


@contextlib.contextmanager
def predict_scope(threads):
    """Run the block with `threads` intra-op threads (None = leave the process setting); restore the count afterwards.
    Yields the thread count in effect inside the block."""
    import torch
    before = torch.get_num_threads()
    if threads is not None:
        if int(threads) < 1:
            raise ValueError(f'threads must be >= 1, got {threads!r}')
        torch.set_num_threads(int(threads))
    try:
        yield torch.get_num_threads()
    finally:
        torch.set_num_threads(before)


class LRUMemo(OrderedDict):
    """The runtime's text -> (pooled, token states) memo with least-recently-used order (reads and writes refresh an entry).
    Eviction happens only in trim(), between predict chunks, so a chunk never loses an entry it is about to read."""

    def __init__(self, cap=None):
        super().__init__()
        self.cap = MEMO_MAX if cap is None else int(cap)

    def __getitem__(self, k):
        v = super().__getitem__(k)
        self.move_to_end(k)
        return v

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.move_to_end(k)

    def trim(self):
        """Evict the least recently used entries down to the cap; returns the number evicted."""
        n = max(len(self) - self.cap, 0)
        for _ in range(n):
            self.popitem(last=False)
        return n


class _Warm(dict):
    """student_fast._ST['warm'] (one memo per encoder tag) whose memos are LRUMemo objects."""

    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def setdefault(self, k, default=None):
        if k not in self:
            self[k] = LRUMemo(self.cap)
        return self[k]


def install_memo(cap=None):
    """Replace the runtime's unbounded warm memo by LRU memos (entries already memoised are kept). Returns the warm dict."""
    import student_fast as FA
    warm = FA._ST['warm']
    if isinstance(warm, _Warm):
        if cap is not None:
            warm.cap = int(cap)
            for m in warm.values():
                m.cap = int(cap)
        return warm
    new = _Warm(MEMO_MAX if cap is None else int(cap))
    for tag, memo in warm.items():
        m = new.setdefault(tag)
        m.update(memo)
    FA._ST['warm'] = new
    return new


def trim_memo():
    """Trim every encoder memo to its cap; returns (entries kept, entries evicted)."""
    import student_fast as FA
    kept = evicted = 0
    for m in FA._ST['warm'].values():
        if isinstance(m, LRUMemo):
            evicted += m.trim()
        kept += len(m)
    return kept, evicted
