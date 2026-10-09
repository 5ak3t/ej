"""ej: calibrated typed decisions in one pass, on device.

Give it a `state` (text, or a JSON object as text) and typed questions (choice / noul (yes-no) / score); it returns one
probability distribution per question, without generating tokens.

    import ej
    model = ej.load('/path/to/model.ejpack')               # one local model file (README: release status)
    (probs,) = model.predict([ej.EXAMPLE_RECORD])          # {qid: [p for each option, in option order]}
    adapted = model.adapt(examples=labelled_records)        # optional: adapt to one workflow (ej.adapt)
    adapted.observe(more_labelled_records)                  # fold in labelled records as they arrive

Weights: a packed model file model.ejpack (ej.pack: pickle-free, sha256-checked, nothing downloaded) is the default format;
`ej.load(..., low_memory=True)` streams the encoder's 2/3-bit weights (less memory, slower, identical predictions). A weights
directory (safetensors + JSON, `python -m ej.train fit` output) still loads, and fetches the base model on first use.
The prediction code lives in ej/_runtime (sha256-pinned modules), imported by bare module name from a sys.path entry that ej
adds on load; do not shadow those names with modules of your own. One loaded model per process. Training: ej.train;
evaluation: ej.eval. ej leaves the process as it found it: the runtime's import-time thread count and seed are undone, threads
are set only inside a predict call, and no runtime module can unpickle (ej.scope)."""
from .adapt import AdaptedModel
from .scope import DEFAULT_THREADS
from .records import EXAMPLE_RECORD, NOUL_OPTIONS, RecordError, validate_record, validate_records

__version__ = '0.0.1'
__all__ = ['load', 'Model', 'AdaptedModel', 'EXAMPLE_RECORD', 'NOUL_OPTIONS', 'RecordError', 'validate_record', 'validate_records',
           '__version__']


class Model:
    """A loaded ej model. Create it with ej.load(); call predict(records)."""

    CHUNK = 64  # records per encoder batch in predict (bounded peak memory)

    def __init__(self, state, config, weights_dir, threads=DEFAULT_THREADS, chunk_size=CHUNK):
        """Wrap a fitted state (ej.pack.load.load_pack or ej.loader.load_state) with its info / config.json dict and its path.
        threads: torch intra-op threads used inside predict (restored afterwards; None = the process setting).
        chunk_size: records encoded per batch (None = all records in one batch)."""
        self._state, self.config, self.weights_dir = state, config, weights_dir
        self.threads, self.chunk_size = threads, chunk_size
        self.last_threads = None  # torch threads in effect during the last predict call (measured inside it)

    def predict(self, records, threads=None, chunk_size=None):
        """[{qid: [p_1, ..., p_K]}] per record, each list in the order of that question's options and summing to 1.
        Records are validated first (ej.records) and encoded in chunks of `chunk_size` records (default self.chunk_size).
        Chunking moves probabilities only at the float-noise level (max |dp| ~1e-7 against one batch: records are padded
        together), and so do thread counts (~6e-7 between 1 and 2 threads); 2 threads reproduce the reference numbers."""
        import student  # the runtime is on sys.path once ej.load() has run
        from . import loader, scope
        records = validate_records(records)
        if not records:
            return []
        n = chunk_size if chunk_size is not None else self.chunk_size
        n = len(records) if not n else int(n)
        out = []
        with scope.predict_scope(self.threads if threads is None else threads) as used, loader.e5_pinned(), loader.cold():
            self.last_threads = used
            for b in range(0, len(records), n):
                out += student.predict(self._state, records[b:b + n])
                scope.trim_memo()
        if self.config.get('format') == 'ejpack-v1':
            from .pack import runtime
            runtime.trim()
        return [{qid: [float(p) for p in ps] for qid, ps in d.items()} for d in out]

    def adapt(self, examples=None, config=None):
        """An AdaptedModel for ONE workflow (ej.adapt): `examples` = its records with 'answers': {qid: option key} (the option
        tilt). Without examples -> this model itself (identical predictions)."""
        if not examples:
            return self
        return AdaptedModel(self, examples, config)

    def __repr__(self):
        return f"ej.Model(state={str(self.config.get('state_key'))[:16]}, weights_dir={self.weights_dir!r})"


def load(path, revision=None, verify=True, threads=DEFAULT_THREADS, chunk_size=Model.CHUNK, memo_max=None, low_memory=False):
    """Load an ej model: a model.ejpack file or a directory holding one (the default format, ej.pack), else a weights directory
    (safetensors + JSON) or a Hugging Face repo id (pin `revision` to a commit for reproducibility).
    verify=True checks the runtime and, for known releases, the weights against the hashes shipped in ej.integrity, before
    decoding; loading never unpickles. low_memory=True (packs only): the encoder keeps its 2/3-bit codes and dequantises each
    weight inside forward (identical predictions, lower peak memory, slower). threads / chunk_size: Model defaults for predict;
    memo_max: texts kept in the encoder memo (ej.scope.MEMO_MAX)."""
    from . import loader
    from .pack import is_pack
    target = loader.resolve(path, revision)
    if is_pack(target):
        from .pack.load import load_pack
        state, info = load_pack(target, verify=verify, memo_max=memo_max, low_memory=low_memory)
        return Model(state, info, info['path'], threads=threads, chunk_size=chunk_size)
    if low_memory:
        raise ValueError(f'{target}: low_memory=True needs a packed model (python -m ej.train export --weights DIR --out FILE)')
    state, cfg = loader.load_state(target, verify=verify, memo_max=memo_max)
    return Model(state, cfg, target, threads=threads, chunk_size=chunk_size)
