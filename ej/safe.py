"""ej -- pickle-free weights: the fitted state and the low-bit encoder as <prefix>.safetensors + a JSON skeleton.

The skeleton may be gzipped (<prefix>.json.gz is read when <prefix>.json is absent); classes resolve against ej's own
runtime (ej/_runtime). Weights directories are written by `python -m ej.train fit` and packaged by scripts/hf_layout.py.

export(obj, out_prefix)  every tensor (also inside nn.Modules, nested anywhere; storage aliasing kept) -> <prefix>.safetensors,
                         everything else -> the JSON skeleton <prefix>.json (format: ej/safe_codec.py docstring).
load(prefix)             rebuilds the object with NO pickle / eval / exec; classes only from ALLOWED_CLASSES, via
                         cls.__new__ + __dict__; unknown class -> CodecError; the .safetensors sha256 recorded in the
                         skeleton is checked before the tensors are read.
install(state_prefix, w23_prefix, ckpt_dir)
                         the fitted state ready for the runtime's student.predict, with student_lb._CK pre-seeded at the
                         exact path student_lb.checkpoint computes, so the encoder is never read from a pickle."""
import gzip
import hashlib
import json
import os
import sys

from .safe_codec import FORMAT, CodecError, Encoder
from .safe_decode import Decoder

RUNTIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_runtime')

# Every class export met in the v1 state and encoder files: the runtime classes of the fitted heads / TF-IDF object and the
# torch.nn module classes inside them. Nothing else is ever instantiated by load().
ALLOWED_CLASSES = frozenset({
    'student_attn.FieldScorer', 'student_dn.Head', 'student_feat.Tfidf', 'student_rich.Scorer',
    'torch.nn.modules.activation.GELU', 'torch.nn.modules.container.Sequential', 'torch.nn.modules.dropout.Dropout',
    'torch.nn.modules.linear.Linear',
})
W23_CLASSES = frozenset()  # the encoder checkpoint is dicts / lists / tuples / tensors only


def file_sha256(path):
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def skeleton_path(prefix):
    """<prefix>.json if it exists, else <prefix>.json.gz (FileNotFoundError when neither exists)."""
    for p in (prefix + '.json', prefix + '.json.gz'):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no skeleton {prefix}.json or {prefix}.json.gz')


def read_skeleton_bytes(prefix):
    """The raw (decompressed) JSON bytes of the skeleton of prefix."""
    p = skeleton_path(prefix)
    with (gzip.open(p, 'rb') if p.endswith('.gz') else open(p, 'rb')) as f:
        return f.read()


def export(obj, out_prefix):
    """Write <out_prefix>.safetensors + <out_prefix>.json; returns {'json', 'safetensors', 'classes', 'tensors'}.
    Raises CodecError (listing every offending path) when obj holds functions, generators or other non-data parts."""
    from safetensors.torch import save_file
    enc = Encoder()
    root = enc.encode(obj)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    st = out_prefix + '.safetensors'
    save_file(enc.tensors, st, metadata={'format': FORMAT})
    doc = {'format': FORMAT, 'classes': sorted(enc.classes), 'tensors': len(enc.tensors),
           'safetensors_sha256': file_sha256(st), 'root': root}
    with open(out_prefix + '.json', 'w') as f:
        json.dump(doc, f, allow_nan=False, separators=(',', ':'), ensure_ascii=False)
    return {'json': out_prefix + '.json', 'safetensors': st, 'classes': doc['classes'], 'tensors': len(enc.tensors)}


def load(prefix, runtime_dir=RUNTIME, allow=ALLOWED_CLASSES):
    """The object stored at <prefix>.json[.gz] + <prefix>.safetensors; no pickle. Refuses a class outside allow and a tensor
    file whose sha256 differs from the one recorded in the skeleton."""
    from safetensors.torch import load_file
    doc = json.loads(read_skeleton_bytes(prefix))
    if doc.get('format') != FORMAT:
        raise CodecError(f'{prefix}: format {doc.get("format")!r}, expected {FORMAT!r}')
    bad = sorted(set(doc.get('classes', [])) - set(allow))
    if bad:
        raise CodecError(f'{prefix} names classes outside the allowlist: {bad}; refusing to load')
    st = prefix + '.safetensors'
    if file_sha256(st) != doc['safetensors_sha256']:
        raise CodecError(f'{st}: sha256 differs from the one recorded in its skeleton')
    if runtime_dir and runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    dec = Decoder(load_file(st), allow)
    out = dec.decode(doc['root'])
    if dec.filled:
        print(f'ej.safe.load: nn.Module defaults filled: {dec.filled}', file=sys.stderr)
    return out


def lowbit_path(c, ckpt_dir):
    """The path student_lb.checkpoint(c) computes when $EDGE_CKPT == ckpt_dir (a cache key; no file is read there)."""
    return os.path.join(ckpt_dir, c['dir'], f"{c['lb']}.pt")


def install(state_prefix, w23_prefix, ckpt_dir, key=None, runtime_dir=RUNTIME, verbose=True):
    """Fitted state for the runtime's student.predict from the pickle-free files; pre-seeds student_lb._CK with the decoded
    encoder. key: expected 64-hex state key (default: the key recorded in the file). ckpt_dir is exported as $EDGE_CKPT."""
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    import student_lb as LB
    import student_state as ST
    payload = load(state_prefix, runtime_dir)
    key = key or payload.get('key')
    if payload.get('version') != ST.VERSION or payload.get('key') != key:
        raise ValueError(f'{state_prefix}: records key {str(payload.get("key"))[:16]} / {payload.get("version")}, '
                         f'expected {str(key)[:16]} / {ST.VERSION}')
    if verbose:
        m = payload.get('meta', {})
        print(f'ej: loaded state {str(key)[:16]} (fitted {m.get("saved")}, torch {m.get("torch")}); '
              f'dropped parts: {payload.get("dropped") or "none"}', file=sys.stderr, flush=True)
    state = ST.decode(payload['state'])
    c = state.get('shallow') if isinstance(state, dict) else None
    if c and 'lb' in c:
        ck = load(w23_prefix, runtime_dir, allow=W23_CLASSES)
        if ck.get('variant') != c['lb']:
            raise ValueError(f'{w23_prefix}: encoder variant {ck.get("variant")!r}, the state needs {c["lb"]!r}')
        ckpt_dir = os.path.abspath(ckpt_dir)
        os.environ['EDGE_CKPT'] = ckpt_dir
        LB._CK.clear()
        LB._CK[lowbit_path(c, ckpt_dir)] = ck
        if LB.checkpoint(c) is not ck:  # the path student_lb computes must hit the pre-seeded entry
            raise RuntimeError('student_lb.checkpoint did not return the pre-seeded encoder')
    return state
