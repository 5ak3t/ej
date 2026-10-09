"""Locate, verify and install ej weights from a weights directory (pickle-free), and prepare the runtime environment.

The default weights format is the packed model file model.ejpack (ej.pack, loaded by ej.pack.load); this module serves the
secondary format, the weights directory that `python -m ej.train fit` writes, and the environment both formats share.
Weights directory layout (built by scripts/hf_layout.py; the Hugging Face repository has the same files):
  config.json                              ej version, runtime sha256, e5 revision, state key, sha256 of every file
  state.safetensors + state.json.gz        fitted state (heads, pool, calibration, TF-IDF, vocabularies)
  encoder/w23.safetensors + encoder/w23.json   low-bit encoder (2-bit weights, 3-bit attention, trimmed vocabulary)
Checks before anything is decoded (verify=True): every file's sha256 against config.json; the files against KNOWN_RELEASES
when the state key is listed there (the trust anchor shipped in this package); config.json's runtime sha256 and e5 revision
against this package; every runtime module against _runtime/RUNTIME_SHA256. Decoding itself never unpickles (ej.safe), and
after loading no runtime module can unpickle (ej.scope.forbid_unpickling).

Environment (public names): $EJ_CACHE (default $XDG_CACHE_HOME/ej, else ~/.cache/ej) receives the trimmed tokenizer
vocabulary; $EJ_COLD=1 makes every predict call bypass the prediction-time caches (cold latency); $HF_HOME defaults to the
Hugging Face default location. The runtime modules read the same settings under their internal names (EDGE_CKPT = the weights
directory, EDGE_CACHE, EDGE_COLD); ej sets those from the public names, and an EDGE_* value already in the environment is
still honoured. For a weights directory, the base model intfloat/e5-small-v2 is fetched from the Hub at the pinned revision
E5_REVISION on first use (a pack needs no base model)."""
import contextlib
import json
import os
import sys

from . import integrity
from .integrity import RUNTIME

E5_MODEL = 'intfloat/e5-small-v2'
E5_REVISION = 'ffb93f3bd4047442299a41ebb6fa998a38507c52'
LAYOUT = {'state': 'state', 'encoder': os.path.join('encoder', 'w23')}  # skeleton/tensor prefixes inside the weights dir


def e5_pinned():
    """Context manager: inside an ej call, from_pretrained(E5_MODEL) without a revision gets E5_REVISION (ej.scope.e5_pinned).
    The runtime calls from_pretrained(E5_MODEL) with no revision; the pin keeps that code unchanged and is undone on exit, so
    transformers is never patched outside ej.load / Model.predict."""
    from .scope import e5_pinned as pin
    return pin(E5_MODEL, E5_REVISION)


def cache_dir():
    """ej's cache directory: $EJ_CACHE, else $XDG_CACHE_HOME/ej, else ~/.cache/ej."""
    if os.environ.get('EJ_CACHE'):
        return os.environ['EJ_CACHE']
    return os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'), 'ej')


@contextlib.contextmanager
def cold():
    """Inside a predict call: EJ_COLD=1 switches the runtime's cold mode (its internal EDGE_COLD) on; restored on exit."""
    if os.environ.get('EJ_COLD') != '1' or os.environ.get('EDGE_COLD') == '1':
        yield
        return
    os.environ['EDGE_COLD'] = '1'
    try:
        yield
    finally:
        os.environ.pop('EDGE_COLD', None)


def prepare_environment(weights_dir):
    """Set the variables the runtime reads (EDGE_CKPT, EDGE_CACHE, HF_HOME) and put the runtime on sys.path.
    Must run before the first runtime import: some runtime modules read these variables at import time."""
    os.environ['EDGE_CKPT'] = os.path.abspath(weights_dir)
    os.environ.setdefault('EDGE_CACHE', cache_dir())  # a pre-set EDGE_CACHE (older setups) wins over EJ_CACHE
    os.makedirs(os.environ['EDGE_CACHE'], exist_ok=True)
    if 'HF_HOME' not in os.environ:
        xdg = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
        os.environ['HF_HOME'] = os.path.join(xdg, 'huggingface')
    if RUNTIME not in sys.path:
        sys.path.insert(0, RUNTIME)


def check_runtime_imports():
    """Refuse when a bare-name runtime module (e.g. 'student') was imported from somewhere other than ej's runtime."""
    names = integrity.read_manifest()
    for mod in (sys.modules.get(n[:-3]) for n in names):
        f = getattr(mod, '__file__', None)
        if f and os.path.dirname(os.path.realpath(f)) != os.path.realpath(RUNTIME):
            raise ImportError(f'module {mod.__name__!r} was imported from {f}, not from the ej runtime {RUNTIME}; '
                              'rename your module or import ej first')


def resolve(path_or_repo, revision=None):
    """A local path: path_or_repo itself when it is a file (a model.ejpack) or a directory; else a Hugging Face repo id, whose
    model.ejpack is downloaded and sha256-checked (ej.hub.download; a repo without one is fetched as a snapshot)."""
    if os.path.isdir(path_or_repo) or os.path.isfile(path_or_repo):
        return os.path.abspath(path_or_repo)
    from .hub import download
    return download(path_or_repo, revision)


def read_config(weights_dir):
    """config.json of a weights directory (ValueError when missing)."""
    p = os.path.join(weights_dir, 'config.json')
    if not os.path.exists(p):
        raise ValueError(f'{weights_dir}: no config.json; not an ej weights directory (see scripts/hf_layout.py)')
    with open(p) as f:
        return json.load(f)


def _skeleton_sha(weights_dir, prefix):
    import hashlib
    from .safe import read_skeleton_bytes
    return hashlib.sha256(read_skeleton_bytes(os.path.join(weights_dir, prefix))).hexdigest()


def verify_weights(weights_dir, cfg):
    """All release checks (module docstring). Returns the list of warnings (unknown state key)."""
    for rel, sha in sorted(cfg.get('files', {}).items()):
        integrity.check_sha(os.path.join(weights_dir, rel), sha, 'weights file')
    for p in LAYOUT.values():
        rel = p.replace(os.sep, '/')
        if rel + '.safetensors' not in cfg.get('files', {}):
            raise ValueError(f'config.json lists no sha256 for {rel}.safetensors; refusing to load')
    if cfg.get('runtime_sha256') != integrity.runtime_sha256():
        raise ValueError(f'weights were packaged for runtime {cfg.get("runtime_sha256")}, this ej has '
                         f'{integrity.runtime_sha256()}; install the matching ej version')
    if cfg.get('e5_revision') != E5_REVISION:
        raise ValueError(f'weights expect e5 revision {cfg.get("e5_revision")}, this ej pins {E5_REVISION}')
    integrity.verify_runtime()
    known = integrity.KNOWN_RELEASES.get(str(cfg.get('state_key', ''))[:16])
    if known is None:
        return [f'state {str(cfg.get("state_key"))[:16]} is not in ej.integrity.KNOWN_RELEASES: files checked against '
                'config.json only']
    for rel, sha in known.items():
        if rel.endswith('.json'):
            got = _skeleton_sha(weights_dir, rel[:-5].replace('/', os.sep))
            if got != sha:
                raise ValueError(f'{rel}: skeleton sha256 {got} != known {sha}; refusing to load')
        else:
            integrity.check_sha(os.path.join(weights_dir, rel), sha, 'weights file')
    return []


def load_state(weights_dir, verify=True, memo_max=None):
    """(fitted state, config) from a weights directory, with the runtime ready to predict. Never unpickles.
    The runtime's import-time side effects (2 torch threads, torch.manual_seed(0)) are undone before returning (ej.scope);
    every runtime module gets a torch / pickle whose load refuses (forbid_unpickling); the encoder memo is bounded."""
    from . import safe, scope
    cfg = read_config(weights_dir)
    for w in (verify_weights(weights_dir, cfg) if verify else ['verification skipped (verify=False)']):
        print(f'ej: warning: {w}', file=sys.stderr)
    prepare_environment(weights_dir)
    check_runtime_imports()
    with scope.isolated_import(), e5_pinned():
        from .pack.runtime import drop_standin
        drop_standin()  # a pack loaded earlier in this process left a base-model stand-in; this path builds from the real one
        state = safe.install(os.path.join(weights_dir, LAYOUT['state']), os.path.join(weights_dir, LAYOUT['encoder']),
                             ckpt_dir=weights_dir, key=cfg.get('state_key') if verify else None)
        integrity.forbid_lowbit_pickle()
        import student  # noqa: F401  (runtime entry point; imported here so a bad environment fails at load time)
    check_runtime_imports()
    scope.forbid_unpickling(RUNTIME)
    scope.install_memo(memo_max)
    return state, cfg
