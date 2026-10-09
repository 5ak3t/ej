"""Integrity checks for ej: runtime manifest, release file hashes, and the runtime's no-unpickle guard of the encoder loader.

A mismatch raises ValueError before anything is read. Loading is pickle-free (ej.safe) and, once loaded, no runtime module
can unpickle (ej.scope.forbid_unpickling)."""
import hashlib
import os

RUNTIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_runtime')
MANIFEST = os.path.join(RUNTIME, 'RUNTIME_SHA256')

# Pickle-free weights files of known fitted states, by state key prefix: sha256 of each .safetensors file and of each skeleton's
# JSON bytes (after gunzip, so .json and .json.gz give the same value). A state not listed loads with a warning (files are then
# checked against its own config.json only).
KNOWN_RELEASES = {
    '3b3e66d28fb423f9': {  # ej 0.0.1 (safetensors + JSON layout)
        'state.safetensors': '543e00893159fa3c5128e1c02c9d9070e19c091cfcd01dddcb037b5f5feb3bea',
        'state.json': 'f2dd3a3fad2d0cce525f03bd37ef4e8c5aa1f4d64a5f89c03e9a4f15f0c4bfcf',
        'encoder/w23.safetensors': '99272d36fdcb57423df09ea38b03b2676787dfe738661b4808a592d9c615a3a4',
        'encoder/w23.json': '779151bc2e2a2daf15b0ec495e1dd04bb2aa773932bbfb13a577360cd928d829',
    },
}
# Packed releases (ejpack v1, one file model.ejpack) of known fitted states, by state key prefix: the content digest
# (ej.pack.container.content_digest: every section's sha256 and length plus the header meta except 'source'). A pack whose
# state key is listed here must match it; a pack not listed loads with a warning (sections checked against its own header).
KNOWN_PACKS = {
    '3b3e66d28fb423f9': '68459de15b060169ce05ca4ed76304a40a112ba1e2e45d3e7ee2c48c35a15c8a',  # ej 0.0.1 (file sha256 e990e184...)
}
# Where each known state comes from: the training pool (sha256), the low-bit encoder, and the model-code commit of the
# maintainers' development repository (not public) whose student*/train_* files and pool reproduce the state key
# (scripts/state_key.py). Its prediction code is ej/_runtime.
RELEASE_PROVENANCE = {
    '3b3e66d28fb423f9': {
        'version': '0.0.1',
        'pool_sha256': 'ce1c1a6fecfd62a90317f6efc4f90fd5f9261becb7081fd00235a6f4d5ee9dbe',
        'model_code_commit': 'f46cf7c', 'encoder': 'lowbit-b3b010513f948ceb',
        'key_tree': 'f46cf7c model code with student_lb.CK_DIR = lowbit-b3b010513f948ceb (scripts/state_key.py --ck-dir)',
    },
}


def file_sha256(path):
    """Hex sha256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_sha(path, expected, what):
    """Raise ValueError unless sha256(path) == expected; returns the digest."""
    got = file_sha256(path)
    if got != expected:
        raise ValueError(f'{what} {path}: sha256 {got} != expected {expected}; refusing to load')
    return got


def read_manifest(path=MANIFEST):
    """{file name: sha256} from a sha256sum-format manifest ('#' lines are comments)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                sha, name = line.split()
                out[name] = sha
    return out


def runtime_sha256(path=MANIFEST):
    """sha256 of the runtime manifest file itself: one value that pins every runtime module."""
    return file_sha256(path)


def verify_runtime(runtime_dir=RUNTIME, manifest=MANIFEST):
    """Check every runtime module against the manifest, and that no unlisted .py file sits next to them."""
    want = read_manifest(manifest)
    have = {f for f in os.listdir(runtime_dir) if f.endswith('.py')}
    extra, missing = sorted(have - set(want)), sorted(set(want) - have)
    if extra or missing:
        raise ValueError(f'ej runtime {runtime_dir}: unlisted files {extra}, missing files {missing}')
    for name, sha in want.items():
        check_sha(os.path.join(runtime_dir, name), sha, 'runtime module')
    return len(want)


def forbid_lowbit_pickle():
    """After a pickle-free install: make the runtime's lazy encoder loader refuse any cache miss instead of unpickling."""
    import student_lb as LB
    if getattr(LB.checkpoint, '_ej_guarded', False):
        return

    def checkpoint(c, _orig=LB.checkpoint):
        path = os.path.join(os.environ.get('EDGE_CKPT', ''), c['dir'], f"{c['lb']}.pt")
        if path not in LB._CK:
            raise ValueError(f'encoder {path} was not installed from safetensors; refusing to unpickle it')
        return _orig(c)
    checkpoint._ej_guarded = True
    LB.checkpoint = checkpoint  # student_lb.encoder looks the name up in its module globals at call time
