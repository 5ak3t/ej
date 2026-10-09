"""Write a fitted state as an ej weights directory (pickle-free), loadable with ``ej.load(DIR)``:
  state.safetensors + state.json.gz          the fitted state (ej.safe.export of the runtime's portable state encoding)
  encoder/w23.safetensors + encoder/w23.json  the low-bit encoder the state was fitted on
  config.json                                ej version, runtime sha256, e5 revision, state key, sha256 of every file, pool sha256
The state key is a sha256 over the state format version, every runtime student*/train_* module, every ej.train module and the
pool bytes, so a new pool or a code change gives a new key. scripts/hf_layout.py turns such a directory into a Hub upload."""
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
import time

from .. import integrity, loader, safe
from .env import HERE, RUNTIME


def sources():
    """Hashed source files: runtime student*/train_* modules, then the ej.train modules (sorted basenames within each)."""
    rt = sorted(f for f in os.listdir(RUNTIME) if f.endswith('.py') and f.startswith(('student', 'train_')))
    tr = sorted(os.path.basename(f) for f in glob.glob(os.path.join(HERE, '*.py')))
    return [os.path.join(RUNTIME, f) for f in rt] + [os.path.join(HERE, f) for f in tr]


def state_key(pool):
    """64-hex key of a fit of this code on this pool."""
    import student_state as ST
    h = hashlib.sha256(ST.VERSION.encode() + b'\x00')
    for p in sources():
        h.update(os.path.relpath(p, os.path.dirname(HERE)).encode() + b'\x00')
        with open(p, 'rb') as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    h.update(b'pool\x00')
    with open(pool, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _gz(src, dst):
    """Deterministic gzip (no file name, mtime 0, level 9)."""
    with open(src, 'rb') as fi, open(dst, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)


def write(state, out, pool, meta=None):
    """Write the weights directory `out` (new or empty) for a fitted, compacted state; returns its config dict."""
    import numpy as np
    import student_lb as LB
    import student_state as ST
    import torch
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError(f'{out} is not empty; refusing to overwrite')
    os.makedirs(os.path.join(out, 'encoder'), exist_ok=True)
    key = state_key(pool)
    enc, dropped, notes = ST.encode(state)
    payload = {'version': ST.VERSION, 'key': key, 'dropped': dropped, 'notes': notes, 'state': enc,
               'meta': dict(meta or {}, torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0],
                            saved=time.strftime('%Y-%m-%dT%H:%M:%S'))}
    st = safe.export(payload, os.path.join(out, 'state'))
    w = safe.export(LB.checkpoint(state['shallow']), os.path.join(out, 'encoder', 'w23'))
    bad = sorted(set(st['classes']) - safe.ALLOWED_CLASSES) + sorted(w['classes'])
    if bad:
        raise ValueError(f'the fitted state holds classes outside ej.safe.ALLOWED_CLASSES: {bad}')
    _gz(st['json'], st['json'] + '.gz')
    os.remove(st['json'])
    files = ['state.safetensors', 'state.json.gz', 'encoder/w23.safetensors', 'encoder/w23.json']
    skel = {rel: hashlib.sha256(safe.read_skeleton_bytes(os.path.join(out, rel[:-5]))).hexdigest()
            for rel in ('state.json', 'encoder/w23.json')}
    from .. import __version__
    cfg = {'ej_version': __version__, 'format': 'rev-safe-v1', 'runtime_sha256': integrity.runtime_sha256(),
           'e5_model': loader.E5_MODEL, 'e5_revision': loader.E5_REVISION, 'state_key': key,
           'encoder': state['shallow']['dir'], 'pool_sha256': integrity.file_sha256(pool), 'dropped_parts': dropped,
           'layout': dict(loader.LAYOUT, encoder='encoder/w23'),
           'files': {rel: integrity.file_sha256(os.path.join(out, rel)) for rel in files}, 'skeleton_json_sha256': skel}
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')
    return cfg
