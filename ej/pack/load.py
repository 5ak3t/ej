"""LOAD a pack for prediction (what ej.load runs for a model.ejpack): verify, read, install into the runtime.

Checks before anything is decoded: the header digest and every section's sha256 (container.Reader); with verify=True also the
runtime modules against their manifest and, when the pack's state key is listed in ej.integrity.KNOWN_PACKS (the trust anchor
shipped in this package), the pack's content digest against the listed one. Decoding never unpickles, and afterwards no
runtime module can unpickle (ej.scope.forbid_unpickling). Nothing is downloaded: the tokenizer and the encoder config are
inside the pack and are written once to ej's cache directory ($EJ_CACHE)."""
import os
import sys

from .. import integrity, loader, scope
from . import container as C
from . import read as RD
from . import runtime as RT
from . import tok as TK


def check_known(r):
    """Warnings for a verified pack (empty for a known release); raises ValueError on a known key with other contents."""
    key = str(r.meta.get('state_key', ''))[:16]
    want = integrity.KNOWN_PACKS.get(key)
    if want is None:
        return [f'state {key} is not in ej.integrity.KNOWN_PACKS: sections checked against the pack header only']
    if r.digest != want:
        raise ValueError(f'pack content digest {r.digest} != known {want} for state {key}; refusing to load')
    return []


def load_pack(path, verify=True, memo_max=None, low_memory=False, lru=0):
    """(fitted state, info dict) from a pack file (or a directory holding model.ejpack), with the runtime ready to predict.
    low_memory=True keeps the encoder's 2/3-bit codes and dequantises each weight inside forward (bit-identical predictions,
    less peak memory, slower); lru: dequantised weights kept in that mode (0 = none)."""
    p = RD.locate(path)
    if p is None:
        raise ValueError(f'{path}: no {RD.NAME} found')
    ckpt_dir = os.path.dirname(p)
    loader.prepare_environment(ckpt_dir)
    loader.check_runtime_imports()
    if verify:
        integrity.verify_runtime()
    with scope.isolated_import(), loader.e5_pinned():
        import student  # noqa: F401  (the runtime; its import-time seed / threads are undone by isolated_import)
        import student_state as ST
        payload, state, ck, r = RD.read(p, unpack=False)
        warnings = check_known(r) if verify else ['verification skipped (verify=False)']
        if payload.get('version') != ST.VERSION:
            raise C.PackError(f'{p}: state format {payload.get("version")!r}, this ej reads {ST.VERSION!r}')
        tok_dir = TK.materialise({n: r.section(n) for n in TK.SECTIONS}, loader.cache_dir())
        RT.install(state, ck, tok_dir, ckpt_dir, 'stream' if low_memory else 'fp32', lru)
        integrity.forbid_lowbit_pickle()
    for w in warnings:
        print(f'ej: warning: {w}', file=sys.stderr)
    info = {'format': C.FORMAT, 'path': p, 'state_key': payload['key'], 'file_sha256': r.file_sha256,
            'content_digest': r.digest, 'bytes': os.path.getsize(p), 'low_memory': bool(low_memory)}
    r.close()
    del r
    loader.check_runtime_imports()
    scope.forbid_unpickling(integrity.RUNTIME)
    scope.install_memo(memo_max)
    RT.trim()
    return state, info
