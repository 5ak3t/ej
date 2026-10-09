"""READ an ejpack v1 file, pickle-free: header digest and every section's sha256 are checked when the file is opened
(container.Reader), then the encoder codes, the hashed vocabularies and the object graph are decoded. No side effects: installing
the result into the runtime is runtime.install's job. The allowlisted classes of the state are runtime classes, so the runtime
directory must be on sys.path (ej.loader.prepare_environment) before read()."""
import os

from . import codes as CD
from . import container as C
from . import hvocab as HV
from . import table as TB

NAME = 'model.ejpack'
VOCABS = ('voc', 'rvoc')


def locate(path):
    """The pack file of path: path itself when it is a file, else <path>/model.ejpack; None when there is none."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    p = os.path.join(path, NAME)
    return os.path.abspath(p) if os.path.isfile(p) else None


def is_pack(path):
    """True when path is an ejpack file (by its magic bytes) or a directory holding model.ejpack."""
    p = locate(path)
    if p is None:
        return False
    with open(p, 'rb') as f:
        return f.read(len(C.MAGIC)) == C.MAGIC


def read(path, unpack=True):
    """(payload, state, ck, reader) decoded from the pack file path. payload = the state's metadata (version, key, meta,
    dropped, notes); ck = the encoder checkpoint; unpack=False keeps the encoder codes bit-packed (entries hold 'raw' instead
    of 'q'; encoder.build takes either)."""
    r = C.Reader(path)
    meta = r.meta
    secs = {n: r.section(n) for n in r.header['sections']}
    codes = CD.decode(secs, meta['encoder']['codes'], unpack)
    obj = TB.decode(secs, meta['tables'])
    payload, state, ck = obj['payload'], obj['state'], obj['ck']
    if payload.get('key') != meta['state_key'] or payload.get('version') != meta['state_version']:
        raise C.PackError('payload key / version differ from the header')
    for k in VOCABS:
        state[k] = HV.HashedVocab(secs[f'{k}.hash'], secs[f'{k}.slots'], secs[f'{k}.runs'], **meta['vocab'][k])
    ck['codes'] = codes
    ck = {k: ck[k] for k in meta['ck_keys']}
    return payload, state, ck, r
