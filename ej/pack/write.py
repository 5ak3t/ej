"""WRITE an ejpack v1 file: one file holding a fitted state, its low-bit encoder checkpoint and the trimmed tokenizer.

Sections (container.py layout):
  enc.q2 / enc.q3 / enc.s / enc.z        bit-packed encoder codes + fp16 steps / offsets          (codes.py)
  heads.q8 / heads.s16 / heads.negz      per-row int8 head matrices + fp16 row scales + -0.0 plane (table.py, quant.py)
  cross.q8 / cross.s16 / cross.negz      int8 cross weights per block of 64 + fp16 block scales    (table.py, quant.py)
  voc.* / rvoc.*                         hashed cross keys (W-byte hashes, slot table)             (hvocab.py)
  rest.safetensors / rest.json.gz        every other tensor (F16 when exact, else F32) + the skeleton (ej's pickle-free codec)
  tok.vocab.txt.gz / tok.config.json     the trimmed tokenizer + BERT config                       (tok.py)
plus the header meta (state key, descriptors). Packing is deterministic: the same inputs give the same bytes.
from_weights_dir() converts an ej weights directory (`python -m ej.train fit` output) without unpickling anything; it is what
`python -m ej.train export` runs."""
import os

from . import codes as CD
from . import container as C
from . import hvocab as HV
from . import table as TB
from . import tok as TK
from .read import NAME, VOCABS

PAYLOAD_KEYS = ('version', 'key', 'meta', 'dropped', 'notes')


def write(path, payload, state, ck, tok_files, source=None):
    """Write the pack file path from payload (the state's metadata: version, key, meta, dropped, notes), the decoded state,
    the encoder checkpoint ck and tok_files ({'vocab.txt', 'config.json': bytes}); returns {'path', 'bytes', 'sections'}."""
    c = state.get('shallow') or {}
    if c.get('lb') != ck.get('variant'):
        raise ValueError(f"state needs encoder {c.get('lb')!r}, checkpoint is {ck.get('variant')!r}")
    head = {k: payload.get(k) for k in PAYLOAD_KEYS}
    sections, vmeta = {}, {}
    enc_secs, enc_desc = CD.encode(ck['codes'])
    sections.update(enc_secs)
    for k in VOCABS:
        v = HV.encode(state[k])
        sections.update({f'{k}.hash': v['hash'], f'{k}.slots': v['slots'], f'{k}.runs': v['runs']})
        vmeta[k] = {'width': v['width'], 'n': v['n']}
    rest_state = {k: (None if k in VOCABS else v) for k, v in state.items()}
    rest_ck = {k: v for k, v in ck.items() if k != 'codes'}
    tsecs, tdesc, _ = TB.encode({'payload': head, 'state': rest_state, 'ck': rest_ck})
    sections.update(tsecs)
    sections.update(TK.sections(tok_files, len(ck['ids'])))
    meta = {'state_key': head['key'], 'state_version': head['version'], 'ck_keys': list(ck), 'source': source or {},
            'encoder': {'variant': ck['variant'], 'dir': c.get('dir'), 'codes': enc_desc}, 'vocab': vmeta, 'tables': tdesc}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    order = sorted(sections, key=lambda n: (not n.startswith('enc.'), n))
    header = C.write(path + '.tmp', [(n, sections[n]) for n in order], meta)
    os.replace(path + '.tmp', path)
    return {'path': path, 'bytes': os.path.getsize(path), 'sections': {n: s['length'] for n, s in header['sections'].items()}}


def from_weights_dir(weights_dir, out, verify=True):
    """Write the pack of an ej weights directory (state.* + encoder/w23.* + config.json) to out (a file path, or a directory
    that receives model.ejpack). Reads safetensors + JSON only; the tokenizer files come from the pinned base tokenizer."""
    from .. import loader, safe, scope
    if os.path.isdir(out) or out.endswith(os.sep):
        out = os.path.join(out, NAME)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; refusing to overwrite')
    cfg = loader.read_config(weights_dir)
    if verify:
        loader.verify_weights(weights_dir, cfg)
    loader.prepare_environment(weights_dir)
    with scope.isolated_import():
        import student_state as ST
        payload = safe.load(os.path.join(weights_dir, loader.LAYOUT['state']))
        state = ST.decode(payload['state'])
        ck = safe.load(os.path.join(weights_dir, loader.LAYOUT['encoder']), allow=safe.W23_CLASSES)
    tok_files = TK.files_from_base(ck['ids'], loader.E5_MODEL, loader.E5_REVISION)
    source = {'weights_files': cfg.get('files', {}), 'runtime_sha256': cfg.get('runtime_sha256')}
    return write(out, payload, state, ck, tok_files, source)
