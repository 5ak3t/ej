"""The packed format (ej.pack, ejpack v1) on a SMALL synthetic pack: no weights needed.
Falsified if: a decoded tensor, code or vocabulary lookup differs by any bit from what was written; writing is not
deterministic; a corrupted section or header is not refused before decoding; a known state key with other contents loads;
anything is unpickled while reading, building and running the encoder; the low-memory (streaming) encoder differs by any bit
from the default one."""
import json
import os
import subprocess
import sys

import pytest
import torch

import synthpack as SP
from ej.pack import bits as B
from ej.pack import container as C
from ej.pack import encoder as EN
from ej.pack import heads as HD
from ej.pack import lowmem as LM
from ej.pack import read as RD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def same(a, b):
    """Bitwise equality of two tensors (float32 compared as int32 views, so -0.0 != 0.0)."""
    if a.dtype == torch.float32:
        a, b = a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def dequant(c):
    """The runtime's dequantisation formula (student_lbq.dequant)."""
    return ((c['q'].float() - c['z'].float()) * c['s'].float()).reshape(c['shape'])


@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    path = tmp_path_factory.mktemp('pack') / 'model.ejpack'
    rep, st, ck = SP.write(path)
    return path, rep, st, ck


def test_bit_packing_round_trip():
    g = torch.Generator().manual_seed(0)
    for bits in (1, 2, 3, 8):
        q = torch.randint(0, 1 << bits, (1000,), generator=g, dtype=torch.uint8)
        buf = B.pack(q.numpy(), bits)
        assert len(buf) == B.nbytes(1000, bits) and (B.unpack(buf, bits, 1000) == q.numpy()).all()
        assert torch.equal(LM.unpack_codes(torch.frombuffer(bytearray(buf), dtype=torch.uint8), bits, 1000), q)
        if bits in (2, 3):  # the encoder's widths: word re-layout, whole-stream and chunked (chunk: a multiple of 8 and 32 // b)
            for chunk in (80, 40960):
                assert torch.equal(LM.from_words(LM.raw_to_words(LM.pack_q(q, bits), bits, 1000, chunk), bits, 1000), q.int())
    with pytest.raises(ValueError):
        B.pack([4], 2)


def test_round_trip_is_bit_exact_and_small(pack):
    path, rep, st, ck = pack
    assert rep['bytes'] == os.path.getsize(path) < 200_000
    payload, st2, ck2, r = RD.read(str(path))
    assert payload == SP.payload() and set(st2) == set(st)
    for n, c in ck['codes'].items():
        d = ck2['codes'][n]
        assert same(d['q'], c['q']) and same(d['s'], c['s']) and same(d['z'], c['z']) and d['shape'] == c['shape']
    assert all(same(ck2['other'][n], v) for n, v in ck['other'].items()) and ck2['ids'] == ck['ids']
    assert same(st2['w'], st['w']) and all(same(a, b) for a, b in zip(st2['rich'].parameters(), st['rich'].parameters()))
    assert same(st2['misc']['f32'], st['misc']['f32']) and same(st2['misc']['f16'], st['misc']['f16'])
    assert same(st2['misc']['ints'], st['misc']['ints']) and st2['misc']['name'] == 'synthetic'
    for k in ('voc', 'rvoc'):
        assert len(st2[k]) == len(st[k]) and all(st2[k][key] == col for key, col in st[k].items())
        assert ('0123456789abcdef', 'not-a-token') not in st2[k] and {s for s, _ in st2[k]} == set(SP.SLOTS)
        assert st2[k].false_hit_bound() < 1e-9
    kinds = {d['kind'] for d in r.meta['tables']['q8'].values()}
    assert kinds == {'rows', 'blocks'}  # the head matrices and the cross vector were stored as exact int8


def test_writing_is_deterministic(pack, tmp_path):
    path = pack[0]
    SP.write(tmp_path / 'again.ejpack')
    assert open(path, 'rb').read() == open(tmp_path / 'again.ejpack', 'rb').read()


def _corrupt(src, dst, where):
    data = bytearray(open(src, 'rb').read())
    header, data0 = C.parse(memoryview(bytes(data)))
    s = header['sections']['heads.q8']
    pos = 60 if where == 'header' else data0 + s['offset'] + s['length'] // 2
    data[pos] ^= 0x01
    open(dst, 'wb').write(bytes(data))


@pytest.mark.parametrize('where, message', [('section', "section 'heads.q8': sha256"), ('header', 'header digest')])
def test_corruption_is_refused_before_decoding(pack, tmp_path, where, message):
    bad = tmp_path / 'model.ejpack'
    _corrupt(pack[0], bad, where)
    with pytest.raises(C.PackError, match=message):
        RD.read(str(bad))


def test_load_refuses_a_corrupted_pack(pack, tmp_path):
    import ej
    _corrupt(pack[0], tmp_path / 'model.ejpack', 'section')
    with pytest.raises(C.PackError, match='sha256'):
        ej.load(str(tmp_path))  # a directory holding model.ejpack


def test_known_state_key_with_other_contents_is_refused(tmp_path):
    from ej import integrity
    from ej.pack.load import check_known
    key = next(iter(integrity.KNOWN_PACKS)) + '0' * 48
    SP.write(tmp_path / 'm.ejpack', key=key)
    with pytest.raises(ValueError, match='content digest'):
        check_known(C.Reader(str(tmp_path / 'm.ejpack')))
    SP.write(tmp_path / 'u.ejpack')
    assert 'not in ej.integrity.KNOWN_PACKS' in check_known(C.Reader(str(tmp_path / 'u.ejpack')))[0]


def test_writer_refuses_a_mismatched_encoder(tmp_path):
    from ej.pack import write as W
    st, ck = SP.state(), SP.checkpoint()
    ck['variant'] = 'w2'
    with pytest.raises(ValueError, match='needs encoder'):
        W.write(str(tmp_path / 'x.ejpack'), SP.payload(), st, ck, SP.tok_files())


def _encoders(path):
    _, _, ck, r = RD.read(str(path), unpack=False)
    cfg = SP.config()
    return ck, EN.build(ck, cfg, 'fp32'), EN.build(ck, cfg, 'stream'), EN.build(ck, cfg, 'stream', lru=4)


def test_low_memory_encoder_is_bit_identical(pack):
    path, _, _, ck_ref = pack
    _, fp32, stream, cached = _encoders(path)
    for n, c in ck_ref['codes'].items():
        assert same(fp32.get_parameter(n), dequant(c)), n
    ids = torch.tensor([[2, 5, 9, 31, 33, 3, 0], [2, 7, 7, 3, 0, 0, 0]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        a = fp32(input_ids=ids, attention_mask=mask).last_hidden_state
        for m in (stream, cached, cached):
            assert same(m(input_ids=ids, attention_mask=mask).last_hidden_state, a)
    assert cached.lru.hits > 0
    held = sum(t.numel() * t.element_size() for t in stream.buffers()) + sum(p.numel() * 4 for p in stream.parameters())
    assert held < sum(p.numel() * 4 for p in fp32.parameters()) / 3  # codes, not fp32 weights


def test_int8_heads_are_swapped_only_when_exact(pack):
    _, st, _, _ = RD.read(str(pack[0]))
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(3))
    before = st['rich'](x)
    assert HD.install(st) == {'rich': (2, 0)}
    assert isinstance(st['rich'][0], HD.Int8Linear) and same(st['rich'](x), before)
    lin = torch.nn.Linear(16, 4)  # random weights: not int8-exact, kept as fp32
    assert HD._swap(torch.nn.Sequential(lin)) == (0, 1)


def test_no_unpickling_while_reading_building_and_running(pack, tmp_path):
    code = f"""
import json, sys, pickle, torch
calls = []
sys.addaudithook(lambda ev, args: calls.append(ev) if ev == 'pickle.find_class' else None)
real = (torch.load, pickle.load, pickle.loads)
def counting(f):
    def g(*a, **k):
        calls.append(f.__name__); return f(*a, **k)
    return g
torch.load, pickle.load, pickle.loads = (counting(f) for f in real)
from transformers import AutoConfig, BertTokenizerFast
import synthpack as SP
from ej.pack import encoder as EN, read as RD, tok as TK
for low in (False, True):
    _, st, ck, r = RD.read({str(pack[0])!r}, unpack=False)
    d = TK.materialise({{n: r.section(n) for n in TK.SECTIONS}}, {str(tmp_path)!r})
    m = EN.Infer(EN.build(ck, AutoConfig.from_pretrained(d), 'stream' if low else 'fp32'))
    b = BertTokenizerFast(d + '/vocab.txt', do_lower_case=True)(['the late order', 'refund please'], padding=True,
                                                                 return_tensors='pt')
    h = m(input_ids=b['input_ids'], attention_mask=b['attention_mask']).last_hidden_state
    st['rich'](torch.zeros(1, 16))
print(json.dumps(dict(calls=calls, shape=list(h.shape))))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, 'tests')]), HF_HUB_OFFLINE='1')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r == {'calls': [], 'shape': [2, 5, 128]}


def test_low_memory_needs_a_pack(tmp_path):
    import ej
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='low_memory=True needs a packed model'):
        ej.load(str(tmp_path), low_memory=True)
    assert not RD.is_pack(str(tmp_path)) and RD.locate(str(tmp_path)) is None
