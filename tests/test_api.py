"""ej tests that need no weights: public API, record validation, integrity refusals, codec refusals, hf_layout refusal."""
import copy
import json
import os
import shutil
import subprocess
import sys

import pytest

import ej
from ej import integrity, loader, safe
from ej.safe_codec import CodecError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_public_api():
    assert ej.__version__ == '0.0.1.post2'
    assert callable(ej.load) and hasattr(ej.Model, 'predict') and hasattr(ej.Model, 'adapt')
    assert hasattr(ej.AdaptedModel, 'observe') and hasattr(ej.AdaptedModel, 'predict')
    assert ej.validate_records([ej.EXAMPLE_RECORD]) == [ej.EXAMPLE_RECORD]


@pytest.mark.parametrize('mutate, message', [
    (lambda r: r.pop('state'), 'state must be a string'),
    (lambda r: r.update(questions={}), 'non-empty'),
    (lambda r: r['questions']['route'].update(type='multi'), 'type must be one of'),
    (lambda r: r['questions']['route'].update(options=r['questions']['route']['options'][:1]), 'at least 2'),
    (lambda r: r['questions']['route']['options'][1].update(key='returns'), 'unique'),
    (lambda r: r['questions']['needs_human']['options'][0].update(key='no'), "'false' and 'true'"),
    (lambda r: r['questions']['urgency']['options'][0].pop('text'), 'an option is'),
])
def test_record_validation_refuses(mutate, message):
    r = copy.deepcopy(ej.EXAMPLE_RECORD)
    mutate(r)
    with pytest.raises(ej.RecordError, match=message):
        ej.validate_records([r])


def test_records_must_be_a_list():
    with pytest.raises(ej.RecordError):
        ej.validate_records(ej.EXAMPLE_RECORD)


def test_runtime_manifest_matches_and_detects_tampering(tmp_path):
    assert integrity.verify_runtime() == 39
    rt = tmp_path / 'rt'
    shutil.copytree(integrity.RUNTIME, rt)
    with open(rt / 'student.py', 'a') as f:
        f.write('# edited\n')
    with pytest.raises(ValueError, match='sha256'):
        integrity.verify_runtime(str(rt))


def test_check_sha_refuses_tampered_file(tmp_path):
    p = tmp_path / 'w.bin'
    p.write_bytes(b'weights')
    good = integrity.file_sha256(p)
    assert integrity.check_sha(p, good, 'file') == good
    p.write_bytes(b'weights!')
    with pytest.raises(ValueError, match='refusing'):
        integrity.check_sha(p, good, 'file')


def _fake_weights(d, files_sha=None):
    (d / 'encoder').mkdir()
    for rel in ('state.safetensors', 'state.json.gz', 'encoder/w23.safetensors', 'encoder/w23.json'):
        (d / rel).write_bytes(b'not real weights')
    cfg = {'runtime_sha256': integrity.runtime_sha256(), 'e5_revision': loader.E5_REVISION, 'state_key': '0' * 64,
           'files': files_sha or {rel: integrity.file_sha256(d / rel) for rel in
                                  ('state.safetensors', 'state.json.gz', 'encoder/w23.safetensors', 'encoder/w23.json')}}
    (d / 'config.json').write_text(json.dumps(cfg))
    return cfg


def test_load_refuses_without_config(tmp_path):
    with pytest.raises(ValueError, match='config.json'):
        ej.load(str(tmp_path))


def test_load_refuses_tampered_weights_before_decoding(tmp_path):
    cfg = _fake_weights(tmp_path)
    (tmp_path / 'state.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sha256'):
        ej.load(str(tmp_path))
    assert cfg['files']


def test_load_refuses_other_runtime_or_e5_revision(tmp_path):
    cfg = _fake_weights(tmp_path)
    for field, value, msg in (('runtime_sha256', 'f' * 64, 'runtime'), ('e5_revision', 'main', 'e5 revision')):
        bad = dict(cfg, **{field: value})
        (tmp_path / 'config.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=msg):
            ej.load(str(tmp_path))


def _export_small(tmp_path, obj):
    prefix = str(tmp_path / 'obj')
    safe.export(obj, prefix)
    return prefix


def test_codec_roundtrip_and_refusals(tmp_path):
    import torch
    obj = {'a': torch.arange(6, dtype=torch.float32).reshape(2, 3), 'b': [1, 2.5, 'x', None], 'm': torch.nn.Linear(2, 2)}
    prefix = _export_small(tmp_path, obj)
    back = safe.load(prefix)
    assert torch.equal(back['a'], obj['a']) and back['b'] == obj['b']
    assert torch.equal(back['m'].weight, obj['m'].weight)
    with pytest.raises(CodecError, match='allowlist'):
        safe.load(prefix, allow=frozenset())
    with open(prefix + '.safetensors', 'ab') as f:
        f.write(b'\0')
    with pytest.raises(CodecError, match='sha256'):
        safe.load(prefix)


def test_codec_refuses_callables(tmp_path):
    with pytest.raises(CodecError, match='unsupported'):
        safe.export({'f': lambda x: x}, str(tmp_path / 'bad'))


def test_hf_layout_refuses_inside_git_repo(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git not installed')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'hf_layout.py'), str(tmp_path / 'out'),
                        '--from-safe', str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and 'inside a git work tree' in r.stderr
    assert not (tmp_path / 'out').exists()


def test_cold_switch_maps_the_public_name(monkeypatch):
    monkeypatch.delenv('EDGE_COLD', raising=False)
    monkeypatch.setenv('EJ_COLD', '1')
    with loader.cold():
        assert os.environ.get('EDGE_COLD') == '1'
    assert 'EDGE_COLD' not in os.environ
    monkeypatch.setenv('EJ_COLD', '0')
    with loader.cold():
        assert 'EDGE_COLD' not in os.environ


def test_cache_dir_public_name(monkeypatch, tmp_path):
    monkeypatch.setenv('EJ_CACHE', str(tmp_path))
    assert loader.cache_dir() == str(tmp_path)
