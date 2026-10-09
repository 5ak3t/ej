"""ej.load with a Hugging Face repo id (ej.hub), without network: the Hub calls are replaced by local copies of a SMALL
synthetic pack. Falsified if: a repo id does not resolve to its model.ejpack alone (at the requested revision); a pack of a
known state whose file sha256 differs from ej.integrity.KNOWN_PACK_FILES is returned; a repo without model.ejpack is not
fetched as a snapshot; a local path reaches the Hub."""
import os
import subprocess
import sys

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError

import synthpack as SP
from ej import hub, integrity, loader


def fake_download(src, calls):
    def hf_hub_download(repo_id, filename, revision=None, **kw):
        calls.append((repo_id, filename, revision))
        return str(src)
    return hf_hub_download


def test_repo_id_downloads_only_the_pack(tmp_path, monkeypatch, capsys):
    SP.write(tmp_path / 'model.ejpack')
    calls = []
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download(tmp_path / 'model.ejpack', calls))
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kw: pytest.fail('snapshot of a repo with a pack'))
    got = loader.resolve('someone/some-ej', revision='v0.0.1')
    assert got == str(tmp_path / 'model.ejpack')
    assert calls == [('someone/some-ej', 'model.ejpack', 'v0.0.1')]
    assert 'not in ej.integrity.KNOWN_PACK_FILES' in capsys.readouterr().err  # synthetic state: warned, not refused


def test_known_state_with_other_file_bytes_is_refused(tmp_path, monkeypatch):
    key = next(iter(integrity.KNOWN_PACK_FILES)) + '0' * 48
    SP.write(tmp_path / 'model.ejpack', key=key)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download(tmp_path / 'model.ejpack', []))
    with pytest.raises(ValueError, match='sha256 .* != known'):
        hub.download(hub.DEFAULT_REPO, revision='v0.0.1')


def test_repo_without_pack_falls_back_to_a_snapshot(tmp_path, monkeypatch):
    def missing(**kw):
        response = httpx.Response(404, request=httpx.Request('GET', 'https://huggingface.co/x/resolve/main/model.ejpack'))
        raise RemoteEntryNotFoundError('no model.ejpack', response=response)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', missing)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda repo_id, revision=None: f'/snap/{repo_id}@{revision}')
    assert hub.download('someone/weights-dir', 'abc123') == '/snap/someone/weights-dir@abc123'


def test_local_paths_never_reach_the_hub(tmp_path, monkeypatch):
    SP.write(tmp_path / 'model.ejpack')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kw: pytest.fail('local path went to the Hub'))
    assert loader.resolve(str(tmp_path / 'model.ejpack')) == str(tmp_path / 'model.ejpack')
    assert loader.resolve(str(tmp_path)) == str(tmp_path)


def test_release_constants_agree():
    assert hub.DEFAULT_REPO == '5ak3t/ej' and hub.PACK_FILE == 'model.ejpack'
    assert set(integrity.KNOWN_PACK_FILES) == set(integrity.KNOWN_PACKS)
    assert all(len(v) == 64 for v in integrity.KNOWN_PACK_FILES.values())


def test_hf_layout_builds_a_pack_upload_directory(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SP.write(tmp_path / 'model.ejpack')
    out = tmp_path / 'hf'
    r = subprocess.run([sys.executable, os.path.join(root, 'scripts', 'hf_layout.py'), str(out), '--pack',
                        str(tmp_path / 'model.ejpack')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    files = sorted(os.path.relpath(os.path.join(d, n), out) for d, _, ns in os.walk(out) for n in ns)
    assert files == ['LICENSES/CC-BY-SA-4.0.txt', 'LICENSES/MIT-e5-small-v2.txt', 'NOTICE', 'README.md', 'SHA256SUMS',
                     'model.ejpack']
    sums = dict(line.split()[::-1] for line in (out / 'SHA256SUMS').read_text().splitlines())
    assert sorted(sums) == [f for f in files if f != 'SHA256SUMS']
    assert all(integrity.file_sha256(str(out / f)) == h for f, h in sums.items())
    card = (out / 'README.md').read_text()
    assert card.startswith('---\nlicense: cc-by-sa-4.0\nbase_model: intfloat/e5-small-v2\n')
    for line in ('library_name: ej', 'pipeline_tag: text-classification', '- zero-shot-classification', '- en'):
        assert line in card.split('\n---\n')[0], line
