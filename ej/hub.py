"""Fetch ej weights from the Hugging Face Hub: `ej.load('5ak3t/ej', revision='v0.0.1')`.

A repo id resolves to its packed model file: only `model.ejpack` is downloaded (hf_hub_download, into the Hugging Face cache,
$HF_HOME), and before it is returned its sha256 is checked against ej.integrity.KNOWN_PACK_FILES when the pack's state key is
listed there (ej.pack.load then checks every section and the content digest against ej.integrity.KNOWN_PACKS). A repo without
a model.ejpack falls back to a snapshot of the whole repo (the weights-directory format). Pin `revision` to a tag or a commit
for reproducibility; HF_HUB_OFFLINE=1 serves a filled cache without network."""
import os
import sys

from . import integrity

PACK_FILE = 'model.ejpack'
DEFAULT_REPO = '5ak3t/ej'  # the published 0.0.1 weights (revision v0.0.1)


def check_pack_file(path):
    """Raise ValueError when path is a pack of a known state whose file sha256 differs from KNOWN_PACK_FILES; returns its
    sha256. A pack of a state not listed there is returned with a warning (ej.pack.load checks its sections)."""
    from .pack import container as C
    got = integrity.file_sha256(path)
    r = C.Reader(path)
    try:
        key = str(r.meta.get('state_key', ''))[:16]
    finally:
        r.close()
    want = integrity.KNOWN_PACK_FILES.get(key)
    if want is None:
        print(f'ej: warning: {path}: state {key} is not in ej.integrity.KNOWN_PACK_FILES: file sha256 not checked',
              file=sys.stderr)
    elif got != want:
        raise ValueError(f'{path}: sha256 {got} != known {want} for state {key}; refusing to load')
    return got


def download(repo_id, revision=None):
    """Local path of repo_id's model.ejpack (downloaded once into the Hugging Face cache and sha256-checked), or of a
    snapshot of the repo when it holds no model.ejpack (weights-directory format)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import RemoteEntryNotFoundError
    try:
        path = hf_hub_download(repo_id=repo_id, filename=PACK_FILE, revision=revision)
    except RemoteEntryNotFoundError:
        return snapshot_download(repo_id=repo_id, revision=revision)
    check_pack_file(path)
    return os.path.abspath(path)
