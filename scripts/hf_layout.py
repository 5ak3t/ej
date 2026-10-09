#!/usr/bin/env python3
"""Build the Hugging Face upload directory for ej weights. Builds files only: it never uploads anything.

usage: python scripts/hf_layout.py OUT_DIR --from-safe DIR [--model-commit SHA] [--remote origin]
  --from-safe DIR   an ej weights directory (`python -m ej.train fit` output: state.*, encoder/w23.*, config.json), or an
                    export directory holding state-<key16>.json + .safetensors and lowbit-<id>/w23.json + .safetensors
  --model-commit    for weights fitted outside this repository: the model-code commit whose student*/train_* files + pool
                    reproduce the state key (scripts/state_key.py); recorded as `model_commit`
Provenance: config.json `code_commit` is this repository's HEAD, which must be clean AND contained in a branch
of the remote (`git ls-remote`), so the named commit can be fetched by anyone; otherwise the build refuses.
OUT_DIR must be new or empty and must not be inside a git work tree (weights never go into git). It receives:
  state.safetensors, state.json.gz, encoder/w23.safetensors, encoder/w23.json   (the weights; no pickle)
  config.json   ej version, code commit, runtime sha256, e5 model + revision, state key, sha256 of every weights file
  README.md     the model card (--card, default docs/model-cards/ej-0.0.1.md) with Hugging Face YAML front matter
  NOTICE, LICENSES/*, SHA256SUMS (sha256sum of every other file)"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ej import integrity, loader, safe  # noqa: E402

FRONT_MATTER = """---
license: cc-by-sa-4.0
base_model: intfloat/e5-small-v2
library_name: ej
language:
- en
pipeline_tag: text-classification
tags:
- typed-decisions
- calibration
- on-device
- safetensors
---

"""


def inside_git(path):
    """True when path (or its nearest existing parent) is inside a git work tree."""
    p = os.path.realpath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    while True:
        if os.path.exists(os.path.join(p, '.git')):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def write_gz(src, dst):
    """Deterministic gzip of src (no file name, mtime 0, level 9)."""
    with open(src, 'rb') as fi, open(dst, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)


def from_safe(src):
    """(state prefix, encoder prefix, extra config) of an ej weights directory or of an export directory."""
    if os.path.exists(os.path.join(src, 'config.json')):
        cfg = loader.read_config(src)
        loader.verify_weights(src, cfg)  # every file's sha256 and the runtime, before anything is copied
        keep = {k: cfg[k] for k in ('pool_sha256', 'encoder', 'dropped_parts') if k in cfg}
        return os.path.join(src, 'state'), os.path.join(src, 'encoder', 'w23'), keep
    st = sorted(glob.glob(os.path.join(src, 'state-*.safetensors')))
    w = sorted(glob.glob(os.path.join(src, 'lowbit-*', 'w23.safetensors')))
    if len(st) != 1 or len(w) != 1:
        sys.exit(f'hf_layout: {src} must be an ej weights directory or hold one state-*.safetensors and one lowbit-*/w23.safetensors')
    return st[0][:-len('.safetensors')], w[0][:-len('.safetensors')], {}


def _git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)


def pushed_commit(repo=ROOT, remote='origin'):
    """HEAD of `repo` when the work tree is clean and HEAD is contained in a branch of `remote`; else (None, reason)."""
    head = _git(repo, 'rev-parse', 'HEAD')
    if head.returncode:
        return None, 'not a git repository'
    head = head.stdout.strip()
    if _git(repo, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        return None, f'{repo} has uncommitted changes'
    heads = _git(repo, 'ls-remote', '--heads', remote)
    if heads.returncode:
        return None, f'cannot list the branches of remote {remote!r}: {heads.stderr.strip()[:200]}'
    for line in heads.stdout.splitlines():
        sha, ref = line.split()
        if _git(repo, 'merge-base', '--is-ancestor', head, sha).returncode == 0:
            return head, ref
    return None, f'HEAD {head[:12]} is in no branch of remote {remote!r}; push it first'


def ej_version():
    """__version__ from ej/__init__.py (read as text)."""
    with open(os.path.join(ROOT, 'ej', '__init__.py')) as f:
        return re.search(r"__version__ = '([^']+)'", f.read()).group(1)


def runtime_source():
    """The source commit recorded in the runtime manifest header."""
    with open(integrity.MANIFEST) as f:
        m = re.search(r'commit ([0-9a-f]{7,40})', f.read())
    return m.group(1) if m else 'unknown'


def place(state_prefix, w23_prefix, out):
    """Copy the weights into the clean names; returns (state key, {rel: sha256 of skeleton JSON bytes})."""
    os.makedirs(os.path.join(out, 'encoder'), exist_ok=True)
    shutil.copyfile(state_prefix + '.safetensors', os.path.join(out, 'state.safetensors'))
    shutil.copyfile(w23_prefix + '.safetensors', os.path.join(out, 'encoder', 'w23.safetensors'))
    shutil.copyfile(safe.skeleton_path(w23_prefix), os.path.join(out, 'encoder', 'w23.json'))
    raw = safe.read_skeleton_bytes(state_prefix)
    with tempfile.NamedTemporaryFile(dir=out, delete=False) as t:
        t.write(raw)
    write_gz(t.name, os.path.join(out, 'state.json.gz'))
    os.remove(t.name)
    sk = {}
    for prefix in ('state', 'encoder/w23'):
        body = safe.read_skeleton_bytes(os.path.join(out, prefix))
        doc = json.loads(body)
        if integrity.file_sha256(os.path.join(out, prefix + '.safetensors')) != doc['safetensors_sha256']:
            sys.exit(f'hf_layout: {prefix}.safetensors does not match the sha256 recorded in its skeleton')
        sk[prefix + '.json'] = hashlib.sha256(body).hexdigest()
        if prefix == 'state':
            key = doc['root']['d']['key']
    return key, sk


def main():
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--from-safe', required=True)
    ap.add_argument('--card', default=os.path.join(ROOT, 'docs', 'model-cards', 'ej-0.0.1.md'))
    ap.add_argument('--model-commit')
    ap.add_argument('--remote', default='origin')
    a = ap.parse_args()
    out = os.path.realpath(a.out)
    if inside_git(out):
        sys.exit(f'hf_layout: {out} is inside a git work tree; refusing (weights never go into git)')
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f'hf_layout: {out} is not empty; refusing')
    commit, where = pushed_commit(ROOT, a.remote)
    if commit is None:
        sys.exit(f'hf_layout: code_commit not reachable: {where}; refusing')
    os.makedirs(out, exist_ok=True)
    state_prefix, w23_prefix, extra = from_safe(a.from_safe)
    key, sk = place(state_prefix, w23_prefix, out)
    weights = ['state.safetensors', 'state.json.gz', 'encoder/w23.safetensors', 'encoder/w23.json']
    cfg = {'ej_version': ej_version(), 'format': 'rev-safe-v1', 'code_repo': 'https://github.com/5ak3t/ej',
           'code_commit': commit, 'code_branch': where, **({'model_commit': a.model_commit} if a.model_commit else {}), **extra,
           'runtime_sha256': integrity.runtime_sha256(), 'runtime_source_commit': runtime_source(),
           'e5_model': loader.E5_MODEL, 'e5_revision': loader.E5_REVISION, 'state_key': key,
           'layout': {'state': 'state', 'encoder': 'encoder/w23'},
           'files': {rel: integrity.file_sha256(os.path.join(out, rel)) for rel in weights},
           'skeleton_json_sha256': sk}
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')
    with open(a.card) as f:
        card = f.read()
    with open(os.path.join(out, 'README.md'), 'w') as f:
        f.write(FRONT_MATTER + card)
    shutil.copyfile(os.path.join(ROOT, 'NOTICE'), os.path.join(out, 'NOTICE'))
    shutil.copytree(os.path.join(ROOT, 'LICENSES'), os.path.join(out, 'LICENSES'))
    files = sorted(os.path.relpath(os.path.join(d, n), out) for d, _, ns in os.walk(out) for n in ns)
    with open(os.path.join(out, 'SHA256SUMS'), 'w') as f:
        for rel in files:
            f.write(f'{integrity.file_sha256(os.path.join(out, rel))}  {rel}\n')
    print(f'hf_layout: {out}: {len(files) + 1} files, state {key[:16]}, runtime {cfg["runtime_sha256"][:16]}; '
          'nothing was uploaded')
    for rel in weights:
        print(f'  {cfg["files"][rel]}  {rel}  ({os.path.getsize(os.path.join(out, rel))} bytes)')


if __name__ == '__main__':
    main()
