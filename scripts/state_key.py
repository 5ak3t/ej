#!/usr/bin/env python3
"""Recompute a released state key from the model-code tree it was fitted with and its training pool (provenance).

usage: python scripts/state_key.py CODE_DIR POOL_JSONL [--expect KEY_OR_PREFIX] [--ck-dir lowbit-<id>]
       python scripts/state_key.py --git REPO COMMIT POOL_JSONL [--expect ...]   (reads REPO's edge/ at COMMIT via git)
--ck-dir: set student_lb.py's encoder switch `CK_DIR = '...'` to this directory first (exactly one such line must exist):
the one substitution the 0.0.1 fit tree made (f46cf7c code, encoder lowbit-b3b010513f948ceb).
The key (student_state.state_key of that tree) = sha256 over VERSION + NUL, then for every edge/student*.py and
edge/train_*.py file sorted by basename: basename + NUL + sha256(bytes), then 'pool' + NUL + the pool file's bytes. VERSION
is read from that tree's student_state.py. Exit status 1 when --expect is given and does not match the key's prefix.
For ej 0.0.1 (state 3b3e66d28fb423f9): model-code commit f46cf7c of the maintainers' development repository (not public)
with --ck-dir lowbit-b3b010513f948ceb + its training pool (sha256 ce1c1a6f...). Weights trained with `python -m ej.train`
carry a key over this repository's code instead (ej.train.export.state_key)."""
import argparse
import hashlib
import re
import subprocess
import sys


def key_from_files(files, pool_bytes_iter):
    """files: {basename: bytes} of the tree's edge/*.py; pool_bytes_iter: iterable of pool byte chunks. -> 64-hex key."""
    m = re.search(r"^VERSION = '([^']+)'", files['student_state.py'].decode(), re.M)
    if not m:
        raise ValueError('student_state.py defines no VERSION')
    h = hashlib.sha256(m.group(1).encode() + b'\x00')
    for f in sorted(n for n in files if n.endswith('.py') and (n.startswith('student') or n.startswith('train_'))):
        h.update(f.encode() + b'\x00')
        h.update(hashlib.sha256(files[f]).digest())
    h.update(b'pool\x00')
    for b in pool_bytes_iter:
        h.update(b)
    return h.hexdigest()


def set_ck_dir(files, ck_dir):
    """files with student_lb.py's `CK_DIR = '<dir>'` line pointed at ck_dir (ValueError unless exactly one such line)."""
    src = files['student_lb.py'].decode()
    lines = re.findall(r"^CK_DIR = '([^']+)'", src, re.M)
    if len(lines) != 1:
        raise ValueError(f'student_lb.py: {len(lines)} CK_DIR lines, expected 1')
    return {**files, 'student_lb.py': src.replace(f"CK_DIR = '{lines[0]}'", f"CK_DIR = '{ck_dir}'", 1).encode()}


def files_from_dir(code_dir):
    """{basename: bytes} of the .py files directly in code_dir."""
    import os
    return {n: open(os.path.join(code_dir, n), 'rb').read() for n in os.listdir(code_dir) if n.endswith('.py')}


def files_from_git(repo, commit, sub='edge'):
    """{basename: bytes} of the .py files directly in <sub>/ at `commit` of the git repository `repo`."""
    names = subprocess.check_output(['git', '-C', repo, 'ls-tree', '--name-only', f'{commit}:{sub}'], text=True).split()
    return {n: subprocess.check_output(['git', '-C', repo, 'show', f'{commit}:{sub}/{n}']) for n in names if n.endswith('.py')}


def pool_chunks(path):
    """The pool file in 1 MiB chunks."""
    with open(path, 'rb') as fh:
        yield from iter(lambda: fh.read(1 << 20), b'')


def main(argv=None):
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('args', nargs='+')
    ap.add_argument('--git', action='store_true')
    ap.add_argument('--expect')
    ap.add_argument('--ck-dir')
    a = ap.parse_args(argv)
    if a.git:
        repo, commit, pool = a.args
        files = files_from_git(repo, commit)
    else:
        code_dir, pool = a.args
        files = files_from_dir(code_dir)
    if a.ck_dir:
        files = set_ck_dir(files, a.ck_dir)
    key = key_from_files(files, pool_chunks(pool))
    print(key)
    if a.expect and not key.startswith(a.expect):
        print(f'state_key: {key[:16]} != expected {a.expect[:16]}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
