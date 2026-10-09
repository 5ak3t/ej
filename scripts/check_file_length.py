#!/usr/bin/env python3
"""Fail when a tracked text file is longer than MAX lines (licence texts excepted). usage: python scripts/check_file_length.py"""
import os
import subprocess
import sys

MAX = 300
EXEMPT = ('LICENSE', 'LICENSES/')


def main():
    """Print every file over the limit and exit 1 if there is one."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        files = subprocess.check_output(['git', '-C', root, 'ls-files', '--cached', '--others', '--exclude-standard'],
                                        text=True).split()
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(d, n), root) for d, _, ns in os.walk(root) if '.git' not in d for n in ns]
    bad = []
    for rel in files:
        if rel.startswith(EXEMPT):
            continue
        try:
            with open(os.path.join(root, rel), encoding='utf-8') as f:
                n = sum(1 for _ in f)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if n > MAX:
            bad.append((n, rel))
    for n, rel in sorted(bad, reverse=True):
        print(f'{rel}: {n} lines > {MAX}')
    print(f'check_file_length: {len(files)} files, {len(bad)} over {MAX} lines')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
