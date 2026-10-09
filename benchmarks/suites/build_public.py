#!/usr/bin/env python3
"""Rebuild the public benchmark suites (td, zs_td, zs_massive; development and final) from the public datasets.

usage: python benchmarks/suites/build_public.py RAW OUT [--split dev|final|both]
RAW holds the raw downloads: typed-decisions/all-train.parquet + all-test.parquet (LocalLLaMA/typed-decisions) and
massive/validation.parquet + test.parquet (mteb/amazon_massive_intent, en). OUT receives <split>/<suite>.jsonl; the sha256
of each file is printed and compared with SHA256SUMS (exit 1 on any mismatch). The ticket suites are built from a private
corpus and cannot be rebuilt here.
Construction (sources.py and leakfree.py are the converters that built the published suites; only path defaults, docstrings
and licence labels were edited):
  td    = Typed Decisions cases of the 3 training workflows: dev = train split, hash bucket 0 of 5 (sha256 of the id);
          final = test split.
  zs_td = workflow security_incidents: dev = train split, final = test split.
  zs_massive = MASSIVE validation (dev) / test (final), 1,000 records in seeded hash order, leak-free option sets
          (leakfree.py --mode cps --shuffle --n 1000).
Requires pyarrow and numpy. Final suites are sealed test sets: rebuild them to verify the checksums, not to tune on them."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ZS_WORKFLOW = 'security_incidents'


def bucket(rid, n=5):
    """Development hash bucket of a record id."""
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) % n


def typed_decisions(raw, split):
    """{suite: records} for td and zs_td of one split."""
    import pyarrow.parquet as pq
    import sources as S
    part = 'all-train' if split == 'dev' else 'all-test'
    rows = [S.typed_decisions(r) for r in pq.read_table(f'{raw}/typed-decisions/{part}.parquet').to_pylist()]
    seen = [r for r in rows if r['workflow'] != ZS_WORKFLOW]
    td = [r for r in seen if bucket(r['id']) == 0] if split == 'dev' else seen
    return {'td': td, 'zs_td': [r for r in rows if r['workflow'] == ZS_WORKFLOW]}


def zs_massive(raw, split):
    """The leak-free MASSIVE suite of one split."""
    import leakfree as L
    rows = L.load_rows(raw, 'massive', 'validation' if split == 'dev' else 'test', 1000, True)
    recs, _ = L.exchangeable_choice(rows, 'massive', L.INSTR['massive'], 2, 8, 'cps')
    return recs


def expected():
    """{'split/suite.jsonl': sha256} from SHA256SUMS."""
    out = {}
    with open(os.path.join(HERE, 'SHA256SUMS')) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                sha, name = line.split()
                out[name] = sha
    return out


def main(argv=None):
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('raw')
    ap.add_argument('out')
    ap.add_argument('--split', default='dev', choices=('dev', 'final', 'both'))
    a = ap.parse_args(argv)
    want, bad = expected(), 0
    for split in (('dev', 'final') if a.split == 'both' else (a.split,)):
        suites = dict(typed_decisions(a.raw, split), zs_massive=zs_massive(a.raw, split))
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        for name, recs in suites.items():
            rel = f'{split}/{name}.jsonl'
            path = os.path.join(a.out, rel)
            with open(path, 'w') as f:
                for r in recs:
                    f.write(json.dumps(r) + '\n')
            sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            ok = sha == want.get(rel)
            bad += not ok
            print(json.dumps({'suite': rel, 'records': len(recs), 'sha256': sha, 'matches_SHA256SUMS': ok}))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
