#!/usr/bin/env python3
"""Run models on one benchmark suite and score them with benchmarks/scoring.py (check, rows, summary).

usage: python benchmarks/run_bench.py SUITE_PATH MODEL [MODEL ...] [--out DIR] [--sample N] [--final]
  SUITE_PATH  a suite JSONL in the ej input format with 'gold' answers (gold is stripped before any model sees a record)
  MODEL       a name from benchmarks/rivals/__init__.py RIVALS ('ej' = this package); run each in its own environment
  --sample N  N records spread over the suite's sources (deterministic hash order); for adapter smoke tests
  --final     required for a suite under $BENCH_FINAL_DIR (sealed test suites: run once per release, never for tuning)
Writes DIR/<suite>.<model>.preds.jsonl ({'id', 'source', 'latency_ms', 'probs'}) and DIR/<suite>.<model>.summary.json;
prints one JSON line per model. Only the summary files are aggregates; keep the preds files private when the suite is.
Timing (one protocol for every model): model load + first record (warm-up, not timed), then wall-clock ms per
record, one record per call; torch threads = BENCH_THREADS (default 1). The summary records the threads observed inside the
timed calls (`threads_measured`; `threads_ok` is False when they differ from BENCH_THREADS), the box (CPU model, cores) and
the 1-minute load average before and after the timed loop, so latencies from different boxes or loads are not ranked
together. EJ_COLD=1 (ej only) bypasses the prediction-time caches: `cache` = 'cold' in the summary."""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import scoring  # noqa: E402


def is_final(path):
    """True when path lies under $BENCH_FINAL_DIR (realpath comparison)."""
    final = os.environ.get('BENCH_FINAL_DIR')
    if not final:
        return False
    real, d = os.path.realpath(path), os.path.realpath(final)
    return real == d or real.startswith(d + os.sep)


def sample(recs, n):
    """n records, round-robin over sources (source, then workflow), each source in seeded hash order."""
    by = {}
    for r in sorted(recs, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest()):
        by.setdefault((r.get('source'), r.get('workflow')), []).append(r)
    keys, out, i = sorted(by, key=str), [], 0
    while len(out) < min(n, len(recs)):
        for k in keys:
            if i < len(by[k]) and len(out) < n:
                out.append(by[k][i])
        i += 1
    return out


def box():
    """{'cpu': model name, 'cores': logical CPUs, 'python': version} of this machine."""
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next((ln.split(':', 1)[1].strip() for ln in f if ln.startswith('model name')), cpu)
    except OSError:
        pass
    return {'cpu': cpu, 'cores': os.cpu_count(), 'python': platform.python_version()}


def loadavg():
    """1-minute load average (None where the OS has none)."""
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def run(recs, model, out_dir, tag):
    """Predict recs one record per call, check and score them; writes the preds and summary files; returns the summary."""
    import rivals
    blind = [scoring.blind(r) for r in recs]
    t0 = time.time()
    predict = rivals.get(model)
    first = predict(blind[:1])  # loads the model + warm-up (not timed)
    load_s = time.time() - t0
    preds, lat, seen = list(first), [None], set()
    want, load_before = int(os.environ.get('BENCH_THREADS', '1')), loadavg()
    for b in blind[1:]:  # one record per call: per-record latency
        t1 = time.perf_counter()
        preds += predict([b])
        lat.append((time.perf_counter() - t1) * 1000)
        seen.add(rivals.threads_used(model))
    timed = sorted(x for x in lat if x is not None)
    for p, r in zip(preds, recs):
        scoring.check(p, r)
    with open(os.path.join(out_dir, f'{tag}.{model}.preds.jsonl'), 'w') as f:
        for p, r, ms in zip(preds, recs, lat):
            f.write(json.dumps({'id': r['id'], 'source': r.get('source'), 'latency_ms': ms, 'probs': p}) + '\n')
    line = {'rival': model, 'suite': tag, 'records': len(recs), **scoring.summary(scoring.rows(recs, preds)),
            'ms_per_record': round(sum(timed) / max(len(timed), 1), 1),
            'p50_ms': round(timed[len(timed) // 2], 1) if timed else None, 'load_plus_warmup_s': round(load_s, 1),
            'threads': want, 'threads_measured': sorted(seen, key=str),
            'threads_ok': seen <= {want, None}, 'cache': 'cold' if os.environ.get('EJ_COLD') == '1' else 'warm',
            'box': box(), 'loadavg_1m': [load_before, loadavg()]}
    with open(os.path.join(out_dir, f'{tag}.{model}.summary.json'), 'w') as f:
        json.dump(line, f, indent=1)
    return line


def main():
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('suite')
    ap.add_argument('models', nargs='+')
    ap.add_argument('--out', default='bench-out')
    ap.add_argument('--sample', type=int)
    ap.add_argument('--final', action='store_true')
    a = ap.parse_args()
    if is_final(a.suite) and not a.final:
        sys.exit(f'refusing to run on {a.suite}: suites under $BENCH_FINAL_DIR need --final')
    if a.final and not is_final(a.suite):
        sys.exit('--final given for a suite outside $BENCH_FINAL_DIR')
    from rivals import RIVALS
    unknown = [m for m in a.models if m not in RIVALS]
    if unknown:
        sys.exit(f'unknown model(s) {unknown}; known: {sorted(RIVALS)}')
    recs = scoring.load(a.suite)
    if a.sample:
        recs = sample(recs, a.sample)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.suite))[0] + (f'-s{a.sample}' if a.sample else '')
    for model in a.models:
        print(json.dumps(run(recs, model, a.out, tag)), flush=True)


if __name__ == '__main__':
    main()
