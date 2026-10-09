"""Memory and per-record latency of a packed model, one configuration per fresh process (Linux: reads /proc/self/status).

  python benchmarks/pack_memory.py mem --weights model.ejpack --records recs.jsonl [--low-memory] [--threads 2] --out m.json
      RSS after `import torch` alone, after ej.load, peak during predict (VmHWM reset via /proc/self/clear_refs just before
      predict) and after; live tensor bytes (gc walk over every torch.Tensor, each storage counted once) after load and after
      predict. Predicts all records in one call. Set EJ_COLD=1 for cold runs.
  python benchmarks/pack_memory.py lat --weights model.ejpack --records td.jsonl [--low-memory] [--n 50] --out l.json
      predict([r]) at 1 thread for the first n records after one warm-up call on record n; wall clock per record.
The output JSON records the box (CPU model, logical cores, 1-minute load average), Python and torch versions. Records are
read blind (only id / state / questions are passed); the per-record predictions are not written."""
import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time


def status(key):
    """A /proc/self/status field in MB (VmRSS, VmHWM)."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 1024
    return None


def reset_peak():
    """Reset VmHWM to the current RSS (Linux >= 4.0); False when not permitted."""
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


def tensor_mb():
    """MB of every live CPU tensor storage reachable as a Python object (each storage counted once)."""
    import torch
    seen, tot = set(), 0
    for o in gc.get_objects():
        if isinstance(o, torch.Tensor) and not o.is_meta:
            try:
                st = o.untyped_storage()
            except RuntimeError:
                continue
            k = (st.data_ptr(), st.nbytes())
            if k[0] and k not in seen:
                seen.add(k)
                tot += k[1]
    return tot / 1e6


def box():
    """CPU model, logical cores, load average, versions."""
    import torch
    cpu = next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo') if line.startswith('model name')), '?')
    return {'cpu': cpu, 'logical_cores': os.cpu_count(), 'loadavg_1m': os.getloadavg()[0], 'python': platform.python_version(),
            'torch': torch.__version__, 'cold': os.environ.get('EJ_COLD') == '1'}


def blind(path, n=None):
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [{k: r[k] for k in ('id', 'state', 'questions') if k in r} for r in recs[:n]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=('mem', 'lat'))
    ap.add_argument('--weights', required=True)
    ap.add_argument('--records', required=True, help='JSONL records (gold, if present, is dropped)')
    ap.add_argument('--low-memory', action='store_true')
    ap.add_argument('--threads', type=int, default=2, help='mem: predict threads (lat always uses 1)')
    ap.add_argument('--n', type=int, default=50, help='lat: records timed')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    import torch  # noqa: F401
    out = {'mode': a.mode, 'low_memory': a.low_memory, 'rss_mb_torch': status('VmRSS')}
    import ej
    t = time.perf_counter()
    m = ej.load(a.weights, low_memory=a.low_memory)
    out.update(load_s=time.perf_counter() - t, rss_mb_load=status('VmRSS'), peak_mb_load=status('VmHWM'))
    if a.mode == 'mem':
        recs = blind(a.records)
        out['tensor_mb_load'] = tensor_mb()
        out['peak_reset'] = reset_peak()
        m.predict(recs, threads=a.threads)
        out.update(records=len(recs), threads=a.threads, rss_mb_peak_predict=status('VmHWM'), rss_mb_after=status('VmRSS'),
                   tensor_mb_after=tensor_mb())
    else:
        recs = blind(a.records, a.n + 1)
        m.predict([recs[a.n]], threads=1)  # warm-up
        ms = []
        for r in recs[:a.n]:
            t = time.perf_counter()
            m.predict([r], threads=1)
            ms.append((time.perf_counter() - t) * 1000)
        out.update(records=len(ms), threads_measured=[m.last_threads], ms_mean=statistics.mean(ms),
                   ms_median=statistics.median(ms))
    out['box'] = box()
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), file=sys.stderr)


if __name__ == '__main__':
    main()
