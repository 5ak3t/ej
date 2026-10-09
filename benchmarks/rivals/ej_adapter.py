"""ej itself, through its public API: ej.load(EJ_WEIGHTS, revision=EJ_REVISION).predict(records, threads=BENCH_THREADS).
EJ_WEIGHTS = a local model.ejpack or weights directory (required; the weights are not yet published) or a repo id; EJ_REVISION
pins a Hub revision. Threads are set INSIDE each predict call (ej scopes them to the call) and the count in
effect there is reported by threads_used(), so a latency labelled "1 thread" is checked, not assumed.
All questions of a record are answered in one pass; probabilities are used as returned (they already sum to 1).
EJ_COLD=1 in the environment bypasses ej's prediction-time caches (cold latency)."""
import functools
import os


@functools.lru_cache(maxsize=1)
def load():
    """The ej model named by EJ_WEIGHTS / EJ_REVISION (loaded once per process)."""
    import ej
    weights = os.environ.get('EJ_WEIGHTS')
    if not weights:
        raise RuntimeError('set EJ_WEIGHTS to a local model.ejpack (the weights are not yet published)')
    return ej.load(weights, revision=os.environ.get('EJ_REVISION'), threads=int(os.environ.get('BENCH_THREADS', '1')))


def predict(records):
    """[{qid: probs}] for records in the ej input format."""
    return load().predict(records)


def threads_used():
    """torch threads in effect inside the last predict call (None before the first call)."""
    return load().last_threads
