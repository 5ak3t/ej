"""The packed model end to end; runs only when EJ_WEIGHTS names a model.ejpack (or a directory holding one).
Optional: EJ_REFERENCE_WEIGHTS = a weights directory (safetensors + JSON) of the same state, for the cross-format parity test
(it builds from the base model, so it needs the Hugging Face cache or network); EJ_PARITY_RECORDS = a JSONL file of records
to compare on (default: 16 built-in records). Every prediction runs in a fresh interpreter, one at a time.
Falsified if: the pack and the weights directory, or the default and the low-memory runtime, give predictions that differ by
any amount (max |dp| must be exactly 0.0); loading or predicting from the pack touches the Hugging Face cache or the network;
the quickstart does not run from the pack."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get('EJ_WEIGHTS')
REFERENCE = os.environ.get('EJ_REFERENCE_WEIGHTS')
RECORDS = os.environ.get('EJ_PARITY_RECORDS')
pytestmark = pytest.mark.skipif(not WEIGHTS, reason='set EJ_WEIGHTS to a model.ejpack')

PREDICT = """
import json, os, sys
import ej
recs = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
m = ej.load(sys.argv[1], low_memory=sys.argv[3] == '1')
json.dump({'preds': m.predict(recs), 'info': {k: m.config.get(k) for k in ('format', 'state_key', 'low_memory')}},
          open(sys.argv[4], 'w'))
"""


def builtin_records():
    """16 records: the example record and 15 variants (other states, question subsets)."""
    import ej
    texts = ['The parcel never arrived and tracking has not moved for nine days.', 'I was billed twice this month.',
             'Can you change the delivery address on order 5521?', 'The app logs me out every few minutes.',
             'My new kettle leaks from the base; I want a replacement.']
    out = [ej.EXAMPLE_RECORD]
    for i in range(15):
        r = json.loads(json.dumps(ej.EXAMPLE_RECORD))
        r['id'] = f'builtin-{i}'
        r['state'] = json.dumps({'customer_tier': ['gold', 'silver', 'basic'][i % 3], 'message': texts[i % 5]})
        if i % 4 == 3:
            r['questions'].pop('urgency')
        out.append(r)
    return out


def run_predict(weights, records, out, low_memory=False, env=None):
    """Predictions of `weights` on `records` from a fresh interpreter: (preds, info)."""
    rec = str(out) + '.records.jsonl'
    with open(rec, 'w') as f:
        f.writelines(json.dumps({k: r[k] for k in ('id', 'state', 'questions') if k in r}) + '\n' for r in records)
    e = dict(os.environ, PYTHONPATH=ROOT, **(env or {}))
    p = subprocess.run([sys.executable, '-c', PREDICT, weights, rec, '1' if low_memory else '0', str(out)], env=e,
                       capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, p.stderr[-3000:]
    with open(out) as f:
        d = json.load(f)
    return d['preds'], d['info']


def max_dp(a, b):
    assert len(a) == len(b) and all(set(x) == set(y) for x, y in zip(a, b))
    assert all(len(x[q]) == len(y[q]) for x, y in zip(a, b) for q in x)
    return max(abs(u - v) for x, y in zip(a, b) for q in x for u, v in zip(x[q], y[q]))


@pytest.fixture(scope='module')
def records():
    if RECORDS:
        with open(RECORDS) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    else:
        recs = builtin_records()
    assert len(recs) >= 15
    return recs


@pytest.fixture(scope='module')
def packed(records, tmp_path_factory):
    """Default-runtime predictions from the pack, run offline with an empty Hugging Face home (nothing may be fetched)."""
    d = tmp_path_factory.mktemp('packed')
    hf = d / 'hf'
    hf.mkdir()
    preds, info = run_predict(WEIGHTS, records, d / 'p.json', env={'HF_HOME': str(hf), 'HF_HUB_OFFLINE': '1',
                                                                   'TRANSFORMERS_OFFLINE': '1', 'EJ_CACHE': str(d / 'c')})
    return preds, info, sorted(os.listdir(hf))


def test_pack_loads_offline_without_any_hub_file(packed, records):
    preds, info, hf_files = packed
    assert info == {'format': 'ejpack-v1', 'state_key': info['state_key'], 'low_memory': False}
    assert hf_files == [] and len(preds) == len(records)
    assert all(abs(sum(p) - 1.0) < 1e-9 for d in preds for p in d.values())


def test_low_memory_predictions_are_identical(packed, records, tmp_path):
    low, info = run_predict(WEIGHTS, records, tmp_path / 'low.json', low_memory=True)
    assert info['low_memory'] is True
    assert max_dp(packed[0], low) == 0.0


@pytest.mark.skipif(not REFERENCE, reason='set EJ_REFERENCE_WEIGHTS to a weights directory of the same state')
def test_pack_matches_the_weights_directory(packed, records, tmp_path):
    ref, info = run_predict(REFERENCE, records, tmp_path / 'ref.json')
    assert info['state_key'] == packed[1]['state_key']
    assert max_dp(packed[0], ref) == 0.0


def test_quickstart_runs_from_the_pack():
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'examples', 'quickstart.py'), '--weights', WEIGHTS],
                       env=dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE='1'), capture_output=True, text=True,
                       timeout=1800)
    assert p.returncode == 0, p.stderr[-3000:]
    for line in ('route (choice)', 'needs_human (noul)', 'urgency (score)', '<- argmax'):
        assert line in p.stdout


@pytest.mark.skipif(not REFERENCE, reason='set EJ_REFERENCE_WEIGHTS to a weights directory of the same state')
def test_export_of_the_weights_directory_reproduces_the_pack(tmp_path):
    from ej.pack import container as C
    from ej.pack.read import locate
    out = tmp_path / 'model.ejpack'
    p = subprocess.run([sys.executable, '-m', 'ej.train', 'export', '--weights', REFERENCE, '--out', str(out)],
                       env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stderr[-3000:]
    assert C.Reader(str(out)).digest == C.Reader(locate(WEIGHTS)).digest  # same contents; only 'source' may differ
