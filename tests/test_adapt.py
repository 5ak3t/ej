"""Workflow adaptation (ej.adapt, ej.adapt_math). The first tests use a stand-in model and need no weights; the parity tests at
the end run only when EJ_WEIGHTS names a model.ejpack (or a weights directory)."""
import copy
import hashlib
import json
import os

import pytest
import torch

import ej
from ej import adapt_math as AM
from ej.adapt import AdaptedModel, _rows

WEIGHTS = os.environ.get('EJ_WEIGHTS')
HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = [{'key': k, 'text': f'option {k}'} for k in ('a', 'b', 'c')]


class Fake:
    """Deterministic stand-in for ej.Model.predict: a softmax of hashed scores, biased toward each question's first option."""

    def __init__(self, nan_ids=()):
        self.nan_ids, self.calls, self.seen = set(nan_ids), 0, []

    def predict(self, records):
        self.calls += 1
        self.seen += records
        out = []
        for r in records:
            d = {}
            for qid, q in r['questions'].items():
                h = hashlib.sha256(f'{r["state"]}|{qid}'.encode()).digest()
                z = torch.tensor([h[j] / 64.0 + (1.5 if j == 0 else 0.0) for j in range(len(q['options']))], dtype=torch.float64)
                p = torch.softmax(z, -1).tolist()
                d[qid] = [float('nan')] * len(p) if r.get('id') in self.nan_ids else p
            out.append(d)
        return out


def rec(i, answer=None, n_opts=3, qid='q'):
    opts = [{'key': f'k{j}', 'text': f'option {j}'} for j in range(n_opts)] if n_opts != 3 else OPTS
    r = {'id': f'r{i}', 'state': f'request number {i}', 'questions': {qid: {'type': 'choice', 'instructions': 'Pick one.',
                                                                            'options': opts}}}
    if answer is not None:
        r['answers'] = {qid: answer}
    return r


def maxdiff(a, b):
    return max(abs(x - y) for u, v in zip(a, b) for q in u for x, y in zip(u[q], v[q]))


def test_no_examples_returns_the_model_itself():
    m = ej.Model(None, {}, None)
    assert m.adapt() is m and m.adapt([]) is m and m.adapt(examples=None) is m


def test_examples_without_answers_change_nothing():
    base, recs = Fake(), [rec(i) for i in range(6)]
    ad = AdaptedModel(base, examples=recs[:3])
    assert ad.offsets == {} and ad.n_labelled == 0 and ad.predict(recs[3:]) == base.predict(recs[3:])


def test_option_tilt_learns_the_label_prior():
    base = Fake()
    ex = [rec(i, 'c') for i in range(8)]
    ad = AdaptedModel(base, examples=ex)
    rows, lp = _rows(ex, base.predict(ex))
    assert ad.offsets == AM.fit_tilt(rows, [2] * 8, lp, sb=0.5)
    assert ad.offsets[('q', 'c')] > 0 > ad.offsets[('q', 'a')]
    p0, p1 = base.predict([rec(99)])[0]['q'], ad.predict([rec(99)])[0]['q']
    assert p1[2] > p0[2] and abs(sum(p1) - 1) < 1e-12


def test_the_base_model_never_sees_the_answers():
    base = Fake()
    AdaptedModel(base, examples=[rec(i, 'b') for i in range(4)])
    assert base.seen and all('answers' not in r and 'gold' not in r for r in base.seen)


def test_gold_format_and_bad_answers():
    r = rec(0)
    r['gold'] = {'q': {'label': 1}}
    assert ej.adapt.answers_of(r) == {'q': 1}
    with pytest.raises(ej.RecordError, match='not one of its option keys'):
        AdaptedModel(Fake(), examples=[rec(0, 'zzz')])


def test_observe_in_batches_equals_adapting_at_once():
    base = Fake()
    ex = [rec(i, 'abc'[i % 3] if i % 4 else 'c') for i in range(30)] + [rec(40 + i, 'k5', n_opts=6, qid='wide') for i in range(5)]
    once = AdaptedModel(base, examples=ex)
    online = AdaptedModel(base, examples=ex[:7])
    for s in range(7, len(ex), 9):
        online.observe(ex[s:s + 9])
    assert online.n_labelled == once.n_labelled == 35 and set(online.offsets) == set(once.offsets)
    assert max(abs(online.offsets[k] - once.offsets[k]) for k in once.offsets) < 1e-6
    q = [rec(90), rec(91, n_opts=6, qid='wide')]
    assert maxdiff(online.predict(q), once.predict(q)) < 1e-6


def test_numerical_edge_cases():
    base = Fake(nan_ids={'r3'})
    ex = [rec(i, 'b') for i in range(6)] + [rec(50, 'k16', n_opts=17, qid='big')]  # r3: NaN zero-shot output, skipped
    ad = AdaptedModel(base, examples=ex)
    assert ad.n_labelled == 6 and len([k for k in ad.offsets if k[0] == 'big']) == 17
    assert all(v == v and abs(v) != float('inf') for v in ad.offsets.values())
    for r, d in zip(ex, ad.predict(ex)):
        assert r['id'] == 'r3' or all(abs(sum(p) - 1) < 1e-9 for p in d.values())


def test_records_are_not_modified():
    ex = [rec(i, 'a') for i in range(3)]
    keep = copy.deepcopy(ex)
    AdaptedModel(Fake(), examples=ex).observe(ex).predict(ex)
    assert ex == keep


@pytest.fixture(scope='module')
def model():
    if not WEIGHTS:
        pytest.skip('set EJ_WEIGHTS to a model.ejpack (or a weights directory)')
    return ej.load(WEIGHTS)


@pytest.fixture(scope='module')
def reference(model):
    """(records, reference predictions for the loaded weights' state key); skips for weights without a reference."""
    with open(os.path.join(HERE, 'data', 'adapt_reference.json')) as f:
        d = json.load(f)
    key = str(model.config.get('state_key'))[:8]
    if key not in d['reference']:
        pytest.skip(f'no adaptation reference for state {key}')
    return d['records'], d['reference'][key]


def test_identity_without_examples_on_real_weights(model, reference):
    recs, ref = reference
    assert model.adapt() is model and model.adapt().predict(recs[11:]) == model.predict(recs[11:])
    assert maxdiff(model.predict(recs[11:]), ref['zero_shot']) < 1e-6


def test_matches_the_reference_implementation(model, reference):
    recs, ref = reference
    got = model.adapt(examples=recs[:6]).predict(recs[11:])
    assert maxdiff(got, ref['examples']) <= 1e-6
    assert maxdiff(got, ref['zero_shot']) > 1e-4  # the adaptation does something
