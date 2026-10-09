"""End-to-end prediction; runs only when EJ_WEIGHTS names a model.ejpack (or a weights directory: `python -m ej.train fit`
output, which fetches the base model e5-small-v2 from the Hub on first use, or HF_HUB_OFFLINE=1 with a filled cache)."""
import math
import os

import pytest

import ej

WEIGHTS = os.environ.get('EJ_WEIGHTS')
pytestmark = pytest.mark.skipif(not WEIGHTS, reason='set EJ_WEIGHTS to a model.ejpack (or a weights directory)')

PLAIN = {'id': 'plain-text', 'state': 'Hi, I was charged twice for my March invoice. Please refund the duplicate payment.',
         'questions': {'topic': {'type': 'choice', 'instructions': 'What is the message about?',
                                 'options': [{'key': 'billing', 'text': 'A payment, charge or invoice problem'},
                                             {'key': 'shipping', 'text': 'Delivery or shipping of an order'},
                                             {'key': 'login', 'text': 'Login or profile settings'}]},
                       'refund': {'type': 'noul', 'instructions': 'The customer asks for money back.',
                                  'options': ej.NOUL_OPTIONS}}}


@pytest.fixture(scope='module')
def model():
    return ej.load(WEIGHTS)


def test_predict_returns_valid_distributions(model):
    records = [ej.EXAMPLE_RECORD, PLAIN]
    preds = model.predict(records)
    assert len(preds) == len(records)
    for r, p in zip(records, preds):
        assert set(p) == set(r['questions'])
        for qid, q in r['questions'].items():
            v = p[qid]
            assert len(v) == len(q['options'])
            assert all(isinstance(x, float) and math.isfinite(x) and 0.0 <= x <= 1.0 for x in v)
            assert abs(sum(v) - 1.0) < 1e-9


def test_predict_is_deterministic(model):
    assert model.predict([ej.EXAMPLE_RECORD, PLAIN]) == model.predict([ej.EXAMPLE_RECORD, PLAIN])


def test_batch_composition_moves_probabilities_only_at_float_noise_level(model):
    one = model.predict([ej.EXAMPLE_RECORD])[0]
    batched = model.predict([ej.EXAMPLE_RECORD, PLAIN])[0]
    assert max(abs(a - b) for q in one for a, b in zip(one[q], batched[q])) < 1e-5


def test_predict_validates_input(model):
    with pytest.raises(ej.RecordError):
        model.predict([{'state': 'x', 'questions': {}}])
