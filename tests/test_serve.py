"""ej.serve / ej.serve_wire: the System One wire format and the HTTP server. Most tests need no weights (a stand-in predict
function); the last ones run only when EJ_WEIGHTS names a model.ejpack."""
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

import ej
from ej import serve, serve_wire as W

BODY = {'state': {'msg': 'The blender arrived cracked. Replace it before Friday.', 'tier': 'gold'},
        'questions': {'route': {'type': 'choice', 'instructions': 'Which team?',
                                'criteria': {'returns': 'Returns and replacements', 'billing': 'Billing problems', 'other': None}},
                      'upset': {'type': 'noul', 'instructions': 'The customer is upset.'},
                      'shipped': {'type': 'noul', 'instructions': {'q': 'Shipped?'},
                                  'criteria': {'true': 'The text says so', 'false': 'It does not'}},
                      'urgency': {'type': 'score', 'instructions': 'How urgent?', 'criteria': ['low', 'medium', 'high']}}}


def test_to_record_maps_every_type():
    r = W.to_record(BODY)
    assert ej.validate_record(r) is r
    assert json.loads(r['state']) == BODY['state']
    q = r['questions']
    assert q['route']['options'][2] == {'key': 'other', 'text': 'other'}  # null description: the key is the text
    assert q['upset']['options'] == ej.NOUL_OPTIONS
    assert [o['key'] for o in q['shipped']['options']] == ['false', 'true'] and q['shipped']['options'][1]['text'] == 'The text says so'
    assert json.loads(q['shipped']['instructions']) == {'q': 'Shipped?'}
    assert [(o['key'], o['text']) for o in q['urgency']['options']] == [('0', 'low'), ('1', 'medium'), ('2', 'high')]


@pytest.mark.parametrize('mutate, message', [
    (lambda b: b.pop('state'), 'missing "state"'),
    (lambda b: b.update(questions={}), 'non-empty'),
    (lambda b: b.update(gold='returns'), 'unknown fields'),
    (lambda b: b['questions']['route'].update(answer='returns'), 'unknown fields'),
    (lambda b: b['questions']['route'].update(type='multi'), 'type must be one of'),
    (lambda b: b['questions']['route'].update(criteria={'a': 'x'}), 'at least 2'),
    (lambda b: b['questions']['urgency'].update(criteria={'0': 'low', '1': 'high'}), 'list of at least 2'),
    (lambda b: b['questions']['shipped'].update(criteria={'yes': 'a', 'no': 'b'}), '"false" and "true"'),
    (lambda b: b['questions']['upset'].pop('instructions'), 'missing "instructions"'),
])
def test_to_record_refuses(mutate, message):
    b = json.loads(json.dumps(BODY))
    mutate(b)
    with pytest.raises(W.WireError, match=message):
        W.to_record(b)


def test_answers_carry_every_reader_field():
    r = W.to_record(BODY)
    probs = {'route': [0.2, 0.5, 0.3], 'upset': [0.3, 0.7], 'shipped': [0.9, 0.1], 'urgency': [0.1, 0.3, 0.6]}
    a = W.response('m', r, probs, 12, False)
    assert a['model'] == 'm' and a['usage'] == {'input_tokens': 12, 'output_tokens': 0} and a['truncated'] is False
    ans = a['answers']
    assert ans['route'] == {'type': 'choice', 'choice': 'billing', 'confidence': 0.5,
                            'probabilities': {'returns': 0.2, 'billing': 0.5, 'other': 0.3}}
    assert ans['upset']['noul'] == 0.7 and ans['upset']['choice'] == 'true'
    assert ans['shipped']['probabilities'] == {'false': 0.9, 'true': 0.1} and ans['shipped']['choice'] == 'false'
    assert ans['urgency']['score'] == 2 and ans['urgency']['probabilities'] == {'0': 0.1, '1': 0.3, '2': 0.6}


def _fake_engine(overflow, lengths):
    def predict(record):
        return {qid: [1 / len(q['options'])] * len(q['options']) for qid, q in record['questions'].items()}, dict(lengths)
    return serve.Engine('fake@1', predict, 256, overflow, {'threads': 1})


@pytest.fixture
def url_of():
    servers = []

    def start(engine, max_bytes=serve.MAX_REQUEST_BYTES):
        s = serve.make_server(engine, port=0, max_bytes=max_bytes)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        return f'http://127.0.0.1:{s.server_address[1]}'
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _post(url, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(url + '/v1/systemone', data, {'content-type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_answers_and_reports_health(url_of):
    url = url_of(_fake_engine('reject', {'query: a': 10, 'passage: b': 256}))
    code, a = _post(url, BODY)
    assert code == 200 and a['model'] == 'fake@1' and set(a['answers']) == set(BODY['questions'])
    assert a['usage']['input_tokens'] == 266 and a['truncated'] is False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + '/health', timeout=30) as r:
        h = json.loads(r.read())
    assert h['overflow'] == 'reject' and h['max_tokens_per_text'] == 256 and h['threads'] == 1


def test_server_overflow_policies(url_of):
    lengths = {'query: long state': 300, 'passage: b': 20}
    code, a = _post(url_of(_fake_engine('reject', lengths)), BODY)
    assert code == 422 and a['error']['type'] == 'context_length_exceeded'
    assert 'maximum context length' in a['error']['message'] and '300' in a['error']['message']
    code, a = _post(url_of(_fake_engine('truncate', lengths)), BODY)
    assert code == 200 and a['truncated'] is True and a['usage']['input_tokens'] == 256 + 20


def test_server_refuses_bad_requests(url_of):
    url = url_of(_fake_engine('reject', {}), max_bytes=4096)
    assert _post(url, None, raw=b'not json')[0] == 400
    assert _post(url, {'state': 'x', 'questions': {'q': {'type': 'multi', 'instructions': 'i'}}})[0] == 400
    code, a = _post(url, {'state': 'x' * 5000, 'questions': BODY['questions']})
    assert code == 413 and a['error']['type'] == 'request_too_large'
    with pytest.raises(ValueError):
        serve.Engine('x', None, 256, 'drop')


def test_traced_records_untruncated_lengths_and_restores():
    from transformers import BertTokenizerFast
    from transformers import tokenization_utils_base as tub
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {t: i for i, t in enumerate(['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'a', 'b'])}
    core = Tokenizer(models.WordPiece(vocab, unk_token='[UNK]'))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = BertTokenizerFast(tokenizer_object=core, unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]')
    before = tub.PreTrainedTokenizerBase.__dict__['__call__']
    with serve.traced(8) as seen:
        out = tok(['a ' * 20, 'b'], truncation=True, max_length=8)
        tok(['a ' * 30], truncation=True, max_length=4)  # another window: not recorded
    assert len(out['input_ids'][0]) == 8  # the runtime's own call is unchanged (still truncated)
    assert seen == {'a ' * 20: 22, 'b': 3}
    assert tub.PreTrainedTokenizerBase.__dict__['__call__'] is before


WEIGHTS = os.environ.get('EJ_WEIGHTS')
weights_only = pytest.mark.skipif(not WEIGHTS, reason='set EJ_WEIGHTS to a model.ejpack (or a weights directory)')


@pytest.fixture(scope='module')
def engine():
    old = os.environ.get('EJ_COLD')
    e = serve.load_engine(WEIGHTS, os.environ.get('EJ_REVISION'), threads=2)
    yield e
    if old is None:
        os.environ.pop('EJ_COLD', None)
    else:
        os.environ['EJ_COLD'] = old


@weights_only
def test_served_probabilities_equal_predict(engine):
    a = engine.answer(BODY)
    record = W.to_record(BODY)
    (ref,) = ej.load(WEIGHTS, revision=os.environ.get('EJ_REVISION')).predict([record])
    for qid, q in record['questions'].items():
        got = [a['answers'][qid]['probabilities'][o['key']] for o in q['options']]
        assert got == pytest.approx(ref[qid], abs=1e-6)
    assert a['truncated'] is False and a['usage']['input_tokens'] > 0


@weights_only
def test_served_overflow_is_refused_then_truncated(engine):
    long_body = dict(BODY, state='word ' * 300)
    with pytest.raises(serve.Overflow, match='maximum context length is 256'):
        engine.answer(long_body)
    engine.overflow = 'truncate'
    try:
        assert engine.answer(long_body)['truncated'] is True
        long_field = dict(BODY, state={'a': 'short', 'b': 'word ' * 300})
        assert engine.answer(long_field)['truncated'] is True  # a long JSON field is caught too
    finally:
        engine.overflow = 'reject'
