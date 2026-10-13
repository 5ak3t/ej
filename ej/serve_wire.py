"""The System One wire format (POST /v1/systemone, as served by Jev-compatible engines) <-> ej records. Pure functions, no model.

Request body: {"model"?: str, "state": text or JSON value, "questions": {qid: question}} with
  choice: {"type": "choice", "instructions", "criteria": {key: description or null}}   (null: the key is the option text)
  noul:   {"type": "noul",   "instructions", "criteria"?: {"false": text, "true": text}} (absent: ej.NOUL_OPTIONS)
  score:  {"type": "score",  "instructions", "criteria": [level text, ...]}             (ordered, lowest level first)
Instructions, descriptions and the state may be JSON values instead of text; they are serialised with json.dumps (ej reads
text). Nothing else in the request reaches the model.

Response: {"model", "answers": {qid: answer}, "usage": {"input_tokens", "output_tokens": 0}, "truncated": bool} with
  choice: {"type", "choice": key, "probabilities": {key: p}, "confidence"}
  noul:   {"type", "noul": P(true), "choice": "false" | "true", "probabilities": {"false": p, "true": p}, "confidence"}
  score:  {"type", "score": level index, "probabilities": {"0": p, ...}, "confidence"}
Probabilities are ej's own, unrounded and unrenormalised.

Input length: the server, not this module, decides what happens to a text longer than ej's encoder window (ej.serve).
"""
import json

from .records import NOUL_OPTIONS, TYPES, validate_record


class WireError(ValueError):
    """A request that does not follow the System One wire format."""


def _text(v, where):
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise WireError(f'{where}: not text or a JSON value') from None


def _options(qid, t, criteria):
    where = f'questions.{qid}.criteria'
    if t == 'choice':
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise WireError(f'{where}: a choice question needs an object of at least 2 options')
        return [{'key': str(k), 'text': str(k) if d is None else _text(d, f'{where}.{k}')} for k, d in criteria.items()]
    if t == 'score':
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise WireError(f'{where}: a score question needs a list of at least 2 levels')
        return [{'key': str(i), 'text': _text(d, f'{where}[{i}]')} for i, d in enumerate(criteria)]
    if criteria is None:
        return [dict(o) for o in NOUL_OPTIONS]
    if not isinstance(criteria, dict) or set(criteria) != {'false', 'true'}:
        raise WireError(f'{where}: noul criteria must be an object with exactly "false" and "true"')
    return [{'key': k, 'text': _text(criteria[k], f'{where}.{k}')} for k in ('false', 'true')]


def to_record(body):
    """The ej input record for one request body (raises WireError on a malformed body)."""
    if not isinstance(body, dict):
        raise WireError('the body must be a JSON object')
    extra = set(body) - {'model', 'state', 'questions'}
    if extra:
        raise WireError(f'unknown fields {sorted(extra)}')
    if 'state' not in body:
        raise WireError('missing "state"')
    qs = body.get('questions')
    if not isinstance(qs, dict) or not qs:
        raise WireError('"questions" must be a non-empty object')
    questions = {}
    for qid, q in qs.items():
        if not isinstance(q, dict) or q.get('type') not in TYPES:
            raise WireError(f'questions.{qid}: type must be one of {TYPES}')
        extra = set(q) - {'type', 'instructions', 'criteria'}
        if extra:
            raise WireError(f'questions.{qid}: unknown fields {sorted(extra)}')
        if q.get('instructions') is None:
            raise WireError(f'questions.{qid}: missing "instructions"')
        questions[qid] = {'type': q['type'], 'instructions': _text(q['instructions'], f'questions.{qid}.instructions'),
                          'options': _options(qid, q['type'], q.get('criteria'))}
    record = {'state': _text(body['state'], 'state'), 'questions': questions}
    try:
        return validate_record(record)
    except ValueError as e:
        raise WireError(str(e)) from None


def _argmax(keys, ps):
    i = max(range(len(ps)), key=lambda j: (ps[j], -j))
    return keys[i], ps[i]


def answers(record, probs):
    """{qid: wire answer} from ej's predict output for this record ({qid: [p per option, in option order]})."""
    out = {}
    for qid, q in record['questions'].items():
        keys, ps = [o['key'] for o in q['options']], [float(p) for p in probs[qid]]
        key, conf = _argmax(keys, ps)
        a = {'type': q['type'], 'probabilities': dict(zip(keys, ps)), 'confidence': conf}
        if q['type'] == 'score':
            a['score'] = int(key)
        else:
            a['choice'] = key
        if q['type'] == 'noul':
            a['noul'] = a['probabilities']['true']
        out[qid] = a
    return out


def response(model, record, probs, input_tokens, truncated):
    """The response body for one answered request."""
    return {'model': model, 'answers': answers(record, probs),
            'usage': {'input_tokens': int(input_tokens), 'output_tokens': 0}, 'truncated': bool(truncated)}
