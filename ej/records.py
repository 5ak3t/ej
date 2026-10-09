"""Input records for ej: validation and a ready-made example.

Record: {'state': text (free text, or a JSON object serialised as text),
         'questions': {qid: {'type': 'choice' | 'noul' | 'score', 'instructions': text,
                             'options': [{'key': key, 'text': description}, ...]}}}
Optional fields ('id', 'source', anything else) are ignored by prediction. 'noul' questions have exactly the two options
'false' and 'true' (in either order); 'score' options are ordered levels, lowest first."""
import json

TYPES = ('choice', 'noul', 'score')
NOUL_OPTIONS = [{'key': 'false', 'text': 'No, the statement does not hold.'},
                {'key': 'true', 'text': 'Yes, the statement holds.'}]

EXAMPLE_RECORD = {
    'id': 'example-0001',
    'state': json.dumps({'customer_tier': 'gold', 'channel': 'email', 'order_total_eur': 412.5, 'days_since_delivery': 3,
                         'message': 'The blender arrived with a cracked jug. I want a replacement before Friday, '
                                    'otherwise refund me. This is the second time this happens.'}),
    'questions': {
        'route': {'type': 'choice', 'instructions': 'Which team should handle this request?',
                  'options': [{'key': 'returns', 'text': 'Returns and replacements for damaged or wrong items'},
                              {'key': 'billing', 'text': 'Billing, invoices and payment problems'},
                              {'key': 'tech', 'text': 'Technical support for using the product'},
                              {'key': 'sales', 'text': 'Sales questions before a purchase'}]},
        'needs_human': {'type': 'noul', 'instructions': 'The customer is upset enough that a human agent should reply.',
                        'options': NOUL_OPTIONS},
        'urgency': {'type': 'score', 'instructions': 'How urgent is this request?',
                    'options': [{'key': '0', 'text': 'Low: can wait a week'},
                                {'key': '1', 'text': 'Medium: answer within two days'},
                                {'key': '2', 'text': 'High: answer today'}]},
    },
}


class RecordError(ValueError):
    """A record that does not follow the ej input format."""


def _check_question(where, q):
    if not isinstance(q, dict):
        raise RecordError(f'{where}: a question must be a dict')
    if q.get('type') not in TYPES:
        raise RecordError(f'{where}: type must be one of {TYPES}, got {q.get("type")!r}')
    if not isinstance(q.get('instructions'), str):
        raise RecordError(f'{where}: instructions must be a string')
    opts = q.get('options')
    if not isinstance(opts, list) or len(opts) < 2:
        raise RecordError(f'{where}: options must be a list of at least 2 options')
    keys = []
    for i, o in enumerate(opts):
        if not isinstance(o, dict) or not isinstance(o.get('key'), str) or not isinstance(o.get('text'), str):
            raise RecordError(f'{where}.options[{i}]: an option is ' + '{"key": str, "text": str}')
        keys.append(o['key'])
    if len(set(keys)) != len(keys):
        raise RecordError(f'{where}: option keys must be unique, got {keys}')
    if q['type'] == 'noul' and sorted(keys) != ['false', 'true']:
        raise RecordError(f"{where}: a noul question has exactly the options 'false' and 'true', got {keys}")


def validate_record(r, i=0):
    """Raise RecordError unless r is a valid input record; returns r."""
    where = f'record {r.get("id", i)!r}' if isinstance(r, dict) else f'record {i}'
    if not isinstance(r, dict):
        raise RecordError(f'{where}: a record must be a dict')
    if not isinstance(r.get('state'), str):
        raise RecordError(f'{where}: state must be a string (serialise a JSON object with json.dumps)')
    qs = r.get('questions')
    if not isinstance(qs, dict) or not qs:
        raise RecordError(f'{where}: questions must be a non-empty dict ' + '{qid: question}')
    for qid, q in qs.items():
        if not isinstance(qid, str):
            raise RecordError(f'{where}: question ids must be strings')
        _check_question(f'{where}.{qid}', q)
    return r


def validate_records(records):
    """Raise RecordError unless records is a list of valid records; returns the list."""
    if not isinstance(records, (list, tuple)):
        raise RecordError('records must be a list of records')
    for i, r in enumerate(records):
        validate_record(r, i)
    return list(records)
