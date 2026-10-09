"""Jev (TypeSafe AI, hosted, closed weights) through its public System One API: POST https://api.typesafe.ai/v1/systemone
{model, state, questions} with a bearer key. Model id JEV_MODEL (default 'jev-latest', = jev-1.13.0 on 2026-10-07); the
resolved version each response reports is logged, and a version other than EXPECTED is warned about on stderr.

Wire format = common.to_wire (the same as every other System One rival): choice criteria {key: description or null},
score criteria = ordered level texts, noul criteria {false/true: text} only when the options carry real text. Answers, per
the public docs (search summaries, 2026-10-07): choice -> {choice, probabilities {option key: p}, confidence} (<= 255
options); score -> {score, legend, probabilities {"0".."k-1": p}, confidence} (2-10 levels); noul -> {noul: p(yes)}. They
go through common.from_response (option-key order, renormalise, EPS smoothing) after check_answer has verified the keys.

Secrets and money: the key is read from KEY_PATH ($JEV_API_KEY_FILE; keep it private, mode 0600) inside this module only, sent only in the auth header, and
redacted from every error text. Each response is cached on disk under CACHE_DIR (0700 dir, 0600 files) by sha256 of the
exact request bytes, so a re-run never pays twice (set JEV_NO_CACHE=1 to bypass reads, e.g. a determinism check). Usage
tokens, resolved model, latency, tries, cache hits and format checks go to USAGE_LOG, one JSON line per record. No gold
or label ever leaves this machine: records carrying 'gold' are refused and the request body is checked before sending.
Retries: 429 / 5xx / timeouts / connection errors, at most TRIES attempts, exponential backoff (Retry-After honoured,
capped); timeout TIMEOUT s per attempt. A retried request may be billed more than once by the provider."""
import atexit
import functools
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error

from . import common, systemone_http

URL = 'https://api.typesafe.ai'
EXPECTED = 'jev-1.13.0'
_HOME = os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'), 'ej-bench', 'jev')
KEY_PATH = os.environ.get('JEV_API_KEY_FILE', os.path.join(os.path.expanduser('~'), '.config', 'jev', 'api_key'))
CACHE_DIR = os.environ.get('JEV_CACHE_DIR', os.path.join(_HOME, 'cache'))
USAGE_LOG = os.environ.get('JEV_USAGE_LOG', os.path.join(_HOME, 'usage.jsonl'))
TRIES, TIMEOUT, MAX_WAIT = 5, 60.0, 60.0
MAX_OPTIONS, SCORE_LEVELS = 255, (2, 10)  # public docs
FORBIDDEN = {'gold', 'label', 'labels', 'answer', 'target'}
TOTALS = {'records': 0, 'cached': 0, 'http_attempts': 0, 'usage': {}}


@functools.lru_cache(maxsize=1)
def _key():
    with open(KEY_PATH) as f:
        k = f.read().strip()
    if not k:
        raise RuntimeError(f'{KEY_PATH} is empty')
    return k


def redact(text):
    """Remove the key (and anything key- or token-shaped) from text before it is raised or printed."""
    text = str(text)
    try:
        text = text.replace(_key(), '[REDACTED]')
    except OSError:
        pass
    # the key prefix as a character class, so a plain grep for the prefix over the repo finds only real leaks
    text = re.sub(r'api[k]ey_[A-Za-z0-9_\-]+', '[REDACTED]', text)
    return re.sub(r'(?i)(bearer\s+)\S+', r'\1[REDACTED]', text)


def _private_dir(path):
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private(path, text, mode='w'):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == 'a' else os.O_TRUNC), 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, mode) as f:
        f.write(text)


def assert_blind(r, body):
    """No gold / label field may reach the API (run_bench strips 'gold' already; this re-checks record and body).
    Explicit raises, not `assert`, so the check survives python -O. Option keys and question ids are data, not fields."""
    if 'gold' in r:
        raise AssertionError(f"record {r.get('id')} still carries gold: refusing to send it")
    if set(body) != {'model', 'state', 'questions'}:
        raise AssertionError(f'unexpected body fields {sorted(body)}')

    def walk(x, where):
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in FORBIDDEN and not where.endswith(('criteria', 'questions')):
                    raise AssertionError(f'{where}.{k}: label-like field in the request body')
                walk(v, f'{where}.{k}')
        elif isinstance(x, list):
            for v in x:
                walk(v, where)
    walk(body, 'body')


def body_of(r, model):
    """The API request body for record r (option limits checked, gold refused)."""
    for qid, q in r['questions'].items():
        k = len(q['options'])
        if q['type'] == 'choice' and k > MAX_OPTIONS:
            raise ValueError(f"{r['id']}/{qid}: {k} options > Jev's {MAX_OPTIONS}")
        if q['type'] == 'score' and not SCORE_LEVELS[0] <= k <= SCORE_LEVELS[1]:
            raise ValueError(f"{r['id']}/{qid}: {k} score levels outside Jev's {SCORE_LEVELS}")
    body = {'model': model, 'state': common.state_text(r), 'questions': common.wire_questions(r)}
    assert_blind(r, body)
    return body


def _wait(err, attempt):
    ra = err.headers.get('retry-after') if isinstance(err, urllib.error.HTTPError) and err.headers else None
    try:
        return min(float(ra), MAX_WAIT) if ra else min(2.0 ** attempt, MAX_WAIT)
    except ValueError:
        return min(2.0 ** attempt, MAX_WAIT)


def post(data):
    """-> (response dict, attempts). Retries 429 / 5xx / timeouts; every error text is redacted."""
    headers = {'authorization': f'Bearer {_key()}'}
    last = None
    for attempt in range(TRIES):
        try:
            return systemone_http.call(URL, None, TIMEOUT, headers=headers, data=data), attempt + 1
        except urllib.error.HTTPError as e:
            detail = redact(e.read().decode(errors='replace')[:500]) if e.fp else ''
            last = f'HTTP {e.code} {redact(e.reason)}: {detail}'
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f'Jev API refused the request: {last}') from None
            time.sleep(_wait(e, attempt))
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last = f'{type(e).__name__}: {redact(e)}'
            time.sleep(_wait(e, attempt))
    raise RuntimeError(f'Jev API failed after {TRIES} tries: {last}') from None


def check_answer(q, ans):
    """Format check of one Jev answer against our question. -> list of soft issues; raises on a mapping-breaking one."""
    issues, t = [], q['type']
    if ans.get('type') not in (None, t):
        raise ValueError(f"answer type {ans.get('type')} for a {t} question")
    if t == 'noul':
        p = ans.get('noul')
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError(f'noul answer without a probability in [0, 1]: {ans}')
        return issues
    probs = ans.get('probabilities')
    if not isinstance(probs, dict) or not probs:
        raise ValueError(f'{t} answer without probabilities: {sorted(ans)}')
    want = [o['key'] for o in q['options']] if t == 'choice' else [str(i) for i in range(len(q['options']))]
    extra = set(map(str, probs)) - set(want)
    if extra:
        raise ValueError(f'{t} answer has unknown keys {sorted(extra)[:5]} (expected {want[:5]}...)')
    if set(want) - set(map(str, probs)):
        issues.append(f'missing keys {sorted(set(want) - set(map(str, probs)))}')
    s = sum(float(v) for v in probs.values())
    if abs(s - 1) > 1e-2:
        issues.append(f'raw sum {s:.4f}')
    if t == 'choice' and ans.get('choice') is not None and ans['choice'] != max(probs, key=probs.get):
        issues.append('choice != argmax')
    if t == 'score' and isinstance(ans.get('score'), (int, float)):
        ev = sum(int(k) * float(v) for k, v in probs.items())
        if abs(ev - ans['score']) > 0.02:
            issues.append(f'score {ans["score"]} != E[level] {ev:.3f}')
    return issues


def _add_usage(u):
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)):
            TOTALS['usage'][k] = TOTALS['usage'].get(k, 0) + v


def _report():
    if TOTALS['records']:
        print('jev usage: ' + json.dumps(TOTALS), file=sys.stderr)


atexit.register(_report)


def one(r, model):
    """Probabilities for one record: cached response or one API call, checked and logged."""
    body = body_of(r, model)
    data = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()  # option order kept (no sort_keys)
    sha = hashlib.sha256(data).hexdigest()
    _private_dir(CACHE_DIR)
    path = os.path.join(CACHE_DIR, sha + '.json')
    t0, tries, cached = time.perf_counter(), 0, False
    if os.path.exists(path) and os.environ.get('JEV_NO_CACHE') != '1':
        resp, cached = json.load(open(path)), True
    else:
        resp, tries = post(data)
        _write_private(path + '.tmp', json.dumps(resp))
        os.replace(path + '.tmp', path)
    ms = (time.perf_counter() - t0) * 1000
    answers = resp.get('answers') or {}
    missing = set(r['questions']) - set(answers)
    if missing:
        raise ValueError(f"{r['id']}: no answer for {sorted(missing)}")
    checks = {qid: check_answer(q, answers[qid]) for qid, q in r['questions'].items()}
    if resp.get('model') != EXPECTED and not cached:
        print(f"jev: served by {resp.get('model')}, expected {EXPECTED}", file=sys.stderr)
    TOTALS['records'] += 1
    TOTALS['cached'] += cached
    TOTALS['http_attempts'] += tries
    if not cached:
        _add_usage(resp.get('usage'))
    log = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'id': r['id'], 'sha256': sha, 'cached': cached,
           'tries': tries, 'ms': round(ms, 1), 'model': resp.get('model'), 'usage': resp.get('usage'),
           'types': [q['type'] for q in r['questions'].values()], 'checks': {k: v for k, v in checks.items() if v}}
    os.makedirs(os.path.dirname(USAGE_LOG) or '.', exist_ok=True)
    _write_private(USAGE_LOG, json.dumps(log) + '\n', mode='a')
    return common.from_response(r, answers)


def predict(records, model=None):
    """[{qid: probs}] per record, in each question's option order."""
    model = model or os.environ.get('JEV_MODEL', 'jev-latest')
    return [one(r, model) for r in records]
