"""Any Jev-compatible System One server over HTTP: POST {url}/v1/systemone {model, state, questions} -> answers.

Covers rivals whose documented inference is a server: DecidaBERT-large (`decida serve --model decidabert=helmo/DecidaBERT-large
--max-len 2048`, from git+https://github.com/heldernoid/decida -- NOT the unrelated PyPI package 'decida'), llama.cpp's
/v1/systemone (Julia-1, lev, OpenJev, Tev1 GGUFs), laya-serve, kev / decider / von servers.
Env: SYSTEMONE_URL (default http://127.0.0.1:8000), SYSTEMONE_MODEL (sent as "model"), SYSTEMONE_TIMEOUT (s, default 120).
Proxies are bypassed for localhost URLs. Probabilities are used as the server returns them (often rounded to 4 dp; the
common.finalize smoothing keeps a rounded 0 finite)."""
import json
import os
import urllib.parse
import urllib.request

from . import common


def _opener(url):
    host = urllib.parse.urlparse(url).hostname or ''
    if host in ('localhost', '127.0.0.1', '::1'):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def call(url, body, timeout, headers=None, data=None):
    """POST body (or the pre-encoded bytes `data`) to {url}/v1/systemone; extra `headers` (e.g. auth) are added as given."""
    req = urllib.request.Request(url.rstrip('/') + '/v1/systemone', data=data or json.dumps(body).encode(),
                                 headers={'content-type': 'application/json', **(headers or {})}, method='POST')
    with _opener(url).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def predict(records, url=None, model=None):
    """[{qid: probs}] per record, in each question's option order."""
    url = url or os.environ.get('SYSTEMONE_URL', 'http://127.0.0.1:8000')
    model = model or os.environ.get('SYSTEMONE_MODEL')
    timeout = float(os.environ.get('SYSTEMONE_TIMEOUT', '120'))
    out = []
    for r in records:
        body = {'state': common.state_text(r), 'questions': common.wire_questions(r)}
        if model:
            body['model'] = model
        out.append(common.from_response(r, call(url, body, timeout)['answers']))
    return out
