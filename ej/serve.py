"""Serve ej over HTTP in the System One wire format: POST /v1/systemone (ej.serve_wire), GET /health.

    python -m ej.serve --weights 5ak3t/ej --revision v0.0.1 --port 8000 --threads 2 [--overflow reject|truncate]

Any client of a Jev-compatible /v1/systemone server can then query ej (benchmark harnesses: S1MB's llama-cpp adapter,
JevBench's typesafe adapter, the Decision Index http engine). One request is one ej record: all its questions are answered
in one predict call. Requests are served one at a time (one loaded model per process); predictions are deterministic for a
fixed --threads.

Input length (--overflow). ej's encoder reads every text of a record (the state, each sentence or JSON field of the state,
the instructions, each option, instructions + option, and texts derived from them) in a window of MAX_LEN = 256 tokens
(student_enc.MAX_LEN, [CLS]/[SEP] included) and truncates a longer text. The server records the untruncated length of every
text the encoder tokenizes for its window during the request (traced(); the runtime's encoder cache is bypassed so no text
is skipped), then:
  reject (default)  any text longer than the window -> HTTP 422 'maximum context length', the prediction is discarded;
  truncate          ej's native behaviour, the response carries "truncated": true.
Not covered by the window rule, and part of the released architecture rather than input shortening: the token-level reader
looks at the first 64 tokens of each text (student_tok.MAX_LEN), and a state is split into at most 24 chunks
(student_feat.MAX_CHUNKS) for it; the whole-state pass still reads the full state within its window.

Errors: 400 malformed request, 413 body larger than --max-request-bytes, 422 input longer than the window (reject mode),
500 prediction failure. Bound to 127.0.0.1 by default; there is no authentication, do not expose it to a network you do not
trust."""
import argparse
import contextlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import serve_wire as W

MAX_REQUEST_BYTES = 16 * 1024 * 1024


class Overflow(ValueError):
    """At least one encoder text is longer than the window (reject mode)."""


class Engine:
    """predict_fn(record) -> ({qid: probs}, {text: untruncated tokens}) for every text the encoder read for its window."""

    def __init__(self, name, predict_fn, window, overflow='reject', settings=None):
        if overflow not in ('reject', 'truncate'):
            raise ValueError("overflow must be 'reject' or 'truncate'")
        self.name, self.predict_fn, self.window, self.overflow = name, predict_fn, int(window), overflow
        self.settings = dict(settings or {})
        self._lock = threading.Lock()

    def health(self):
        return {'model': self.name, 'endpoint': '/v1/systemone', 'overflow': self.overflow,
                'max_tokens_per_text': self.window, **self.settings}

    def answer(self, body):
        """The response body for one request body (raises W.WireError, Overflow)."""
        record = W.to_record(body)
        with self._lock:
            probs, lengths = self.predict_fn(record)
        long_ = [n for n in lengths.values() if n > self.window]
        if long_ and self.overflow == 'reject':
            raise Overflow(f'maximum context length is {self.window} tokens per encoded text; {len(long_)} text(s) are '
                           f'longer (longest {max(long_)} tokens): refusing instead of truncating')
        return W.response(self.name, record, probs, sum(min(n, self.window) for n in lengths.values()), bool(long_))


@contextlib.contextmanager
def traced(window):
    """Inside the block, every tokenizer call made with max_length=window and truncation records {text: untruncated
    length} into the yielded dict (class attribute patched for the block only, restored on exit; ej.scope.e5_pinned does
    the same for from_pretrained)."""
    from transformers import tokenization_utils_base as tub
    cls = tub.PreTrainedTokenizerBase
    orig, seen = cls.__dict__['__call__'], {}

    def wrapped(self, text=None, *args, **kw):
        if text is not None and kw.get('truncation') and kw.get('max_length') == window:
            batch = [text] if isinstance(text, str) else list(text)
            todo = sorted({t for t in batch if isinstance(t, str) and t not in seen})
            if todo:
                ids = orig(self, todo, add_special_tokens=kw.get('add_special_tokens', True), verbose=False)['input_ids']
                seen.update(zip(todo, map(len, ids)))
        return orig(self, text, *args, **kw)
    cls.__call__ = wrapped
    try:
        yield seen
    finally:
        cls.__call__ = orig


def _handler(engine, max_bytes):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code, kind, message):
            self._send(code, {'error': {'type': kind, 'message': message}})

        def do_GET(self):
            if self.path.rstrip('/') == '/health':
                return self._send(200, engine.health())
            self._error(404, 'not_found', 'GET /health or POST /v1/systemone')

        def do_POST(self):
            if self.path.rstrip('/') != '/v1/systemone':
                return self._error(404, 'not_found', 'POST /v1/systemone')
            try:
                n = int(self.headers.get('Content-Length') or -1)
            except ValueError:
                n = -1
            if n < 0 or n > max_bytes:
                self.close_connection = True
                return self._error(413, 'request_too_large', f'body larger than {max_bytes} bytes or without a length')
            try:
                body = json.loads(self.rfile.read(n).decode('utf-8'))
            except (UnicodeDecodeError, ValueError):
                return self._error(400, 'invalid_request', 'the body is not JSON')
            try:
                return self._send(200, engine.answer(body))
            except W.WireError as e:
                return self._error(400, 'invalid_request', str(e))
            except Overflow as e:
                return self._error(422, 'context_length_exceeded', str(e))
            except Exception as e:  # noqa: BLE001 - one failed request must not stop the server
                return self._error(500, 'prediction_failed', type(e).__name__)

        def log_message(self, fmt, *args):  # no request log (request bodies are never logged either)
            pass
    return Handler


def make_server(engine, host='127.0.0.1', port=8000, max_bytes=MAX_REQUEST_BYTES):
    """An HTTPServer for engine (port 0 = any free port: read server.server_address)."""
    return HTTPServer((host, port), _handler(engine, max_bytes))


def load_engine(weights, revision=None, threads=2, overflow='reject', low_memory=False):
    """An Engine around ej.load(weights, revision); the window is the loaded runtime's own (student_enc.MAX_LEN)."""
    import ej
    model = ej.load(weights, revision=revision, threads=threads, low_memory=low_memory)
    import student_enc as E  # the runtime is importable by bare name once ej.load() has run
    import student_feat as F
    import student_tok as TOK
    os.environ['EJ_COLD'] = '1'  # every text is tokenized in every call, so traced() sees all of them

    def predict(record):
        with traced(E.MAX_LEN) as seen:
            (probs,) = model.predict([record])
        return probs, dict(seen)
    name = f'{weights}@{revision}' if revision else str(weights)
    settings = {'ej_version': ej.__version__, 'threads': threads, 'low_memory': bool(low_memory), 'encoder_cache': 'off',
                'state_key': model.config.get('state_key'), 'token_reader_window': TOK.MAX_LEN,
                'max_state_chunks': F.MAX_CHUNKS}
    return Engine(name, predict, E.MAX_LEN, overflow, settings)


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m ej.serve', description=__doc__.split('\n\n')[0])
    p.add_argument('--weights', default='5ak3t/ej', help='model.ejpack, a weights directory or a Hub repo id')
    p.add_argument('--revision', default=None, help='Hub revision (pin it, e.g. v0.0.1)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--threads', type=int, default=2, help='torch threads inside each predict call')
    p.add_argument('--overflow', choices=('reject', 'truncate'), default='reject')
    p.add_argument('--low-memory', action='store_true')
    p.add_argument('--max-request-bytes', type=int, default=MAX_REQUEST_BYTES)
    a = p.parse_args(argv)
    engine = load_engine(a.weights, a.revision, a.threads, a.overflow, a.low_memory)
    server = make_server(engine, a.host, a.port, a.max_request_bytes)
    print(json.dumps({'listening': f'http://{a.host}:{server.server_address[1]}', **engine.health()}), file=sys.stderr,
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
