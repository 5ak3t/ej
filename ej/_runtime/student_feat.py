"""Lexical features without an encoder: state tokenisation (field-wise path=value tokens for JSON states, binned numbers), a
TF-IDF space over word and character 3-gram tokens, JSON parsing and state-option lexical overlap features."""
import json
import math
import re

import numpy as np

WORD = re.compile(r'[a-z0-9]+')


def words(text):
    """Lower-case alphanumeric words; snake_case / kebab-case / camel pieces are split by the regex."""
    return WORD.findall(text.lower())


def _num(x):
    if isinstance(x, bool) or x is None:
        return str(x).lower()
    if float(x).is_integer() and abs(x) <= 10:
        return str(int(x))
    return ('-' if x < 0 else '') + f'~{round(math.log2(abs(x) + 1))}'


def _leaves(obj, path):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f'{path}.{k}' if path else str(k))
    elif isinstance(obj, list):
        yield f'{path}#n', _num(len(obj))
        for v in obj:
            yield from _leaves(v, path)
    else:
        yield path, obj


def parse_json(state):
    s = state.strip()
    if not s.startswith('{'):
        return None
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def state_tokens(state):
    """Tokens of a state for the wide (cross) component: JSON -> field=value tokens; text -> word unigrams + bigrams."""
    obj = parse_json(state)
    if obj is None:
        w = words(state)
        return sorted(set(w) | {f'{a}_{b}' for a, b in zip(w, w[1:])})
    out = set()
    for p, v in _leaves(obj, ''):
        if isinstance(v, str):
            if len(v) <= 32:
                out.add(f'{p}={v.lower()}')
            out.update(f'{p}:{t}' for t in words(v))
        else:
            out.add(f'{p}={_num(v)}')
    return sorted(out)


def lex_terms(text):
    """Terms of the TF-IDF space: words plus character 3-grams inside words (catches 'lightoff' ~ 'lights off')."""
    out = []
    for w in words(text):
        out.append('w:' + w)
        p = f'<{w}>'
        out.extend('c:' + p[i:i + 3] for i in range(len(p) - 2))
    return out


class Tfidf:
    """Sublinear TF-IDF with an IDF learnt from training texts; vectors are L2-normalised sparse dicts."""

    def __init__(self, texts):
        df = {}
        for t in texts:
            for term in set(lex_terms(t)):
                df[term] = df.get(term, 0) + 1
        n = len(texts)
        self.idf = {k: math.log((1 + n) / (1 + v)) + 1 for k, v in df.items() if v >= 2}
        self.default = math.log(1 + n) + 1  # unseen term: maximal idf

    def vec(self, text):
        tf = {}
        for term in lex_terms(text):
            tf[term] = tf.get(term, 0) + 1
        v = {k: (1 + math.log(c)) * self.idf.get(k, self.default) for k, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}

    def size_mb(self):
        return sum(len(k) + 4 for k in self.idf) / 2 ** 20


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())


SENT = re.compile(r'(?<=[.!?])\s+')
MAX_CHUNKS = 24


def chunks(state):
    """Field-wise chunks of a state for late interaction: JSON -> one readable 'path: value' line per leaf; text -> sentences."""
    obj = parse_json(state)
    if obj is None:
        out = [s for s in SENT.split(state.strip()) if s]
    else:
        out = [f"{p.replace('.', ' ').replace('_', ' ')}: {v}" for p, v in _leaves(obj, '') if not p.endswith('#n')]
    return (out or [state])[:MAX_CHUNKS]


def lexical(tf, items):
    """(N, K, 2) lexical features per option: TF-IDF cosine(state, option text) and cosine(instructions, option text)."""
    K = max(len(i['opts']) for i in items)
    out = np.zeros((len(items), K, 2), dtype=np.float32)
    cache = {}

    def vec(t):
        if t not in cache:
            cache[t] = tf.vec(t)
        return cache[t]
    for n, i in enumerate(items):
        s, q = vec(i['state']), vec(i['instr'])
        for j, o in enumerate(i['opts']):
            ov = vec(o)
            out[n, j] = (cos(s, ov), cos(q, ov))
    return out
