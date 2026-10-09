"""Shared helpers for rival adapters: evaluator records <-> the System One wire format, and output normalisation.

Benchmark record (the ej input format): {'id', 'state', 'questions': {qid: {'type', 'instructions', 'options': [{'key', 'text'}]}}}.
Every adapter returns, per record, {qid: [p_0, ..., p_{k-1}]} in the record's OPTION ORDER (choice: given options; noul: the
record's own order of the 'false'/'true' options; score: ordered levels), finite, >= 0, summing to 1 (scoring.check)."""
import json
import math
import os

GENERIC_NOUL = {'No, the statement does not hold.', 'Yes, the statement holds.'}  # sources.NOUL_TEXT (no real criteria)
EPS = 1e-6  # additive smoothing after renormalising: a rival's 4-dp-rounded 0.0 is not scored as -log(1e-15)


def setup_torch():
    """Light local CPU: one thread unless BENCH_THREADS says otherwise. NB transformers >= 5 loads a checkpoint in
    its STORED dtype by default; the xsmall zero-shot NLI checkpoint is stored in fp16 and fp16 matmuls on this CPU are
    ~15x slower (3.5 s vs ~0.2 s per 3-pair forward, measured), so HF adapters load in float32."""
    import torch
    torch.set_num_threads(int(os.environ.get('BENCH_THREADS', '1')))
    return torch


def state_text(r):
    """The record state as text (a dict state is serialised as JSON)."""
    s = r['state']
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def option_desc(o):
    """Description of a choice option for the wire format: None when the text only repeats the key."""
    k, t = o['key'], (o['text'] or '').strip()
    if t.startswith(k + ': '):
        t = t[len(k) + 2:].strip()
    return None if t in ('', k) else t


def generic_noul(q):
    """True when a noul question carries only the generic yes/no option texts."""
    return all(o['text'] in GENERIC_NOUL for o in q['options'])


def to_wire(q):
    """One evaluator question -> one System One question (Jev / TypeSafe wire format)."""
    t, ins = q['type'], q['instructions']
    if t == 'choice':
        return {'type': 'choice', 'instructions': ins, 'criteria': {o['key']: option_desc(o) for o in q['options']}}
    if t == 'score':
        return {'type': 'score', 'instructions': ins, 'criteria': [o['text'] for o in q['options']]}
    w = {'type': 'noul', 'instructions': ins}
    if not generic_noul(q):
        w['criteria'] = {o['key']: o['text'] for o in q['options']}
    return w


def wire_questions(r):
    """All questions of record r in the System One wire format."""
    return {qid: to_wire(q) for qid, q in r['questions'].items()}


def finalize(v):
    """Clip to >= 0, renormalise (uniform if degenerate), add EPS smoothing. Returns plain floats summing to 1."""
    v = [float(x) if x is not None and math.isfinite(float(x)) and float(x) > 0 else 0.0 for x in v]
    s = sum(v)
    if s <= 0:
        v, s = [1.0] * len(v), float(len(v))
    v = [x / s + EPS for x in v]
    s = sum(v)
    return [x / s for x in v]


def softmax(z):
    """Softmax of a list of logits."""
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def from_answer(q, ans):
    """A System One answer object (dict) -> probabilities in the record's option order."""
    probs = ans.get('probabilities') or {}
    probs = {str(k).lower() if isinstance(k, bool) else str(k): v for k, v in probs.items()}
    if q['type'] == 'choice':
        return finalize([probs.get(o['key'], 0.0) for o in q['options']])
    if q['type'] == 'score':
        return finalize([probs.get(str(i), 0.0) for i in range(len(q['options']))])
    if 'true' in probs and 'false' in probs:
        pt = probs['true'] / max(probs['true'] + probs['false'], 1e-300)
    else:
        pt = ans.get('noul_raw') if ans.get('noul_raw') is not None else ans['noul']
    return finalize([pt if o['key'] == 'true' else 1.0 - pt for o in q['options']])


def from_response(r, answers):
    """{qid: answer} (System One response 'answers') -> {qid: probs} for record r."""
    return {qid: from_answer(q, answers[qid]) for qid, q in r['questions'].items()}


def to_dict(x):
    """Pydantic answer objects (von, kev) -> dicts."""
    if hasattr(x, 'model_dump'):
        return x.model_dump()
    if hasattr(x, 'dict') and not isinstance(x, dict):
        return x.dict()
    return x


def unround(module):
    """Make a rival module's output rounding (round(x, n) with n >= 4) a no-op so its answers keep full precision.
    Integer rounding and coarser rounding are untouched; the model and its calibration are unchanged."""
    import builtins
    module.round = lambda x, n=None: x if n is not None and n >= 4 else builtins.round(x, n)
