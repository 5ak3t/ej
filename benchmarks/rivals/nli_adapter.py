"""NLI zero-shot classifiers (Yin et al. 2019 recipe, as in the transformers zero-shot pipeline) for typed questions.

Premise = the state. choice / score / described noul: one hypothesis per option, "<instructions> <option>", and the answer is
the softmax over the options' ENTAILMENT logits (the pipeline's single-label rule). noul without option descriptions: one
hypothesis "<instructions>", P(true) = softmax([contradiction, entailment])[entailment] (the pipeline's multi-label rule; for
2-class models 'not_entailment' plays contradiction). Premise truncated to fit 512 tokens ('only_first').
Models: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 (tested), cross-encoder/nli-deberta-v3-xsmall (tested),
MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c and facebook/bart-large-mnli (same code; not run here)."""
import functools

from . import common

MODELS = {'nli-xsmall-zs': ('MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33', '262ae02f29173eec1c250f90804dc7edc677dcff'),
          'nli-ce-xsmall': ('cross-encoder/nli-deberta-v3-xsmall', 'a150876415327c80daeff35ca6f68f5ed8cf5c24'),
          'nli-large-c': ('MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c', 'b2730f16019076bb0009481121efbe4705e0e378'),
          'bart-mnli': ('facebook/bart-large-mnli', 'd7645e127eaf1aefc7862fd59a17a5aa8558b8ce')}
MAX_LEN, CHUNK = 512, 32


@functools.lru_cache(maxsize=1)
def load(name):
    """Load the model once per process (cached)."""
    torch = common.setup_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    repo, rev = MODELS.get(name, (name, None))
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    labels = {i: str(l).lower() for i, l in model.config.id2label.items()}
    ent = next(i for i, l in labels.items() if l.startswith('entail'))
    con = next((i for i, l in labels.items() if l.startswith('contra')), None)
    if con is None:
        con = next(i for i, l in labels.items() if i != ent)  # 2-class: not_entailment
    return torch, tok, model, ent, con


def label_text(o):
    """Option text used in the hypothesis."""
    d = common.option_desc(o)
    if d is None or o['text'].lower().startswith(o['key'].lower()):
        return o['text']
    return f"{o['key'].replace('_', ' ')}: {o['text']}"


def hypotheses(q):
    """-> (list of hypotheses, mode) with mode 'options' (softmax over entailment) or 'binary' (entail vs contradiction)."""
    if q['type'] == 'noul' and common.generic_noul(q):
        return [q['instructions']], 'binary'
    return [f"{q['instructions']} {label_text(o)}".strip() for o in q['options']], 'options'


def logits(name, premise, hyps):
    """NLI logits for (premise, hypothesis) pairs, in chunks."""
    torch, tok, model, _, _ = load(name)
    rows = []
    with torch.inference_mode():
        for i in range(0, len(hyps), CHUNK):
            h = hyps[i:i + CHUNK]
            enc = tok([premise] * len(h), h, truncation='only_first', max_length=MAX_LEN, padding=True, return_tensors='pt')
            rows += model(**enc).logits.float().tolist()
    return rows


def predict(records, name='nli-xsmall-zs'):
    """[{qid: probs}] per record, in each question's option order."""
    _, _, _, ent, con = load(name)
    out = []
    for r in records:
        prem, pr = common.state_text(r), {}
        for qid, q in r['questions'].items():
            hyps, mode = hypotheses(q)
            z = logits(name, prem, hyps)
            if mode == 'options':
                pr[qid] = common.finalize(common.softmax([row[ent] for row in z]))
            else:
                pt = common.softmax([z[0][con], z[0][ent]])[1]
                pr[qid] = common.finalize([pt if o['key'] == 'true' else 1 - pt for o in q['options']])
        out.append(pr)
    return out
