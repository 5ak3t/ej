"""GLiClass v3 (knowledgator/gliclass-edge-v3.0, 32.7M) via its own package: gliclass.ZeroShotClassificationPipeline.

Install: pip install --no-deps gliclass==0.1.20 (pure Python; needs torch + transformers >= 5, tqdm, packaging). One call per
question: text = state, labels = the options (unique texts), prompt = the question's instructions (the pipeline's task
description), threshold 0 so every label's score comes back. choice / score / described noul: the sigmoid scores are turned
back into logits and softmaxed over the options (a single-label read of the same forward pass). noul without option
descriptions: label = the instructions, P(true) = its sigmoid score (the documented multi-label use)."""
import functools
import math

from . import common

MODELS = {'gliclass-edge': ('knowledgator/gliclass-edge-v3.0', 'df03993a2ed98e5e4a0d2dd7efbbd105abe874cf')}


@functools.lru_cache(maxsize=1)
def load(name):
    """Load the model once per process (cached)."""
    common.setup_torch()
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
    repo, rev = MODELS.get(name, (name, None))
    model = GLiClassModel.from_pretrained(repo, revision=rev).float().eval()
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, add_prefix_space=True)
    return ZeroShotClassificationPipeline(model, tok, max_classes=256, classification_type='multi-label', device='cpu',
                                          progress_bar=False)


def unique_labels(q):
    """Label texts for the options, made unique by trailing spaces."""
    seen, labels = set(), []
    for o in q['options']:
        t = o['text'] if o['text'].lower().startswith(o['key'].lower()) or common.option_desc(o) is None \
            else f"{o['key'].replace('_', ' ')}: {o['text']}"
        while t in seen:
            t += ' '
        seen.add(t)
        labels.append(t)
    return labels


def scores(pipe, text, labels, prompt):
    """Sigmoid score per label from one pipeline call."""
    res = pipe(text, labels, threshold=0.0, prompt=prompt)[0]
    by = {d['label']: d['score'] for d in res}
    return [by[l] for l in labels]


def logit(p):
    """Logit of a probability (clipped away from 0 and 1)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def predict(records, name='gliclass-edge'):
    """[{qid: probs}] per record, in each question's option order."""
    pipe = load(name)
    out = []
    for r in records:
        text, pr = common.state_text(r), {}
        for qid, q in r['questions'].items():
            if q['type'] == 'noul' and common.generic_noul(q):
                pt = scores(pipe, text, [q['instructions']], None)[0]
                pr[qid] = common.finalize([pt if o['key'] == 'true' else 1 - pt for o in q['options']])
            else:
                s = scores(pipe, text, unique_labels(q), q['instructions'])
                pr[qid] = common.finalize(common.softmax([logit(x) for x in s]))
        out.append(pr)
    return out
