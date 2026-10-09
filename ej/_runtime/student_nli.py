"""NLI reader: entailment margins of cross-encoder/nli-deberta-v3-xsmall between state texts and per-option hypotheses, turned
into expert features. The cross-encoder is a fit-time teacher; at prediction time its logits come from the distilled reader
(student_dn)."""
import glob
import hashlib
import json
import os

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
import numpy as np  # noqa: E402
import sentencepiece as spm  # noqa: E402
import torch  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as pb  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

import student_enc as E  # noqa: E402
import student_feat as F  # noqa: E402
import student_rr as RR  # noqa: E402

MODEL, BITS, NV = 'cross-encoder/nli-deberta-v3-xsmall', 4, 32000
MAX_LEN, BATCH = 160, 32
NF = 9
IO = True  # text view also reads 'instructions + option' hypotheses (choice / score)
USE_FACTS = False  # fact view: LOGO-positive on the 3 pool workflows (yes/no .69 -> .47) but ANTI-transferred to the dev held-out
#   workflow -> off; 3 workflows cannot certify transfer
TEXT_JSON = False  # text view on JSON states: no transfer in the pool LOGO probe (1.2311 = uniform) -> off (saves its passes)
NR = 6  # weight rows: question type x (structured state); T passed as type + 3 * json
L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
_M = {}


def load():
    """(segmenter restricted to the NV kept pieces, 4-bit model)."""
    if 'm' not in _M:
        d = glob.glob(f"{os.environ['HF_HOME']}/hub/models--{MODEL.replace('/', '--')}/snapshots/*/")[0]
        mp = pb.ModelProto(); mp.ParseFromString(open(d + 'spm.model', 'rb').read())
        for i, p in enumerate(mp.pieces):
            if i >= NV and p.type == 1:
                p.type = 5  # UNUSED: never emitted by the segmenter
        sp = spm.SentencePieceProcessor(model_proto=mp.SerializeToString())
        mod = AutoModelForSequenceClassification.from_pretrained(d).float().eval()
        with torch.no_grad():
            for m in mod.modules():
                if E._q(m):
                    m.weight.copy_(E._quant(m.weight, BITS))
        _M['m'] = (sp, mod)
    return _M['m']


def size_mb():
    """Device bytes: 4-bit (+ fp16 scale per 32) for Linear/Embedding weights, the word embedding counted for the NV kept rows only;
    fp32 for biases / LayerNorms."""
    mod = load()[1]
    wemb = mod.deberta.embeddings.word_embeddings.weight
    qn = sum(m.weight.numel() for m in mod.modules() if E._q(m)) - wemb.numel() + NV * wemb.shape[1]
    tot = sum(p.numel() for p in mod.parameters()) - wemb.numel() + NV * wemb.shape[1]
    return (qn * (BITS + 16 / E.GROUP) + 32 * (tot - qn)) / 8 / 2 ** 20


def _ids(sp, a, b):
    ia, ib = sp.encode(a)[:MAX_LEN - 4 - min(len(sp.encode(b)), 40)], sp.encode(b)[:40]
    return [1] + ia + [2] + ib + [2]


def _read(base):
    if not os.path.exists(base + '.json'):
        return {}, []
    return json.load(open(base + '.json')), list(np.load(base + '.npy'))


def margins(pairs):
    """(N, 3) logits [contradiction, entailment, neutral] per (premise, hypothesis); disk cache keyed by sha256 of the pair."""
    base = f"{E.CACHE}/nli_{MODEL.replace('/', '_')}#q{BITS}#v{NV}#{MAX_LEN}"
    os.makedirs(E.CACHE, exist_ok=True)
    index, vals = _read(base)
    keys = [hashlib.sha256((p + '\x00' + h).encode()).hexdigest()[:24] for p, h in pairs]
    todo = {k: ph for k, ph in zip(keys, pairs) if k not in index}
    if todo:
        sp, mod = load()
        enc = sorted(((k, _ids(sp, *ph)) for k, ph in todo.items()), key=lambda x: (len(x[1]), x[0]))
        new = {}
        with torch.no_grad():
            for i in range(0, len(enc), BATCH):
                ch = enc[i:i + BATCH]; L = max(len(c[1]) for c in ch)
                ids = torch.tensor([c[1] + [0] * (L - len(c[1])) for c in ch])
                z = mod(input_ids=ids, attention_mask=(ids > 0).long()).logits
                new.update({k: v for (k, _), v in zip(ch, z.tolist())})
                if len(new) >= 20000 or i + BATCH >= len(enc):  # merge with what other processes saved meanwhile
                    index, vals = _read(base)
                    for k, v in new.items():
                        if k not in index:
                            index[k] = len(vals); vals.append(v)
                    tmp = base + f'.{os.getpid()}'
                    np.save(tmp + '.npy', np.array(vals, dtype=np.float32).reshape(-1, 3)); json.dump(index, open(tmp + '.json', 'w'))
                    os.replace(tmp + '.npy', base + '.npy'); os.replace(tmp + '.json', base + '.json'); new = {}
    return np.array([vals[index[k]] for k in keys], dtype=np.float32).reshape(-1, 3)


def text_of(state):
    """Natural-language part of a state: the text itself, or a JSON state's string leaves with >= 4 words ('' if none)."""
    obj = F.parse_json(state)
    if obj is None:
        return state
    out = []

    def walk(x):
        if isinstance(x, dict):
            [walk(v) for _, v in sorted(x.items())]
        elif isinstance(x, list):
            [walk(v) for v in x]
        elif isinstance(x, str) and len(x.split()) >= 4:
            out.append(x)
    walk(obj)
    return ' '.join(out)


def _m(z):
    return z[..., 1] - z[..., 0]  # entailment - contradiction


def features(items, mu):
    """(N, K, NF) label-agnostic per-option NLI features (zeros on padded slots / missing views)."""
    K = max(len(i['opts']) for i in items)
    fs = {i['state']: RR.facts(i['state']) for i in items if i['json'] and USE_FACTS}
    tx = {i['state']: text_of(i['state']) if TEXT_JSON or not i['json'] else '' for i in items}
    pairs = set()
    for i in items:
        hyp = list(i['opts']) + ([i['instr']] if i['type'] == 1 else [])
        pairs.update((tx[i['state']], h) for h in hyp + io_hyps(i) if tx[i['state']])
        pairs.update((t, h) for f in fs.get(i['state'], []) for t in (f[0], f[2]) for h in fact_hyps(i))
    pairs = sorted(pairs)
    pos = {p: n for n, p in enumerate(pairs)}
    Z = torch.tensor(margins(pairs)) if pairs else torch.zeros(0, 3)
    out = torch.zeros(len(items), K, NF)
    for n, i in enumerate(items):
        k, st = len(i['opts']), i['state']
        pol = torch.tensor([-1.0, 1.0]) if i['type'] == 1 and k == 2 else torch.zeros(k)
        c = lambda v: v - v.mean()
        if tx[st]:
            e = _m(Z[[pos[(tx[st], o)] for o in i['opts']]])
            t = _m(Z[pos[(tx[st], i['instr'])]]) if i['type'] == 1 else torch.tensor(0.0)
            out[n, :k, 0:4] = torch.stack([c(e), c(torch.log_softmax(Z[[pos[(tx[st], o)] for o in i['opts']]], -1)[:, 1]),
                                           pol * t, pol * torch.tanh(t / 4)], 1)
            if io_hyps(i):
                zio = Z[[pos[(tx[st], h)] for h in io_hyps(i)]]
                out[n, :k, 7:9] = torch.stack([c(_m(zio)), c(torch.log_softmax(zio, -1)[:, 1])], 1)
        f = fs.get(st, [])
        if f:
            sg = torch.tensor([1.0 if x[1].endswith('+') else -1.0 for x in f])
            om = 1 - sg * mu[torch.tensor([RR.RELS.index(x[1][:-1] + '+') // 2 for x in f])]
            hyp = fact_hyps(i)
            a = _m(Z[[pos[(x[0], h)] for x in f for h in hyp]]).view(len(f), len(hyp))
            d = a - _m(Z[[pos[(x[2], h)] for x in f for h in hyp]]).view(len(f), len(hyp))  # holds-minus-fails contrast
            w = om / om.sum().clamp(min=1e-6)
            g = torch.stack([d.mean(0), w @ d, w @ a], 1)  # (len(hyp), 3): mean / surprise-weighted contrast, weighted support
            out[n, :k, 4:7] = pol[:, None] * g if i['type'] == 1 else g - g.mean(0, keepdim=True)
    return out


def io_hyps(i):
    """Text-view hypotheses 'instructions + option' for choice / score questions (the question gives short option texts context)."""
    return [f"{i['instr']} {o}" for o in i['opts']] if IO and i['type'] != 1 else []


def fact_hyps(i):
    """Hypotheses read against the fact premises: a yes/no question's statement itself (signed by option polarity later); else
    'instructions + option' per option (the question context disambiguates short option texts)."""
    return [i['instr']] if i['type'] == 1 else [f"{i['instr']} {o}" for o in i['opts']]


def rows(items):
    """Weight row per question: type + 3 * structured state (the fact view exists only for JSON states)."""
    return torch.tensor([i['type'] + 3 * i['json'] for i in items])


def logits(W, X, M, T):
    return (X * W[T][:, None, :]).sum(-1).masked_fill(~M, -1e9)


def train(X, M, T, y, l2):
    """Per-type conditional logit, L2 (one value per type: the loss separates by type) on standardised features (scale folded back),
    L-BFGS (convex; NR x NF parameters)."""
    sd = X[M].std(0).clamp(min=1e-6)
    lam = torch.as_tensor(l2, dtype=torch.float32).expand(NR)[:, None]
    W = torch.zeros(NR, NF, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits(W, X / sd, M, T), y) + (lam * W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (W.detach() / sd).half().float()


def logo_l2(X, M, T, y, gid):
    """L2 per question type by leave-one-group-out NLL (transfer to an unseen source/workflow), decided inside fit."""
    rep = {}
    for l2 in L2_GRID:
        tot = torch.zeros(NR)
        for g in sorted(set(gid)):
            o = torch.tensor(np.asarray(gid) == g)
            nll = torch.nn.functional.cross_entropy(logits(train(X[~o], M[~o], T[~o], y[~o], l2), X[o], M[o], T[o]), y[o],
                                                    reduction='none')
            tot += torch.zeros(NR).index_add(0, T[o], nll)
        rep[l2] = (tot / torch.bincount(T, minlength=NR).clamp(min=1)).tolist()
    best = [min(L2_GRID, key=lambda l: rep[l][t]) for t in range(NR)]
    return best, {t: {l: round(rep[l][t], 4) for l in L2_GRID} for t in range(NR)}


def fit_all(X, M, T, y, hold, gid):
    """The NLI expert inside fit: L2 by LOGO; held-out logits (seen keys), LOGO logits per group (unseen keys), full refit."""
    l2, rep = logo_l2(X, M, T, y, gid)
    tr, ho = torch.tensor(~hold), torch.tensor(hold)
    out = {'ho': logits(train(X[tr], M[tr], T[tr], y[tr], l2), X[ho], M[ho], T[ho]).masked_fill(~M[ho], 0), 'logo': {},
           'full': train(X, M, T, y, l2), 'rep': {'l2': l2, 'logo': rep}}
    for g in sorted(set(gid)):
        o = torch.tensor(np.asarray(gid) == g)
        out['logo'][g] = logits(train(X[~o], M[~o], T[~o], y[~o], l2), X[o], M[o], T[o]).masked_fill(~M[o], 0)
    return out
