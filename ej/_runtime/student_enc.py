"""Sentence encoder: intfloat/e5-small-v2 with weight-only k-bit quantisation (symmetric round-to-nearest, one fp16 scale per
group of weights), mean-pooled L2-normalised embeddings, and an on-disk embedding cache under $EDGE_CACHE."""
import hashlib
import json
import os

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

GROUP = 32
MAX_LEN = 256
BATCH = 32
CACHE = os.environ.get('EDGE_CACHE', os.path.expanduser('~/.cache/ej'))
_ENC = {}


def _quant(w, bits):
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, GROUP)
    s = (g.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax).half().float()
    return (torch.round(g / s).clamp(-qmax, qmax) * s).reshape(w.shape)


def _q(m):
    return isinstance(m, (torch.nn.Embedding, torch.nn.Linear))


def load(model, bits):
    key = f'{model}#q{bits}'
    if key not in _ENC:
        enc = AutoModel.from_pretrained(model).float().eval()
        if bits < 16:
            with torch.no_grad():
                for m in enc.modules():
                    if _q(m):
                        m.weight.copy_(_quant(m.weight, bits))
        _ENC[key] = (AutoTokenizer.from_pretrained(model), enc)
    return key


def size_mb(model, bits):
    enc = _ENC[load(model, bits)][1]
    qn = sum(m.weight.numel() for m in enc.modules() if _q(m))
    tot = sum(p.numel() for p in enc.parameters())
    return (qn * (bits + 16 / GROUP if bits < 16 else 32) + 32 * (tot - qn)) / 8 / 2 ** 20


def embed(model, bits, texts):
    """(N, d) float32 embeddings; cached on disk per (model, bits) by sha256 of the text."""
    key = load(model, bits)
    os.makedirs(CACHE, exist_ok=True)
    base = f'{CACHE}/{key.replace("/", "_")}'
    index = json.load(open(base + '.json')) if os.path.exists(base + '.json') else {}
    mat = np.load(base + '.npy') if os.path.exists(base + '.npy') else None
    h = [hashlib.sha256(t.encode()).hexdigest()[:24] for t in texts]
    todo = [t for t, k in dict(zip(texts, h)).items() if k not in index]
    if todo:
        tok, enc = _ENC[key]
        new = []
        with torch.no_grad():
            for i in range(0, len(todo), BATCH):
                b = tok(todo[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
                hs = enc(**b).last_hidden_state
                m = b['attention_mask'].unsqueeze(-1).float()
                new.append(torch.nn.functional.normalize((hs * m).sum(1) / m.sum(1), dim=-1).numpy())
        new = np.concatenate(new).astype(np.float32)
        start = 0 if mat is None else len(mat)
        mat = new if mat is None else np.concatenate([mat, new])
        for j, t in enumerate(todo):
            index[hashlib.sha256(t.encode()).hexdigest()[:24]] = start + j
        tmp = base + f'.{os.getpid()}'
        np.save(tmp + '.npy', mat); json.dump(index, open(tmp + '.json', 'w'))
        os.replace(tmp + '.npy', base + '.npy'); os.replace(tmp + '.json', base + '.json')
    return mat[[index[k] for k in h]]
