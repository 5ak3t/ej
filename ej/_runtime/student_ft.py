"""Partial fine-tuning plumbing: splits the quantised encoder into a frozen lower stack and a trainable top stack, caching the
lower stack's token states on disk so that only the top layers run."""
import copy
import glob
import hashlib
import json
import os

import numpy as np
import torch

import student_enc as E


class _Pass(torch.nn.Module):
    """Replaces the embedding block of the top stack: its input is already a hidden state of layer n-K."""

    def forward(self, input_ids=None, inputs_embeds=None, **kw):
        return inputs_embeds


def split(model, bits, k):
    """(tokenizer, frozen lower stack, trainable top stack) from the k-bit encoder."""
    tok, enc = E._ENC[E.load(model, bits)]
    n = len(enc.encoder.layer)
    lower = copy.deepcopy(enc); lower.encoder.layer = lower.encoder.layer[:n - k]; lower.pooler = None
    top = copy.deepcopy(enc); top.embeddings = _Pass(); top.encoder.layer = top.encoder.layer[n - k:]; top.pooler = None
    for p in lower.parameters():
        p.requires_grad_(False)
    return tok, lower.eval(), top


def _key(t):
    return hashlib.sha256(t.encode()).hexdigest()[:24]


def lower_states(tok, lower, texts, max_len, tag, batch=64):
    """{text: (len, d) fp16 token states after the frozen lower stack}; disk cache in shards under E.CACHE/<tag>/."""
    d = f'{E.CACHE}/{tag}'; os.makedirs(d, exist_ok=True)
    index = {}
    for f in sorted(glob.glob(f'{d}/*.json')):
        mat = np.load(f[:-5] + '.npy', mmap_mode='r')
        for k, (a, n) in json.load(open(f)).items():
            index[k] = (mat, a, n)
    uniq = sorted(set(texts), key=len); todo = [t for t in uniq if _key(t) not in index]
    if todo:
        rows, idx, at = [], {}, 0
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                b = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
                h = lower(**b).last_hidden_state.half().numpy()
                for t, hh, n in zip(chunk, h, b['attention_mask'].sum(1).tolist()):
                    rows.append(hh[:n]); idx[_key(t)] = (at, n); at += n
        name = f'{d}/{os.getpid()}_{len(glob.glob(d + "/*.json"))}'
        np.save(name + '.tmp.npy', np.concatenate(rows)); os.replace(name + '.tmp.npy', name + '.npy')
        json.dump(idx, open(name + '.tmp', 'w')); os.replace(name + '.tmp', name + '.json')
        mat = np.load(name + '.npy', mmap_mode='r')
        for k, (a, n) in idx.items():
            index[k] = (mat, a, n)
    out = {}
    for t in uniq:
        mat, a, n = index[_key(t)]
        out[t] = torch.from_numpy(np.array(mat[a:a + n]))
    return out


def size_top_mb(top, bits):
    """(size of the top stack at k-bit as counted in E.size_mb, size at fp16)."""
    qn = sum(m.weight.numel() for m in top.modules() if E._q(m)); tot = sum(p.numel() for p in top.parameters())
    return (qn * (bits + 16 / E.GROUP) + 32 * (tot - qn)) / 8 / 2 ** 20, tot * 16 / 8 / 2 ** 20
