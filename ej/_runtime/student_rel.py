"""Relational evidence tokens of a structured state: comparisons, sums and memberships between JSON fields rendered as tokens
and crossed with option slots, used as sparse features by an expert."""
import itertools

import student_feat as F

MAX_STR = 64
MAX_NUM = 40  # numeric leaves kept per state (pairs grow quadratically)


def _walk(obj, path, num, strs, lists, objs):
    if isinstance(obj, dict):
        objs.append((path, obj))
        for k, v in obj.items():
            _walk(v, f'{path}.{k}' if path else str(k), num, strs, lists, objs)
    elif isinstance(obj, list):
        scal = [v for v in obj if not isinstance(v, (dict, list))]
        if scal:
            lists.setdefault(path, set()).update(str(v).lower() for v in scal)
        for v in obj:
            if isinstance(v, (dict, list)):
                _walk(v, f'{path}[]', num, strs, lists, objs)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        num.setdefault(path, float(obj))
    elif isinstance(obj, str) and len(obj) <= MAX_STR:
        strs.setdefault(path, obj.strip().lower())


def _derived(objs):
    """Sums and sums of pairwise products of numeric fields over the elements of each list of objects."""
    out = {}
    for path, o in objs:
        if not path.endswith('[]'):
            continue
        ks = sorted(k for k, v in o.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
        for k in ks:
            out[f'{path}.{k}#sum'] = out.get(f'{path}.{k}#sum', 0.0) + float(o[k])
        for a, b in itertools.combinations(ks, 2):
            out[f'{path}.{a}*{b}#sum'] = out.get(f'{path}.{a}*{b}#sum', 0.0) + float(o[a]) * float(o[b])
    return out


def _cmp(x, y):
    d = abs(x - y)
    m = max(abs(x), abs(y))
    if d <= 1e-9 * max(m, 1.0):
        return 'eq'
    s = 'lt' if x < y else 'gt'
    if d <= 0.05 or d <= 1e-3 * m:
        return s + '_tiny'
    return s + ('_small' if d <= 0.02 * m else '_large')


def tokens(state):
    """Sorted relational tokens of a JSON state ([] for plain text)."""
    obj = F.parse_json(state)
    if obj is None:
        return []
    num, strs, lists, objs = {}, {}, {}, []
    _walk(obj, '', num, strs, lists, objs)
    num.update(_derived(objs))
    keys = sorted(num)[:MAX_NUM]
    out = set()
    for a, b in itertools.combinations(keys, 2):
        out.add(f'~{a}|{b}={_cmp(num[a], num[b])}')
    sk = sorted(strs)
    for a, b in itertools.combinations(sk, 2):
        if strs[a] == strs[b]:
            out.add(f'={a}|{b}')
    for b, vals in lists.items():  # membership: the true links + one any/none summary per list (absence is the default)
        hits = [a for a in sk if a != b and strs[a] in vals] + [a for a in keys if not a.endswith('#sum') and _numstr(num[a]) in vals]
        out.update(f'@{a}|{b}=in' for a in hits)
        out.add(f'@*|{b}={"in" if hits else "none"}')
    return sorted(out)


def _numstr(x):
    return str(int(x)) if float(x).is_integer() else str(x)


# ---- relational EXPERT: a sparse conditional logit over (option slot x [bias, field tokens, relational tokens]) with BINARY
# features (the wide expert scales tokens by 1/sqrt(#tokens), which shrinks ~250 relational tokens per invoice state to ~0.06 each
# and lets its single L2 erase them; pool-only probe, 20% held-out by id: invoice duplicate .39 -> .15, matches_order .66 -> .40).
# Silent on unseen option slots (as the field-attention expert); the pool weighs it per key.
REL_L2 = (1e-4, 1e-3, 1e-2, 1e-1)


def _feat(item, cache):
    """Structured (JSON) states only: plain-text states carry no field relations (the wide expert already covers their words)."""
    if item['state'] not in cache:
        js = F.parse_json(item['state']) is not None
        cache[item['state']] = ['__bias__'] + F.state_tokens(item['state']) + tokens(item['state']) if js else []
    return cache[item['state']]


def vocab(items, slot):
    cnt, cache = {}, {}
    for i in items:
        for j in range(len(i['opts'])):
            s = slot(i, j)
            for t in _feat(i, cache):
                cnt[(s, t)] = cnt.get((s, t), 0) + 1
    return {k: n for n, k in enumerate(sorted(k for k, c in cnt.items() if c >= 2))}


def bags(voc, items, K, slot):
    """embedding_bag inputs (binary weights) over the (N, K) option grid."""
    import torch
    idx, off, cache = [], [], {}
    for i in items:
        f = _feat(i, cache)
        for j in range(K):
            off.append(len(idx))
            if j < len(i['opts']):
                s = slot(i, j)
                idx.extend(voc[(s, t)] for t in f if (s, t) in voc)
    return (torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long), torch.ones(len(idx)), None)
