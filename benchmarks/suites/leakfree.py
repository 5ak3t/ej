#!/usr/bin/env python3
"""Leak-free (EXCHANGEABLE) candidate sampling for sampled-choice suites. It writes new files and never touches raw data. Usage:
  python benchmarks/suites/leakfree.py --source massive --split validation --n 1000 --out OUT/dev/zs_massive.jsonl [--mode cps]
  python benchmarks/suites/leakfree.py --source banking77|clinc150|goemotions --n 3000 --out FILE [--mode cps|quota|full] [--balance] [--shuffle]

Why. sources.sampled_choice draws gold + (k-1) distractors UNIFORMLY from the other labels, so P(gold = j | S) = pi_j / sum_S pi
(pi = gold shares): the option strings' frequency ACROSS items reveals gold (an option-count rule on a non-leak-free zs_massive: .631 vs
chance .240). The prefix selection (rows[:CAP] / [:1000]) makes it worse: those files are sorted, so many labels are NEVER gold in
the suite (MASSIVE val first 1000: 39 of 59 option labels; banking77 first 3000: 24 of 77; clinc first 3000: 30 of 150) and occur
only as distractors.

Construction (mode 'cps'). A suite is leak-free iff P(gold = j | S) = 1/|S| for all j in S: then no function of the option sets,
pooled over any number of items, beats chance. Draw the distractors by CONDITIONAL POISSON SAMPLING (CPS): given gold y and size k,
  P(D | y) = prod_{l in D} w_l / e_{k-1}(w without y),       e_r = elementary symmetric polynomial,
with weights w^(k) solved (iterative proportional fitting; Chen, Dempster & Liu 1994, Biometrika 81:457) so that the size-k CPS
inclusion probabilities over ALL labels equal k pi_l. Then P(S, gold = y) = pi_y prod_D w / e_{k-1}(w_-y)
= prod_S w / (k e_k(w)), independent of y in S: exactly exchangeable. In particular every label occurs as a distractor (k-1) times
as often as it occurs as gold (frequency-matched), and labels that are never gold are never options. Requires k max(pi) <= 1: k is
capped at floor(1 / max pi) (reported), or --balance drops records of over-represented labels (deterministic, by record-id hash)
until max pi <= 1 / (kmax + 1) (margin). k and the shuffle use the same per-record seed as sources.sampled_choice, so the
leak-free suite has the same texts, ids, instructions and (uncapped) option counts as the leaky one; only the distractor law
changes (in expectation every label is a distractor (k-1) times as often as it is gold).
Mode 'quota' (finite-sample exact): with FIXED gold quotas n_l a pooled method knows P(l is gold | l in S_i, all sets) =
(remaining quota) / (remaining occurrences), so any spread of the realised counts c_l = n_l + D_l leaks a little (equal quotas,
clinc: count_auc .487, gold slightly RARER, because a distractor's count always includes the item it sits in). 'quota' makes the
realised counts exactly proportional, c_l = n_l + T_l with T_l = (total distractor slots) pi_l (largest remainder), filling items
in a gold-independent hash order by CPS over the remaining quotas; then n_l / c_l is constant and occurrence counts carry nothing.
Mode 'full': every label of the suite is an option of every item (a fixed per-source option set; trivially exchangeable).
--shuffle: choose the n records by a seeded hash order instead of the file prefix (covers the label space)."""
import argparse, hashlib, json, os, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sources as S  # noqa: E402  (read-only use of the record format helpers)

INSTR = {'massive': 'What does the user want the assistant to do?', 'banking77': 'Which banking request is this?',
         'clinc150': 'Which request does the user make?', 'goemotions': 'Which emotion does the writer express?'}


def esym(w, r):
    e = np.zeros(r + 1); e[0] = 1.0
    for x in w:
        e[1:] = e[1:] + x * e[:-1]
    return e


def inclusion(w, k):
    """First-order inclusion probabilities of size-k CPS with weights w."""
    e = esym(w, k)
    em = np.zeros((len(w), k)); em[:, 0] = 1.0
    for j in range(1, k):
        em[:, j] = e[j] - w * em[:, j - 1]
    return w * em[:, k - 1] / e[k]


def solve_weights(pi, k, iters=2000, tol=1e-12):
    """w with inclusion(w, k) = k pi (pi > 0, k max pi <= 1). Returns (w, max abs error)."""
    target, w = k * pi, pi / pi.max()
    for _ in range(iters):
        p = inclusion(w, k)
        if np.abs(p - target).max() < tol:
            break
        w = w * target / p; w = w / w.max()
    return w, float(np.abs(inclusion(w, k) - target).max())


def cps_draw(w, r, rng):
    """r distinct indices with P(D) ~ prod_{l in D} w_l (sequential draw-by-suffix with elementary symmetric polynomials)."""
    n = len(w)
    E = np.zeros((n + 1, r + 1)); E[n, 0] = 1.0
    for j in range(n - 1, -1, -1):
        E[j] = E[j + 1]; E[j, 1:] += w[j] * E[j + 1, :-1]
    out, need = [], r
    for j in range(n):
        if need == 0:
            break
        if rng.random() < w[j] * E[j + 1, need - 1] / E[j, need]:
            out.append(j); need -= 1
    return out


def hkey(rid):
    return int(hashlib.sha256(('leakfree:' + rid).encode()).hexdigest()[:12], 16)


def balance(rows, kmax):
    """Drop the highest-hash records of over-represented labels until max gold share <= 1 / (kmax + 1)."""
    rows = list(rows)
    while True:
        by = {}
        for r in rows:
            by.setdefault(r[2], []).append(r)
        cap = len(rows) // (kmax + 1)
        over = {l: v for l, v in by.items() if len(v) > cap}
        if not over:
            return rows
        drop = {r[0] for l, v in over.items() for r in sorted(v, key=lambda r: hkey(r[0]))[cap:]}
        rows = [r for r in rows if r[0] not in drop]


def exchangeable_choice(rows, source, instructions, kmin=2, kmax=8, mode='cps'):
    """rows: [(rid, text, label)] -> (records in the sources.sampled_choice format, report). pi = the rows' gold shares."""
    labs = sorted({r[2] for r in rows})
    cnt = np.array([sum(r[2] == l for r in rows) for l in labs], float); pi = cnt / cnt.sum()
    kcap = min(kmax, int(np.floor(1.0 / pi.max() + 1e-9)))
    ws = {k: solve_weights(pi, k) for k in range(kmin, kcap + 1)} if mode == 'cps' else {}
    ks = [min(S.rng_for(rid).randint(kmin, kmax), kcap) for rid, _, _ in rows]  # same k draw as sources.sampled_choice
    if mode == 'quota':  # realised-exact frequency matching: label l fills T_l ~ n_l distractor slots (largest remainder)
        t = sum(k - 1 for k in ks) * pi; R = np.floor(t).astype(int)
        R[np.argsort(-(t - R))[:sum(k - 1 for k in ks) - R.sum()]] += 1
        dist = {}
        for n in sorted(range(len(rows)), key=lambda n: hkey(rows[n][0])):  # random order w.r.t. gold
            w = R.astype(float); w[labs.index(rows[n][2])] = 0.0
            d = cps_draw(w, min(ks[n] - 1, int((w > 0).sum())), np.random.default_rng(hkey(rows[n][0])))
            R[d] -= 1; dist[n] = [labs[j] for j in d]
    out, capped = [], 0
    for n, (rid, text, label) in enumerate(rows):
        rng = S.rng_for(rid)
        capped += rng.randint(kmin, kmax) > kcap; k = ks[n]
        if mode == 'full':
            opts = list(labs)
        elif mode == 'quota':
            opts = [label] + dist[n]
        else:
            y = labs.index(label); w = ws[k][0].copy(); w[y] = 0.0  # distractors: CPS over the other labels (same w^(k))
            opts = [label] + [labs[j] for j in cps_draw(w, k - 1, np.random.default_rng(hkey(rid)))]
        rng.shuffle(opts)
        q = {'type': 'choice', 'instructions': instructions, 'options': [{'key': o, 'text': S.human(o)} for o in opts]}
        out.append({'id': rid, 'source': source, 'state': text, 'questions': {'intent': q},
                    'gold': {'intent': {'label': opts.index(label), 'probs': None}}})
    if mode == 'quota':
        capped = {'capped': capped, 'short_items': sum(len(dist[n]) < ks[n] - 1 for n in dist), 'unfilled_slots': int(R.sum())}
    rep = {'n': len(out), 'labels': len(labs), 'max_pi': round(float(pi.max()), 4), 'kcap': kcap, 'capped': capped,
           'ipf_max_err': max([e for _, e in ws.values()], default=0.0), 'mode': mode}
    return out, rep


def load_rows(raw, source, split, n, shuffle):
    """(rid, text, label) rows with the builder's ids; prefix (builder) or seeded hash order (--shuffle)."""
    if source == 'massive':
        m = pq.read_table(f'{raw}/massive/{split}.parquet').to_pylist()
        rows = [(f'massive-{split}-{r["id"]}', r['text'], r['label']) for r in m]
    elif source == 'banking77':
        rows = [(f'b77-{i}', r['text'], r['label_text']) for i, r in enumerate(pq.read_table(f'{raw}/banking77/train.parquet').to_pylist())]
    elif source == 'clinc150':
        names = json.load(open(f'{raw}/clinc_oos-plus-intent.json'))
        lab = lambda r: 'none of these' if names[str(r['intent'])] == 'oos' else names[str(r['intent'])]
        rows = [(f'clinc-{i}', r['text'], lab(r)) for i, r in enumerate(pq.read_table(f'{raw}/clinc/train.parquet').to_pylist())]
    elif source == 'goemotions':
        emo = json.load(open(f'{raw}/go_emotions-simplified-labels.json')); en = [emo[k] for k in sorted(emo, key=int)]
        rows = [(f'ge-{r["id"]}', r['text'], en[r['labels'][0]]) for r in pq.read_table(f'{raw}/goemotions/train.parquet').to_pylist()
                if len(r['labels']) == 1]
    else:
        raise ValueError(source)
    return (sorted(rows, key=lambda r: hkey(r[0])) if shuffle else rows)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True); ap.add_argument('--source', required=True)
    ap.add_argument('--split', default='train'); ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--mode', default='cps', choices=('cps', 'quota', 'full')); ap.add_argument('--balance', action='store_true')
    ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--kmin', type=int, default=2)
    ap.add_argument('--kmax', type=int, default=8); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = load_rows(a.raw, a.source, a.split, a.n, a.shuffle)
    rows = balance(rows, a.kmax) if a.balance else rows
    recs, rep = exchangeable_choice(rows, a.source, INSTR[a.source], a.kmin, a.kmax, a.mode)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({**rep, 'out': a.out, 'sha256': hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}))


if __name__ == '__main__':
    main()
