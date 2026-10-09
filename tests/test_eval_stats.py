"""Falsification tests of the evaluation statistics (ej.eval.stats); no weights, no data.

Falsified if: a perfectly calibrated synthetic suite is not 'calibrated' (observed ECE15 above the null q95) or a clearly
overconfident one is; the record-cluster CI changes when every question is duplicated inside its record; the group-macro t
interval covers the true macro < .94 of the time on a zs_wide-like design (groups per source 83 / 43 / 23 / 5, questions per group
12 / 48 / 36 / 12, beta-binomial truth), for all sources, the real ones or the 5-group source; a one-group source gets a
zero-width interval; the primary statistic includes the synthetic source; or a CA threshold certified on one suite is claimed
for another on which its risk exceeds .10."""
import math
import random

from ej.eval import stats as ES

DESIGN = {'sni': (83, 12, .447, .240), 'glmsyn': (43, 48, .371, .123), 'sgd': (23, 36, .457, .160), 'abcd': (5, 12, .183, .109)}


def _rows(n, shift, seed=0):
    rng, out = random.Random(seed), []
    for i in range(n):
        c = rng.uniform(.4, .95)
        y = 0 if rng.random() < max(c - shift, 0) else 1
        out.append((f'r{i // 2}', [c, 1 - c], y))
    return out


def test_floor_separates_calibrated_from_overconfident():
    assert ES.ece_floor(_rows(600, 0.0))['calibrated'] is True
    bad = ES.ece_floor(_rows(600, 0.25))
    assert bad['calibrated'] is False and bad['ece15'] > bad['null_q95']


def test_cluster_ci_ignores_question_duplication():
    rs = _rows(400, .05)
    assert ES.cluster_ci(rs, draws=300) == ES.cluster_ci([x for x in rs for _ in range(2)], draws=300)


def _beta(m, sd):
    c = m * (1 - m) / (sd * sd) - 1
    return max(m * c, .05), max((1 - m) * c, .05)


def test_macro_t_coverage_on_a_zs_wide_like_design():
    rng, reps = random.Random(2), 1500
    cover = {'all': 0, 'real': 0, 'abcd': 0}
    truth = {'all': sum(v[2] for v in DESIGN.values()) / 4, 'real': sum(DESIGN[s][2] for s in ES.REAL) / 3, 'abcd': DESIGN['abcd'][2]}
    for _ in range(reps):
        acc = {}
        for s, (G, q, m, sd) in DESIGN.items():
            a, b = _beta(m, math.sqrt(max(sd * sd - m * (1 - m) / q, 1e-4)))
            for g in range(G):
                p = rng.betavariate(a, b)
                acc[f'{s}/g{g}'] = sum(rng.random() < p for _ in range(q)) / q
        for k, keep in (('all', None), ('real', ES.REAL), ('abcd', ('abcd',))):
            r = ES.macro_t(acc, keep)
            cover[k] += r['lo'] <= truth[k] <= r['hi']
    cov = {k: v / reps for k, v in cover.items()}
    assert all(v >= 0.94 for v in cov.values()), cov


def test_macro_t_degenerate_and_primary():
    assert ES.macro_t({'a/g': .5})['lo'] is None  # no zero-width CI from one group
    r = ES.macro_t({'sni/1': .2, 'sni/2': .6, 'glmsyn/1': 1.0, 'glmsyn/2': 1.0, 'sgd/1': .3, 'sgd/2': .5}, ES.REAL)
    assert r['sources'] == ['sgd', 'sni'] and r['groups'] == 4 and abs(r['macro'] - .4) < 1e-12 and r['lo'] < .4 < r['hi']
    assert 'glmsyn' not in ES.REAL


def _ca_rows(n, shift, seed):
    rng = random.Random(seed)
    return [(f'r{i}', [c, 1 - c], 0 if rng.random() < max(c - shift, 0) else 1) for i in range(n) for c in [rng.uniform(.5, 1)]]


def test_ca_threshold_is_not_claimed_across_suites():
    assert ES.claim_allowed('td', 'td') and not ES.claim_allowed('td', 'zs_wide')
    a = _ca_rows(2000, 0.0, 1)
    tr = ES.ca_transfer(a, _ca_rows(2000, 0.3, 2))
    assert tr['t'] is not None and tr['risk_b'] > ES.RISK and tr['claim_ok'] is False, tr
    same = ES.ca_transfer(a, _ca_rows(2000, 0.0, 3))
    assert same['risk_b'] <= ES.RISK + 0.03, same


def test_suite_stats_report_scope(tmp_path):
    st = ES.suite_stats(_rows(200, 0.0), draws=50)
    assert set(st['ci']) == {'nll', 'acc', 'ece15'} and 'calibrated' in st['ece_floor'] and 'this suite only' in st['ca_scope']


def test_ledger_chain_detects_edits(tmp_path):
    led = str(tmp_path / 'ledger.jsonl')
    for i in range(3):
        ES.ledger_append(led, {'event': 'compare', 'i': i})
    assert ES.ledger_verify(led) and len(ES.ledger_rows(led)) == 3
    lines = open(led).read().splitlines()
    open(led, 'w').write('\n'.join(lines[:1] + lines[2:]) + '\n')
    assert not ES.ledger_verify(led)
