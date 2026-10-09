"""Falsification tests of the seed-aware keep rule (ej.eval.compare) and the evaluator command line; no weights, no data.

Falsified if: a NULL candidate (same code, other fit seeds) is kept (BETTER) above the nominal one-sided .025 in simulation, or the
95% CI covers the true difference < .93 of the time (S = 3, fit-noise SD .0065 of a difference, .0030 of it record-level; normal
and t5 noise); a candidate truly worse by the margin (+.010) passes EQUAL > .05 (+ Monte-Carlo slack); the single-fit rule (S = 1,
prior fit noise) keeps a d = -.012 candidate with record SE .003 that a single-fit record bootstrap would keep; or the command
line (score with saved predictions, then compare with a ledger) loses a comparison or accepts an edited ledger."""
import json
import os
import random
import subprocess
import sys

import numpy as np

from ej.eval import compare as DC
from ej.eval import stats as ES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_null_candidate_is_not_kept_and_ci_covers():
    for tdf in (None, 5):
        r = DC.simulate(3, 0.0, reps=600, seed=7, tdf=tdf)
        assert r['BETTER'] <= 0.025 and r['WORSE'] <= 0.025 and r['cover'] >= 0.93, r


def test_tost_margin_controls_false_equivalence():
    for dl in (0.010, -0.010):
        r = DC.simulate(3, dl, reps=600, seed=8)
        assert r['EQUAL'] <= 0.07, (dl, r)  # nominal .05 at the margin; 600 reps -> MC SE ~ .009


def test_single_fit_rule_uses_the_prior_fit_noise():
    rng = np.random.default_rng(0)
    Z = [rng.normal(0, 0.003 * np.sqrt(300), (1, 300))]  # one suite, record SE of the difference ~ .003
    e = DC.estimate(np.array([-0.012]), Z, [1.0], DC.SD_FIT)
    v = DC.verdict(e)
    assert e['fit_noise'] == 'prior' and v['verdict'] == 'INCONCLUSIVE' and v['ci95'][1] > 0, v
    assert DC.verdict(DC.estimate(np.array([-0.012]), Z, [1.0]))['fit_noise'] == 'NOT included'


def test_verdict_semantics():
    e = {'d': -0.02, 'se': 0.003, 'df': 30.0, 'S': 6}
    assert DC.verdict(e)['verdict'] == 'BETTER' and DC.verdict({**e, 'd': 0.0})['verdict'] == 'EQUAL'
    assert DC.verdict({**e, 'd': 0.02})['verdict'] == 'WORSE' and DC.verdict({**e, 'd': -0.008})['verdict'] == 'INCONCLUSIVE'
    acc = DC.verdict({**e, 'd': 0.05}, DC.ACC_D, DC.ACC_D, lower_better=False)
    assert acc['verdict'] == 'BETTER' and acc['p_better'] < 0.001


def _write(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _recs(n, prefix, src=None):
    out = []
    for i in range(n):
        g = None if src is None else src[i % len(src)]
        qs = {f'q{j}': {'type': 'choice', 'instructions': 'pick', 'options': [{'key': k, 'text': k} for k in 'abc']} for j in range(2)}
        out.append({'id': f'{prefix}{i}', 'state': 'x', 'questions': qs, 'source': g[0] if g else prefix,
                    'workflow': g[1] if g else 'w', 'gold': {q: {'label': (i + j) % 3, 'probs': None} for j, q in enumerate(qs)}})
    return out


def _preds(recs, salt):
    out = []
    for r in recs:
        rng, d = random.Random(f'{salt}{r["id"]}'), {}
        for q, x in r['questions'].items():
            w = [rng.random() + 0.2 for _ in x['options']]
            d[q] = [v / sum(w) for v in w]
        out.append({'id': r['id'], 'probs': d})
    return out


def _run(*args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, '-m', 'ej.eval', *args], env=env, capture_output=True, text=True, timeout=600)


def test_score_and_compare_command_line(tmp_path):
    suites = {s: _recs(30, s) for s in ('tickets', 'td', 'zs_td', 'zs_massive')}
    suites['zs_wide'] = _recs(60, 'w', [(s, f'g{g}') for s in ('sni', 'sgd', 'abcd', 'glmsyn') for g in range(3)])
    args = []
    for s, recs in suites.items():
        _write(tmp_path / f'{s}.jsonl', recs)
        _write(tmp_path / f'{s}.p.jsonl', _preds(recs, 'one'))
        args += ['--suite', str(tmp_path / f'{s}.jsonl'), '--preds', str(tmp_path / f'{s}.p.jsonl')]
    dump = str(tmp_path / 'a.json')
    r = _run('score', *args, '--groups', 'zs_wide', '--real', 'sni,sgd,abcd', '--draws', '50', '--dump', dump)
    assert r.returncode == 0, r.stderr
    one = json.loads(r.stdout.splitlines()[-1])
    assert one['metric_suites'] == ['tickets', 'td', 'zs_td', 'zs_massive']
    for s in ('td', 'zs_wide'):
        assert set(one[s]['ci']) == {'nll', 'acc', 'ece15'} and 'calibrated' in one[s]['ece_floor']
        assert 'this suite only' in one[s]['ca_scope']
    w = one['zs_wide']
    assert w['primary'] == 'macro_real' and w['macro_real_sources'] == ['abcd', 'sgd', 'sni'] and w['macro_real_groups'] == 9
    led, out = str(tmp_path / 'ledger.jsonl'), str(tmp_path / 'd.json')
    for _ in range(2):
        r = _run('compare', '--ref', dump, '--cand', dump, '--groups', 'zs_wide', '--groups-file', str(tmp_path / 'zs_wide.jsonl'),
                 '--real', 'sni,sgd,abcd', '--ledger', led, '--tag', 't1', '--out', out)
        assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    assert rep['selection']['n_comparisons'] == 2 and rep['metric']['d'] == 0.0 and rep['zs_wide_macro_real']['d'] == 0.0
    assert rep['suites_in_metric'] == ['tickets', 'td', 'zs_td', 'zs_massive'] and ES.ledger_verify(led)
    lines = open(led).read().splitlines()
    open(led, 'w').write('\n'.join(lines[1:]) + '\n')  # delete the first row -> the chain breaks, compare refuses
    bad = _run('compare', '--ref', dump, '--cand', dump, '--ledger', led, '--tag', 't1')
    assert bad.returncode != 0 and 'hash chain broken' in bad.stderr
