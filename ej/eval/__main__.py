"""Command line of ej.eval (``python -m ej.eval --help``)."""
import argparse
import json
import sys


def _csv(v):
    return tuple(x for x in v.split(',') if x) if v else None


def cmd_score(a):
    """score: print the report as one JSON line (and write --out / --dump)."""
    from . import score as SC
    if bool(a.weights) == bool(a.preds):
        sys.exit('ej.eval score: give either --weights DIR or one --preds FILE per --suite')
    if a.preds and len(a.preds) != len(a.suite):
        sys.exit('ej.eval score: one --preds file per --suite, in the same order')
    predict = SC.predictor(a.weights, a.threads) if a.weights else None
    rep, dump = SC.score(a.suite, predict, a.preds, a.groups, _csv(a.real), a.draws)
    if a.dump:
        SC.write_dump(dump, a.dump)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rep, f, indent=1)
    print(json.dumps(rep))


def cmd_compare(a):
    """compare: the seed-aware verdicts of --cand against --ref."""
    from . import compare as C
    from . import metrics as M
    from . import stats as ES
    ref, cand = [json.load(open(p)) for p in a.ref], [json.load(open(p)) for p in a.cand]
    grp = None
    if a.groups:
        if not a.groups_file:
            sys.exit('ej.eval compare: --groups needs --groups-file (the suite file with source / workflow fields)')
        grp = {r['id']: M.group_of(r) for r in M.load(a.groups_file)}
    n_cmp = 1
    if a.ledger:
        if ES.ledger_rows(a.ledger) and not ES.ledger_verify(a.ledger):
            sys.exit(f'{a.ledger}: hash chain broken (a row was edited or removed)')
        ES.ledger_append(a.ledger, {'event': 'compare', 'tag': a.tag, 'ref': a.ref, 'cand': a.cand})
        n_cmp = sum(r.get('event') == 'compare' and r.get('tag') == a.tag for r in ES.ledger_rows(a.ledger))
    rep = C.score(ref, cand, _csv(a.suites), grp, a.groups, _csv(a.real), n_cmp, a.prior_sd)
    rep['inputs'] = {'ref': a.ref, 'cand': a.cand}
    print(json.dumps({k: rep[k] for k in ('suites_in_metric', 'metric', 'selection')}, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rep, f, indent=1)


def cmd_power(a):
    """power: verdict rates by number of fits per arm (choice of S)."""
    from . import compare as C
    for S in (1, 2, 3, 4, 5, 6, 8):
        row = {f'D={dl:+.3f}': C.simulate(S, dl, a.reps, seed=S) for dl in (-0.020, -0.010, 0.0, 0.010)}
        print(json.dumps({'S': S, **{k: {kk: v[kk] for kk in ('BETTER', 'EQUAL', 'cover')} for k, v in row.items()}}), flush=True)


def cmd_null(a):
    """null: falsification run, NULL candidates (same code, other seeds) must not be kept."""
    from . import compare as C
    for S in (1, 3, 5):
        for tdf in (None, 5):
            print(json.dumps({'S': S, 'noise': 't5' if tdf else 'normal', **C.simulate(S, 0.0, a.reps, 10 + S, tdf=tdf)}), flush=True)


def parser():
    """The argument parser."""
    ap = argparse.ArgumentParser(prog='python -m ej.eval', description='ej evaluator: score suites, compare two arms of fits.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score', help='score suite files with a weights directory or saved predictions')
    s.add_argument('--suite', action='append', required=True, help='suite JSONL (ej records + gold); repeat for several')
    s.add_argument('--weights', help='an ej model: model.ejpack (or a directory holding it) or a weights directory')
    s.add_argument('--preds', action='append', help='predictions JSONL per suite ({"id", "probs"}, benchmarks/run_bench.py)')
    s.add_argument('--groups', help='suite name reported as group-macro accuracy over source/workflow (left out of metric)')
    s.add_argument('--real', help='comma-separated sources of the primary group statistic (macro_real); default all')
    s.add_argument('--threads', type=int, help='torch threads inside predict (default: the model default, 2)')
    s.add_argument('--draws', type=int, default=1000, help='bootstrap draws of the record-cluster CIs')
    s.add_argument('--dump', help='write per-question predictions (input of compare; keep private for private suites)')
    s.add_argument('--out', help='also write the report as indented JSON')
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser('compare', help='seed-aware comparison of two arms (S score dumps each, seed i paired with seed i)')
    c.add_argument('--ref', nargs='+', required=True)
    c.add_argument('--cand', nargs='+', required=True)
    c.add_argument('--suites', help='comma-separated suites of the metric (default: all in the dumps except --groups)')
    c.add_argument('--groups', help='suite with source/workflow groups: group-macro accuracy verdicts')
    c.add_argument('--groups-file', help='that suite file (for the record -> group map)')
    c.add_argument('--real', help='comma-separated sources of macro_real (default all)')
    c.add_argument('--prior-sd', type=float, default=0.0065, help='prior fit-noise SD of a metric difference, used when S = 1')
    c.add_argument('--ledger', help='append this run to a sha256-chained ledger and count comparisons per --tag')
    c.add_argument('--tag', default='default', help='ledger tag of a selection campaign (Bonferroni N = its compare runs)')
    c.add_argument('--out', help='write the full report as JSON')
    c.set_defaults(fn=cmd_compare)
    for name, fn, h in (('power', cmd_power, 'simulated verdict rates by fits per arm'),
                        ('null', cmd_null, 'simulated NULL candidates (falsification of the keep rule)')):
        p = sub.add_parser(name, help=h)
        p.add_argument('--reps', type=int, default=2000)
        p.set_defaults(fn=fn)
    return ap


def main(argv=None):
    """Entry point."""
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
