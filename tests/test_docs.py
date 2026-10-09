"""Claim lint of the public documents. Each test fails when an unsupported claim appears, a required statement goes missing,
or the documents disagree with the code (version, results files, workflow)."""
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

import ej

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = 'docs/model-cards/ej-0.0.1.md'
DOCS = ['README.md', 'BENCHMARKS.md', 'CHANGELOG.md', 'AGENTS.md', 'NOTICE', CARD, 'docs/data-card.md', 'docs/releases/0.0.1.md',
        'docs/architecture.md', 'docs/training.md', 'docs/adaptation.md', 'benchmarks/README.md', 'benchmarks/METHOD.md',
        'benchmarks/results/README.md', 'benchmarks/suites/README.md']


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def flat(rel):
    """The document with line breaks collapsed (sentences wrap across lines)."""
    return re.sub(r'\s+', ' ', read(rel))


def test_versions_agree_and_no_quickstart_needs_unpublished_weights():
    version = re.search(r'^version = "([^"]+)"', read('pyproject.toml'), re.M).group(1)
    assert version == ej.__version__ == '0.0.1'
    readme = read('README.md')
    assert f'version = {{{version}}}' in readme and '## 0.0.1' in read('CHANGELOG.md')
    assert 'not yet published' in readme and 'not yet published' in flat(CARD)
    assert not re.search(r"ej\.load\(\s*['\"][\w.-]+/[\w.-]+['\"]", readme), 'README loads a hub repo id'
    assert 'huggingface.co/5ak3t' not in readme + read('pyproject.toml') + read('examples/quickstart.py')


@pytest.mark.parametrize('rel', DOCS)
def test_no_release_history_or_internal_ids(rel):
    text = read(rel)
    for bad in ('1038d03e', 'v1.0.0', 'v1.0.1', '1.0.1', '14a3e64f', 'withdrawn', 'codecraf8'):
        assert bad not in text, (rel, bad)
    assert not re.search(r'\b[AD]-0\d\d\b|\bround \d|\baudit M-\d', text), rel


@pytest.mark.parametrize('rel', DOCS)
def test_no_unsupported_rankings(rel):
    text = flat(rel).lower()
    for claim in ('smallest', 'fastest', 'lowest mean ece', 'best-calibrated', 'best calibrated', 'state of the art'):
        assert claim not in text, (rel, claim)


def test_notice_names_creator_link_licence_uri_and_modified_per_source():
    notice = read('NOTICE')
    block = notice[notice.index('Training data.'):notice.index('Distillation signals')]
    entries = [e for e in re.split(r'\n- ', block)[1:] if not e.startswith('In-house')]
    assert len(entries) == 4, entries
    for e in entries:
        assert 'Creator:' in e and 'Link: https://' in e and re.search(r'Licence: .*https://', e, re.S) and 'Modified:' in e, e
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        assert re.search(rf'^{q}\s', notice, re.M), q
    assert 'CC BY-NC 4.0' in flat('NOTICE') and 'Not used: Amazon counterfactual' in flat('NOTICE')


def test_zs_td_and_selection_statements():
    for rel in (CARD, 'benchmarks/results/README.md'):
        t = flat(rel)
        assert 'zs_td is ONE held-out workflow' in t and 'neither unbiased nor unseen-workflow evidence' in t, rel
    assert 'Unseen-workflow evidence = zs_wide final macro_real.' in flat(CARD)
    assert 'Development numbers are selected' in flat(CARD) and 'selected and optimistic' in flat('README.md')
    for rel in DOCS:
        t = flat(rel).lower()
        assert 'never used for model selection' not in t and 'the unbiased read' not in t, rel


def test_adaptation_numbers_match_the_results_file():
    d = json.load(open(os.path.join(ROOT, 'benchmarks/results/adaptation/ej-0.0.1.few-shot.json')))
    doc, readme = flat('docs/adaptation.md'), flat('README.md')
    w = d['zs_wide']['k8']
    assert f"+{w['d_macro']:.3f}".replace('0.', '.') in doc and 'covers 0' in doc
    assert '111 public-source + 43 GLM-synthetic' in doc + readme and 'count prior' in doc
    assert '+.008 [−.0001, +.016]' in doc and '+.008 [−.0001, +.016]' in readme
    assert 'record-cluster bootstrap' in doc and '**k (few-shot)**' in readme


def test_calibration_ca_scope_and_transductive_statement():
    card, readme = flat(CARD), flat('README.md')
    assert 'this suite only' in card and 'this suite only' in readme
    statement = "Zero-shot `predict` is record-independent: each record's output depends only on that record"
    for t in (card, readme):
        assert statement in t and 'opt-in, per-workflow transductive mode' in t


def test_accuracy_numbers_name_their_kind():
    for rel in ('README.md', CARD):
        for line in read(rel).splitlines():
            if 'fit-to-fit SD' in line:  # the zs_td statement: one workflow, where micro and macro coincide
                continue
            if re.search(r'\baccuracy\b', line) and re.search(r'(?<![\w.])0?\.\d{2,}', line):
                assert re.search(r'micro|macro', line), (rel, line)


def test_sizes_name_their_measure():
    for rel in ('README.md', CARD, 'docs/releases/0.0.1.md'):
        t = flat(rel)
        assert 'counted' in t and 'on disk' in t and 'resident' in t, rel


def test_latency_numbers_come_from_measured_one_thread_summaries():
    d = os.path.join(ROOT, 'benchmarks', 'results', 'latency')
    summaries = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith('.summary.json')]
    assert summaries
    readme = read('README.md')
    for s in summaries:
        assert s['threads'] == 1 and s['threads_measured'] == [1] and s['threads_ok'] and s['rival'] == 'ej'
        assert s['weights'].startswith('ej 0.0.1') and f"{s['ms_per_record']:,.1f}" in readme, s['ms_per_record']
    assert 'not comparable' in readme and 'not comparable' in flat('BENCHMARKS.md')


def test_results_tables_match_the_results_file():
    r = json.load(open(os.path.join(ROOT, 'benchmarks/results/ej-0.0.1.final.json')))
    readme, bench = flat('README.md'), flat('BENCHMARKS.md')
    for s in ('td', 'zs_td', 'zs_massive', 'tickets', 'tickets_ood'):
        acc = str(Decimal(str(r[s]['acc'])).quantize(Decimal('.001'), ROUND_HALF_UP)).lstrip('0')
        assert acc in readme and acc in bench, (s, acc)
    real = r['zs_wide']['macro_real']
    assert f'macro_real {real:.3f}'.replace('0.', '.') in readme


def test_suite_checksums_and_builders():
    sums = read('benchmarks/suites/SHA256SUMS')
    for suite in ('td', 'zs_td', 'zs_massive'):
        for split in ('dev', 'final'):
            assert re.search(rf'^[0-9a-f]{{64}}  {split}/{suite}\.jsonl$', sums, re.M), (split, suite)
    for f in ('build_public.py', 'sources.py', 'leakfree.py'):
        compile(read(f'benchmarks/suites/{f}'), f, 'exec')


def test_ci_runs_tests_without_deploying():
    wf = read('.github/workflows/tests.yml')
    on = wf[wf.index('\non:'):wf.index('\npermissions:')]
    assert 'pull_request' in on and 'workflow_dispatch' in on and 'push' not in on and 'schedule' not in on
    assert 'contents: read' in wf and 'pytest' in wf
    for word in ('deploy', 'secrets.', 'upload', 'publish', 'twine', 'huggingface-cli'):
        assert word not in wf.replace('no deployment', ''), word


def test_documents_link_to_existing_files():
    for rel in DOCS:
        for target in re.findall(r'\]\(([^)#:]+)\)', read(rel)):
            assert os.path.exists(os.path.normpath(os.path.join(ROOT, os.path.dirname(rel), target))), (rel, target)
