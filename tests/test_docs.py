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
        'docs/training.md', 'docs/adaptation.md', 'benchmarks/README.md', 'benchmarks/METHOD.md',
        'benchmarks/results/README.md', 'benchmarks/suites/README.md']


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def flat(rel):
    """The document with line breaks collapsed (sentences wrap across lines)."""
    return re.sub(r'\s+', ' ', read(rel))


HF_REPO = 'https://huggingface.co/5ak3t/ej'
PACK_SHA256 = 'e990e1846cba43f8405a969c606057f2fd6e2076595f4a34d202e8fc531891b0'
ARCH_LINE = 'Architecture and method: technical report forthcoming.'


def test_versions_agree_and_weights_point_to_the_hub():
    version = re.search(r'^version = "([^"]+)"', read('pyproject.toml'), re.M).group(1)
    assert version == ej.__version__ == '0.0.1'
    readme = read('README.md')
    assert f'version = {{{version}}}' in readme and '## 0.0.1' in read('CHANGELOG.md')
    for rel in DOCS + ['examples/quickstart.py', 'benchmarks/rivals/ej_adapter.py']:
        assert 'not yet published' not in flat(rel).lower() and 'no hosting' not in flat(rel).lower(), rel
    for rel in ('README.md', CARD, 'docs/releases/0.0.1.md'):
        t = flat(rel)
        assert HF_REPO in t and 'v0.0.1' in t and PACK_SHA256 in t and 'model.ejpack' in t, rel
    for rel in ('README.md', CARD):
        assert re.search(r"""ej\.load\(['"]5ak3t/ej['"], revision=['"]v0\.0\.1['"]\)""", read(rel)), rel
    from ej import integrity
    assert integrity.KNOWN_PACK_FILES['3b3e66d28fb423f9'] == PACK_SHA256
    notes = flat('docs/releases/0.0.1.md')
    assert re.search(r'tag `v0\.0\.1` = Hub commit `[0-9a-f]{40}`', notes), 'release notes name the Hub commit of the tag'


def test_architecture_lives_in_the_technical_report():
    assert not os.path.exists(os.path.join(ROOT, 'docs', 'architecture.md'))
    for rel in ('README.md', CARD, 'docs/training.md', 'BENCHMARKS.md', 'docs/releases/0.0.1.md'):
        t = flat(rel)
        assert ARCH_LINE in t, rel
        for word in ('GPTQ', 'expert', 'int8', '2/3-bit', '2-bit', '3-bit', 'log-linear', 'lapse', 'docs/architecture.md'):
            assert word not in t, (rel, word)


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


def test_ci_tests_and_release_workflows():
    import yaml
    tests = yaml.safe_load(read('.github/workflows/tests.yml'))
    on = tests[True]  # YAML 1.1 reads the key `on` as True
    assert on['push'] == {'branches': ['main']} and 'pull_request' in on and 'workflow_dispatch' in on
    assert tests['permissions'] == {'contents': 'read'} and set(tests['jobs']) == {'lint-and-unit', 'weights'}
    assert tests['env']['EJ_HF_REPO'] == '5ak3t/ej' and tests['env']['EJ_HF_REVISION'] == 'v0.0.1'
    assert tests['env']['EJ_PACK_SHA256'] == PACK_SHA256
    text = read('.github/workflows/tests.yml')
    assert 'secrets.' not in text and 'download.pytorch.org/whl/cpu' in text and 'torch==2.5.1' in text
    assert 'sha256sum -c' in text and text.count('pytest') >= 2 and 'EJ_WEIGHTS' in text
    rel = yaml.safe_load(read('.github/workflows/release.yml'))
    assert rel[True] == {'push': {'tags': ['v*']}} and rel['jobs']['tests']['uses'] == './.github/workflows/tests.yml'
    assert rel['jobs']['release']['needs'] == 'tests' and rel['jobs']['release']['permissions'] == {'contents': 'write'}
    text = read('.github/workflows/release.yml')
    assert 'python -m build' in text and 'gh release create "$GITHUB_REF_NAME" dist/* SHA256SUMS' in text
    for word in ('secrets.', 'twine upload', 'pypi-publish@'):
        assert word not in text, word
    for f in ('tests.yml', 'release.yml'):
        uses = re.findall(r'uses: (actions/[\w-]+)@(\S+)', read(f'.github/workflows/{f}'))
        assert all(v in ('v4', 'v5') for _, v in uses), uses


def test_documents_link_to_existing_files():
    for rel in DOCS:
        for target in re.findall(r'\]\(([^)#:]+)\)', read(rel)):
            assert os.path.exists(os.path.normpath(os.path.join(ROOT, os.path.dirname(rel), target))), (rel, target)
