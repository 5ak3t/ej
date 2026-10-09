"""ej.train without training anything (no GPU, no data): the command line, the work-directory setup, the bare-name modules the
runtime's fit imports, pool validation and the state key. Each environment test runs in a fresh interpreter, because setting up
a training environment points the runtime at the work directory for the whole process."""
import json
import os
import subprocess
import sys

import pytest

import ej

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _py(code, tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE='1', EJ_CACHE=str(tmp_path / 'cache'))
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _pool(path, n=6):
    with open(path, 'w') as f:
        for i in range(n):
            r = dict(ej.EXAMPLE_RECORD, id=f'p{i}', source='toy', workflow='w')
            r['gold'] = {q: {'label': 0, 'probs': None} for q in r['questions']}
            f.write(json.dumps(r) + '\n')
    return str(path)


def test_command_line_help():
    for mod in ('ej.train', 'ej.eval'):
        r = subprocess.run([sys.executable, '-m', mod, '--help'], env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0 and 'usage: python -m ' + mod in r.stdout, r.stderr
    r = subprocess.run([sys.executable, '-m', 'ej.train', 'fit', '--pool', 'x.jsonl'], env=dict(os.environ, PYTHONPATH=ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and '--out DIR is required' in r.stderr


def test_environment_and_bare_name_modules(tmp_path):
    pool = _pool(tmp_path / 'pool.jsonl')
    out = _py(f"""
import json, os, sys
from ej.train import env
work = env.setup({pool!r}, {str(tmp_path / 'work')!r})
import student_hcf, train_hcf, student_hdn, train_merge as TM, student_state as ST
from ej.train import fit, export, hcf, hdn
key = export.state_key(env.pool_path())
print(json.dumps(dict(hcf=train_hcf is hcf, hdn=student_hdn is hdn, pairs=list(train_hcf.PAIRS), parts=list(TM.PARTS),
    pool=TM.POOL == env.pool_path(), ckpt=os.environ['EDGE_CKPT'].startswith(work), n=fit.check_pool(fit._records()), key=key,
    key2=export.state_key(env.pool_path()), nsrc=len(export.sources()), version=ST.VERSION)))
""", tmp_path)
    assert out['hcf'] and out['hdn'] and out['pool'] and out['ckpt'] and out['n'] == 6
    assert out['pairs'][0] == 'p01' and out['parts'][0] == 'full' and len(out['key']) == 64 and out['key'] == out['key2']
    assert out['nsrc'] >= 45 and out['version'] == 'edge-state-v1'


def test_state_key_follows_the_pool(tmp_path):
    a, b = _pool(tmp_path / 'a.jsonl', 6), _pool(tmp_path / 'b.jsonl', 7)
    keys = [_py(f"""
import json
from ej.train import env, export
env.setup({p!r}, {str(tmp_path / ('w' + str(k)))!r})
print(json.dumps(export.state_key(env.pool_path())))
""", tmp_path) for k, p in enumerate((a, b))]
    assert keys[0] != keys[1]


def test_pool_validation(tmp_path):
    pool = _pool(tmp_path / 'pool.jsonl', 2)
    out = _py(f"""
import json
from ej.train import env, fit
env.setup({pool!r}, {str(tmp_path / 'work')!r})
recs = fit._records()
errs = []
for bad in ({{'source': ''}}, {{'gold': {{}}}}):
    try:
        fit.check_pool([dict(recs[0], **bad)])
    except ValueError as e:
        errs.append(str(e))
print(json.dumps(errs))
""", tmp_path)
    assert len(out) == 2 and 'source' in out[0] and 'gold.label' in out[1]


SEED_PROBE = '''
import json, random, sys
from ej.train import seed
installed = seed.install(int(sys.argv[1]))
import numpy as np, torch
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 2))
net.train(); out = net(torch.randn(3, 4)).sum().item()
perm = np.random.default_rng(5).permutation(10).tolist()
r = random.Random(3).random()
g = torch.randn(2, generator=torch.Generator().manual_seed(0)).tolist()
print(json.dumps({'installed': installed, 'out': out, 'perm': perm, 'r': r, 'gen': g}))
'''


def _seed_probe(s):
    r = subprocess.run([sys.executable, '-c', SEED_PROBE, str(s)], env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_fit_seed_replicates():
    """Seed 0 patches nothing; seeds >= 1 are reproducible, differ from each other and leave explicit Generator draws alone."""
    base, s1, s1b, s2 = _seed_probe(0), _seed_probe(1), _seed_probe(1), _seed_probe(2)
    assert base['installed'] is False and s1 == s1b and s1['installed'] is True
    for k in ('out', 'perm', 'r'):
        assert s1[k] != base[k] and s2[k] != s1[k], k
    assert s1['gen'] == base['gen'] == s2['gen']


def test_seed_claims_its_work_directory(tmp_path):
    from ej.train import seed
    from ej.train.seed import splitmix64
    assert splitmix64(0) == 0xE220A8397B1DCDAF  # SplitMix64 reference value
    seed.claim(str(tmp_path), 0)
    (tmp_path / 'ckpt').mkdir()
    (tmp_path / 'ckpt' / 'dd-cafe.pt').write_bytes(b'x')
    with pytest.raises(RuntimeError, match='separate --work'):
        seed.claim(str(tmp_path), 3)
