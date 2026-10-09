"""Process hygiene, no unpickling, bounded memory. The weights tests run only when EJ_WEIGHTS
names a model.ejpack (or a weights directory); each runs in a fresh interpreter so the process state before `import ej` is known.
Falsified if: ej.load changes torch's thread count, RNG state or transformers' from_pretrained; a "1 thread" predict runs with
another thread count; anything is unpickled while importing ej, loading the weights and predicting; a runtime module
can still call torch.load; the encoder memo grows past its cap; chunked predictions differ from one batch by more than 1e-6."""
import json
import os
import re
import subprocess
import sys

import pytest

from ej import scope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get('EJ_WEIGHTS')
needs_weights = pytest.mark.skipif(not WEIGHTS, reason='set EJ_WEIGHTS to a model.ejpack (or a weights directory)')
RECORDS = """
def records(n, tag='r'):
    import ej
    out = []
    for i in range(n):
        r = json.loads(json.dumps(ej.EXAMPLE_RECORD))
        r['state'] = r['state'].replace('blender', f'blender model {tag}{i}')
        out.append(r)
    return out
"""


def run(code):
    """Run code in a fresh interpreter (the repository on sys.path); returns the JSON object printed on its last stdout line."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, '-c', 'import json\n' + RECORDS + code], env=env, capture_output=True, text=True,
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_lru_memo_bounds_and_order():
    m = scope.LRUMemo(cap=3)
    for k in 'abcd':
        m[k] = k
    assert m['a'] == 'a'  # a read refreshes the entry
    assert m.trim() == 1 and list(m) == ['c', 'd', 'a']
    w = scope._Warm(2)
    assert isinstance(w.setdefault('E'), scope.LRUMemo) and w.setdefault('E') is w['E']


def test_guarded_module_refuses_unpickling_only():
    import pickle
    import torch
    gt = scope.guarded_module(torch, ('load',))
    gp = scope.guarded_module(pickle, ('load', 'loads', 'Unpickler'))
    assert gt.zeros(2).tolist() == [0.0, 0.0] and gt.Tensor is torch.Tensor and gp.dumps(1) == pickle.dumps(1)
    for f in (lambda: gt.load('x'), lambda: gp.loads(b''), lambda: gp.Unpickler(None)):
        with pytest.raises(scope.UnpicklingRefused):
            f()
    assert torch.load is not gt.load and pickle.loads is not gp.loads  # the real modules are untouched


def test_package_has_no_unpickling_call_site():
    """The prediction package (ej/*.py, ej/eval and ej/pack) never unpickles. ej/train reads checkpoints it wrote itself."""
    for sub in ('', 'eval', 'pack'):
        src = os.path.join(ROOT, 'ej', sub)
        for name in os.listdir(src):
            if name.endswith('.py'):
                text = open(os.path.join(src, name)).read()
                assert not re.search(r'torch\.load\(|pickle\.loads?\(|weights_only=False', text), name


@needs_weights
def test_load_leaves_threads_rng_and_transformers_unchanged():
    r = run(f"""
import torch, transformers
torch.set_num_threads(3); torch.manual_seed(7)
rng, own = torch.get_rng_state().clone(), {{c: getattr(transformers, c).__dict__.get('from_pretrained')
                                         for c in ('AutoModel', 'AutoTokenizer', 'AutoConfig')}}
import ej
m = ej.load({WEIGHTS!r})
after = dict(threads=torch.get_num_threads(), rng=bool(torch.equal(rng, torch.get_rng_state())),
             fp=all(getattr(transformers, c).__dict__.get('from_pretrained') is f for c, f in own.items()))
m.predict([ej.EXAMPLE_RECORD], threads=1); t1 = m.last_threads
m.predict([ej.EXAMPLE_RECORD]); t2 = m.last_threads
print(json.dumps(dict(after, t1=t1, t2=t2, threads_end=torch.get_num_threads(), rng_end=bool(torch.equal(rng, torch.get_rng_state())))))
""")
    assert r == {'threads': 3, 'rng': True, 'fp': True, 't1': 1, 't2': 2, 'threads_end': 3, 'rng_end': True}


@needs_weights
def test_no_unpickling_when_loading_and_predicting():
    r = run(f"""
import sys, pickle, torch
calls = []
sys.addaudithook(lambda ev, args: calls.append(ev) if ev == 'pickle.find_class' else None)
real = (torch.load, pickle.load, pickle.loads)
def counting(f):
    def g(*a, **k):
        calls.append(f.__name__); return f(*a, **k)
    return g
torch.load, pickle.load, pickle.loads = (counting(f) for f in real)
import ej
from ej import scope
m = ej.load({WEIGHTS!r})
m.predict([ej.EXAMPLE_RECORD] + records(3))
torch.load, pickle.load, pickle.loads = real
mods = scope.runtime_modules(ej.integrity.RUNTIME)
unguarded = [x.__name__ for x in mods if getattr(x.__dict__.get('torch'), '__name__', None) == 'torch'
             and not getattr(x.__dict__.get('torch'), '_ej_guarded', False)]
import student_lb
try:
    student_lb.torch.load('/nonexistent'); refused = False
except scope.UnpicklingRefused:
    refused = True
print(json.dumps(dict(calls=calls, unguarded=unguarded, refused=refused, n=len(mods))))
""")
    assert r['calls'] == [] and r['unguarded'] == [] and r['refused'] and r['n'] >= 22


@needs_weights
def test_memo_is_bounded_and_chunking_is_float_noise_only():
    r = run(f"""
import resource, ej
from ej import scope
m = ej.load({WEIGHTS!r}, memo_max=150)
sizes = []
for c in range(8):
    m.predict(records(25, tag=f'c{{c}}-'))
    sizes.append(scope.trim_memo()[0])
recs = records(40, tag='x')
one = m.predict(recs, chunk_size=0)
chunked = m.predict(recs, chunk_size=7)
dp = max(abs(a - b) for x, y in zip(one, chunked) for q in x for a, b in zip(x[q], y[q]))
print(json.dumps(dict(sizes=sizes, dp=dp, rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
""")
    assert max(r['sizes']) <= 150, r['sizes']
    assert r['dp'] <= 1e-6, r['dp']
