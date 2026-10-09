"""kev (jaredpalmer/kev-0.8b, kev-4b) via its own server code path, in process (same as POST /v1/systemone).

Environment: its own venv with kev @ fe64b127 (github.com/jaredpalmer/kev), torch 2.8 cpu, transformers 5.18. Path:
SystemOneRequest -> to_record(prepare(req)) -> Server.probs; the checkpoint's own temperature
stays on, and the FULL-PRECISION calibrated distributions are used (kev's API body rounds to 4 dp).
KEV_DTYPE=fp32 is kev's documented evaluation path (default here); kev-4B needs ~10 GB download + RAM -> GPU host."""
import functools
import os

from . import common

RUNS = {'kev-0.8b': 'jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f',
        'kev-4b': 'jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52'}


@functools.lru_cache(maxsize=1)
def load(run):
    """Load the model once per process (cached)."""
    os.environ.setdefault('KEV_DTYPE', 'fp32')
    common.setup_torch()
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.device import default_device
    from kev.serve import Server
    dev = default_device()
    ck = Checkpoint(RUNS.get(run, run))
    tok, model = ck.load(dev, LoadOptions.from_env())
    return Server(ck, tok, model, dev)


def predict(records, run='kev-0.8b'):
    """[{qid: probs}] per record, in each question's option order."""
    from kev.api import SystemOneRequest, to_record
    from kev.serve import prepare
    srv = load(run)
    out = []
    for r in records:
        req = SystemOneRequest(state=common.state_text(r), questions=common.wire_questions(r))
        rec, meta = to_record(prepare(req))
        ps, _ = srv.probs(rec)
        full = {mm['id']: {'probabilities': dict(zip([str(k).lower() if isinstance(k, bool) else str(k) for k in mm['keys']], p))}
                for mm, p in zip(meta, ps)}
        out.append(common.from_response(r, full))
    return out
