"""Von 1.3 (wfzyx/von, ModernBERT-large + Option-Marker head, 395M) via its own SDK: von.api.system_one(state, questions).

NOT RUN HERE (needs a GPU host or a big CPU budget: 1.6 GB backbone + 1.6 GB head download; disk here is ~5 GB).
Install: pip install "von-sdk==1.3.7" (weights from the hub, sha 498ceba3). Documented inference, shipped calibration map.
Changes: VON_NOUL_DECISION=raw, so a yes/no answer is the calibrated posterior P(true) (`noul_raw`) instead of the 'band'
decision rule that pushes every answer to 0.15-0.2 / 0.8-0.85 (a committed decision, not a probability); and the backend's
4-dp output rounding is disabled."""
import functools
import os

from . import common


@functools.lru_cache(maxsize=1)
def load():
    """Load the model once per process (cached)."""
    os.environ.setdefault('VON_NOUL_DECISION', 'raw')
    common.setup_torch()
    import von.api
    import von.backends.option_marker_backend as omb
    common.unround(omb)
    return von.api


def predict(records):
    """[{qid: probs}] per record, in each question's option order."""
    api = load()
    out = []
    for r in records:
        res = api.system_one(common.state_text(r), common.wire_questions(r))
        answers = {qid: common.to_dict(a) for qid, a in common.to_dict(res)['answers'].items()}
        out.append(common.from_response(r, answers))
    return out
