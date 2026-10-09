"""decider (Mapika/decider-2b v11, Mapika/decider-12b v2) via its own package: Decider(path).system_one(state, questions).

NOT RUN HERE (needs GPU: 3.8 GB / 23.9 GB bf16 weights; the Qwen3.5 linear-attention layers want flash-linear-attention/Triton).
Install on the GPU host: pip install "decider-ai==1.8.2" (>= 1.7.0 needed for decider-12b's Gemma softcapping), or use the
decider/ inference subset shipped inside each HF repo. Documented defaults: independent=True (one row per question), isolated
Score levels, the checkpoint's per-type temperatures. Only change: decider.systemone's 4-dp output rounding is disabled."""
import functools

from . import common

MODELS = {'decider-2b': ('Mapika/decider-2b', '533964dae8be954c5b5e19fa4948e48408094c1e'),
          'decider-12b': ('Mapika/decider-12b', '8ac1efa708b71b86ae33b01d2a8d7a3ddcb48e66')}


@functools.lru_cache(maxsize=1)
def load(name):
    """Load the model once per process (cached)."""
    common.setup_torch()
    from huggingface_hub import snapshot_download
    import decider.systemone
    from decider.infer import Decider
    common.unround(decider.systemone)
    repo, rev = MODELS.get(name, (name, None))
    return Decider(snapshot_download(repo, revision=rev))


def predict(records, name='decider-2b'):
    """[{qid: probs}] per record, in each question's option order."""
    d = load(name)
    out = []
    for r in records:
        res = d.system_one(common.state_text(r), common.wire_questions(r))
        out.append(common.from_response(r, res['answers']))
    return out
