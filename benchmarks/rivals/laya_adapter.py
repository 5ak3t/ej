"""laya (convaiinnovations/laya-typed-decisions) via its own package: laya.load(repo).predict(state, questions).

Environment: its own venv with laya 0.3.26 and torch (cpu). Documented inference, all questions of a record in one call, checkpoint's own
temperatures. Only change: laya.agent's 4-dp output rounding is disabled (common.unround) so probabilities keep full precision."""
import functools

from . import common

DEFAULT = 'convaiinnovations/laya-typed-decisions'
REVISION = 'e929ae5cf69bc34259cd2f95c9e91145b818b1f0'  # pinned snapshot


@functools.lru_cache(maxsize=2)
def load(repo=DEFAULT, revision=REVISION):
    """Load the model once per process (cached)."""
    common.setup_torch()
    import laya
    import laya.agent
    common.unround(laya.agent)
    try:
        return laya.load(repo, revision=revision)
    except TypeError:  # older laya without a revision argument
        return laya.load(repo)


def predict(records, repo=DEFAULT, revision=REVISION):
    """[{qid: probs}] per record, in each question's option order."""
    agent = load(repo, revision)
    out = []
    for r in records:
        res = agent.predict(common.state_text(r), common.wire_questions(r))
        out.append(common.from_response(r, res['answers']))
    return out
