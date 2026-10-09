"""Small causal LLM as a constrained classifier: option log-likelihood (lm-eval-harness style multiple choice).

Prompt = state, question, the option list; the answer distribution is the softmax over options of the SUM of the option
continuation's token log-probabilities, i.e. the model's own probability of each allowed answer string, renormalised over
the allowed set. noul without described options: continuations "No" / "Yes". State truncated to MAX_STATE tokens.
Qwen3-0.6B (hybrid chat model) is read through its chat template with thinking off; base models use a plain prompt.
Exact prefix caching: the state is encoded once per record, each question once, each option on a copy of that cache (same
log-probabilities as scoring every full sequence; checked by `selfcheck`). Tested here with the cached
Qwen/Qwen3.5-0.8B-Base; Qwen3-0.6B itself (1.5 GB download) is not run."""
import copy
import functools
import os

from . import common

MODELS = {'qwen3-0.6b': ('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca', True),
          'qwen3.5-0.8b-base': ('Qwen/Qwen3.5-0.8B-Base', 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68', False)}
MAX_STATE = 768
SPLIT = '\x00STATE_END\x00'


@functools.lru_cache(maxsize=1)
def load(name):
    """Load the model once per process (cached)."""
    torch = common.setup_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev, chat = MODELS.get(name, (name, None, False))
    dtype = getattr(torch, os.environ.get('LL_DTYPE', 'float32'))
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype).eval()
    return torch, tok, model, chat


def options(q):
    """Answer strings scored for question q."""
    if q['type'] == 'noul' and common.generic_noul(q):
        return ['No' if o['key'] == 'false' else 'Yes' for o in q['options']]
    return [o['text'] if common.option_desc(o) is None or o['text'].lower().startswith(o['key'].lower())
            else f"{o['key'].replace('_', ' ')}: {o['text']}" for o in q['options']]


def prompt_parts(tok, chat, state, q, opts):
    """-> (state part, question part): their concatenation is the full prompt, split right after the state."""
    ask = q['instructions'] if not (q['type'] == 'noul' and common.generic_noul(q)) else \
        f"Is the following statement true? {q['instructions']}"
    body = f"{state}\n\n{SPLIT}Question: {ask}\nOptions:\n" + ''.join(f"- {o}\n" for o in opts) + "Answer with one option."
    if chat:
        body = tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    else:
        body += "\nAnswer:"
    a, b = body.split(SPLIT)
    return a, b


def step(name, ids, cache):
    """Feed ids on top of cache (mutated). -> (log-probs per fed position, cache)."""
    torch, _, model, _ = load(name)
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
    return torch.log_softmax(out.logits[0].float(), -1), out.past_key_values


def score_options(name, cache, last_lp, conts):
    """Summed log-likelihood of each continuation on top of the cached prompt."""
    out = []
    for c in conts:
        total = float(last_lp[c[0]])
        if len(c) > 1:
            lp, _ = step(name, c[:-1], copy.deepcopy(cache))
            total += sum(float(lp[j, t]) for j, t in enumerate(c[1:]))
        out.append(total)
    return out


def truncate_state(tok, text):
    """The state text cut to MAX_STATE tokens."""
    ids = tok(text, add_special_tokens=False)['input_ids']
    return text if len(ids) <= MAX_STATE else tok.decode(ids[:MAX_STATE]) + ' ...'


def encode(tok, text):
    """Token ids of text without special tokens."""
    return tok(text, add_special_tokens=False)['input_ids']


def predict(records, name='qwen3.5-0.8b-base'):
    """[{qid: probs}] per record, in each question's option order."""
    _, tok, _, chat = load(name)
    out = []
    for r in records:
        state, pr, s_cache = truncate_state(tok, common.state_text(r)), {}, None
        for qid, q in r['questions'].items():
            opts = options(q)
            a, b = prompt_parts(tok, chat, state, q, opts)
            if s_cache is None:  # the state part is identical for every question of the record
                _, s_cache = step(name, encode(tok, a), None)
            lp, q_cache = step(name, encode(tok, b), copy.deepcopy(s_cache))
            conts = [encode(tok, ('' if chat else ' ') + o) for o in opts]
            pr[qid] = common.finalize(common.softmax(score_options(name, q_cache, lp[-1], conts)))
        out.append(pr)
    return out


def selfcheck(name, record):
    """Max |cached - uncached| option log-likelihood over the record's questions (uncached = one full forward each)."""
    torch, tok, model, chat = load(name)
    state, worst = truncate_state(tok, common.state_text(record)), 0.0
    for q in record['questions'].values():
        opts = options(q)
        a, b = prompt_parts(tok, chat, state, q, opts)
        pre = encode(tok, a) + encode(tok, b)
        _, s_cache = step(name, encode(tok, a), None)
        lp, q_cache = step(name, encode(tok, b), s_cache)
        conts = [encode(tok, ('' if chat else ' ') + o) for o in opts]
        cached = score_options(name, q_cache, lp[-1], conts)
        for c, v in zip(conts, cached):
            full, _ = step(name, pre + c, None)
            ref = sum(float(full[len(pre) - 1 + j, t]) for j, t in enumerate(c))
            worst = max(worst, abs(ref - v))
    return worst
