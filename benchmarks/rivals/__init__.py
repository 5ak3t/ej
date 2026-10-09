"""Model adapters for the ej benchmark. Each adapter: predict(records) -> [{qid: probs}] in the record's option order.
Modules import their heavy dependencies lazily, so this registry imports anywhere; run each model in the environment named
in its row (its own package versions). STATUS: 'tested' = run end to end with this code, 'not run' = written, not run
(needs a GPU or a large download)."""
import importlib

OWN = 'this repository (pip install -e .)'
HF = 'torch + transformers (pyproject versions)'
GPU = 'GPU host (see METHOD.md)'

# name: (module, kwargs, environment, status)
RIVALS = {
    'ej': ('ej_adapter', {}, OWN, 'tested'),
    'laya': ('laya_adapter', {}, 'own venv: laya 0.3.26', 'tested'),
    'kev-0.8b': ('kev_adapter', {'run': 'kev-0.8b'}, 'own venv: kev @ fe64b127', 'tested'),
    'kev-4b': ('kev_adapter', {'run': 'kev-4b'}, 'own venv: kev @ fe64b127', 'not run'),
    'decider-2b': ('decider_adapter', {'name': 'decider-2b'}, GPU, 'not run'),
    'decider-12b': ('decider_adapter', {'name': 'decider-12b'}, GPU, 'not run'),
    'von': ('von_adapter', {}, GPU, 'not run'),
    'decidabert': ('systemone_http', {'model': 'decidabert'}, 'any (HTTP client; server on a GPU host)', 'not run'),
    'systemone-http': ('systemone_http', {}, 'any (HTTP client)', 'tested'),
    'nli-xsmall-zs': ('nli_adapter', {'name': 'nli-xsmall-zs'}, HF, 'tested'),
    'nli-ce-xsmall': ('nli_adapter', {'name': 'nli-ce-xsmall'}, HF, 'tested'),
    'nli-large-c': ('nli_adapter', {'name': 'nli-large-c'}, HF, 'not run'),
    'bart-mnli': ('nli_adapter', {'name': 'bart-mnli'}, HF, 'not run'),
    'gliclass-edge': ('gliclass_adapter', {'name': 'gliclass-edge'}, HF + ' + gliclass==0.1.20', 'tested'),
    'qwen3-0.6b': ('llm_ll_adapter', {'name': 'qwen3-0.6b'}, HF, 'not run'),
    'qwen3.5-0.8b-base': ('llm_ll_adapter', {'name': 'qwen3.5-0.8b-base'}, HF, 'tested'),
    'julia-1': ('julia_adapter', {}, 'any (stdlib client; llama-server binary via LLAMA_SERVER)', 'tested'),
    'openthai-systemone': ('openthai_adapter', {}, HF, 'tested'),
    'jev': ('jev_adapter', {}, 'any (stdlib HTTPS client; hosted API, key file via JEV_API_KEY_FILE)', 'tested'),
}


def get(name):
    """-> predict(records) for a registered model."""
    mod, kw, _, _ = RIVALS[name]
    m = importlib.import_module(f'{__name__}.{mod}')
    return lambda records: m.predict(records, **kw)


def threads_used(name):
    """Threads in effect inside the model's last predict call: the adapter's threads_used() when it has one, else torch's
    process setting when torch is loaded, else None (servers and remote APIs: not observable from here)."""
    import sys
    m = importlib.import_module(f'{__name__}.{RIVALS[name][0]}')
    if hasattr(m, 'threads_used'):
        return m.threads_used()
    t = sys.modules.get('torch')
    return t.get_num_threads() if t is not None else None
