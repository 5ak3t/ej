"""Julia-1 (SupersonicLabs/Julia-1, 144M mmBERT-small decision model) through llama.cpp's POST /v1/systemone.

Serving path = the ggml-org GGUF card's documented one (llama.cpp PR #29818): llama-server built from source, CPU only, at
llama.cpp b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea (`cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
-DGGML_NATIVE=ON; cmake --build build --target llama-server`). The GGUF stores decision type 'laya' (2 head blocks,
max_head_tokens 256) and the 'systemone' prompt template; the server answers each question in one forward pass and returns
full-precision softmax probabilities (no 4-dp rounding). Requests go through systemone_http (same wire format and
common.from_response mapping as every other adapter).

If JULIA_URL (default http://127.0.0.1:8090) does not answer /health, the first call starts one llama-server process
(threads = BENCH_THREADS, ctx = ubatch = 8192 so a whole prompt fits one batch, as the laya path requires) and stops it by its
own PID at exit. JULIA_GGUF picks the file (default BF16, the closest to the fp32 reference; Q8_0 is the smaller build).
Julia's own Python runtime (julia/ in the model repo) is NOT used: it is downloaded repo code (never executed here)."""
import atexit
import functools
import os
import subprocess
import time
import urllib.request

from . import systemone_http

SERVER = os.environ.get('LLAMA_SERVER', 'llama-server')  # path of the llama-server binary built as above
LLAMA_CPP_COMMIT = 'b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea'
GGUF_REPO = ('ggml-org/Julia-1-GGUF', '16fee17949206fbf58da9347daea44d792a81211')
GGUF = {'bf16': 'Julia-1-BF16.gguf', 'q8_0': 'Julia-1-Q8_0.gguf'}
SOURCE = ('SupersonicLabs/Julia-1', 'a85b127321d580d65176c89ced8273f305745d85')  # GGUF .src_sha


def gguf_path(name):
    """Local path of a GGUF: an existing file path as given, else the pinned file from the Hugging Face Hub."""
    if os.path.exists(name):
        return name
    from huggingface_hub import hf_hub_download
    return hf_hub_download(GGUF_REPO[0], GGUF.get(name, name), revision=GGUF_REPO[1])


def healthy(url, timeout=2.0):
    """True when the server at url answers /health with 200."""
    try:
        with systemone_http._opener(url).open(urllib.request.Request(url.rstrip('/') + '/health'), timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def ensure_server(url):
    """Start llama-server for Julia-1 unless one already answers at url. -> url."""
    if healthy(url):
        return url
    port = url.rsplit(':', 1)[-1].strip('/')
    gguf = os.environ.get('JULIA_GGUF', 'bf16')
    threads = os.environ.get('BENCH_THREADS', '1')
    cmd = [SERVER, '-m', gguf_path(gguf), '--host', '127.0.0.1', '--port', port, '-t', threads, '-tb', threads,
           '-c', '8192', '-b', '8192', '-ub', '8192', '-np', '1']
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(stop, proc)
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f'llama-server exited with {proc.returncode}: {" ".join(cmd)}')
        if healthy(url):
            return url
        time.sleep(0.5)
    stop(proc)
    raise RuntimeError('llama-server did not become healthy within 120 s')


def stop(proc):
    """Terminate the server this module started (by its own PID), then wait for it."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def predict(records):
    """[{qid: probs}] per record, in each question's option order."""
    url = ensure_server(os.environ.get('JULIA_URL', 'http://127.0.0.1:8090'))
    return systemone_http.predict(records, url=url)
