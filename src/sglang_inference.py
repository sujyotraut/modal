import asyncio
import subprocess
import time

import modal
import modal.experimental

GPU_TYPE = "L4"
NUMBER_OF_GPU = 1

PORT = 8080
MINUTES = 60
MAX_INPUTS = 8
BATCH = ",".join(map(str, range(1, MAX_INPUTS + 1)))

MODEL_NAME = "Qwen/Qwen3-14B-AWQ"

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name(
    name="huggingface-cache-v2",
    create_if_missing=True,
    version=2
)

sglang_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .entrypoint([])
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            # improve compatibility with snapshots
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

app = modal.App("sglang_server")

with sglang_image.imports():
    import requests

def warmup():
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
        ).raise_for_status()


def sleep():
    requests.post(
        f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
    ).raise_for_status()


def wake_up():
    requests.post(
        f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
    ).raise_for_status()


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(1)
    raise TimeoutError(f"SGLang server not ready within timeout of {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)

CMD = [
    "sglang", "serve",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--model-path", MODEL_NAME,
    # Flash attn on by default
    # "--context-length", "8192",
    # "--context-length", "131072",
    # Completely on the GPU, no CPU offloading
    "--kv-cache-dtype", "fp8_e4m3",
    "--max-running-requests", str(MAX_INPUTS),
    "--cuda-graph-bs-decode", "[1,2,3,4,5,6,7,8]",
    "--cuda-graph-max-bs-decode", str(MAX_INPUTS),
    "--cuda-graph-backend-prefill", "disabled"
]

@app.cls(
    image=sglang_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    timeout=20 * MINUTES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
class SGLangServer:
    @modal.enter(snap=True)
    def init_sglang(self):
        self.process = subprocess.Popen(CMD)
        print("Waiting for SGLang server to be ready...")
        wait_ready(self.process)
        print("SGLang server is ready. Warming up...")
        warmup()
        print("SGLang server is warmed up. Taking a snapshot...")
        sleep()
    
    @modal.enter(snap=False)
    def wake_up(self):
        print("Waking up SGLang server...")
        wake_up()

    @modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.process.terminate()

# @app.function(
#     image=sglang_image,
#     gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
#     volumes={HF_CACHE_PATH: HF_CACHE_VOL},
#     timeout=20 * MINUTES,
# )
# @modal.concurrent(max_inputs=10)
# @modal.web_server(port=SGLANG_SERVER_PORT, startup_timeout=20 * MINUTES)
# def serve():
#     import subprocess

#     cmd = [
#         "sglang", "serve",
#         "--host", "0.0.0.0",
#         "--port", str(SGLANG_SERVER_PORT),
#         "--model-path", MODEL_NAME,
#         # Flash attn on by default
#         # "--context-length", "8192",
#         # "--context-length", "131072",
#         # Completely on the GPU, no CPU offloading
#         "--kv-cache-dtype", "fp8_e4m3",
#     ]

#     subprocess.Popen(cmd)
