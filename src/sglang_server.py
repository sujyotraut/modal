import subprocess
import time

import modal

GPU_TYPE = "L40S"
NUMBER_OF_GPU = 1

PORT = 8080
MINUTES = 60
MAX_INPUTS = 8
BATCH = "[" + ",".join(map(str, range(1, MAX_INPUTS + 1))) + "]"

SGLANG_VERSION = "v0.5.16"

MODEL_NAME = "cyankiwi/Qwen3.6-27B-AWQ-INT4"
# MODEL_REVISION = "e5cc0400fb2403c437c2c40a7c52fb5ae93fda18"

# MODEL_NAME = "cyankiwi/GLM-4.7-Flash-AWQ-4bit"
# MODEL_REVISION = "25624b53414e585bcf7dcb9584667c3106c6089b"

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name(
    name="huggingface-cache",
    create_if_missing=True,
    version=2
)

sglang_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry(f"lmsysorg/sglang:{SGLANG_VERSION}-cu130-runtime")
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

app = modal.App("sglang-server")

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

@app.server(
    port=PORT,
    max_containers=1,
    image=sglang_image,
    unauthenticated=True,
    enable_memory_snapshot=True,
    startup_timeout=30 * MINUTES,
    target_concurrency=MAX_INPUTS,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    experimental_options={"enable_gpu_snapshot": True},
)
class SGLangServer:
    @modal.enter(snap=True)
    def init_sglang(self):
        self.process = subprocess.Popen([
            "sglang", "serve",
            "--model-path", MODEL_NAME,
            "--tp-size", "1",
            "--reasoning-parser", "qwen3",
            "--mem-fraction-static", "0.8",
            "--host", "0.0.0.0",
            "--port", str(PORT),
        ])

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

    @modal.exit()
    def stop(self):
        self.process.terminate()
