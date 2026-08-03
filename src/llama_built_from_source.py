import os
import modal

NUMBER_OF_GPU = 1
GPU_TYPE = "L40S"
# GPU_TYPE = "A100"

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10

LLAMACPP_VERSION = "b10201"
LLAMACPP_GIT_URL = "https://github.com/ggml-org/llama.cpp"

llama_cpp_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    # modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu26.04", add_python="3.14")
    .entrypoint([])
    .workdir("/root")
    .apt_install("git", "cmake", "build-essential", "libssl-dev", "curl", "ccache")
    .run_commands(f"git clone --branch {LLAMACPP_VERSION} --depth 1 {LLAMACPP_GIT_URL}")
    .workdir("llama.cpp")
    .run_commands(
        # Slower build and more portability since it complies for all cuda GPUs
        # "cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF",
        # Faster build and less portability since it only complies for this exact GPU
        # NOTE: In practice both shows similar build time and cost but `-DGGML_CUDA=ON` 
        # shows faster inference speed (40t/s) compare to it's counterpart (35t/s)
        "cmake -B build -DGGML_CUDA=ON",
        "cmake --build build --config Release -j 64",
        gpu=GPU_TYPE,
    )
    .add_local_file(local_path="models.ini", remote_path="/root/models.ini", copy=True)
    .env(
        {
            "PATH": "$PATH:/root/llama.cpp/build/bin",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name(
    name="huggingface-cache-gguf",
    create_if_missing=True,
    version=2
)

app = modal.App("llama-server")

@app.server(
    image=llama_cpp_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("llama-secret")],
    # Container configuration
    port=PORT,
    min_containers=0,
    max_containers=1,
    startup_timeout=10 * MINUTES,
    scaledown_window=1 * MINUTES,
    target_concurrency=MAX_INPUTS,
    unauthenticated=True
)
class LlamaServer:
    @modal.enter()
    def enter(self):
        import subprocess

        print("Entering Llama server...")

        subprocess.Popen([
            "llama-server",
            "--models-max", "1",
            "--models-preset", "/root/models.ini",
            "--api-key", os.environ["LLAMA_API_KEY"],
            "--host", "0.0.0.0",
            "--port", str(PORT),
        ])

    @modal.exit()
    def exit(self):
        print("Exiting Llama server...")

@app.local_entrypoint()
async def main():
    import time
    import requests

    print("Starting llama server...")

    url = await LlamaServer.get_url.aio()
    print(url)

    deadline = time.time() + 10 * MINUTES
    while time.time() < deadline:
        try:
            res = requests.get(f"{url}/v1/models", timeout=5)
            if res.status_code == 200:
                print(res.json())
                break
            print(f"Got {res.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed ({e}), retrying...")
        time.sleep(5)
    else:
        print("Server didn't become ready in time")