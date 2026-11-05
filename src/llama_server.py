import modal
import os

PARALLEL = 2
UBATCH_SIZE = 1024

GPU_TYPE = "A10"
AUTOLOAD_MODEL = "GLM-4.7-Flash"

PORT = 9931
MINUTES = 60
MAX_INPUTS = 10
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
UNAUTHENTICATED = True
STARTUP_TIMEOUT = 10 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

LLAMA_VERSION = "b12110"
LLAMACPP_INSTALL_SCRIPT = f"curl -LsSf https://llama.app/install.sh | sh"

llama_cpp_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu26.04", add_python="3.14")
    .entrypoint([])
    .apt_install("curl")
    .uv_pip_install("requests")
    .env({ "LLAMA_VERSION": LLAMA_VERSION })
    .run_commands(LLAMACPP_INSTALL_SCRIPT, gpu=f"{GPU_TYPE}:1")
    .add_local_file(local_path="models.ini", remote_path="/root/models.ini", copy=True)
    .env({ "PATH": f"$PATH:/root/.local/bin" })
)

hf_cache_vol = modal.Volume.from_name(
    name="huggingface-cache-gguf",
    create_if_missing=True,
    version=2
)

app = modal.App("llama-cpp")

def start_server():
    import json, subprocess

    print("Entering Llama server...")

    subprocess.Popen([
        "llama", "serve",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--models-max", "1",
        "--parallel", str(PARALLEL),
        "--ubatch-size", str(UBATCH_SIZE),
        "--models-preset", "/root/models.ini",
    ])

    max_retry = STARTUP_TIMEOUT
    api_key = os.environ["LLAMA_API_KEY"]

    subprocess.run([
        "curl",
        "--retry-delay", "1",
        "--retry-connrefused",
        "--retry", str(max_retry),
        "--retry-max-time", str(max_retry),
        "-H", f"Authorization: Bearer {api_key}",
        f"http://127.0.0.1:{PORT}/health"
    ])

    # payload = json.dumps({
    #     "model": AUTOLOAD_MODEL,
    #     "messages": [{"role": "user", "content": "warmup"}],
    #     "max_tokens": 1,
    # })

    # subprocess.Popen([
    #     "curl",
    #     "-d", payload,
    #     "--retry-delay", "1",
    #     "--retry", str(max_retry),
    #     "--retry-max-time", str(max_retry),
    #     "-H", f"Authorization: Bearer {api_key}",
    #     "-H", "Content-Type: application/json",
    #     f"http://127.0.0.1:{PORT}/v1/chat/completions"
    # ])

    payload = json.dumps({ "model": AUTOLOAD_MODEL })

    subprocess.Popen([
        "curl",
        "-X", "POST",
        "-d", payload,
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        f"http://127.0.0.1:{PORT}/models/load"
    ])


@app.local_entrypoint()
async def main():
   url =  await Server.get_url.aio()
   print(url)


@app.server(
    port=PORT,
    gpu=f"{GPU_TYPE}:1",
    image=llama_cpp_image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("llama-server-secret")],
    experimental_options={"enable_gpu_snapshot": True},
    enable_memory_snapshot=True,
    # Container configuration
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    target_concurrency=MAX_INPUTS,
    unauthenticated=UNAUTHENTICATED,
    startup_timeout=STARTUP_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Server:
    @modal.enter(snap=True)
    def enter(self):
        start_server()

    @modal.exit()
    def exit(self):
        print("Exiting Llama server...")
