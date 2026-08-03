import modal

GPU_TYPE = "L40S"
NUMBER_OF_GPU = 1
# 16 - cost efficient
# 64 - faster builds
BUILD_CPU = 64

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10
STARTUP_TIMEOUT = 10 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

LLAMACPP_VERSION = "b10238"
LLAMACPP_GIT_URL = "https://github.com/ggml-org/llama.cpp.git"

# LLAMACPP_VERSION = "tqp-v0.3.0"
# LLAMACPP_GIT_URL = "https://github.com/TheTom/llama-cpp-turboquant.git"

REPO_NAME = LLAMACPP_GIT_URL.rstrip("/").split("/")[-1].removesuffix(".git")

def build_llamacpp():
    import subprocess

    print(f"Building {REPO_NAME} from source...")

    subprocess.run(
        ["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF"],
        check=True
    )

    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "-j", str(BUILD_CPU)],
        check=True
    )

llama_cpp_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu26.04", add_python="3.14")
    .entrypoint([])
    .workdir("/root")
    .apt_install("git", "cmake", "build-essential", "libssl-dev", "curl", "ccache", "nodejs", "npm")
    .uv_pip_install("requests")
    .run_commands(f"git clone --branch {LLAMACPP_VERSION} --depth 1 {LLAMACPP_GIT_URL}")
    .workdir(REPO_NAME)
    .run_function(build_llamacpp, cpu=BUILD_CPU)
    .add_local_file(local_path="models.ini", remote_path="/root/models.ini", copy=True)
    .env({ "PATH": f"$PATH:/root/{REPO_NAME}/build/bin" })
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
    secrets=[modal.Secret.from_name("llama-server-secret")],
    experimental_options={"enable_gpu_snapshot": True},
    enable_memory_snapshot=True,
    # Container configuration
    port=PORT,
    min_containers=0,
    max_containers=1,
    unauthenticated=True,
    target_concurrency=MAX_INPUTS,
    startup_timeout=STARTUP_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class LlamaServer:
    @modal.enter(snap=True)
    def enter(self):
        import subprocess

        print("Entering Llama server...")

        subprocess.Popen([
            "llama-server",
            "--models-max", "1",
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--models-preset", "/root/models.ini",
        ])

        url = f"http://127.0.0.1:{PORT}"
        server_health_check(url)

    @modal.exit()
    def exit(self):
        print("Exiting Llama server...")


def server_health_check(url: str) -> bool:
    import time
    import requests

    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            res = requests.get(f"{url}/health", timeout=5)
            if res.status_code == 200:
                return True
            print(f"Got {res.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed ({e}), retrying...")
        time.sleep(5)
    
    print(f"Startup timeout ({STARTUP_TIMEOUT/60} minutes) reached, server health check failed.")
    return False


@app.local_entrypoint()
async def main():
    print("Starting llama server...")

    url = LlamaServer.get_url()
    server_health_check(url)
