import modal

GPU_TYPE = "L40S"
NUMBER_OF_GPU = 1
# 16 - cost efficient
# 64 - faster builds
BUILD_CPU = 64

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10

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
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--models-preset", "/root/models.ini",
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