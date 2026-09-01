from dataclasses import dataclass
import requests
import modal
import time
import os

@dataclass
class ServerConfig:
    parallel: int
    ubatch_size: int
    preload_model: str

    @property
    def batch_size(self) -> int:
        return self.ubatch_size * self.parallel

# 16 - cost efficient
# 64 - faster builds
BUILD_CPU = 64

l4_config = ServerConfig(
    parallel=2,
    ubatch_size=512,
    preload_model="GLM-4.7-Flash"
)

l40s_config = ServerConfig(
    parallel=4,
    ubatch_size=1024,
    preload_model="GLM-4.7-Flash"
)

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
UNAUTHENTICATED = True
STARTUP_TIMEOUT = 10 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

LLAMACPP_VERSION = "xsn/dflash2"
LLAMACPP_GIT_URL = "https://github.com/ggml-org/llama.cpp.git"

# LLAMACPP_VERSION = "tqp-v0.3.0"
# LLAMACPP_GIT_URL = "https://github.com/TheTom/llama-cpp-turboquant.git"

REPO_NAME = LLAMACPP_GIT_URL.rstrip("/").split("/")[-1].removesuffix(".git")

def build_llamacpp():
    import subprocess

    print(f"Building {REPO_NAME} from source...")

    subprocess.run(
        ["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_IS_DEV=OFF"],
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
    port=PORT,
    gpu="L4:1",
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
class L4:
    @modal.enter(snap=True)
    def enter(self):
        start_server(l4_config)

    @modal.exit()
    def exit(self):
        print("Exiting Llama server (L4)...")


@app.server(
    port=PORT,
    gpu="L40S:1",
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
class L40S:
    @modal.enter(snap=True)
    def enter(self):
        start_server(l40s_config)

    @modal.exit()
    def exit(self):
        print("Exiting Llama server (L40S)...")


@app.server(
    port=PORT,
    gpu="B300:1",
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
class B300:
    @modal.enter(snap=True)
    def enter(self):
        start_server(ServerConfig(
            parallel=4,
            ubatch_size=2048,
            preload_model="GLM-4.7-Flash"
        ))

    @modal.exit()
    def exit(self):
        print("Exiting Llama server (B300)...")

def start_server(config :ServerConfig):
    import subprocess

    print("Entering Llama server...")

    subprocess.Popen([
        "llama-server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--models-max", "1",
        "--parallel", str(config.parallel),
        "--batch-size", str(config.batch_size),
        "--ubatch-size", str(config.ubatch_size),
        "--models-preset", "/root/models.ini",
    ])

    url = f"http://127.0.0.1:{PORT}"
    api_key = os.environ["LLAMA_API_KEY"]
    health_check(url, api_key)


def health_check(url: str, api_key: str) -> bool:
    deadline = time.time() + STARTUP_TIMEOUT
    attempt = 0

    print("Starting health check")
    while time.time() < deadline:
        try:
            headers = { "Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{url}/health", headers=headers, timeout=1)
            if response.ok:
                print(f"Health check, passed [attempt={attempt}, status_code={response.status_code}]")
                return True

            print(f"Health check, unhealthy [attempt={attempt}, status_code={response.status_code}]")
        except requests.exceptions.RequestException:
            print(f"Health check, unreachable [attempt={attempt}]")

        attempt += 1
        time.sleep(2)

    print(f"Health check, failed [attempt={attempt}, timeout={STARTUP_TIMEOUT}]")
    return False


@app.local_entrypoint()
async def main():
    print("Starting llama server...")

    api_key = "MY_API_KEY"
    l4_url = await L4.get_url.aio()
    health_check(l4_url, api_key)
