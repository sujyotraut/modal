import requests
import modal
import time
import os

MODEL_FAMILY = "swift"
MODEL_QUANT = "IQ2_XS"
# CONTEXT_SIZE = 65536
# CONTEXT_SIZE = 131072
CONTEXT_SIZE = 262144

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
UNAUTHENTICATED = True
STARTUP_TIMEOUT = 10 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

STRATA_VERSION = "v0.1.43.1"
STRATA_GIT_URL = "https://github.com/Niko1221/Strata.git"
REPO_NAME = STRATA_GIT_URL.rstrip("/").split("/")[-1].removesuffix(".git")

strata_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu26.04", add_python="3.14")
    .entrypoint([])
    .workdir("/root")
    .apt_install("git", "cmake", "build-essential", "libssl-dev", "curl", "ccache", "nodejs", "npm")
    .uv_pip_install("requests")
    # .run_commands(f"git clone --branch {STRATA_VERSION} --depth 1 {STRATA_GIT_URL}")
    # .workdir(REPO_NAME)
    # .run_function(build_llamacpp, cpu=BUILD_CPU)
    # .env({ "PATH": f"$PATH:/root/{REPO_NAME}/build/bin" })
)

strata_data_cache_vol = modal.Volume.from_name(
    name="strata-data-cache",
    create_if_missing=True,
    version=2
)

strata_cache_vol = modal.Volume.from_name(
    name="strata-cache",
    create_if_missing=True,
    version=2
)

app = modal.App("strata-server")

@app.server(
    port=PORT,
    gpu="L4:1",
    image=strata_image,
    volumes={
        "/root/Strata": strata_cache_vol,
        "/root/Strata-data": strata_data_cache_vol
    },
    secrets=[modal.Secret.from_name("strata-server-secret")],
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
        start_server()

    @modal.exit()
    def exit(self):
        print("Exiting Strata server (L4)...")

def start_server():
    import subprocess, os

    print("Entering Strata server...")

    if not os.path.exists("/root/Strata/.git"):
        subprocess.run(["git", "clone", "--branch", STRATA_VERSION, "--depth", "1", STRATA_GIT_URL, "/root/Strata"], check=True)

    subprocess.Popen([
        "/root/Strata/setup.sh",
        "--api-key", os.environ["STRATA_API_KEY"],
        "--data-dir", "/root/Strata-data",
        "--context", str(CONTEXT_SIZE),
        "--family", MODEL_FAMILY,
        "--model", MODEL_QUANT,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--no-browser",
        "--yes"
    ])

    # url = f"http://127.0.0.1:{PORT}"
    # api_key = os.environ["STRATA_API_KEY"]
    # health_check(url, api_key)


@app.local_entrypoint()
async def main():
    print("Starting strata server...")

    api_key = "MY_API_KEY"
    l4_url = await L4.get_url.aio()
