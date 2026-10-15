import modal

MODEL_FAMILY = "swift"
MODEL_QUANT = "IQ2_XS"
MODEL_VISION = "yes"
# CONTEXT_SIZE = 65536
# CONTEXT_SIZE = 131072
CONTEXT_SIZE = 262144

GPU_TYPE = "L4"

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
UNAUTHENTICATED = True
STARTUP_TIMEOUT = 30 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

STRATA_VERSION = "v0.1.43.1"
STRATA_GIT_URL = "https://github.com/Niko1221/Strata.git"
STRATA_DATA_DIR = "/root/Strata-data"

strata_data_cache_vol = modal.Volume.from_name(
    name="strata-data-cache",
    create_if_missing=True,
    version=2
)

# TODO: Extract modal family, quant, vision and context size a function arguments
def build_strata(gpu_type: str):
    import subprocess, os

    strata_root = f"/root/Strata_{gpu_type}"

    subprocess.run(
        ["git", "clone", "--branch", STRATA_VERSION, "--depth", "1", STRATA_GIT_URL, strata_root],
        check=True
    )

    subprocess.run([
        f"/root/Strata_{gpu_type}/setup.sh",
        "--api-key", os.environ["STRATA_API_KEY"],
        "--data-dir", STRATA_DATA_DIR,
        "--context", str(CONTEXT_SIZE),
        "--vision", MODEL_VISION,
        "--family", MODEL_FAMILY,
        "--model", MODEL_QUANT,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--no-browser",
        "--no-start",
        "--yes"
    ], check=True)

strata_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu26.04", add_python="3.14")
    .entrypoint([])
    .workdir("/root")
    .apt_install("git", "cmake", "build-essential", "libssl-dev", "curl", "ccache", "nodejs", "npm")
    .uv_pip_install("requests")
    .run_function(
        build_strata,
        gpu=GPU_TYPE,
        args=[GPU_TYPE],
        volumes={ STRATA_DATA_DIR: strata_data_cache_vol },
        secrets=[modal.Secret.from_name("strata-server-secret")]
    )
)

app = modal.App("strata-server")

@app.server(
    port=PORT,
    gpu=f"{GPU_TYPE}:1",
    image=strata_image,
    volumes={ STRATA_DATA_DIR: strata_data_cache_vol },
    secrets=[modal.Secret.from_name("strata-server-secret")],
    experimental_options={"enable_gpu_snapshot": True},
    # memory=48 * 1024, # 48GB
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
        start_server(GPU_TYPE)

    @modal.exit()
    def exit(self):
        print(f"Exiting Strata server ({GPU_TYPE})...")

def start_server(gpu_type: str):
    import subprocess

    print(f"Entering Strata server ({gpu_type})...")
    subprocess.Popen([f"/root/Strata_{gpu_type}/setup.sh"])

    # url = f"http://127.0.0.1:{PORT}"
    # api_key = os.environ["STRATA_API_KEY"]
    # health_check(url, api_key)


@app.local_entrypoint()
async def main():
    print("Starting strata server...")
