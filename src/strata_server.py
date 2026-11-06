import modal

# Swift
MODEL_FAMILY = "swift"
MODEL_QUANT = "IQ2_XS"

# Unsloth
# MODEL_FAMILY = "unsloth"
# MODEL_QUANT = "UD-Q4_K_XL"

MODEL_VISION = "on"
CONTEXT_SIZE = 262144
# CONTEXT_SIZE = 131072
# CONTEXT_SIZE = 65536

GPU_TYPE = "L40S"
STRATA_VERSION = "latest"
STRATA_INSTALL_SCRIPT = f"curl -fsSL https://github.com/Niko1221/Strata/releases/{STRATA_VERSION}/download/install.sh -o install.sh && sh install.sh"

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
UNAUTHENTICATED = True
STARTUP_TIMEOUT = 30 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

STRATA_DATA_DIR = "/root/Strata-data"
strata_data_cache_vol = modal.Volume.from_name(
    name="strata-data-cache",
    create_if_missing=True,
    version=2
)

strata_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu26.04", add_python="3.14")
    .entrypoint([])
    .apt_install("curl")
    .uv_pip_install("requests")
    .run_commands(STRATA_INSTALL_SCRIPT, gpu=GPU_TYPE)
    .env({ "PATH": f"$PATH:/root/.local/bin" })
)

app = modal.App("strata")

@app.server(
    port=PORT,
    gpu=f"{GPU_TYPE}:1",
    image=strata_image,
    volumes={ STRATA_DATA_DIR: strata_data_cache_vol },
    secrets=[modal.Secret.from_name("strata-server-secret")],
    # experimental_options={"enable_gpu_snapshot": True},
    # memory=48 * 1024, # 48GB
    # enable_memory_snapshot=True,
    # Container configuration
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    target_concurrency=MAX_INPUTS,
    unauthenticated=UNAUTHENTICATED,
    startup_timeout=STARTUP_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Server:
    # @modal.enter(snap=True)
    @modal.enter()
    def enter(self):
        start_server(GPU_TYPE)

    @modal.exit()
    def exit(self):
        print(f"Exiting Strata server ({GPU_TYPE})...")

def start_server(gpu_type: str):
    import subprocess, os

    print(f"Entering Strata server ({gpu_type})...")

    subprocess.Popen([
        "strata", "launch",
        "--access", "lan",
        "--port", str(PORT),
        "--api-key", os.environ["STRATA_API_KEY"],
        "--data-dir", STRATA_DATA_DIR,
        "--context", str(CONTEXT_SIZE),
        "--vision", MODEL_VISION,
        "--family", MODEL_FAMILY,
        "--model", MODEL_QUANT,
        "--drafting", "dflash2",
        "--mtp", "on",
        "--yes"
    ])


@app.local_entrypoint()
async def main():
    url = Server.get_url.aio()
    print(url)
