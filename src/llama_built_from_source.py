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

app = modal.App("llama_server")


@app.function(
    image=llama_cpp_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("llama-secret")],
    # Container configuration
    min_containers=0,
    max_containers=1,
    timeout=30 * MINUTES,
    startup_timeout=10 * MINUTES,
    scaledown_window=1 * MINUTES,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    subprocess.run(["llama-server", "--version"])

    cmd = [
        "llama-server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--models-preset", "/root/models.ini",
        "--api-key", os.environ["LLAMA_API_KEY"],
    ]

    subprocess.Popen(cmd)
