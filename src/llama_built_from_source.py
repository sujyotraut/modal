import os
import modal

GPU_TYPE = "L40S"
NUMBER_OF_GPU = 1

PORT = 8080
MINUTES = 60
MAX_INPUTS = 10

llama_cpp_image = (
    # NOTE: T4 GPU doesn't support cuda 13, only cuda 12.x
    # modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .entrypoint([])
    .workdir("/root")
    .apt_install("git", "cmake", "build-essential", "libssl-dev", "curl", "ccache")
    .run_commands("git clone https://github.com/ggml-org/llama.cpp")
    .workdir("llama.cpp")
    .run_commands(
        "cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF",
        "cmake --build build --config Release -j 64",
        gpu="L4",
    )
    .add_local_file(local_path="models.ini", remote_path="/root/models.ini", copy=True)
    .env(
        {
            "PATH": "$PATH:/root/llama.cpp/build/bin",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

# hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

hf_cache_vol = modal.Volume.from_name(
    name="huggingface-cache-v2", create_if_missing=True, version=2
)

app = modal.App("llama_server")


@app.function(
    image=llama_cpp_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("llama-secret")],
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "llama-server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--models-preset", "/root/models.ini",
        "--api-key", os.environ["LLAMA_API_KEY"],
    ]

    subprocess.Popen(cmd)


@app.function(
    image=llama_cpp_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("llama-secret")],
)
def benchmark():
    import subprocess

    # TODO: Add benchmark testing to get get maximize speed
    cmd = [
        "llama-bench",
        "-hf", "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL",
    ]

    subprocess.run(cmd)


@app.local_entrypoint()
def main():
    benchmark.remote()


if __name__ == "__main__":
    main()
