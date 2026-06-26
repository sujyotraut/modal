import os
import modal

# MODEL_NAME = "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF:UD-Q4_K_XL"
MODEL_NAME = "google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0"
# MODEL_NAME = "ggml-org/SmolLM3-3B-GGUF:Q4_K_M"

GPU_TYPE = "L4"
NUMBER_OF_GPU = 1

MINUTES = 60
LLAMA_SERVER_PORT = 8080

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
        # "cmake --build build --config Release -j${nproc}",
        # f"cmake --build build --config Release -j {os.cpu_count()}",
        gpu="L4",
    )
    .env(
        {
            "PATH": "$PATH:/root/llama.cpp/build/bin",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("llama_server")


@app.function(
    image=llama_cpp_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=5 * MINUTES,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=LLAMA_SERVER_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "llama-server",
        "-hf",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(LLAMA_SERVER_PORT),
    ]

    subprocess.Popen(cmd)


@app.function(
    image=llama_cpp_image,
    gpu=f"{GPU_TYPE}:{NUMBER_OF_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def benchmark():
    import subprocess

    cmd = [
        "llama-bench",
        "-hf",
        MODEL_NAME,
    ]

    subprocess.run(cmd)


@app.local_entrypoint()
def main():
    benchmark.remote()


if __name__ == "__main__":
    main()
