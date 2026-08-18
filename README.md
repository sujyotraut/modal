# Modal Llama Server

A serverless LLM inference setup using [Modal](https://modal.com/) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

This project builds `llama.cpp` with CUDA support inside a Modal image and exposes an OpenAI-compatible-style LLM server through Modal. It supports GPU-backed inference on **NVIDIA L4** and **L40S** instances, with Hugging Face GGUF models loaded on demand.

## ✨ Features

* 🚀 Serverless LLM inference with Modal
* 🎮 NVIDIA **L4** and **L40S** GPU configurations
* ⚡ CUDA-enabled `llama.cpp` built from source
* 📦 GGUF models downloaded from Hugging Face
* 💾 Persistent Hugging Face model cache using a Modal Volume
* 🧠 GPU and memory snapshots for faster startup
* 🔐 API-key based authentication
* ❤️ Automatic server health checks
* 📈 Configurable concurrency and batching
* 🔄 Automatic container scale-down when idle
* 🧩 Multiple model presets through `models.ini`

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      Client          │
                    │  HTTP / LLM Request  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │        Modal         │
                    │   Serverless GPU     │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌──────────────┐             ┌──────────────┐
        │   NVIDIA L4  │             │  NVIDIA L40S  │
        │  2 parallel  │             │  4 parallel  │
        └──────┬───────┘             └──────┬───────┘
               │                            │
               └────────────┬───────────────┘
                            ▼
                   ┌─────────────────┐
                   │    llama.cpp    │
                   │  llama-server   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  GGUF Models    │
                   │ Hugging Face    │
                   └─────────────────┘
```

## 📁 Project Structure

```text
.
├── src/
│   ├── llama_server.py       # Modal deployment and llama.cpp server
│   └── preload_model.py      # Local model preloading utility
├── models.ini                # llama.cpp model configurations
├── pyproject.toml             # Python project configuration
├── .python-version
└── README.md
```

## ⚙️ Requirements

* Python **3.14+**
* A [Modal](https://modal.com/) account
* Modal CLI
* NVIDIA GPU access through Modal
* Hugging Face access for the configured models

The project currently requires Python `>=3.14`.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sujyotraut/modal.git
cd modal
```

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Or create and activate a virtual environment with Python 3.14+ and install the required packages.

### 3. Configure Modal

Authenticate the Modal CLI:

```bash
modal setup
```

Create the required Modal secret:

```bash
modal secret create llama-server-secret LLAMA_API_KEY="your-api-key"
```

The deployment expects a secret named `llama-server-secret` containing `LLAMA_API_KEY`.

## 🖥️ Deploy the Server

Deploy the Modal application with:

```bash
modal deploy src/llama_server.py
```

For development/testing:

```bash
modal run src/llama_server.py
```

The application defines two server configurations:

| GPU         | Parallel Requests | Micro Batch Size | Batch Size |
| ----------- | ----------------: | ---------------: | ---------: |
| NVIDIA L4   |                 2 |              512 |       1024 |
| NVIDIA L40S |                 4 |             1024 |       4096 |

These values are configured in `src/llama_server.py`.

## 🤖 Models

Models are configured in `models.ini`.

Currently available presets include:

* `ThinkingCap-Qwen3.6-27B-MTP`
* `ThinkingCap-Qwen3.6-27B`
* `Qwen3.6-35B-A3B-MTP`
* `Qwen3.6-35B-A3B`
* `Qwen3.6-27B-MTP`
* `Qwen3.6-27B`
* `gemma-4-31B`
* `GLM-4.7-Flash`
* `DeepSeek-R1-Distill-Qwen-32B`

The models use GGUF repositories hosted on Hugging Face, with several configurations using speculative decoding / draft MTP.

### Default model

The current Modal server configuration uses:

```text
GLM-4.7-Flash
```

The L4 configuration uses two-way parallelism, while the L40S configuration uses four-way parallelism.

## 🧠 llama.cpp Configuration

The container builds `llama.cpp` from source with CUDA enabled.

Important build configuration:

```text
GGML_CUDA=ON
GGML_NATIVE=OFF
```

The project currently pins llama.cpp to:

```text
b10456
```

The build uses a CUDA 13.3 development image and installs the required compilation tools before building llama.cpp.

## 💾 Model Cache

A persistent Modal Volume is used for the Hugging Face cache:

```text
huggingface-cache-gguf
```

This allows downloaded model files to persist between containers instead of being downloaded from scratch every time.

## ❤️ Health Checks

Before the server is considered ready, the deployment performs a health check against:

```text
GET /health
```

with bearer-token authentication.

The server waits up to **10 minutes** for the health check to succeed.

## 📊 Scaling Configuration

The current deployment is configured with:

```text
min_containers = 0
max_containers = 1
target_concurrency = 10
scaledown_window = 1 minute
```

This means the deployment can scale down completely when idle and run up to one container at a time.

## 🔐 Authentication

Authentication is enabled for the deployed server.

The API key is supplied through the Modal secret:

```text
llama-server-secret
```

with:

```text
LLAMA_API_KEY
```

Do **not** commit API keys or secrets to the repository.

## 🧪 Local Model Preloading

`src/preload_model.py` provides a utility for starting a local llama server and requesting a model load.

The default model in this utility is:

```text
Qwen3.5-4B
```

It communicates with the server using:

```text
POST /models/load
GET  /models
```

and waits until the requested model reports a `loaded` status.

Run it with:

```bash
python src/preload_model.py
```

> Update the model name, API key, and local server path before using this utility in a different environment.

## 🔧 Customizing the Deployment

### Change the GPU

In `src/llama_server.py`, modify the Modal server configuration:

```python
@app.server(
    gpu="L4:1",
    ...
)
```

or:

```python
@app.server(
    gpu="L40S:1",
    ...
)
```

### Change the model

Update the `preload_model` value:

```python
l4_config = ServerConfig(
    parallel=2,
    ubatch_size=512,
    preload_model="GLM-4.7-Flash"
)
```

You can use any model preset defined in `models.ini`.

### Add a model

Add a new section to `models.ini`:

```ini
[MyModel]
hf-repo = username/my-model-GGUF:Q4_K_M
```

For models supporting speculative decoding, additional configuration can be added:

```ini
[MyModel-MTP]
hf-repo = username/my-model-GGUF:Q4_K_M
spec-type = draft-mtp
spec-draft-n-max = 3
```

## 🛠️ Useful Commands

```bash
# Authenticate with Modal
modal setup

# Run locally through Modal
modal run src/llama_server.py

# Deploy to Modal
modal deploy src/llama_server.py

# Check Modal applications
modal app list

# Check Modal volumes
modal volume list

# Check Modal secrets
modal secret list
```

## 📌 Configuration

| Setting           | Current Value   |
| ----------------- | --------------- |
| Python            | `>=3.14`        |
| llama.cpp         | `b10456`        |
| Port              | `8080`          |
| Max inputs        | `10`            |
| Min containers    | `0`             |
| Max containers    | `1`             |
| Startup timeout   | `10 minutes`    |
| Scale-down window | `1 minute`      |
| Authentication    | Enabled         |
| Model cache       | Modal Volume    |
| Default model     | `GLM-4.7-Flash` |

## ⚠️ Notes

* GPU availability and pricing depend on Modal's current infrastructure.
* Model download times depend on model size and cache state.
* Large GGUF models may require a different GPU configuration.
* `models.ini` contains model references; the actual model weights are downloaded from Hugging Face.
* Keep API keys and Modal secrets outside the repository.
* The repository is currently configured as a minimal Python project and does not declare Python dependencies in `pyproject.toml`.

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the deployment.
5. Open a pull request.

## 📄 License

No license is currently specified in the repository. Add a `LICENSE` file before distributing the project as open-source software.

## 🔗 Links

* [Repository](https://github.com/sujyotraut/modal)
* [Modal](https://modal.com/)
* [llama.cpp](https://github.com/ggml-org/llama.cpp)
* [Hugging Face](https://huggingface.co/)

---

Built with **Modal + llama.cpp + GGUF**.
