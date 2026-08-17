# Llama Server on Modal

GPU-accelerated [`llama.cpp`](https://github.com/ggml-org/llama.cpp) inference deployed with [Modal](https://modal.com/).

The repository builds `llama.cpp` with CUDA support, runs `llama-server` on Modal GPU infrastructure, and provides configurable GGUF model presets through `models.ini`.

## Features

* CUDA-enabled `llama.cpp` built from source
* Modal GPU deployments
* NVIDIA **L4** and **L40S** configurations
* GGUF models hosted on Hugging Face
* Configurable model presets
* Persistent Hugging Face model cache
* GPU and memory snapshots
* Automatic server health checks
* Configurable concurrency and container scaling
* Speculative decoding support for compatible models

## Repository Structure

```text
.
├── src/
│   ├── llama_server.py
│   └── preload_model.py
├── models.ini
├── pyproject.toml
├── .python-version
├── .gitignore
└── README.md
```

## Requirements

* Python 3.14+
* A Modal account
* Modal CLI
* Access to the required Hugging Face models
* A CUDA-capable Modal GPU

The project currently specifies Python `>=3.14` and does not declare Python dependencies in `pyproject.toml`.

## Installation

Clone the repository:

```bash
git clone https://github.com/sujyotraut/modal.git
cd modal
```

Create a virtual environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

Authenticate with Modal:

```bash
modal setup
```

## Model Configuration

Models are defined in `models.ini`.

The configuration includes models such as:

* ThinkingCap-Qwen3.6-27B
* Qwen3.6-35B-A3B
* Qwen3.6-27B
* Gemma 4 31B
* GLM-4.7-Flash
* DeepSeek-R1-Distill-Qwen-32B

The models are sourced from Hugging Face GGUF repositories.

For example:

```ini
[GLM-4.7-Flash]
hf-repo = unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL
```

### Common Model Settings

Global defaults include:

```ini
[*]
jinja = on
flash-attn = on
cache-reuse = 512
cache-type-k = q8_0
cache-type-v = q8_0
n-gpu-layers = all
kv-unified = on
```

Some models additionally use speculative decoding:

```ini
spec-type = draft-mtp
spec-draft-n-max = 3
```

## Modal Deployment

The main deployment is implemented in:

```text
src/llama_server.py
```

The application is named:

```text
llama-server
```

Deploy it with:

```bash
modal deploy src/llama_server.py
```

For local Modal development:

```bash
modal run src/llama_server.py
```

## GPU Configurations

Two Modal server configurations are currently defined.

### NVIDIA L4

```python
ServerConfig(
    parallel=2,
    ubatch_size=512,
    preload_model="GLM-4.7-Flash"
)
```

Effective batch size:

```text
2 × 512 = 1024
```

### NVIDIA L40S

```python
ServerConfig(
    parallel=4,
    ubatch_size=1024,
    preload_model="GLM-4.7-Flash"
)
```

Effective batch size:

```text
4 × 1024 = 4096
```

Both configurations currently preload `GLM-4.7-Flash`.

## llama.cpp Build

The deployment builds `llama.cpp` from source with CUDA enabled.

The current revision is:

```text
b10456
```

The build uses:

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF
cmake --build build --config Release -j 64
```

The container is based on:

```text
nvidia/cuda:13.3.1-devel-ubuntu26.04
```

and uses Python 3.14.

## Hugging Face Cache

A persistent Modal Volume is used for model caching:

```text
huggingface-cache-gguf
```

It is mounted at:

```text
/root/.cache/huggingface
```

This allows downloaded models to persist between container lifecycles.

## Server Configuration

The server listens on:

```text
0.0.0.0:8080
```

Relevant deployment settings include:

| Setting            |        Value |
| ------------------ | -----------: |
| Port               |       `8080` |
| Max inputs         |         `10` |
| Minimum containers |          `0` |
| Maximum containers |          `1` |
| Startup timeout    | `10 minutes` |
| Scale-down window  |   `1 minute` |
| llama.cpp          |     `b10456` |
| L4 GPU             |     `1 × L4` |
| L40S GPU           |   `1 × L40S` |

The deployment also enables GPU and memory snapshots.

## Authentication

The Modal application references the secret:

```text
llama-server-secret
```

The server-side health check reads:

```text
LLAMA_API_KEY
```

and sends it as a Bearer token when checking:

```text
/health
```

Do not commit API keys or other credentials to the repository.

## Health Checks

After starting `llama-server`, the deployment waits for the server's health endpoint:

```text
GET /health
```

The health check retries every two seconds until the server becomes healthy or the startup timeout is reached.

## Changing the Default Model

The model loaded by each GPU configuration can be changed in `src/llama_server.py`.

For example:

```python
l4_config = ServerConfig(
    parallel=2,
    ubatch_size=512,
    preload_model="Qwen3.6-27B"
)
```

The selected model must correspond to a preset in `models.ini`.

## Adding a Model

Add a model preset to `models.ini`:

```ini
[MyModel]
hf-repo = organization/model-GGUF:Q4_K_M
```

For a model supporting draft-MTP speculative decoding:

```ini
[MyModel-MTP]
hf-repo = organization/model-GGUF:Q4_K_M
spec-type = draft-mtp
spec-draft-n-max = 3
```

## Local Model Preloading

`src/preload_model.py` provides a separate local utility for starting a llama server and loading a model.

The current configuration uses:

```python
MODEL_NAME = "Qwen3.5-4B"
```

and a local server on:

```text
localhost:1234
```

The script starts the server, performs a health check, requests the model to load, and waits until its status becomes `loaded`.

The script currently contains a machine-specific path:

```text
/home/sujyot/containers/llama-server/models.ini
```

This should be changed when running the utility on another machine.

## Performance Tuning

The primary batching parameters are:

```python
parallel
ubatch_size
```

The effective batch size is:

```text
batch_size = parallel × ubatch_size
```

Increasing these values can improve throughput but also increases GPU memory requirements.

When tuning performance, consider:

* GPU memory
* Model size
* Quantization
* Context length
* Concurrent requests
* Batch size
* Target latency

## Troubleshooting

### Server does not start

Check the Modal application logs:

```bash
modal app logs llama-server
```

Look for:

* CUDA initialization errors
* Model download failures
* GPU out-of-memory errors
* `llama-server` startup failures
* Missing Modal secrets
* Health-check failures

### Model cannot be loaded

Verify that the model name matches a section in `models.ini`.

For example:

```text
GLM-4.7-Flash
```

must correspond to:

```ini
[GLM-4.7-Flash]
```

### GPU out of memory

Try:

* A smaller quantization
* A smaller model
* Lower `parallel`
* Lower `ubatch_size`
* A GPU with more memory

## Security

Review the deployment's authentication settings before exposing it to production traffic.

The current Modal configuration sets:

```python
unauthenticated=True
```

while the internal health check uses `LLAMA_API_KEY`. This means deployment-level access control should be considered separately from the health-check authentication mechanism.

Never commit:

* API keys
* Modal secrets
* Hugging Face tokens
* Private credentials

## Contributing

Contributions and improvements are welcome.

Before submitting a change:

1. Keep model configuration changes in `models.ini`.
2. Keep deployment logic in `src/llama_server.py`.
3. Avoid committing credentials or machine-specific secrets.
4. Test configuration changes before deploying to Modal.
5. Clearly describe GPU, model, and performance-related changes in pull requests.

## Related Projects

* [llama.cpp](https://github.com/ggml-org/llama.cpp)
* [Modal](https://modal.com/)
* [Hugging Face](https://huggingface.co/)

## Repository

[github.com/sujyotraut/modal](https://github.com/sujyotraut/modal)

---

A Modal-based deployment setup for running configurable `llama.cpp` GGUF inference workloads on GPU infrastructure.
