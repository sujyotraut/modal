Modal Llama Server

Deploy and run a Llama-based inference server on Modal
, with model preloading and configuration managed through a lightweight Python project.

Overview

This project provides the building blocks for running a Llama model on Modal's serverless infrastructure.

The repository is organized around:

src/llama_server.py — server/inference entry point.
src/preload_model.py — model initialization and preloading logic.
models.ini — model configuration.
pyproject.toml — Python project and dependency configuration.

The goal is to keep model startup and serving logic separated, making it easier to deploy models without maintaining dedicated GPU infrastructure.
 
Project Structure
.
├── src/
│   ├── llama_server.py
│   └── preload_model.py
├── models.ini
├── pyproject.toml
├── .python-version
├── .gitignore
└── README.md

Requirements
Python
A Modal
 account
Modal CLI
The dependencies specified in pyproject.toml
Getting Started
1. Clone the repository
git clone https://github.com/sujyotraut/modal.git
cd modal

2. Set up the Python environment

If you use uv:

uv sync


Alternatively, install the project using your preferred Python environment manager.

3. Authenticate with Modal

Install the Modal client if necessary:

pip install modal


Then authenticate:

python -m modal setup


Follow the prompts to connect your local environment to your Modal account.

Configuration

Model-related configuration is kept in:

models.ini


Update this file according to the model you want to serve and the configuration expected by the application.

Keep credentials, API keys, and other secrets out of models.ini and source control. Use Modal's secret management facilities for sensitive values.

Running the Server

The main server entry point is:

src/llama_server.py


Depending on the Modal application definition, deployment can typically be initiated with:

modal deploy src/llama_server.py


For development or testing, you can run the application directly with:

modal run src/llama_server.py


Check the functions and entry points defined in llama_server.py if the project uses a different Modal command or invocation pattern.

Model Preloading

Model initialization is separated into:

src/preload_model.py


Preloading the model can help avoid repeatedly performing expensive initialization when new compute instances are started.

The exact model-loading behavior is controlled by the implementation and the configuration in models.ini.

Why Modal?

Modal
 provides on-demand cloud compute that is particularly useful for GPU-backed workloads.

Using Modal allows this project to:

Run GPU workloads without managing physical servers.
Scale compute based on demand.
Package the inference environment alongside the application.
Keep model-serving infrastructure relatively simple.
Spin up compute only when it is needed.
Development

After making changes to the source code, test the application locally through Modal:

modal run src/llama_server.py


For deployment:

modal deploy src/llama_server.py


Review the Modal dashboard and application logs when debugging startup, model-loading, or inference issues.

Troubleshooting
Model fails to load

Check:

The model configuration in models.ini.
Required Python dependencies.
GPU/resource requirements.
Any required Modal secrets or environment variables.
Modal application logs.
Deployment fails

Make sure you are authenticated:

python -m modal setup


Then verify that the Modal application entry point is the one defined in src/llama_server.py.

Slow startup

Large language models can require significant time and resources to initialize. The repository's separate preload module is intended to keep model initialization logic isolated and reusable.

Contributing

Contributions are welcome.

Fork the repository.
Create a feature branch.
git checkout -b feature/my-change

Make your changes.
Test the Modal application.
Commit your changes.
git commit -m "Add my change"

Push the branch and open a pull request.
License

No license is currently documented in the repository. Add a LICENSE file and update this section once a license has been selected.

Author

Sujyot Raut

Repository:
https://github.com/sujyotraut/modal
:::{"fallbackMarkdown":"","reference":{"matched_text":" ","prefix":null,"start_idx":5123,"end_idx":5123,"safe_urls":[],"refs":[],"alt":"","prompt_text":null,"type":"sources_footnote","sources":[{"title":"GitHub - sujyotraut/modal · GitHub","url":"https://github.com/sujyotraut/modal","attribution":"GitHub"}],"has_images":false},"showLoginRequiredCard":false}
