# lmc — LM Studio Model Control

[![Python Version](https://img.shields.io/pypi/pyversions/lmc)](https://pypi.org/project/lmc/)
[![PyPI - Version](https://img.shields.io/pypi/v/lmc)](https://pypi.org/project/lmc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`lmc` is a lightweight CLI tool for managing [LM Studio](https://lmstudio.ai) models on your local machine or remote server. It provides simple commands to list available models, check their loading status, and dynamically load/unload them—perfect for optimizing memory usage during development or production workflows. 

`lmc` was built using [RecursiveSI/flint](https://huggingface.co/recursivesi/flint) and [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

## Features

- `lmc list` — list all models available on the server, and whether they are currently loaded
- `lmc load <model_key>` — load a model into memory on the LM Studio server
- `lmc unload <model_key>` — unload a model instance from memory

## Installation

Requires Python 3.9+.

```bash
pip install lmc
```

Or from source:

```bash
pip install .
```

## Usage

### List models

```bash
lmc list
```

Shows a table of all discovered models with quantization, load status, size, and context info. Loaded models are highlighted in green.

### Load a model

```bash
lmc load <model_key>
```

Example:

```bash
lmc load qwen3.5-9b
lmc load recursivesi/flint
```

Optional flags:

```bash
lmc load qwen3.5-9b --context-length 65536
lmc load qwen3.5-9b --flash-attention
lmc load qwen3.5-9b --offload-kv-cache-to-gpu
```

### Unload a model

```bash
lmc unload <model_key>
```

Example:

```bash
lmc unload qwen3.5-9b
```

You can also unload by instance ID:

```bash
lmc unload <instance_id>
```

## Configuration

The CLI talks to the LM Studio server via its REST API:

- **Default host:** `http://localhost:1234`

If your LM Studio server runs on a different port or you want to connect remotely (e.g. via Tailscale), override the host with the `--host` flag:

```bash
lmc --host <your-lm-studio-host> list
```

Or edit the `DEFAULT_HOST` constant at the top of `lmc/api.py` to set a default for all invocations.

## Troubleshooting

### Connection Error / "Failed to list models"
Make sure LM Studio is running before using `lmc`:
- Open LM Studio and wait for it to finish loading
- Default server URL must be accessible at `http://localhost:1234` (or your configured host)

### Can't Find My Model Key?
Model keys are shown in the `lmc list` output. They look like:
- `qwen3.5-9b`
- `recursivesi/flint`
- Or any model name from HuggingFace, etc.

### Remote Connection Issues (Tailscale)
Ensure your Tailscale network is active and the remote host is reachable:
```bash
tailscale status  # verify you're connected
lmc --host <your-lm-studio-host> list
```

### Model Won't Load?
- Check available VRAM/GPU memory
- Try reducing `--context-length` if loading fails
- Some models may require more than your system can handle

## License

MIT — see [LICENSE](LICENSE).
