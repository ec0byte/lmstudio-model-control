# Changelog

## [0.1.0] - 2026-10-26

### Added

- Initial release of `lmc`, a CLI to load/unload models on a local [LM Studio](https://lmstudio.ai) server.
- `lmc list` — list all models available on the server, and whether they are currently loaded, with quantization, size, and context info. Loaded models are highlighted.
- `lmc load <model_key>` — load a model into memory on the LM Studio server, with optional `--context-length`, `--flash-attention`, and `--offload-kv-cache-to-gpu` flags.
- `lmc unload <model_key>` — unload a loaded model instance by model key or instance ID.

### Configuration

- Default LM Studio server URL: `http://localhost:1234` (defined via the `DEFAULT_HOST` constant in `lmc/api.py`).
- To use a remote server, override with `--host` / `-H`:

```bash
lmc --host <your-lm-studio-host> list
```

### Requirements

- Python 3.9+
- A running [LM Studio](https://lmstudio.ai) server (default `http://localhost:1234`)
