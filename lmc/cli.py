"""CLI for managing models on a local LM Studio server."""

from typing import Optional

import typer

from . import api

app = typer.Typer(help="Manage models on a local LM Studio server")


@app.callback()
def main(
    ctx: typer.Context,
    host: str = typer.Option(
        api.DEFAULT_HOST,
        "--host",
        "-H",
        help="Base URL of the LM Studio local server (e.g. http://100.x.x.x:1234 when remote over Tailscale)",
    ),
) -> None:
    """Manage models on a local LM Studio server."""
    ctx.obj = host


@app.command(name="list")
def list_models_cmd(ctx: typer.Context) -> None:
    """List models available on the server, and whether they are loaded."""
    client = api.create_client(ctx.obj)
    try:
        models = api.list_models(client)
    except Exception as e:
        raise typer.Exit(f"Failed to list models: {e}")

    if not models:
        typer.echo("No models found.")
        return

    typer.echo()

    header = f"{'MODEL':<45} {'QUANT':<14} {'STATUS':<12} {'SIZE (GB)':>10} {'CONTEXT':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for model in models:
        key = model.get("key", "?")
        quant = model.get("quantization") or {}
        quant_str = quant.get("name") or "?"
        status = "loaded" if model.get("loaded_instances") else "not loaded"
        size_bytes = model.get("size_bytes")
        size = f"{size_bytes / 1e9:.2f}" if size_bytes else "?"
        context_length = None
        instances = model.get("loaded_instances") or []
        if instances:
            context_length = (instances[0].get("config") or {}).get("context_length")
        if context_length is None:
            context_length = model.get("max_context_length")
        context = str(context_length) if context_length is not None else "?"
        key_str = f"{key:<45}"
        if status == "loaded":
            key_str = f"\033[92m{key_str}\033[0m"
        typer.echo(f"{key_str} {quant_str:<14} {status:<12} {size:>10} {context:>8}")
    typer.echo()


@app.command(name="load")
def load_cmd(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model key to load (e.g. qwen3.5-9b or recursivesi/flint)"),
    context_length: Optional[int] = typer.Option(None, "--context-length", help="Maximum context length in tokens"),
    flash_attention: bool = typer.Option(False, "--flash-attention", is_flag=True, help="Enable flash attention"),
    offload_kv_cache_to_gpu: bool = typer.Option(False, "--offload-kv-cache-to-gpu", is_flag=True, help="Offload KV cache to GPU memory"),
) -> None:
    """Load a model into memory on the LM Studio server."""
    client = api.create_client(ctx.obj)
    try:
        result = api.load_model(client, model, context_length=context_length, flash_attention=flash_attention, offload_kv_cache_to_gpu=offload_kv_cache_to_gpu)
    except Exception as e:
        raise typer.Exit(f"Failed to load model: {e}")

    instance_id = result.get("instance_id", "?")
    typer.echo(f"Loaded '{model}'")
    typer.echo(f"  instance_id: {instance_id}")
    load_time = result.get("load_time_seconds")
    if load_time is not None:
        typer.echo(f"  load_time_seconds: {load_time:.2f}")


@app.command(name="unload")
def unload_cmd(
    ctx: typer.Context,
    model_key: str = typer.Argument(..., help="Model key or instance ID to unload"),
) -> None:
    """Unload a model from memory on the LM Studio server."""
    client = api.create_client(ctx.obj)
    try:
        instance_id = api.resolve_loaded_instance(client, model_key)
        result = api.unload_model(client, instance_id)
    except Exception as e:
        raise typer.Exit(f"Failed to unload model: {e}")

    unloaded_id = result.get("instance_id", instance_id)
    typer.echo(f"Unloaded '{unloaded_id}'")


if __name__ == "__main__":
    app()
