"""CLI for managing models on a local LM Studio server."""

from typing import Optional

import typer
from httpx import ConnectError as HttpConnectionError

from . import api


def handle_http_error(
    ctx: typer.Context,
    response: object,  # httpx.Response
    command_name: str,
    entity_key: Optional[str] = None,
) -> None:
    """Convert raw HTTP error into a user-friendly message."""
    status = response.status_code

    # Common LM Studio API errors with friendly descriptions
    entity_label = f"model '{entity_key}'" if entity_key else "resource"

    error_messages = {
        401: "❌ Unauthorized: The LM Studio server is either not running or the host is unreachable.",
        403: "❌ Forbidden: You don't have permission to perform this operation on the server.",
        404: f"❌ Not Found ({command_name.title()}): {entity_label} was not found on the server.",
        429: "⚠️  Rate Limited (429): The LM Studio API has rate-limited your request. Please wait a moment and try again.",
        500: "❌ Server Error (500): Something went wrong on the LM Studio server side.",
        503: "⚠️  Unavailable (503): The LM Studio server is currently unavailable or restarting. Please wait a moment and try again.",
    }

    message = error_messages.get(status, f"❌ HTTP Error {status} from the LM Studio server.")
    typer.echo(message)

    # Show raw JSON response if available
    try:
        body = response.json()
        if isinstance(body, dict):
            typer.echo(f"\n   Server response: {body}")
    except Exception:
        pass


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
        # Wrap in request so we can inspect HTTP errors before raising
        response = client.get("/api/v1/models")

        if response.status_code != 200:
            handle_http_error(
                ctx,
                response,
                command_name="list models",
            )
            return

        models = response.json().get("models", [])

    except HttpConnectionError as e:
        typer.echo(f"❌ Cannot connect to LM Studio server at '{ctx.obj}'.")
        typer.echo(f"   Is the LM Studio server running? Check the default host setting.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to list models: {e}")
        raise typer.Exit(1)

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
        result = api.load_model(
            client=client,
            model=model,
            context_length=context_length,
            flash_attention=flash_attention,
            offload_kv_cache_to_gpu=offload_kv_cache_to_gpu,
        )

        # Check if we got a successful response
        if result["response"].status_code != 200:
            handle_http_error(
                ctx,
                result["response"],
                command_name="load model",
                entity_key=model,
            )
            return

        data = result["data"]
        instance_id = data.get("instance_id", "?") if data else "?"
        typer.echo(f"✅ Loaded '{model}'")
        typer.echo(f"   instance_id: {instance_id}")

        load_time = data.get("load_time_seconds") if data else None
        if load_time is not None:
            typer.echo(f"   load_time_seconds: {load_time:.2f}s")

    except HttpConnectionError as e:
        typer.echo(f"❌ Cannot connect to LM Studio server at '{ctx.obj}'.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to load model '{model}': {e}")
        raise typer.Exit(1)


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

        if result["response"].status_code != 200:
            handle_http_error(
                ctx,
                result["response"],
                command_name="unload model",
                entity_key=model_key,
            )
            return

        unloaded_id = result["data"].get("instance_id") or instance_id
        typer.echo(f"✅ Unloaded '{unloaded_id}'")

    except HttpConnectionError as e:
        typer.echo(f"❌ Cannot connect to LM Studio server at '{ctx.obj}'.")
        raise typer.Exit(1)
    except RuntimeError as e:
        typer.echo(str(e))  # Already user-friendly from api.resolve_loaded_instance()
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to unload model '{model_key}': {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
