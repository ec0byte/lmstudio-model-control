"""HTTP client for the LM Studio local server REST API."""

from typing import Dict, Any, Optional, List

import httpx


DEFAULT_HOST = "http://localhost:1234"


def create_client(base_url: str = DEFAULT_HOST) -> httpx.Client:
    """Create an httpx client pointed at the LM Studio server."""
    return httpx.Client(base_url=base_url, timeout=30.0)


def list_models(client: httpx.Client) -> List[Dict[str, Any]]:
    """GET /api/v1/models - list available and loaded models."""
    response = client.get("/api/v1/models")
    response.raise_for_status()
    return response.json().get("models", [])


def load_model(
    client: httpx.Client,
    model: str,
    context_length: Optional[int] = None,
    flash_attention: bool = False,
    offload_kv_cache_to_gpu: bool = False,
) -> Dict[str, Any]:
    """POST /api/v1/models/load - load a model into memory."""
    body: dict = {"model": model}
    if context_length is not None:
        body["context_length"] = context_length
    if flash_attention:
        body["flash_attention"] = True
    if offload_kv_cache_to_gpu:
        body["offload_kv_cache_to_gpu"] = True

    response = client.post("/api/v1/models/load", json=body)
    return {
        "response": response,
        "data": response.json() if response.status_code < 400 else None,
    }


def resolve_loaded_instance(client: httpx.Client, model_key: str) -> str:
    """Find the loaded instance id matching a model key (or instance id)."""
    models = list_models(client)
    for model in models:
        key = model.get("key")
        instances = model.get("loaded_instances") or []
        for instance in instances:
            if instance.get("id") == model_key or key == model_key:
                return instance["id"]
    raise RuntimeError(f"No loaded instance found matching '{model_key}'")


def unload_model(client: httpx.Client, instance_id: str) -> Dict[str, Any]:
    """POST /api/v1/models/unload - unload a model instance from memory."""
    response = client.post("/api/v1/models/unload", json={"instance_id": instance_id})
    return {
        "response": response,
        "data": response.json() if response.status_code < 400 else None,
    }
