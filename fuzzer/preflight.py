"""Check the endpoint before spending a fuzzing budget."""
from urllib.parse import urlsplit

import httpx


async def check_server(base_url: str, model: str = "") -> dict:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an http(s) server origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, query parameters or fragments")
    if parsed.path.rstrip("/"):
        raise ValueError("use the server origin, without /v1 or another path")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(base_url.rstrip("/") + "/v1/models")
            response.raise_for_status()
            models = [entry["id"] for entry in response.json()["data"]]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot discover models at {base_url}: {exc}") from exc
    if not models:
        raise ValueError("server returned no models")
    if model and model not in models:
        raise ValueError(f"model {model!r} is not served; available: {', '.join(models)}")
    return {"model": model or models[0], "models": models}
