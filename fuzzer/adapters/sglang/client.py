"""Pure client-side helpers for talking to SGLang's HTTP API.

Used by both the legacy ``SglangBaseAdapter`` (``sglang/legacy.py``) and the
new-shape ``Adapter`` (``sglang/adapter.py``). All functions in this module
are pure — they take inputs (an Event, a base URL, etc.) and return outputs
without holding instance state.

Stateful concerns (e.g. the adapter's cached model name) stay on the adapter
class itself; this module just provides the underlying HTTP / payload-build
primitives the class delegates to.

Extracted in Phase 2 of the adapter / strategy split refactor.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Optional


# ---------------------------------------------------------------------------
# Unicode stress pools
# ---------------------------------------------------------------------------

UNICODE_POOLS: dict = {
    "cjk":       ["你好", "世界", "深度学习", "模型", "推理", "令牌"],
    "emoji":     ["🔥", "💡", "🤖", "✨", "🧠", "⚡", "🌍"],
    "combining": ["á", "ë", "ñ", "ô"],
    "rtl_mixed": ["שלום", "hello", "مرحبا", "world"],
}


def interleave_unicode(tokens: list, stress_type: str) -> list:
    """Interleave unicode tokens from the pool every 3rd position."""
    pool = UNICODE_POOLS.get(stress_type, [])
    if not pool:
        return tokens
    result = []
    pool_idx = 0
    for i, tok in enumerate(tokens):
        result.append(tok)
        if (i + 1) % 3 == 0:
            result.append(pool[pool_idx % len(pool)])
            pool_idx += 1
    return result


def stable_hash(s: str) -> int:
    """MD5-based stable hash truncated to 4 decimal digits."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 10_000


def perm_idx(rid_hash: int, position: int) -> int:
    """Map (rid_hash, position) to a pseudo-random but deterministic index."""
    key = f"{rid_hash}:{position}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10_000


# ---------------------------------------------------------------------------
# Payload construction (deterministic from the Event)
# ---------------------------------------------------------------------------

def build_prompt(event) -> str:
    """Build a prompt string that honours prefix_len for real KV prefix sharing.

    The first ``prefix_len`` tokens are deterministic shared tokens so requests
    with identical prefix_len share the same RadixAttention blocks. The
    remaining tokens are unique per (prompt_family_id or request_id).
    """
    total = event.prompt_len or 16
    plen  = min(event.prefix_len or 0, total)
    ulen  = total - plen
    shared   = [f"pre{i}" for i in range(plen)]
    seed_str = getattr(event, "prompt_family_id", None) or event.request_id
    rid_hash = stable_hash(seed_str)
    if getattr(event, "prompt_style", None) == "rejection_heavy":
        unique = [
            f"u{stable_hash(seed_str + ':pos' + str(i))}_{perm_idx(rid_hash, i)}"
            for i in range(ulen)
        ]
    else:
        unique = [f"u{rid_hash}_{perm_idx(rid_hash, i)}" for i in range(ulen)]
    tokens = shared + unique
    if event.unicode_stress is not None:
        tokens = interleave_unicode(tokens, event.unicode_stress)
    return " ".join(tokens)


def build_chat_messages(event) -> list:
    """Build chat messages for /v1/chat/completions with prefix-cache-aware content."""
    seed_str = getattr(event, "prompt_family_id", None) or event.request_id
    rid_hash = stable_hash(seed_str)
    result = []
    for i, msg in enumerate(event.messages or []):
        role = msg.get("role", "user")
        content_len = max(1, msg.get("content_len", 16))
        if role == "system":
            tokens = [f"sys{j}" for j in range(content_len)]
            if event.unicode_stress is not None:
                if event.unicode_stress == "rtl_mixed" or role != "system":
                    tokens = interleave_unicode(tokens, event.unicode_stress)
        elif role == "user":
            if getattr(event, "prompt_style", None) == "rejection_heavy":
                tokens = [
                    f"u{stable_hash(seed_str + ':turn' + str(i) + ':pos' + str(j))}_{perm_idx(rid_hash, j)}"
                    for j in range(content_len)
                ]
            else:
                tokens = [f"u{rid_hash}_t{i}_{j}" for j in range(content_len)]
            if event.unicode_stress is not None:
                tokens = interleave_unicode(tokens, event.unicode_stress)
        else:
            tokens = [f"ast{i}_{j}" for j in range(content_len)]
            if event.unicode_stress is not None:
                tokens = interleave_unicode(tokens, event.unicode_stress)
        result.append({"role": role, "content": " ".join(tokens)})
    return result


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def fetch_server_info_raw(base_url: str) -> Optional[dict]:
    """Fetch /server_info from an SGLang server, trying both known endpoint names.

    SGLang has used both "/server_info" (current) and "/get_server_info" (older).
    We try the current name first so the collector and version-discovery code
    always hit the same endpoint on any deployment.
    """
    import httpx
    for endpoint in ("/server_info", "/get_server_info"):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}{endpoint}", timeout=5.0)
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            continue
    return None


async def check_liveness(base_url: str) -> bool:
    """Return True if the SGLang server is healthy."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health", timeout=3.0)
            return resp.status_code == 200
    except Exception:
        return False


async def fetch_model_name(base_url: str) -> Optional[str]:
    """Discover the served model name via /v1/models. Returns None on error."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/models", timeout=5.0)
            data = resp.json()
            return data["data"][0]["id"]
    except Exception:
        return None


async def fetch_version_endpoint(base_url: str) -> Optional[str]:
    """Fetch SGLang's /version endpoint. Returns None on error."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/version", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("version")
    except Exception:
        pass
    return None


async def flush_cache_endpoint(base_url: str) -> bool:
    """Try the native /flush_cache endpoint. Returns True on 200, False otherwise."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/flush_cache",
                params={"timeout": 5.0},
                timeout=10.0,
            )
            return resp.status_code == 200
    except Exception:
        return False


async def flush_via_large_prompt(base_url: str, model_name: str) -> None:
    """Fallback flush: send a large unique prompt to evict the LRU cache."""
    import httpx
    flush_tokens = " ".join(
        f"flush{uuid.uuid4().hex[:6]}_{i}" for i in range(2048)
    )
    payload: dict = {
        "prompt":      flush_tokens,
        "max_tokens":  1,
        "temperature": 0.0,
        "stream":      False,
    }
    if model_name:
        payload["model"] = model_name
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base_url}/v1/completions",
                json=payload,
                timeout=60.0,
            )
    except Exception:
        pass
    await asyncio.sleep(1.0)


async def discover_version_chain(base_url: str) -> dict:
    """Run the SGLang version-discovery fallback chain.

    Order: /server_info → /version → importlib.metadata → sglang.__version__.
    Returns ``{"sglang_version": <str>}`` or ``{"sglang_version": "unknown"}``
    if every path fails. Never raises.
    """
    info: dict = {}

    # 1. /server_info (or /get_server_info fallback)
    data = await fetch_server_info_raw(base_url)
    if data:
        v = data.get("version") or data.get("sglang_version")
        if v:
            info["sglang_version"] = v

    # 2. /version endpoint
    if "sglang_version" not in info:
        v = await fetch_version_endpoint(base_url)
        if v:
            info["sglang_version"] = v

    # 3. importlib.metadata
    if "sglang_version" not in info:
        try:
            from importlib.metadata import version as _pkg_version
            info["sglang_version"] = _pkg_version("sglang")
        except Exception:
            pass

    # 4. sglang.__version__
    if "sglang_version" not in info:
        try:
            import sglang as _sglang
            info["sglang_version"] = getattr(_sglang, "__version__", "unknown")
        except Exception:
            info["sglang_version"] = "unknown"

    return info
