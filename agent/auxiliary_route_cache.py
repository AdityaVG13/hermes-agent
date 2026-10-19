"""Resolve named auxiliary bindings before lookup, not after a stale cache hit."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def named_route_binding(provider: str, base_url: str | None, api_key: Any):
    from agent import auxiliary_client as aux
    from hermes_cli.runtime_provider import _get_named_custom_provider

    original = (provider or '').strip().lower()
    normalized = aux._normalize_aux_provider(provider)
    if normalized in aux._EXPLICIT_PROVIDER_BRANCHES and original.removeprefix('custom:') not in aux._LOCAL_SERVER_ALIASES:
        return None
    entry = _get_named_custom_provider(original)
    if entry is None and normalized != original:
        entry = _get_named_custom_provider(normalized)
    if entry is None:
        return None
    endpoint = base_url or entry.get('base_url') or ''
    key = aux._normalize_api_key(api_key) or aux._named_custom_api_key(entry, normalized, endpoint)
    return entry, key


def named_binding_cache_key(binding):
    if binding is None:
        return ()
    from agent.auxiliary_client import _runtime_cache_discriminator

    entry, key = binding
    # The entire selected entry includes protocol, endpoint, headers and key-source pointers.
    # Hash it: header values, URL queries and inline keys must never appear in cache diagnostics.
    descriptor = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(',', ':')).encode()).digest()
    # The descriptor already binds the command. Do not expose command arguments
    # (which may contain credentials), or bind the newly created callable's id.
    credential = ('key_cmd',) if entry.get('key_cmd') and callable(key) else _runtime_cache_discriminator('api_key', key)
    return descriptor, credential
