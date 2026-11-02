"""Transport recovery must heal a loopback endpoint that moved after the session froze it.

Session runtime state freezes the resolved base URL; a plugin provider whose loopback
adapter rebinds (restart overlap) strands the session on a dead port: every retry
rebuilds the client against the same dead URL. Recovery must adopt the registry's
current loopback endpoint — and must never touch remote URLs, remote registry
defaults standing in for a user's local override, or static providers.
"""
from types import SimpleNamespace
from typing import Any

import agent.agent_runtime_helpers as arh
from agent.custom_entry_pool import SNAPSHOT_OWNER_KEY, runtime_entry_owner

STALE = "http://127.0.0.1:43220/v1"
LIVE = "http://127.0.0.1:43221/v1"
REMOTE = "https://api.example.com/v1"


class APIConnectionError(Exception):
    """Named to match the transient-transport classification (type name check)."""


def _snapshot(base_url: str) -> dict[str, Any]:
    return {
        "model": "muse-spark-1.3-contributor",
        "provider": "muse-code",
        "requested_provider": "muse-code",
        SNAPSHOT_OWNER_KEY: runtime_entry_owner("muse-code", "muse-code", base_url),
        "base_url": base_url,
        "api_mode": "codex_responses",
        "api_key": "test-key",
        "request_overrides": {},
        "client_kwargs": {"api_key": "test-key", "base_url": base_url},
        "reasoning_echo_flag": False,
    }


def _make_agent(base_url: str) -> SimpleNamespace:
    agent = SimpleNamespace(
        provider="muse-code",
        requested_provider="muse-code",
        model="muse-spark-1.3-contributor",
        base_url=base_url,
        api_key="test-key",
        api_mode="codex_responses",
        request_overrides={},
        _reasoning_echo_flag=False,
        _transport_cache={},
        _client_kwargs={"api_key": "test-key", "base_url": base_url},
        _primary_runtime=_snapshot(base_url),
        client=object(),
        _fallback_activated=False,
        log_prefix="[test] ",
    )
    agent._retire_shared_openai_client = lambda *a, **k: None
    agent._vprint = lambda *a, **k: None
    agent._is_openrouter_url = lambda: False
    agent.created: dict[str, Any] = {}  # type: ignore[annotation]
    agent._create_openai_client = (
        lambda kwargs, reason=None, shared=False: agent.created.update(kwargs, _reason=reason) or object()
    )
    return agent


def _run_recovery(agent: SimpleNamespace, *, registry_url: str, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "hermes_cli.providers.get_provider",
        lambda name, **k: SimpleNamespace(base_url=registry_url),
    )
    arh.try_recover_primary_transport(agent, APIConnectionError("connection refused"), retry_count=0, max_retries=3)


def test_a_drifted_loopback_endpoint_is_adopted_before_the_client_rebuild(monkeypatch):
    agent = _make_agent(STALE)
    _run_recovery(agent, registry_url=LIVE, monkeypatch=monkeypatch)

    assert agent.base_url == LIVE, "recovery rebuilt the client against the dead frozen port"
    assert agent._client_kwargs["base_url"] == LIVE
    assert agent.created["base_url"] == LIVE, "the rebuilt client must dial the live endpoint"
    # The stored snapshot must stay healed so turn-start restores do not resurrect the dead URL.
    assert agent._primary_runtime["base_url"] == LIVE
    assert agent._primary_runtime["client_kwargs"]["base_url"] == LIVE
    assert agent._primary_runtime[SNAPSHOT_OWNER_KEY] == runtime_entry_owner(
        "muse-code", "muse-code", LIVE
    )


def test_a_static_provider_freezing_its_own_registry_url_is_a_noop(monkeypatch):
    agent = _make_agent(LIVE)
    _run_recovery(agent, registry_url=LIVE, monkeypatch=monkeypatch)

    assert agent.base_url == LIVE
    assert agent._primary_runtime[SNAPSHOT_OWNER_KEY] == runtime_entry_owner(
        "muse-code", "muse-code", LIVE
    )


def test_a_frozen_remote_url_is_never_rewritten(monkeypatch):
    agent = _make_agent(REMOTE)
    _run_recovery(agent, registry_url="https://api.example.com/v2", monkeypatch=monkeypatch)

    assert agent.base_url == REMOTE, "remote endpoints are outside the loopback healing rule"
    assert agent._primary_runtime["base_url"] == REMOTE


def test_a_loopback_url_is_not_healed_onto_a_remote_registry_default(monkeypatch):
    # A user may point a provider at a local proxy via config the registry cannot see
    # (the snapshot doctrine forbids live config lookups). The registry's remote default
    # must never replace it.
    agent = _make_agent(STALE)
    _run_recovery(agent, registry_url=REMOTE, monkeypatch=monkeypatch)

    assert agent.base_url == STALE, "a remote registry default must not hijack a loopback route"
    assert agent._primary_runtime["base_url"] == STALE
