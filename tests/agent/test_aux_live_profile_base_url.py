"""Request paths must resolve a profile's LIVE base URL, not its declared one.

``ProviderConfig.inference_base_url`` carries the declared endpoint (metadata readers must never
start a live accessor — see ``tests/providers/test_provider_metadata_declared_base_url.py``). A
profile that overrides ``declared_base_url`` is saying its ``base_url`` accessor is an adapter:
muse-code binds a loopback proxy that aliases the tool names Meta refuses for subscription keys.
Building a request client from the registry row bypasses that proxy, and every request declaring
a tool comes back 403 client_attestation_denied.
"""
import providers
from agent.auxiliary_client import _live_profile_base_url
from providers.base import ProviderProfile

_ADAPTER_URL = "http://127.0.0.1:43220/v1"
_DECLARED_URL = "https://api.example.test/v1"


class _AdapterProfile(ProviderProfile):
    """Declares a live/declared split, like muse-code's loopback adapter."""

    name = "adapter-test"
    display_name = "Adapter Test"
    auth_type = "api_key"
    env_vars = ("ADAPTER_TEST_API_KEY",)

    def declared_base_url(self) -> str:
        return _DECLARED_URL


def _adapter_profile(live_url: str = _ADAPTER_URL) -> _AdapterProfile:
    profile = _AdapterProfile(name="adapter-test")
    profile.base_url = live_url
    return profile


def test_split_profile_resolves_the_live_accessor(monkeypatch):
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: _adapter_profile())

    assert _live_profile_base_url("adapter-test") == _ADAPTER_URL


def test_split_profile_without_a_live_difference_is_ignored(monkeypatch):
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: _adapter_profile(_DECLARED_URL))

    assert _live_profile_base_url("adapter-test") == ""


def test_plain_profile_keeps_its_registry_endpoint(monkeypatch):
    plain = ProviderProfile(name="plain-test", base_url="https://plain.example.test/v1")
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: plain)

    assert _live_profile_base_url("plain-test") == ""


def test_unknown_provider_is_inert(monkeypatch):
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: None)

    assert _live_profile_base_url("no-such-provider") == ""


def test_a_raising_declared_url_never_breaks_resolution(monkeypatch):
    class _Exploding(_AdapterProfile):
        def declared_base_url(self) -> str:
            raise RuntimeError("metadata read exploded")

    profile = _Exploding(name="adapter-test")
    profile.base_url = _ADAPTER_URL
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile)

    assert _live_profile_base_url("adapter-test") == ""


# --- the delegation / main-runtime resolver (``resolve_runtime_provider``) -------------------
#
# Subagents and the main agent build their client from the runtime dict, not from
# ``resolve_provider_client``: if that dict echoes the declared endpoint, every delegated request
# that declares a tool is refused with 403 client_attestation_denied.

from hermes_cli.runtime_provider import _apply_live_profile_base_url


def _split(monkeypatch):
    monkeypatch.setattr(providers, "live_base_url_for", lambda name: _ADAPTER_URL)
    monkeypatch.setattr(providers, "declared_base_url_for", lambda name: _DECLARED_URL)


def test_runtime_uses_the_live_accessor_when_it_echoes_the_declared_row(monkeypatch):
    _split(monkeypatch)
    runtime = {"provider": "muse-code", "base_url": _DECLARED_URL}

    fixed = _apply_live_profile_base_url(runtime, "muse-code", None)

    assert fixed["base_url"] == _ADAPTER_URL
    assert runtime["base_url"] == _DECLARED_URL, "the caller's snapshot must not be mutated in place"


def test_runtime_keeps_a_configured_override(monkeypatch):
    _split(monkeypatch)
    runtime = {"provider": "muse-code", "base_url": "https://proxy.internal.example/v1"}

    assert _apply_live_profile_base_url(runtime, "muse-code", None)["base_url"] == (
        "https://proxy.internal.example/v1"
    )


def test_runtime_keeps_an_explicitly_requested_endpoint(monkeypatch):
    _split(monkeypatch)
    runtime = {"provider": "muse-code", "base_url": "https://payg.example.test/v1"}

    assert _apply_live_profile_base_url(runtime, "muse-code", "https://payg.example.test/v1")[
        "base_url"
    ] == "https://payg.example.test/v1"
