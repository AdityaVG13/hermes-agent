"""Auxiliary ``api_key`` routes honor a registry profile's *declared* ``api_mode``.

``ProviderProfile.api_mode_for_model`` documents ``None`` as "keep the static
``api_mode``", and ``provider_route_hooks.model_api_mode`` only falls back to
the declared mode on *exception*. ``_resolve_api_key_branch`` gated its
Responses wrap on ``model_api_mode(profile, model) == "codex_responses"``
though, so a plugin profile that declares ``api_mode="codex_responses"``
without a per-model hook answered ``None`` at that gate and every auxiliary
call for the provider fell back to a plain chat client — the main agent and
``runtime_provider`` already resolve the effective wire as
``model_api_mode(...) or declared`` (runtime_provider.py). The muse-code
subscription profile is the live case: its loopback adapter only speaks the
Responses dialect upstream, so each vision/compression auxiliary call 400'd on
``unknown parameter include``.

These tests pin the contract through the real resolution entry point: a
declared ``codex_responses`` wraps the auxiliary client, a per-model hook
still overrides the declaration, and a chat declaration stays chat.
"""

from __future__ import annotations

import pytest

import providers as _providers
from providers.base import ProviderProfile

from agent.auxiliary_client import CodexAuxiliaryClient

_PROBE_ENV_VAR = "AUX_DECLARED_MODE_PROBE_AUTH"
_PROBE_KEY = "probe-sentinel"


class _StubOpenAI:
    """Stands in for ``openai.OpenAI`` so resolution never touches a real transport.

    Building the real client walks install-manifest readers that the home I/O
    guard (rightly) refuses against a live production home on a dev machine.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")


class _DeclaredResponsesProfile(ProviderProfile):
    """The plugin shape under test: declares the wire, no per-model hook."""


class _HookDowngradesToChatProfile(ProviderProfile):
    """A per-model hook must keep overriding the static declaration."""

    def api_mode_for_model(self, model: str, *, api_key: str | None = None) -> str | None:
        return "chat_completions"


def _probe_profile(cls, name: str, api_mode: str | None) -> ProviderProfile:
    return cls(
        name=name,
        auth_type="api_key",
        env_vars=(_PROBE_ENV_VAR,),
        base_url=f"https://{name}.invalid",
        default_aux_model="probe-model",
        api_mode=api_mode,
    )


@pytest.fixture
def registered(monkeypatch):
    """Register provider profiles for one test on copies of both registries.

    Mirrors the import-time synthesis ``hermes_cli.auth`` performs for plugin ``api_key``
    profiles, which is what routes them into ``_resolve_api_key_branch`` in the first place.
    """
    import agent.auxiliary_client as _ac
    import hermes_cli.auth as _auth
    from hermes_cli.auth_plugin_providers import register_plugin_provider
    from agent import secret_scope as _secret_scope

    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(_auth, "PROVIDER_REGISTRY", dict(_auth.PROVIDER_REGISTRY))
    # Building a real OpenAI client composes the pm runtime env, which reads the
    # machine's live install manifest — the home I/O guard (rightly) refuses that
    # against a production home. A stub answers the wrap assertions just as well.
    monkeypatch.setattr(_ac, "_create_openai_client",
                        lambda *, api_key, base_url, **kw: _StubOpenAI(api_key=api_key, base_url=base_url, **kw))
    scope_token = _secret_scope.set_secret_scope({_PROBE_ENV_VAR: _PROBE_KEY})

    def _register(profile: ProviderProfile) -> None:
        _providers.register_provider(profile)
        register_plugin_provider(profile)

    yield _register
    _secret_scope.reset_secret_scope(scope_token)


def test_declared_codex_responses_wraps_the_auxiliary_client(registered):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_DeclaredResponsesProfile, "aux-declared-responses", "codex_responses"))
    client, model = resolve_provider_client("aux-declared-responses", "probe-model")

    # None from api_mode_for_model means "keep the static api_mode" (providers.base),
    # so the auxiliary route must speak the declared Responses wire.
    assert isinstance(client, CodexAuxiliaryClient)
    assert model == "probe-model"


def test_per_model_hook_still_overrides_the_declaration(registered):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_HookDowngradesToChatProfile, "aux-hook-downgrades", "codex_responses"))
    client, _model = resolve_provider_client("aux-hook-downgrades", "probe-model")

    assert not isinstance(client, CodexAuxiliaryClient)
    assert isinstance(client, _StubOpenAI)


def test_chat_declaration_stays_chat(registered):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_DeclaredResponsesProfile, "aux-declared-chat", "chat_completions"))
    client, _model = resolve_provider_client("aux-declared-chat", "probe-model")

    assert not isinstance(client, CodexAuxiliaryClient)
    assert isinstance(client, _StubOpenAI)
