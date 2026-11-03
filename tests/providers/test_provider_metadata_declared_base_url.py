"""Metadata readers must not invoke a profile's live ``base_url`` accessor.

A profile may implement ``base_url`` as a lazily-started adapter accessor (muse-code binds a
loopback port on read and waits out a port pin). Registry rows and host maps are metadata: they
run in every process that imports the terminal tooling, so touching the live accessor there
costs the adapter start on paths that never issue a request.
"""
from hermes_cli.auth_plugin_providers import _profile_row
from providers.base import ProviderProfile


class _SideEffectProfile(ProviderProfile):
    """``base_url`` raises: any metadata read through it is a bug."""

    name = "side-effect"
    display_name = "Side Effect"
    auth_type = "api_key"
    env_vars = ("SIDE_EFFECT_API_KEY",)

    @property
    def base_url(self) -> str:  # type: ignore[override]
        raise AssertionError("metadata read the live base_url accessor")

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._constructed_base_url = str(value or "")

    def declared_base_url(self) -> str:
        return "https://api.example.test/v1"


def test_profile_row_uses_declared_base_url():
    row = _profile_row(_SideEffectProfile(name="side-effect", env_vars=("SIDE_EFFECT_API_KEY",)))

    assert row is not None, "an api-key profile with env vars must still produce a registry row"
    assert row.inference_base_url == "https://api.example.test/v1"


def test_get_hostname_uses_declared_base_url():
    assert _SideEffectProfile(name="side-effect", env_vars=("SIDE_EFFECT_API_KEY",)).get_hostname() == "api.example.test"


def test_plain_profile_still_reads_its_attribute():
    plain = ProviderProfile(name="plain", base_url="https://plain.example.test/v1")

    assert plain.declared_base_url() == "https://plain.example.test/v1"
    assert plain.get_hostname() == "plain.example.test"
