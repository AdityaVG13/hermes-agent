"""A remembered env-key slot is not an authenticated picker route when its secret is absent."""
import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli.model_switch_providers import _credential_pool_is_usable


@pytest.mark.parametrize("access_token,refresh_token,visible", [
    ("", None, False),
    ("present-test-token", None, True),
    ("", "refreshable-test-grant", True),
])
def test_picker_requires_auth_material_in_a_structured_pool(monkeypatch, access_token, refresh_token, visible):
    entry = PooledCredential.from_dict("deepseek", {
        "id": "remembered-env-slot", "source": "env:DEEPSEEK_API_KEY",
        "access_token": access_token, "refresh_token": refresh_token,
        "last_status": "exhausted",
    })
    pool = CredentialPool("deepseek", [entry])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: pool)
    assert _credential_pool_is_usable("deepseek", for_picker=True) is visible
