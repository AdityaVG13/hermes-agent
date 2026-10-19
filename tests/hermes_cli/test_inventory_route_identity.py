"""Picker route tokens separate identical names without exposing credential material."""
from copy import deepcopy

from hermes_cli.inventory import _apply_capabilities


def test_inventory_route_identity_uses_owner_endpoint_protocol_and_credential_scope(monkeypatch, tmp_path):
    monkeypatch.setattr('agent.models_dev.get_model_capabilities', lambda *a, **k: None)
    monkeypatch.setattr('hermes_cli.inventory._reasoning_catalog_reader', lambda _slug: None)
    config = {'providers': {'lab': {'base_url': 'https://endpoint-a.invalid/v1',
                                  'api_mode': 'chat_completions', 'key_env': 'LAB_A'}}}

    def identify(cfg, slug='custom:lab'):
        rows = [{'slug': slug, 'models': ['shared-model']}]
        _apply_capabilities(rows, metadata_config=cfg)
        return rows[0]['capabilities']['shared-model']['route_id']

    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'a'))
    original = identify(config)
    assert original and 'endpoint-a' not in original and 'LAB_A' not in original
    assert identify(config, 'lab') == original
    for field, value in [('base_url', 'https://endpoint-b.invalid/v1'),
                         ('api_mode', 'codex_responses'), ('key_env', 'LAB_B')]:
        changed = deepcopy(config)
        changed['providers']['lab'][field] = value
        assert identify(changed) != original
    unrelated = deepcopy(config)
    unrelated['providers']['other'] = {'base_url': 'https://other.invalid/v1'}
    assert identify(unrelated) == original
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'b'))
    assert identify(config) != original


def test_picker_effort_mapping_uses_the_selected_transport(monkeypatch):
    monkeypatch.setattr('agent.models_dev.get_model_capabilities', lambda *a, **k: None)
    rows = [{'slug': 'openai-codex', 'models': ['gpt-6.1-sol']}]
    config = {'model': {'provider': 'openai-codex', 'api_mode': 'codex_app_server'}}
    _apply_capabilities(rows, metadata_config=config)
    assert rows[0]['capabilities']['gpt-6.1-sol']['effort_map']['ultra'] == 'ultra'

    config['model']['api_mode'] = 'codex_responses'
    _apply_capabilities(rows, metadata_config=config)
    assert rows[0]['capabilities']['gpt-6.1-sol']['effort_map']['ultra'] == 'max'


def test_same_name_capabilities_stay_in_the_owner_snapshot_under_multiplex(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    bindings = []
    for owner, reasoning in [('a', True), ('b', False), ('a', True)]:
        config = {
            'providers': {'lab': {'base_url': f'https://{owner}.invalid/v1'}},
            'model_overrides': {'custom:lab': {'shared-model': {'supports_reasoning': reasoning}}},
        }
        token = set_hermes_home_override(tmp_path / owner)
        try:
            rows = [{'slug': 'custom:lab', 'models': ['shared-model']}]
            _apply_capabilities(rows, metadata_config=config)
            caps = rows[0]['capabilities']['shared-model']
            assert caps['reasoning'] is reasoning
            bindings.append(caps['route_id'])
        finally:
            reset_hermes_home_override(token)
    assert bindings[0] == bindings[2] and bindings[0] != bindings[1]
