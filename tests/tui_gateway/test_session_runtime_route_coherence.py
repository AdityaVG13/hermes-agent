"""A live Desktop model pick must replace every persisted route identity."""
import json
from types import SimpleNamespace


def test_live_runtime_replaces_stale_gateway_route():
    from hermes_state import SessionDB
    from tui_gateway.server import _runtime_model_config, _stored_session_runtime_overrides

    agent = SimpleNamespace(model='picked-model', provider='openai-codex', base_url='https://chatgpt.com/backend-api/codex',
                            api_mode='codex_responses', reasoning_config={'enabled': True, 'effort': 'max'},
                            service_tier=None)
    old = {'gateway_runtime': {'provider': 'anthropic', 'base_url': 'https://api.anthropic.com',
                               'api_mode': 'anthropic_messages', 'fallback_active': True},
           'browser_model_lock': {'confirmed': True}}
    config = _runtime_model_config(agent, old)
    row = {'model': agent.model, 'model_config': json.dumps(config)}
    route = SessionDB.session_gateway_runtime(row)
    assert (route['provider'], route['base_url'], route['api_mode']) == (
        agent.provider, agent.base_url, agent.api_mode)
    assert not route.get('fallback_active')
    assert config['browser_model_lock'] == old['browser_model_lock']
    assert old['gateway_runtime']['provider'] == 'anthropic', 'do not mutate the caller snapshot'
    assert _stored_session_runtime_overrides(row)['model_override']['provider'] == agent.provider
