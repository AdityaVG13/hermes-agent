"""Canonical model receipts must not leave an obsolete provider in resume metadata."""
import json
import sqlite3

import pytest


@pytest.mark.parametrize('selection', [
    {'provider': 'picked-provider', 'base_url': 'http://127.0.0.2:9/v1', 'api_mode': 'responses'},
    {'provider': 'picked-provider'},
])
def test_model_receipt_replaces_stale_resume_route(monkeypatch, selection):
    import hermes_state_mutation_prepared as prepared
    from hermes_state import SessionDB

    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE sessions(id TEXT PRIMARY KEY,model TEXT,model_config TEXT)')
    old = {'model': 'old-model', 'provider': 'old-provider', 'base_url': 'http://127.0.0.1:9/v1',
           'api_mode': 'old-wire', 'gateway_runtime': {'provider': 'old-provider',
           'base_url': 'http://127.0.0.1:9/v1', 'api_mode': 'old-wire', 'fallback_active': True},
           'browser_model_lock': {'confirmed': True}}
    conn.execute('INSERT INTO sessions VALUES(?,?,?)', ('sid', 'old-model', json.dumps(old)))
    monkeypatch.setattr(prepared, '_commit_policy', lambda *args: ({'target': 'sid'}, 2))
    policy = {'model': 'new-model', 'config_json': json.dumps({'model': selection})}
    try:
        prepared.model_in_transaction(None, conn, 'sid', {}, {'policy': policy})
        model, raw = conn.execute('SELECT model,model_config FROM sessions').fetchone()
        config = json.loads(raw)
        route = SessionDB.session_gateway_runtime({'model': model, 'model_config': raw})
        assert route['provider'] == selection['provider']
        assert (route.get('base_url') or '') == selection.get('base_url', '')
        assert (route.get('api_mode') or '') == selection.get('api_mode', '')
        assert not route.get('fallback_active')
        assert config['model'] == model == 'new-model'
        assert config['browser_model_lock'] == old['browser_model_lock']
    finally:
        conn.close()
