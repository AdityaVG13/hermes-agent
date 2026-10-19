"""Named endpoint edits must not reuse an auxiliary client built for an old route."""
import json

import pytest

import agent.auxiliary_client as aux


@pytest.mark.parametrize('change', ['endpoint', 'protocol', 'credential', 'headers'])
def test_named_route_cache_tracks_resolved_binding(tmp_path, monkeypatch, change):
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(aux, '_client_cache', {})
    # The process-wide keepalive bootstrap discovers the installed payload; it is not
    # part of route resolution. Let the real SDK own its ordinary isolated HTTP client.
    monkeypatch.setattr(aux, '_openai_http_client_kwargs', lambda _url: {})
    entry = {'base_url': 'https://endpoint-a.invalid/v1', 'api_key': 'fixture-key-a',
             'api_mode': 'chat_completions'}

    def save():
        (home / 'config.yaml').write_text(json.dumps({'providers': {'lab': entry}}))

    save()
    first, model = aux._get_cached_client('custom:lab', 'shared-model', False)
    assert model == 'shared-model'
    assert aux._get_cached_client('custom:lab', 'shared-model', False)[0] is first
    before = aux._client_cache_key('custom:lab', async_mode=False, model='shared-model')
    changes = {
        'endpoint': {'base_url': 'https://endpoint-b.invalid/v1'},
        'protocol': {'api_mode': 'codex_responses'},
        'credential': {'api_key': 'fixture-key-b'},
        'headers': {'extra_headers': {'x-route': 'scope-b'}},
    }
    entry.update(changes[change])
    save()
    second, _ = aux._get_cached_client('custom:lab', 'shared-model', False)
    assert second is not None
    after = aux._client_cache_key('custom:lab', async_mode=False, model='shared-model')
    assert second is not first
    assert before != after
    assert 'fixture-key-' not in repr((before, after))
    if change == 'protocol':
        assert isinstance(second, aux.CodexAuxiliaryClient)
    else:
        assert str(second.base_url).rstrip('/') == entry['base_url']
        assert second.api_key == entry['api_key']
        if change == 'headers':
            assert second.default_headers['x-route'] == 'scope-b'
    assert aux._get_cached_client('custom:lab', 'shared-model', False)[0] is second


def test_named_clients_are_distinct_for_multiplexed_profiles(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    monkeypatch.setattr(aux, '_client_cache', {})
    monkeypatch.setattr(aux, '_openai_http_client_kwargs', lambda _url: {})
    clients = []
    for owner in ('a', 'b', 'a'):
        home = tmp_path / owner
        home.mkdir(exist_ok=True)
        entry = {'base_url': f'https://{owner}.invalid/v1', 'api_key': f'fixture-{owner}',
                 'api_mode': 'chat_completions'}
        (home / 'config.yaml').write_text(json.dumps({'providers': {'lab': entry}}))
        token = set_hermes_home_override(home)
        try:
            client, model = aux._get_cached_client('custom:lab', 'shared-model', False)
            assert client is not None
            assert model == 'shared-model'
            assert str(client.base_url).rstrip('/') == entry['base_url']
            assert client.api_key == entry['api_key']
            clients.append(client)
        finally:
            reset_hermes_home_override(token)
    assert clients[0] is clients[2] and clients[0] is not clients[1]


def test_command_credential_source_is_stable_but_never_plaintext_in_cache_key():
    from agent.auxiliary_route_cache import named_binding_cache_key

    entry = {'key_cmd': 'printf fixture-private-a'}
    first = named_binding_cache_key((entry, lambda: 'unused'))
    assert 'fixture-private-a' not in repr(first)
    assert named_binding_cache_key((entry, lambda: 'unused')) == first
    assert named_binding_cache_key(({'key_cmd': 'printf fixture-private-b'}, lambda: 'unused')) != first
