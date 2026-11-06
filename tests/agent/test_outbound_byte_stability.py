"""Outbound payload byte-stability for a fixed transcript (the ``rebuilt()`` law).

Mainplane's harness refuses to continue a session when a new build compiles the same
records into different request bytes (``rebuilt()`` in pkg/harness/harness.go): byte
drift is a session-ending event, not a cache tax to absorb. Hermes sends through
``sanitize_api_messages``, an unconditional deterministic pipeline; this pins the law
that the SAME transcript must sanitize to the SAME bytes on every build — no
wall-clock, no counters, no ordering instability, no input mutation.

A regression here silently re-prices the whole prompt prefix on every request (the
provider sees a conversation it never saw), which is exactly the failure this guard
exists to catch at commit time.
"""

from __future__ import annotations

import copy
import json

from agent.agent_runtime_helpers import sanitize_api_messages


def _fixed_transcript() -> list[dict]:
    """A transcript exercising every sanitize pass: alias ids, an orphan call, a
    mispositioned result, duplicate ids, and an empty non-final assistant row."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "read config and search for x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "config.yaml"}'}},
                {"id": "call_b", "type": "function", "function": {"name": "search_files", "arguments": '{"pattern": "x"}'}},
                # call_c is never answered positionally -> deterministic stub below
                {"id": "call_c", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "date"}'}},
            ],
        },
        # answer for call_a under the `call_id` alias spelling
        {"role": "tool", "name": "read_file", "call_id": "call_a", "content": '{"content": "k: v"}'},
        {"role": "assistant", "content": ""},
        # a result that arrives AFTER the next assistant row (positional orphan)
        {"role": "tool", "name": "search_files", "tool_call_id": "call_b", "content": '{"matches": []}'},
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                {"id": "call_d", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
                # duplicate id: second declaration must be deduped, not duplicated
                {"id": "call_d", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
            ],
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "call_d", "content": "a"},
        {"role": "user", "content": "thanks"},
    ]


def test_same_transcript_sanitizes_to_identical_bytes():
    first = sanitize_api_messages(copy.deepcopy(_fixed_transcript()))
    second = sanitize_api_messages(copy.deepcopy(_fixed_transcript()))
    # Canonical-form equality catches value drift; plain-dump equality also pins
    # list ordering (semantic for the API) and key insertion order drift.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(first) == json.dumps(second)


def test_orphan_stub_content_is_a_fixed_sentinel():
    """The injected answer for a positionally-unanswered call must be byte-fixed:
    a counter, timestamp or call-position in the stub would re-price the prefix."""
    out = sanitize_api_messages(_fixed_transcript())
    stubs = [
        m for m in out
        if m.get("role") == "tool" and "Result unavailable" in str(m.get("content", ""))
    ]
    assert stubs, "fixture expects at least one injected orphan stub"
    for m in stubs:
        assert m["content"] == "[Result unavailable — see context summary above]"


def test_sanitize_does_not_mutate_the_input_transcript():
    """The pipeline must not write through to the caller's rows: shared-state
    mutation would make the second build's input differ from the first's."""
    transcript = _fixed_transcript()
    before = json.dumps(transcript, sort_keys=True)
    sanitize_api_messages(transcript)
    assert json.dumps(transcript, sort_keys=True) == before
