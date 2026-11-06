"""The context gauge: every committed tool result carries the provider-priced size
of the request that spawned it (the usage anchor's prompt+completion tokens),
stamped once at construction — append-only thereafter (the ``rebuilt()`` byte law).

Mainplane's harness stamps ``context 48210`` on every tool result so the model
sees its own budget climb and can act before trimming instead of discovering
silent eviction by failure. Hermes takes the number from the usage anchor: the
provider's real usage for THIS turn's request. No anchor (endpoint omitted
usage): no footer — an estimate would lie. The footer sits outside any
untrusted-data wrapper, like plugin notes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.agent.test_tool_call_incremental_persistence import _make_agent, _mock_tool_call

_ANCHOR = {
    "prompt_tokens": 40_000,
    "completion_tokens": 210,
    "base_count": 3,
    "base_last_role": "assistant",
    "base_last_fp": "f",
    "base_prefix_fp": "p",
}


def _run_one_tool(agent, *, name="read_file", function_result='{"ok": true}', call_id="call_gauge"):
    tool_calls = [_mock_tool_call(name=name, call_id=call_id)]
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock()
    with patch("model_tools.handle_function_call", return_value=function_result):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=tool_calls), messages, "task-1")
    return [m for m in messages if m.get("role") == "tool"]


def test_result_carries_anchored_context_footer():
    agent = _make_agent()
    agent._usage_anchor = dict(_ANCHOR)  # type: ignore[assignment]  # Optional[dict] at runtime (set_usage_anchor)
    [tool] = _run_one_tool(agent)
    assert tool["content"].endswith("[context 40210 tokens]")


def test_no_anchor_means_no_footer():
    agent = _make_agent()
    agent._usage_anchor = None
    [tool] = _run_one_tool(agent)
    assert "[context" not in tool["content"]


def test_zero_usage_anchor_means_no_footer():
    agent = _make_agent()
    agent._usage_anchor = {**_ANCHOR, "prompt_tokens": 0, "completion_tokens": 0}  # type: ignore[assignment]
    [tool] = _run_one_tool(agent)
    assert "[context" not in tool["content"]


def test_footer_sits_outside_the_untrusted_wrapper():
    agent = _make_agent()
    agent._usage_anchor = dict(_ANCHOR)  # type: ignore[assignment]
    # Long enough to trip _UNTRUSTED_WRAP_MIN_CHARS for web_search.
    [tool] = _run_one_tool(agent, name="web_search", function_result="x" * 80)
    content = tool["content"]
    assert "</untrusted_tool_result>" in content
    assert content.endswith("</untrusted_tool_result>\n[context 40210 tokens]")
