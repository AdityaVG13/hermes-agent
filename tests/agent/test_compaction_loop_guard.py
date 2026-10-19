"""A compaction boundary drops tool results, so the loop guard must stop counting repeats of them.

Mid-turn compaction removed a skill's view; the model viewed it again, and the per-turn
identical-result guard counted the views from before the boundary toward its hard stop.
"""

from types import SimpleNamespace

from agent.conversation_compression import _reset_read_dedup_caches
from agent.tool_guardrails import LoopCapConfig, ToolCallGuardrailConfig, ToolCallGuardrailController


def test_compaction_boundary_restarts_the_identical_result_count():
    args, result = {"name": "multi-pass-bug-hunting"}, '{"success":true,"content":"steps"}'
    agent = SimpleNamespace(_tool_guardrails=ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2)))
    for _ in range(2):
        agent._tool_guardrails.after_call("skill_view", args, result, failed=False)
    assert agent._tool_guardrails.before_call("skill_view", args).action == "block"

    _reset_read_dedup_caches("task", session_id="s1", agent=agent)

    assert agent._tool_guardrails.before_call("skill_view", args).action == "allow"


def test_compaction_rearms_reads_without_forgiving_failures_or_turn_caps():
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, exact_failure_block_after=2,
        loop_caps=LoopCapConfig(max_web_searches=1)))
    agent = SimpleNamespace(_tool_guardrails=guard)
    for _ in range(2):
        guard.after_call("terminal", {"command": "broken"}, "failed", failed=True)
    assert guard.before_call("web_search", {"query": "first"}).action == "allow"
    result = "skill body " * 100
    guard.observe_call("skill_view", {"name": "s"}, result)
    assert guard.observe_call("skill_view", {"name": "s"}, result).stub is not None

    _reset_read_dedup_caches("task", session_id="s1", agent=agent)

    assert guard.observe_call("skill_view", {"name": "s"}, result).stub is None
    assert guard.before_call("terminal", {"command": "broken"}).action == "block"
    assert guard.before_call("web_search", {"query": "second"}).action == "block"
