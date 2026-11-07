"""The batching nudge: a structural hint when single-call turns stack up.

2026-11-07 evidence (state.db, this install): 56% of assistant turns carry
exactly one tool call; each pays a full model round trip for an ms-scale
handler, and execute_code carried 0.82% of tool results. Prompt framing is
volitional — this nudge fires from the loop itself, riding the context-gauge
placement rule (appended once, outside untrusted wrappers, byte-stable).

Two live lessons are pinned here:
  1. Append timing differs by path: turn_tool_round appends the in-flight
     row BEFORE dispatch (walker counts it), approval-segmented turns commit
     before their row lands (walker doesn't). The caller's own tool_call_id
     arbitrates — the intermediate "fix" that unconditionally added +1
     double-counted and silenced the nudge for a full day of live use.
  2. Never nag a batch: agent._current_tool_batch_size >= 2 returns None even
     on paths whose rows lag (a 3-call batch turn once got the hint).
"""

import unittest
from types import SimpleNamespace

from agent.tool_executor import _append_gauge_to_tool_message, _batching_nudge_note, _single_call_streak


def asst(call_ids):
    """An assistant row with exactly len(call_ids) tool calls (dict wire shape)."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "name": "read_file", "arguments": "{}"} for cid in call_ids]}


def tool_row(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "{}"}


def transcript(*turn_ids, text_tail=False):
    """[asst, tool, asst, tool, ...] — the persisted session shape."""
    rows = []
    for ids in turn_ids:
        rows.append(asst(ids))
        for cid in ids:
            rows.append(tool_row(cid))
    if text_tail:
        rows.append({"role": "assistant", "content": "done", "tool_calls": None})
    return rows


class TestSingleCallStreak(unittest.TestCase):
    def test_counts_trailing_single_call_turns(self):
        streak, cur = _single_call_streak(transcript(["a"], ["b"], ["c"]), "c")
        self.assertEqual(streak, 3)
        self.assertTrue(cur)

    def test_multi_call_turn_breaks_the_run(self):
        streak, _ = _single_call_streak(transcript(["a"], ["b", "c2"], ["d"]), "d")
        self.assertEqual(streak, 1)

    def test_text_only_turn_breaks_the_run(self):
        streak, _ = _single_call_streak(transcript(["a"], ["b"], text_tail=True), "z")
        self.assertEqual(streak, 0)

    def test_user_turn_breaks_the_run(self):
        msgs = transcript(["a"], ["b"]) + [{"role": "user", "content": "hi"}]
        streak, _ = _single_call_streak(msgs, "z")
        self.assertEqual(streak, 0)

    def test_empty_transcript(self):
        self.assertEqual(_single_call_streak([], "z"), (0, False))

    def test_object_shaped_assistant_rows_count_too(self):
        """The in-memory loop may hold assistant message objects, not dicts."""
        class Msg:
            def __init__(self, calls):
                self.role = "assistant"
                self.content = ""
                self.tool_calls = [type("TC", (), {"id": i})() for i in calls]
        msgs = [Msg(["a"]), tool_row("a"), Msg(["b"]), tool_row("b"), Msg(["c"])]
        streak, cur = _single_call_streak(msgs, "c")
        self.assertEqual(streak, 3)
        self.assertTrue(cur)


class TestBatchingNudgeNote(unittest.TestCase):
    def _agent(self, batch_size=None):
        return SimpleNamespace(**({"_current_tool_batch_size": batch_size} if batch_size is not None else {}))

    def test_fires_on_third_single_own_result_row_present(self):
        """turn_tool_round path: the in-flight row IS in messages — walker counts it."""
        msgs = transcript(["a"], ["b"], ["c"])
        note = _batching_nudge_note(self._agent(1), msgs, "read_file", tool_call_id="c")
        self.assertIsNotNone(note)
        self.assertTrue(note and "parallel" in note and "execute_code" in note, repr(note))

    def test_fires_when_in_flight_row_has_not_landed(self):
        """Approval-segment path: commit runs before the row lands — id arbitrates."""
        msgs = transcript(["a"], ["b"])
        self.assertIsNotNone(
            _batching_nudge_note(self._agent(1), msgs, "terminal", tool_call_id="c"))

    def test_no_double_count_when_row_present(self):
        """The 2026-11-07 bug: +1 on top of a counted row pushes 3 → 4, silencing."""
        msgs = transcript(["a"], ["b"], ["c"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(1), msgs, "read_file", tool_call_id="d"))

    def test_second_single_does_not_fire(self):
        msgs = transcript(["a"], ["b"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(1), msgs, "read_file", tool_call_id="b"))

    def test_fourth_single_never_fires_until_reset(self):
        msgs = transcript(["a"], ["b"], ["c"], ["d"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(1), msgs, "read_file", tool_call_id="d"))

    def test_batch_in_flight_never_nagged(self):
        msgs = transcript(["a"], ["b"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(3), msgs, "read_file", tool_call_id="c"))

    def test_batch_row_in_transcript_breaks_streak_without_attr(self):
        """Even without the batch-size attr, a 2-call row in the transcript resets."""
        msgs = transcript(["a"], ["b", "c"], ["d"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(None), msgs, "read_file", tool_call_id="d"))

    def test_execute_code_call_is_exempt_but_counts_in_streak(self):
        msgs = transcript(["a"], ["b"], ["c"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(1), msgs, "execute_code", tool_call_id="c"))

    def test_errors_and_blocked_results_stay_silent(self):
        msgs = transcript(["a"], ["b"], ["c"])
        self.assertIsNone(
            _batching_nudge_note(self._agent(1), msgs, "read_file",
                                 is_error=True, tool_call_id="c"))
        self.assertIsNone(
            _batching_nudge_note(self._agent(1), msgs, "read_file",
                                 blocked=True, tool_call_id="c"))

    def test_fallback_without_attr_uses_transcript_only(self):
        msgs = transcript(["a"], ["b"], ["c"])
        self.assertIsNotNone(
            _batching_nudge_note(self._agent(), msgs, "read_file", tool_call_id="c"))

    def test_note_is_deterministic_byte_stable(self):
        msgs = transcript(["a"], ["b"], ["c"])
        n1 = _batching_nudge_note(self._agent(1), msgs, "read_file", tool_call_id="c")
        n2 = _batching_nudge_note(self._agent(1), msgs, "read_file", tool_call_id="c")
        self.assertEqual(n1, n2)


class TestGaugePlacementRule(unittest.TestCase):
    def test_append_grows_string_and_list_shapes(self):
        msg = {"role": "tool", "content": '{"ok": true}'}
        _append_gauge_to_tool_message(msg, "[batching hint] x")
        self.assertTrue(msg["content"].endswith("\n[batching hint] x"))
        self.assertIn('{"ok": true}', msg["content"])
        mm = {"role": "tool", "content": [{"type": "text", "text": "a"}]}
        _append_gauge_to_tool_message(mm, "[x]")
        self.assertEqual(mm["content"][-1], {"type": "text", "text": "[x]"})


if __name__ == "__main__":
    unittest.main()
