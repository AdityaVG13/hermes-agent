"""The batching nudge: a structural hint when single-call turns stack up.

2026-11-07 evidence (state.db, this install): 56% of assistant turns carry
exactly one tool call; each pays a full model round trip for an ms-scale
handler, and execute_code carried 0.82% of tool results. Prompt framing alone
is volitional — this nudge fires from the loop itself, once per streak,
exactly at three consecutive single-call turns, and never on errors or
execute_code (which IS batching)."""

import unittest

from agent.tool_executor import _append_gauge_to_tool_message, _batching_nudge_note, _single_call_streak


def asst(n_calls: int, text: str = "working"):
    if n_calls == 0:
        return {"role": "assistant", "content": text}
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": f"c{i}", "function": {"name": "read_file", "arguments": "{}"}}
                           for i in range(n_calls)]}


def tool_result(name="read_file"):
    return {"role": "tool", "name": name, "tool_call_id": "c0", "content": "{}"}


def transcript(*turns):
    """Interleave each assistant turn with its tool results, like a real transcript."""
    out = []
    for n in turns:
        out.append(asst(n))
        for i in range(n):
            out.append(tool_result())
    return out


class TestSingleCallStreak(unittest.TestCase):
    def test_counts_trailing_single_call_turns(self):
        msgs = transcript(1, 1, 1)
        self.assertEqual(_single_call_streak(msgs), 3)

    def test_multi_call_turn_breaks_the_run(self):
        msgs = transcript(1, 1, 2, 1)
        self.assertEqual(_single_call_streak(msgs), 1)

    def test_text_only_turn_breaks_the_run(self):
        msgs = transcript(1, 1) + [asst(0)]
        self.assertEqual(_single_call_streak(msgs), 0)

    def test_user_turn_breaks_the_run(self):
        msgs = transcript(1, 1) + [{"role": "user", "content": "hi"}]
        self.assertEqual(_single_call_streak(msgs), 0)

    def test_empty_transcript(self):
        self.assertEqual(_single_call_streak([]), 0)


class TestBatchingNudgeNote(unittest.TestCase):
    def test_fires_exactly_at_three(self):
        msgs = transcript(1, 1, 1)
        self.assertIsNotNone(_batching_nudge_note(msgs, "read_file"))

    def test_silent_below_three(self):
        self.assertIsNone(_batching_nudge_note(transcript(1, 1), "read_file"))

    def test_silent_past_three_until_reset(self):
        # fires at exactly 3 — a 4th single turn must not re-fire
        self.assertIsNone(_batching_nudge_note(transcript(1, 1, 1, 1), "read_file"))

    def test_execute_code_turns_never_nudge(self):
        self.assertIsNone(_batching_nudge_note(transcript(1, 1, 1), "execute_code"))

    def test_error_and_blocked_results_stay_silent(self):
        msgs = transcript(1, 1, 1)
        self.assertIsNone(_batching_nudge_note(msgs, "read_file", is_error=True))
        self.assertIsNone(_batching_nudge_note(msgs, "read_file", blocked=True))

    def test_note_text_is_deterministic_and_names_both_paths(self):
        msgs = transcript(1, 1, 1)
        n1, n2 = _batching_nudge_note(msgs, "read_file"), _batching_nudge_note(msgs, "read_file")
        self.assertEqual(n1, n2)
        self.assertIn("parallel", n1)
        self.assertIn("execute_code", n1)

    def test_attaches_outside_wrapper_like_the_gauge(self):
        msg = {"role": "tool", "content": "{\"ok\": true}", "tool_call_id": "c0"}
        _append_gauge_to_tool_message(msg, "[batching hint] x")
        self.assertTrue(msg["content"].endswith("\n[batching hint] x"))
        self.assertIn("{\"ok\": true}", msg["content"])


if __name__ == "__main__":
    unittest.main()
