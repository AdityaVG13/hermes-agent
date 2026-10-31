"""Orientation inventory in the compaction summary template.

Post-compaction amnesia: the old ``## Relevant Files`` body asked for a "brief
note" per file, so summaries carried paths without the durable facts, and there
was no record of which skills had been loaded. Every post-compaction turn then
re-read the same files and re-loaded the same skills before doing any work
(observed in long goal sessions: three compactions, each followed by a
~15-call re-orientation ritual). The template must therefore pin:

- ``## Relevant Files``: a per-file gist of durable facts, so a later turn
  re-reads a file only when it needs the exact text.
- ``## Skills Loaded``: name + what was read + the rules actually applied,
  so a later turn does not reload a skill it already has on record.
- The iterative-update form must carry both inventories forward.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _compressor(previous_summary: str | None = None) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True)
    if previous_summary is not None:
        compressor._previous_summary = previous_summary
    return compressor


def test_fresh_template_contains_orientation_inventory():
    prompt = _compressor()._build_summary_prompt("turns", 500, None, "", True)
    assert "## Relevant Files" in prompt
    # The gist directive: facts that make a re-read unnecessary except for exact text.
    assert "re-read" in prompt and "exact text" in prompt
    assert "## Skills Loaded" in prompt
    # The summarizer must be told where skill loads come from.
    assert "skill_view" in prompt


def test_no_user_turn_template_also_contains_inventory():
    prompt = _compressor()._build_summary_prompt("turns", 500, None, "", False)
    assert "## Skills Loaded" in prompt
    assert "exact text" in prompt


def test_update_prompt_carries_inventory_forward():
    prompt = _compressor(previous_summary="## Goal\nold")._build_summary_prompt(
        "turns", 500, None, "", True
    )
    assert "Skills Loaded" in prompt
    # The update instruction must name the inventories it preserves.
    assert "Relevant Files" in prompt
    assert "carry" in prompt.lower()
