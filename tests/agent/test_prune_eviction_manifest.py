"""The eviction manifest: a prune pass announces itself where the model reads it.

``_stamp_eviction_manifest`` appends one deterministic line to the FIRST row a
proactive-prune pass rewrote — ``[pruned: N results elided (tool×k, …) — originals
in session history; recover with session_search]`` — so the model meets the eviction
at the top of the pruned zone instead of discovering silent loss by failure (the
skill re-load treadmill class). The manifest rides an existing row because the
in-place commit requires equal row counts; it never adds, drops or reorders one,
and identical inputs stamp identical bytes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agent.context_compressor import ContextCompressor

LARGE_WINDOW = 1_000_000


def _compressor(**kw: Any) -> ContextCompressor:
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    defaults.update(kw)
    with patch("agent.context_compressor.get_model_context_length", return_value=LARGE_WINDOW):
        return ContextCompressor(**defaults)


def _history(n_pairs: int = 8, big: int = 9_000) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        name = "terminal" if i % 2 == 0 else "read_file"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": '{"a":1}'}}],
        })
        msgs.append({
            "role": "tool", "name": name, "tool_call_id": cid,
            "content": chr(65 + i) * big if i < 4 else "ok",
        })
    return msgs


def _run_prune(messages):
    compressor = _compressor()
    compressor._proactive_prune_rearm_tokens = 0
    return compressor.prune_tool_results_only(messages, current_tokens=1_000_000)


def test_manifest_rides_the_first_changed_row():
    original = _history()
    pruned, count = _run_prune(list(original))
    assert count > 0
    assert len(pruned) == len(original)
    changed = [
        i for i, (o, p) in enumerate(zip(original, pruned))
        if o.get("content") != p.get("content")
    ]
    assert changed, "fixture expects rewritten rows"
    manifest_rows = [i for i in changed if "[pruned:" in str(pruned[i].get("content"))]
    assert manifest_rows == [changed[0]], "exactly one manifest, on the first changed row"
    first_content = pruned[changed[0]]["content"]
    assert "terminal×2" in first_content and "read_file×2" in first_content
    assert "session_search" in first_content


def test_manifest_absent_when_nothing_pruned():
    small = [{"role": "system", "content": "sys"}]
    for i in range(8):
        cid = f"call_{i}"
        small.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
        small.append({"role": "tool", "tool_call_id": cid, "content": "ok"})
    pruned, count = _run_prune(small)
    assert count == 0 and pruned is small
    assert not any("[pruned:" in str(m.get("content", "")) for m in small)


def test_manifest_deterministic():
    a_in, b_in = _history(), _history()
    a_pruned, _ = _run_prune(a_in)
    b_pruned, _ = _run_prune(b_in)
    a_manifests = [str(m.get("content")) for m in a_pruned if "[pruned:" in str(m.get("content"))]
    b_manifests = [str(m.get("content")) for m in b_pruned if "[pruned:" in str(m.get("content"))]
    assert a_manifests == b_manifests
