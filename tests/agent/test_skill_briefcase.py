"""Pinned hot-skills briefcase (prompt-cache campaign, final lever).

Static skill bodies ride a byte-stable block appended to the effective system
message, so the per-account prefix cache serves them across sessions instead
of re-billing them as fresh input on every skill_view. Measured motivation:
skill_view was 14.3% of all new content entering codex prompts (32.5M tok/14d),
one skill fully loaded 1,038 times across 449 sessions.
"""

import json
from pathlib import Path

import pytest

from agent.skill_briefcase import (
    build_briefcase_block,
    is_pinned,
    pinned_skill_names,
)
from agent.turn_context import _effective_system_with_briefcase


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """One flat skills tier with two small skills."""
    for name, body in (
        ("alpha", "---\nname: alpha\ndescription: A\n---\nALPHA BODY"),
        ("beta", "---\nname: beta\ndescription: B\n---\nBETA BODY"),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(body)

    import tools.skills_tool as st

    monkeypatch.setattr(st, "skill_search_dirs", lambda: ([("user", tmp_path)], tmp_path))
    return tmp_path


CFG = {"skills": {"pin": ["alpha", "beta"]}}


def test_unconfigured_returns_nothing(fake_root):
    assert pinned_skill_names({}) == []
    assert build_briefcase_block(config={}) is None


def test_block_contains_config_order_and_is_byte_stable(fake_root):
    names = pinned_skill_names(CFG)
    assert names == ["alpha", "beta"]
    b1 = build_briefcase_block(config=CFG)
    b2 = build_briefcase_block(config=CFG)
    assert b1 is not None and b1 == b2  # determinism: same bytes on every build
    assert b1.index("ALPHA BODY") < b1.index("BETA BODY")  # config order preserved
    assert "do not call skill_view" in b1  # teaches the short-circuit


def test_skill_edit_changes_block_and_stamp(fake_root):
    b1 = build_briefcase_block(config=CFG)
    (fake_root / "alpha" / "SKILL.md").write_text("---\nname: alpha\n---\nALPHA V2")
    b2 = build_briefcase_block(config=CFG)
    assert b2 is not None and b2 != b1
    assert "ALPHA V2" in b2


def test_missing_skill_is_skipped_not_fatal(fake_root):
    cfg = {"skills": {"pin": ["alpha", "ghost"]}}
    block = build_briefcase_block(config=cfg)
    assert block is not None
    assert "ALPHA BODY" in block
    assert "ghost" not in block


def test_is_pinned_matches_resolved_names(fake_root):
    assert build_briefcase_block(config=CFG) is not None
    assert is_pinned("alpha", config=CFG) is True
    assert is_pinned("ghost", config=CFG) is False


def test_effective_system_order_prompt_briefcase_ephemeral():
    out = _effective_system_with_briefcase("BASE PROMPT", "BRIEFCASE BLOCK", "EPHEMERAL")
    assert out.index("BASE PROMPT") < out.index("BRIEFCASE BLOCK") < out.index("EPHEMERAL")
    # no ephemeral, no briefcase: unchanged base
    assert _effective_system_with_briefcase("BASE", None, None) == "BASE"
    # briefcase without ephemeral still lands
    assert "BRIEFCASE" in _effective_system_with_briefcase("BASE", "BRIEFCASE", None)


def test_skill_view_short_circuits_pinned_body(fake_root, monkeypatch):
    monkeypatch.setattr("agent.skill_briefcase._active_config", lambda: CFG)
    assert build_briefcase_block(config=CFG) is not None
    from tools.skills_tool import skill_view

    out = json.loads(skill_view("alpha"))
    assert out.get("pinned") is True
    assert "content" not in out  # body is NOT re-sent
    assert out.get("linked_files") is not None or "path" in out  # references still reachable

    # a reference file still serves in full
    ref = fake_root / "alpha" / "references"
    ref.mkdir()
    (ref / "api.md").write_text("REF BODY")
    served = json.loads(skill_view("alpha", file_path="references/api.md"))
    assert served.get("success") is True
    assert "REF BODY" in served.get("content", "")
