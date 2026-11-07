"""skill_view envelope: ref + digest + stateless if_digest, bounded model-path body.

Compaction-session contract (2026-11-06): the fold receipt emits ledger lines
as ``skill-name#Heading`` / ``skill-name`` with a digest; a post-fold re-read
must be able to answer {"status":"unchanged"} statelessly (the session-scoped
dedup_skill_view cache dies at every compression boundary — this does not).
Model-path serves (preprocess=True) bound the body at 8 KB with a heading-ref
continuation; slash/preload callers keep the full body.
"""

import json
from unittest.mock import patch

import pytest

from tools.skills_tool import skill_view

SKILL_MD = "---\nname: test-skill\ndescription: test skill\n---\n\n# One\nhello\n"


@pytest.fixture
def fake_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD)
    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        yield {"skills_dir": skills_dir, "skill_dir": skill_dir}


def _load(result_str):
    return json.loads(result_str)


class TestRefAndDigest:
    def test_full_serve_carries_ref_and_digest(self, fake_skills):
        d = _load(skill_view("test-skill"))
        assert d["success"] is True
        assert d["ref"] == "test-skill"
        assert len(d["digest"]) == 16
        int(d["digest"], 16)  # hex

    def test_file_serve_carries_ref_and_digest(self, fake_skills):
        d = _load(skill_view("test-skill", file_path="SKILL.md"))
        assert d["success"] is True
        assert d["ref"] == "test-skill/SKILL.md"
        assert len(d["digest"]) == 16

    def test_digest_is_stable_across_calls(self, fake_skills):
        a = _load(skill_view("test-skill"))["digest"]
        b = _load(skill_view("test-skill"))["digest"]
        assert a == b


class TestIfDigest:
    def test_match_returns_unchanged_stub_without_body(self, fake_skills):
        first = _load(skill_view("test-skill"))
        second = _load(skill_view("test-skill", if_digest=first["digest"]))
        assert second["status"] == "unchanged"
        assert second["ref"] == "test-skill"
        assert second["digest"] == first["digest"]
        assert "content" not in second

    def test_mismatch_serves_full_body(self, fake_skills):
        d = _load(skill_view("test-skill", if_digest="0" * 16))
        assert "content" in d
        assert d.get("status") != "unchanged"

    def test_file_serve_if_digest_match_stubs(self, fake_skills):
        first = _load(skill_view("test-skill", file_path="SKILL.md"))
        second = _load(
            skill_view("test-skill", file_path="SKILL.md", if_digest=first["digest"])
        )
        assert second["status"] == "unchanged"
        assert "content" not in second


class TestBodyBudget:
    def test_model_path_body_bounded_with_continuation(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "big-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: big-skill\ndescription: big\n---\n\n# Body\n" + "x" * 20000
        )
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            d = _load(skill_view("big-skill"))
        assert len(d["content"]) <= 8000
        assert d["truncated"] is True
        assert d["total_chars"] > 20000
        assert "SKILL.md#" in d["continuation"]

    def test_slash_path_keeps_full_body(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "big-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: big-skill\ndescription: big\n---\n\n# Body\n" + "x" * 20000
        )
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            d = _load(skill_view("big-skill", preprocess=False))
        assert len(d["content"]) > 20000
        assert not d.get("truncated")

    def test_hand_placed_giant_reads_full_by_contract(self, tmp_path):
        # Above the 100k storage-governance ceiling = hand-placed by the owner;
        # reads in full even on the model path (TestHandPlacedSkillsNoLimit).
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "manual-giant"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: manual-giant\ndescription: giant\n---\n\n# Body\n" + "x" * 150_000
        )
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            d = _load(skill_view("manual-giant"))
        assert len(d["content"]) > 150_000
        assert not d.get("truncated")
        assert len(d["digest"]) == 16
