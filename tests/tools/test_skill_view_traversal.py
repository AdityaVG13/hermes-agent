"""Tests for path traversal prevention in skill_view.

Regression tests for issue #220: skill_view file_path parameter allowed
reading arbitrary files (e.g., ~/.hermes/.env) via path traversal.
"""

import json
import os

import pytest
from unittest.mock import patch

from tools.skills_tool import skill_view


@pytest.fixture()
def fake_skills(tmp_path):
    """Create a fake skills directory with one skill and a sensitive file outside."""
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "test-skill"
    skill_dir.mkdir(parents=True)

    # Create SKILL.md
    (skill_dir / "SKILL.md").write_text("# Test Skill\nA test skill.")

    # Create a legitimate file inside the skill
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "api.md").write_text("API docs here")

    # Create a sensitive file outside skills dir (simulating .env)
    (tmp_path / ".env").write_text("SECRET_API_KEY=sk-do-not-leak")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        yield {"skills_dir": skills_dir, "skill_dir": skill_dir, "tmp_path": tmp_path}


class TestPathTraversalBlocked:
    def test_dotdot_in_skill_name_blocked(self, fake_skills):
        """A traversal skill name must not escape the skills search root.

        Regression: `name` was joined onto each search dir to build the lookup
        path with no `..`/absolute guard (while file_path WAS validated), so
        name="../outside-skill" could select a sibling dir outside SKILLS_DIR.
        """
        tmp_path = fake_skills["tmp_path"]
        outside_skill = tmp_path / "outside-skill"
        outside_skill.mkdir()
        (outside_skill / "SKILL.md").write_text("# Outside Skill\n")
        (outside_skill / ".env").write_text("ESCAPED_SECRET=do-not-leak")

        result = json.loads(skill_view("../outside-skill", file_path=".env"))

        assert result["success"] is False
        assert "traversal" in result["error"].lower()
        assert "do-not-leak" not in json.dumps(result)

    def test_absolute_skill_name_blocked(self, fake_skills):
        """An absolute skill name must not bypass the trusted search root."""
        tmp_path = fake_skills["tmp_path"]
        outside_skill = tmp_path / "outside-absolute"
        outside_skill.mkdir()
        (outside_skill / "SKILL.md").write_text("# Outside Absolute\n")
        (outside_skill / ".env").write_text("ABSOLUTE_SECRET=do-not-leak")

        result = json.loads(skill_view(str(outside_skill), file_path=".env"))

        assert result["success"] is False
        assert "relative path" in result["error"].lower()
        assert "do-not-leak" not in json.dumps(result)

    def test_legit_skill_name_still_works(self, fake_skills):
        """A normal skill name must still resolve after the name guard."""
        result = json.loads(skill_view("test-skill"))
        assert result["success"] is True

    def test_dotdot_in_file_path(self, fake_skills):
        """Direct .. traversal should be rejected."""
        result = json.loads(skill_view("test-skill", file_path="../../.env"))
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    def test_dotdot_nested(self, fake_skills):
        """Nested .. traversal should also be rejected."""
        result = json.loads(skill_view("test-skill", file_path="references/../../../.env"))
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    @pytest.mark.platforms("posix", "windows")
    def test_legitimate_file_still_works(self, fake_skills):
        """Valid paths within the skill directory should work normally."""
        result = json.loads(skill_view("test-skill", file_path="references/api.md"))
        assert result["success"] is True
        assert "API docs here" in result["content"]

    def test_no_file_path_shows_skill(self, fake_skills):
        """Calling skill_view without file_path should return the SKILL.md."""
        result = json.loads(skill_view("test-skill"))
        assert result["success"] is True

    @pytest.mark.parametrize("sibling", [False, True])
    def test_symlink_escape_blocked(self, fake_skills, sibling):
        """Symlinks pointing outside the skill directory should be blocked."""
        skill_dir = fake_skills["skill_dir"]
        outside = fake_skills["skills_dir"] / "test-skill-v2" if sibling else fake_skills["tmp_path"]
        outside.mkdir(exist_ok=True)
        secret = outside / "secret.txt"
        secret.write_text("TOP SECRET DATA")

        symlink = skill_dir / "evil-link"
        try:
            symlink.symlink_to(secret)
        except OSError:
            pytest.skip("Symlinks not supported")

        result = json.loads(skill_view("test-skill", file_path="evil-link"))
        # The resolve() check should catch the symlink escaping
        assert result["success"] is False
        assert "escapes" in result["error"].lower() or "boundary" in result["error"].lower()
        assert "TOP SECRET DATA" not in json.dumps(result)

    def test_sensitive_file_not_leaked(self, fake_skills):
        """Even if traversal somehow passes, sensitive content must not leak."""
        result = json.loads(skill_view("test-skill", file_path="../../.env"))
        assert result["success"] is False
        assert "sk-do-not-leak" not in result.get("content", "")
        assert "sk-do-not-leak" not in json.dumps(result)


class TestDefaultViewSymlinks:
    """The DEFAULT view (SKILL.md body) read had no containment check.

    Found live 2026-11-07 by the adversarial rig: skills/escaper/SKILL.md ->
    /etc/passwd served passwd's content. The file_path path was guarded
    (validate_within_dir resolves symlinks); the body read was not."""

    def test_default_view_blocks_symlinked_skill_md(self, fake_skills):
        secret = fake_skills["tmp_path"] / "outside-secret.md"
        secret.write_text("ROOT:0:0: TOP SECRET BODY\n")
        # overwrite the real SKILL.md with a symlink to the secret
        (fake_skills["skill_dir"] / "SKILL.md").unlink()
        os.symlink(secret, fake_skills["skill_dir"] / "SKILL.md")
        result = json.loads(skill_view("test-skill"))
        assert result.get("success") is False, result
        assert "TOP SECRET BODY" not in json.dumps(result)

    def test_skill_dir_symlink_stays_loadable_but_nested_escape_refuses(self, fake_skills):
        # A symlinked skill DIRECTORY is a supported fleet pattern (gtnh-shorts
        # -> repo/tools/shorts/skill): its own real SKILL.md must serve.
        outside = fake_skills["tmp_path"] / "evil-skill"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: legit-linked\ndescription: e\n---\nLEGIT LINKED BODY\n")
        (outside / "escape.md").write_text("SECOND SECRET\n")
        (fake_skills["skills_dir"] / "sneaky").symlink_to(outside)
        result = json.loads(skill_view("sneaky"))
        assert result.get("success") is True, result
        assert "LEGIT LINKED BODY" in json.dumps(result)
        # but a SKILL.md symlink planted INSIDE any dir still refuses
        secret2 = fake_skills["tmp_path"] / "outside-secret2.md"
        secret2.write_text("TOP SECRET NUMBER TWO\n")
        (outside / "SKILL.md").unlink()
        os.symlink(secret2, outside / "SKILL.md")
        result = json.loads(skill_view("sneaky"))
        assert result.get("success") is False, result
        assert "TOP SECRET NUMBER TWO" not in json.dumps(result)

    def test_file_serve_blocked_when_skill_dir_itself_escapes(self, fake_skills):
        outside = fake_skills["tmp_path"] / "evil-refs"
        outside.mkdir()
        (outside / "passwd.txt").write_text("root:x:0:0\n")
        (fake_skills["skill_dir"] / "refs-link").symlink_to(outside)
        result = json.loads(skill_view("test-skill", file_path="refs-link/passwd.txt"))
        assert result.get("success") is False, result
        assert "root:x" not in json.dumps(result)
