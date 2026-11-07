"""map: first-contact repo skeleton — one bounded call, everything chained.

Council ruling (2026-11-06): the 8th verb. A first read of a project should be
ONE bounded result the model can act on: layout, manifests, entry points, gates
(as verify indexes), context files (as read_file refs) — not an improvised
ls/find/grep scatter. Discovery rides agent.coding_context.detect_project_facts
(the same single source of truth behind verify and the workspace snapshot).
"""

import json
from unittest.mock import patch

from tools.map_tool import _handle_map


def _make_tree(tmp_path):
    (tmp_path / "Makefile").write_text("check:\n\t@echo ok\ntest:\n\t@echo t\n")
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest run"}}')
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / "AGENTS.md").write_text("rules\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('entry')\n")
    (src / "util.py").write_text("x = 1\n" * 50)
    deep = tmp_path / "src" / "nested"
    deep.mkdir()
    (deep / "inner.py").write_text("y = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x(): pass\n")


class TestMapEnvelope:
    def test_carries_facts_and_refs(self, tmp_path):
        _make_tree(tmp_path)
        result = _handle_map({"path": str(tmp_path)})
        assert result.get("success") is True
        # Gates match detect_project_facts AND verify's command_index space
        assert "make test" in result["verify_commands"]
        assert "npm run test" in result["verify_commands"]
        # Context files are read_file refs (chain verbatim)
        assert any("README.md" in r for r in result["context_files"])
        # Entry points found by convention
        assert any("main.py" in e for e in result["entry_points"])
        # Layout present and bounded
        assert any(d["dir"] == "src" for d in result["layout"])

    def test_layout_is_bounded_with_honest_truncation(self, tmp_path):
        for i in range(40):
            (tmp_path / f"pkg_{i:02d}").mkdir()
        result = _handle_map({"path": str(tmp_path), "max_dirs": 10})
        assert len(result["layout"]) <= 10
        assert result["truncated"] is True
        assert result["total_dirs"] >= 40

    def test_empty_dir_is_honest_not_error(self, tmp_path):
        result = _handle_map({"path": str(tmp_path)})
        assert result.get("success") is True
        assert result["verify_commands"] == []
        assert "manifest" in result.get("note", "").lower()

    def test_missing_root_refused(self, tmp_path):
        result = _handle_map({"path": str(tmp_path / "nope")})
        assert "error" in result

    def test_ignores_noise_dirs(self, tmp_path):
        _make_tree(tmp_path)
        (tmp_path / ".git").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "__pycache__").mkdir()
        result = _handle_map({"path": str(tmp_path)})
        names = [d["dir"] for d in result["layout"]]
        assert ".git" not in names and "node_modules" not in names
