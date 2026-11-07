"""read_file accepts line-range refs — search hits chain into reads verbatim.

Council envelope ruling (2026-11-06): every search result carries refs the read
verb consumes verbatim. search_files already prints ``path:line|content`` and
asgrep hits carry ``path#L12-L40``; this makes read_file accept those refs in
its ``path`` argument so the find→read compound is one copy-paste, not a
re-derivation of offsets the model must guess.
"""

from unittest.mock import patch

import pytest

from tools.file_tools import _parse_path_ref


class TestParsePathRef:
    def test_hash_range(self):
        assert _parse_path_ref("src/a.py#L12-L40") == ("src/a.py", 12, 29)

    def test_hash_single_line(self):
        assert _parse_path_ref("src/a.py#L7") == ("src/a.py", 7, None)

    def test_colon_line(self):
        assert _parse_path_ref("src/a.py:12") == ("src/a.py", 12, None)

    def test_plain_path_untouched(self):
        assert _parse_path_ref("src/a.py") == ("src/a.py", None, None)

    def test_windows_drive_colon_is_not_a_ref(self):
        # C:\\path\\to\\f.txt has a colon that is NOT a line ref — must pass through.
        assert _parse_path_ref("C:\\Users\\a\\f.txt") == ("C:\\Users\\a\\f.txt", None, None)

    def test_lsp_colon_range(self):
        assert _parse_path_ref("src/a.py:12-18") == ("src/a.py", 12, 7)

    def test_out_of_range_forms_rejected_to_plain(self):
        # Nonsense suffixes (L0, reversed ranges) fall back to plain path text
        # so a genuinely odd filename still reads instead of erroring.
        assert _parse_path_ref("weird#L0") == ("weird#L0", None, None)
        assert _parse_path_ref("weird#L40-L12") == ("weird#L40-L12", None, None)


class TestHandlerWiring:
    def test_ref_routes_offset_and_limit_into_read(self):
        from tools import file_tools

        captured = {}

        def fake_read_file_tool(**kwargs):
            captured.update(kwargs)
            return {"output": "ok"}

        with patch.object(file_tools, "read_file_tool", fake_read_file_tool):
            file_tools._handle_read_file({"path": "src/a.py#L12-L40"}, task_id="t")

        assert captured["path"] == "src/a.py"
        assert captured["offset"] == 12
        assert captured["limit"] == 29

    def test_single_line_ref_sets_offset_only(self):
        from tools import file_tools

        captured = {}

        def fake_read_file_tool(**kwargs):
            captured.update(kwargs)
            return {"output": "ok"}

        with patch.object(file_tools, "read_file_tool", fake_read_file_tool):
            file_tools._handle_read_file({"path": "src/a.py#L7"}, task_id="t")

        assert captured["path"] == "src/a.py"
        assert captured["offset"] == 7
        assert captured["limit"] == file_tools.DEFAULT_READ_LIMIT

    def test_explicit_offset_wins_over_ref(self):
        from tools import file_tools

        captured = {}

        def fake_read_file_tool(**kwargs):
            captured.update(kwargs)
            return {"output": "ok"}

        with patch.object(file_tools, "read_file_tool", fake_read_file_tool):
            file_tools._handle_read_file(
                {"path": "src/a.py#L12-L40", "offset": 5}, task_id="t"
            )

        # An explicit non-default offset/limit is deliberate — it refines the ref.
        assert captured["offset"] == 5
