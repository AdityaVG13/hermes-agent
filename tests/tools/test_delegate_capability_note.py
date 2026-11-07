"""The child capability note: a spawned child learns its deferred-tool universe.

Children inherit the parent's toolsets but are told only a goal and context — nothing
names the purpose-built tools reachable through the tool_search bridge (AST search,
find-and-read, connectors). A child that doesn't know they exist burns its window on
manual read+grep loops; the note is derived from the child's REAL scope (the same
un-collapsed catalog its bridge dispatch checks), so it never names a tool the child
cannot actually call, and never blocks a spawn on failure.
"""

from unittest.mock import patch

from tools.delegate_tool import _child_capability_note


_DEFS = [{"name": "asgrep_search"}, {"name": "asgrep_read"}, {"name": "genome_snap"},
         {"name": "read_file"}, {"name": "terminal"}]


def _note(deferrable=frozenset({"asgrep_search", "asgrep_read", "genome_snap"}), defs=None, explode=False):
    import model_tools
    import tools.tool_search as ts

    def fake_defs(**_kw):
        if explode:
            raise RuntimeError("catalog unreadable")
        return defs if defs is not None else _DEFS

    with (patch.object(model_tools, "get_tool_definitions", side_effect=fake_defs),
          patch.object(ts, "scoped_deferrable_names", return_value=frozenset(deferrable))):
        return _child_capability_note(["default"], None)


def test_note_names_the_deferred_universe_and_the_discovery_verb():
    note = _note()
    assert "asgrep_search" in note and "genome_snap" in note
    assert "tool_search" in note and "tool_call" in note
    # Core tools are never named: they are already in the child's visible toolset.
    assert "read_file" not in note and "terminal" not in note


def test_note_is_deterministic_for_a_fixed_scope():
    assert _note() == _note()


def test_no_deferrable_tools_means_no_note():
    assert _note(deferrable=frozenset()) == ""


def test_catalog_failure_fails_soft_to_no_note():
    assert _note(explode=True) == ""


def test_long_universes_are_capped():
    many = frozenset(f"tool_{i}" for i in range(12))
    note = _note(deferrable=many)
    assert "+4 more" in note  # tool_6..tool_9 are hidden; string-sort keeps tool_10/tool_11 shown
    assert "tool_7" not in note
