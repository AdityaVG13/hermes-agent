"""`hermes skills health` — surfaces the silent failure classes this fleet actually hit:

- same-name duplicates that make a skill uninvocable (the collision-hiding rule)
- hub lock entries pointing at directories that do not exist (true orphans only —
  the existing `check` false-flags live skills, which must never be uninstalled)
- dead directory symlinks in the skills root (dangling or resolving without SKILL.md)
- husk directories: non-dot dirs holding no SKILL.md at any depth
- a disk catalog cache whose persisted rows disagree with the current library
"""
import json
from unittest.mock import patch

import tools.skills_tool as st


def _root(tmp_path):
    root = tmp_path / "skills"
    root.mkdir(exist_ok=True)
    return root


def _make_skill(root, name, category=None):
    skill_dir = root / (category or "") / name if category else root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )
    return skill_dir


class _StubLockFile:
    """Patches tools.skills_hub.HubLockFile: entries is what list_installed returns."""
    entries = []

    def __init__(self, *a, **kw):
        pass

    def list_installed(self):
        return self.entries


def _health(tmp_path):
    from hermes_cli import skills_health
    with (
        patch("tools.skills_tool.SKILLS_DIR", _root(tmp_path)),
        patch("tools.skills_tool._catalog_cache_path", return_value=tmp_path / "cat.json"),
        patch("tools.skills_hub.HubLockFile", _StubLockFile),
    ):
        st._SKILLS_CACHE.clear()
        return skills_health.collect_health()


def _health_with_cache_file(tmp_path, content: str):
    from hermes_cli import skills_health
    root = _root(tmp_path)
    cache_file = tmp_path / "cat.json"
    if content is not None:
        cache_file.write_text(content, encoding="utf-8")
    with (
        patch("tools.skills_tool.SKILLS_DIR", root),
        patch("tools.skills_tool._catalog_cache_path", return_value=cache_file),
        patch("tools.skills_hub.HubLockFile", _StubLockFile),
    ):
        st._SKILLS_CACHE.clear()
        return skills_health.collect_health()


def test_healthy_library_reports_no_findings(tmp_path):
    _make_skill(_root(tmp_path), "alpha")
    report = _health(tmp_path)
    assert report["findings"] == 0, report
    assert report["totals"]["loadable"] == 1


def test_duplicate_name_is_reported_hidden(tmp_path):
    # the fleet incident class: same name at top level AND in a category with DIVERGED
    # content -> resolution ambiguous, no path-unique winner for the top-level copy ->
    # uninvocable. (Identical twins resolve fine: shallowest rank wins, loser shadowed.)
    root = _root(tmp_path)
    _make_skill(root, "twin")
    twin_dir = _make_skill(root, "twin", category="creative")
    (twin_dir / "SKILL.md").write_text(
        "---\nname: twin\ndescription: A stale, diverged copy.\n---\n\nOld body.\n",
        encoding="utf-8")
    report = _health(tmp_path)
    names = [h["name"] for h in report["hidden"]]
    assert "twin" in names, report


def test_true_orphan_lock_reported_live_lock_not(tmp_path):
    root = _root(tmp_path)
    _make_skill(root, "live-one")
    _StubLockFile.entries = [
        {"name": "ghost", "install_path": "ghost"},
        {"name": "live-one", "install_path": "live-one"},
    ]
    try:
        report = _health(tmp_path)
        orphans = [o["name"] for o in report["orphan_locks"]]
        assert orphans == ["ghost"], report
    finally:
        _StubLockFile.entries = []


def test_dead_symlink_reported(tmp_path):
    import os
    root = _root(tmp_path)
    _make_skill(root, "alpha")
    os.symlink(str(root / "nowhere"), str(root / "dead-link"))
    report = _health(tmp_path)
    assert "dead-link" in [d["name"] for d in report["dead_links"]], report


def test_symlink_without_skill_md_reported(tmp_path):
    import os
    root = _root(tmp_path)
    target = tmp_path / "target-dir"
    target.mkdir()
    (target / "not-a-skill.txt").write_text("x", encoding="utf-8")
    _make_skill(root, "alpha")
    os.symlink(str(target), str(root / "bridge"))
    report = _health(tmp_path)
    assert "bridge" in [d["name"] for d in report["dead_links"]], report


def test_husk_directory_reported(tmp_path):
    root = _root(tmp_path)
    _make_skill(root, "alpha")
    (root / "empty-husk").mkdir()
    report = _health(tmp_path)
    assert "empty-husk" in [h["name"] for h in report["husk_dirs"]], report


def test_cache_rows_disagreement_informational_not_finding(tmp_path):
    _make_skill(_root(tmp_path), "alpha")
    content = json.dumps({"k": {"digest": "x", "rows": [
        {"name": "alpha"}, {"name": "phantom"}]}})
    report = _health_with_cache_file(tmp_path, content)
    assert report["cache"]["state"] == "rows_disagree", report["cache"]
    # expected lifecycle: the digest-keyed file is never served stale, so a stale
    # rows set must not fail the health exit code (only unreadable/torn does)
    assert report["findings"] == 0, report


def test_corrupt_cache_flagged_and_not_repaired(tmp_path):
    _make_skill(_root(tmp_path), "alpha")
    report = _health_with_cache_file(tmp_path, "{not json")
    assert report["cache"]["state"] == "unreadable", report["cache"]
    # health is read-only: the corrupt file must still be on disk, unrepaired
    assert (tmp_path / "cat.json").read_text(encoding="utf-8") == "{not json"
