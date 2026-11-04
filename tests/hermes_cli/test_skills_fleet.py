"""``hermes skills fleet scan`` — cross-machine skills drift classification.

Every regression class here is a live fleet failure from the 2026-11-03
campaign: depth-limited probes that never saw ``mlops/models/*``, skills whose
dir name differs from their frontmatter name (invisible to dir-name keying),
flat+categorized twins (benign and diverged), AppleDouble litter, UTF-16
PowerShell redirect output, and broken symlinks that crashed naive walks.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from hermes_cli import skills_fleet as sf
from typing import Optional


# ── fixture builder ──────────────────────────────────────────────────────────

def make_skill(root: Path, rel: str, name: str | None = None, content: str = "body text",
               mtime: int | None = None) -> Path:
    """Create <root>/<rel>/SKILL.md; *name* defaults to the dir basename."""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    skill_name = name if name is not None else Path(rel).name
    (d / "SKILL.md").write_text(f"---\nname: {skill_name}\ndescription: d\n---\n{content}\n",
                                encoding="utf-8")
    if mtime is not None:
        os.utime(d / "SKILL.md", (mtime, mtime))
    return d


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def local_root(tmp_path: Path) -> Path:
    return tmp_path / "skills"


# ── probe rules contract (generator and local walker share ONE rule set) ─────

def test_probe_rules_unlimited_depth_followlinks_prunes_dotdirs():
    rules = sf.PROBE_RULES
    assert rules["max_depth"] is None, "any depth cap blinds the probe to mlops/models/* twins"
    assert rules["follow_symlinks"] is True, "loader follows skill-dir symlinks; probes must too"
    for dotdir in (".hub", ".archive", ".locks", ".curator_backups"):
        assert dotdir in rules["prunes"], f"{dotdir} must be pruned (state, not skills)"


def test_rules_have_no_leak_of_cache_or_sources_dirs():
    for pruned in ("cache", "_sources", "__pycache__", "node_modules"):
        assert pruned in sf.PROBE_RULES["prunes"]


# ── local walk ───────────────────────────────────────────────────────────────

def test_walk_finds_depth3_dirs_and_keys_by_frontmatter_name(local_root):
    # The audiocraft class: dir "audiocraft", frontmatter name
    # "audiocraft-audio-generation", three levels deep.
    make_skill(local_root, "mlops/models/audiocraft", name="audiocraft-audio-generation")
    rows = {r.name: r for r in sf.local_rows(local_root)}
    assert "audiocraft-audio-generation" in rows
    assert rows["audiocraft-audio-generation"].relpath == "mlops/models/audiocraft"
    assert "audiocraft" not in rows, "rows are keyed by frontmatter name, never dir name"


def test_walk_prunes_dotdirs_and_ignores_appledouble(local_root):
    make_skill(local_root, "real-skill")
    make_skill(local_root, ".hub/bundled/SKILL.md" if False else ".hub/bundled", name="x")
    (local_root / "._real-skill").write_bytes(b"\x00\x05mac metadata")
    (local_root / ".hub").mkdir(exist_ok=True)
    names = {r.name for r in sf.local_rows(local_root)}
    assert names == {"real-skill"}, f"dotdir skills and AppleDouble junk leaked in: {names}"


def test_walk_leaf_rule_skips_nested_skill_md(local_root):
    skill = make_skill(local_root, "parent-skill")
    (skill / "fixtures" / "nested").mkdir(parents=True)
    (skill / "fixtures" / "nested" / "SKILL.md").write_text(
        "---\nname: nested-fixture\n---\n")
    names = {r.name for r in sf.local_rows(local_root)}
    assert names == {"parent-skill"}, "a dir holding SKILL.md is a leaf; nested ones are not skills"


def test_walk_follows_skill_dir_symlinks_flags_them_skips_broken(local_root):
    local_root.mkdir(parents=True, exist_ok=True)
    outside = local_root.parent / "elsewhere"
    make_skill(outside, "linked-skill")
    link = local_root / "linked-skill"
    link.symlink_to(outside / "linked-skill", target_is_directory=True)
    (local_root / "dead-skill").symlink_to(local_root.parent / "gone", target_is_directory=True)
    rows = {r.name: r for r in sf.local_rows(local_root)}
    assert set(rows) == {"linked-skill"}, f"broken link crashed or leaked: {set(rows)}"
    assert rows["linked-skill"].symlink is True


def test_local_rows_hash_matches_sha256(local_root):
    p = make_skill(local_root, "hashed-skill")
    row = next(r for r in sf.local_rows(local_root) if r.name == "hashed-skill")
    assert row.sha == sha_of(p / "SKILL.md")


def test_loader_prune_list_is_covered_by_probe_rules():
    """Contract: if the loader's scanner ever prunes MORE than the fleet probes,
    probes report skills the machine doesn't serve (or vice versa). Rules may
    prune extra, never less."""
    from agent.skill_utils import EXCLUDED_SKILL_DIRS
    missing = set(EXCLUDED_SKILL_DIRS) - set(sf.PROBE_RULES["prunes"])
    assert not missing, f"loader prunes {missing} but fleet probes do not"


@pytest.mark.platforms("posix")
def test_bash_probe_output_matches_local_rows(tmp_path):
    """The generated probe, executed against the same tree as the local
    walker, must produce byte-identical (name, sha, relpath) rows — the
    strongest guard against probe/walker rule drift."""
    import subprocess
    root = tmp_path / "skills"
    make_skill(root, "plain")
    make_skill(root, "deep/a/b/renamed", name="declared-name")   # dir≠name + depth 3
    make_skill(root, "twin", name="tw", content="one")           # benign twin pair
    make_skill(root, "cat/twin", name="tw", content="one")
    make_skill(root, "holder")                                    # leaf rule: nested ghost invisible
    make_skill(root, "holder/fixtures/ghost", name="ghost")
    outside = tmp_path / "elsewhere"
    make_skill(outside, "ext")                                    # followed symlink
    (root / "ext").symlink_to(outside / "ext", target_is_directory=True)
    (root / "dead").symlink_to(tmp_path / "gone", target_is_directory=True)  # dangling
    out = tmp_path / "probe.out"
    subprocess.run(["sh", "-c", sf.render_bash_probe(), "probe", str(root), str(out)],
                   check=True, timeout=60)
    got = {(r.name, r.sha, r.relpath) for r in sf.parse_probe_output(out.read_bytes())}
    want = {(r.name, r.sha, r.relpath) for r in sf.local_rows(root)}
    assert got == want, f"probe rows diverged from walker rows: {got ^ want}"


# ── probe script generation (tested by output contract) ─────────────────────

def test_bash_probe_script_never_depth_limited_and_follows_links():
    script = sf.render_bash_probe()
    assert "find -L" in script, "must follow symlinks like the loader"
    assert "-maxdepth" not in script, "a depth cap is exactly how the fleet twins were missed"
    for dotdir in (".hub", ".archive", ".locks", ".curator_backups"):
        assert f"-name {dotdir}" in script, f"{dotdir} must be pruned by the probe"
    assert "DONE" in script, "sentinel terminates the stream"


def test_ps_probe_script_handles_reparse_points_without_depth_blindness():
    script = sf.render_ps_probe()
    assert "-Depth 2" not in script and "-Depth 3" not in script, "no fixed recursion cap"
    assert "ReparsePoint" in script or "Attributes" in script, "must follow junctions/symlinks"
    assert "Get-FileHash" in script
    assert "DONE" in script


# ── probe output parsing (real-world encodings) ──────────────────────────────

def test_parse_probe_output_utf16le_bom_crlf_done():
    payload = "alpha|" + "ab" * 32 + "|cat/alpha\r\ndelta|" + "cd" * 32 + "|delta\r\nDONE\r\n"
    raw = ("\ufeff" + payload).encode("utf-16")  # BOM + UTF-16 (PS `>` redirect)
    rows = sf.parse_probe_output(raw)
    assert [r.name for r in rows] == ["alpha", "delta"]
    assert rows[0].sha.startswith("abab") and rows[0].sha == rows[0].sha.lower()


def test_parse_probe_output_utf8_junk_lines_skipped():
    payload = ("beta|" + "ef" * 32 + "|beta\n"
               "garbage line without pipes\n"
               "truncated|only-two-parts\n"
               "DONE\npost-done junk that must not appear\n")
    rows = sf.parse_probe_output(payload.encode("utf-8"))
    assert [r.name for r in rows] == ["beta"]


def test_parse_probe_output_missing_sentinel_is_a_machine_failure():
    with pytest.raises(ValueError, match="missing DONE sentinel"):
        sf.parse_probe_output(("alpha|" + "ab" * 32 + "|alpha\n").encode())
    # empty stream is a dead probe too, not a zero-skill machine
    with pytest.raises(ValueError):
        sf.parse_probe_output(b"")


# ── classification ───────────────────────────────────────────────────────────

def _row(name, sha, relpath):
    return sf.FleetRow(name=name, sha=sha, relpath=relpath, symlink=False)


def test_classify_identical_diverged_missing_extra_relocated():
    local = [_row("alpha", "a" * 64, "cat/alpha"),
             _row("beta", "b" * 64, "beta"),
             _row("gamma", "c" * 64, "gamma"),
             _row("eps", "e" * 64, "eps")]
    remote = [_row("alpha", "a" * 64, "alpha"),        # identical + relocated (path differs)
              _row("beta", "z" * 64, "beta"),          # diverged
              _row("delta", "d" * 64, "delta")]        # extra
    rep = sf.classify(local, {"sat": remote})
    m = rep["machines"]["sat"]
    assert m["identical"] == ["alpha"]
    assert m["diverged"] == ["beta"]
    assert m["missing"] == ["eps", "gamma"]  # sorted: deterministic report order
    assert m["extra"] == ["delta"]
    assert m["relocated"] == {"alpha": "alpha"}
    assert rep["ok"] is False


def test_classify_allow_missing_quieted():
    local = [_row("apple-only", "a" * 64, "apple-only")]
    remote = []
    rep = sf.classify(local, {"win": remote}, allowed_missing={"win": {"apple-only"}})
    assert rep["machines"]["win"]["missing"] == []
    assert rep["machines"]["win"]["allowed_missing"] == ["apple-only"]


def test_classify_benign_vs_diverged_twins_on_both_sides():
    local = [_row("t", "a" * 64, "t"), _row("t", "a" * 64, "cat/t"),
             _row("u", "b" * 64, "u")]
    remote = [_row("t", "a" * 64, "t"),
              _row("u", "b" * 64, "u"), _row("u", "x" * 64, "cat/u")]
    rep = sf.classify(local, {"sat": remote})
    assert rep["twins"]["local"]["benign"] == {"t": ["cat/t", "t"]}
    assert rep["twins"]["sat"]["benign"] == {}  # one 't' row on sat — no twin there
    assert rep["twins"]["local"]["diverged"] == {}
    assert rep["twins"]["sat"]["diverged"] == {"u": ["cat/u", "u"]}
    assert rep["critical"] == ["sat: diverged twin 'u' at cat/u, u"]
    assert rep["ok"] is False


def test_classify_ok_when_all_identical():
    rows = [_row("solo", "a" * 64, "solo")]
    rep = sf.classify(rows, {"sat": [_row("solo", "a" * 64, "solo")]})
    assert rep["ok"] is True and rep["critical"] == []


# ── run_scan orchestration with a fake transport ─────────────────────────────

class FakeTransport:
    def __init__(self, outputs: dict[str, bytes], fail: Optional[set[str]] = None):
        self.outputs = outputs
        self.fail = fail or set()
        self.scripts: dict[str, str] = {}

    def __call__(self, machine: dict, script: str) -> bytes:
        name = machine["name"]
        if name in self.fail:
            raise RuntimeError(f"ssh: connect to {machine['host']}: timed out")
        self.scripts[name] = script
        return self.outputs[name]


def test_run_scan_classifies_via_transport(local_root, monkeypatch):
    make_skill(local_root, "same")
    make_skill(local_root, "drifted", content="newer mac body")
    out = (f"same|{sha_of(local_root/'same'/'SKILL.md')}|same\n"
           f"drifted|{'0'*64}|drifted\nDONE\n").encode()
    machines = [{"name": "sat", "host": "sat-host", "root": "/srv/skills", "kind": "bash"}]
    t = FakeTransport({"sat": out})
    rep = sf.run_scan(machines, transport=t, local_root=local_root)
    m = rep["machines"]["sat"]
    assert m["identical"] == ["same"] and m["diverged"] == ["drifted"]
    assert "find -L" in t.scripts["sat"]


def test_run_scan_transport_failure_isolated(local_root):
    make_skill(local_root, "same")
    sha = sha_of(local_root / "same" / "SKILL.md")
    out = (f"same|{sha}|same\nDONE\n").encode()
    machines = [{"name": "down", "host": "down-host", "root": "/s", "kind": "bash"},
                {"name": "up", "host": "up-host", "root": "/s", "kind": "powershell"}]
    t = FakeTransport({"down": b"", "up": out}, fail={"down"})
    rep = sf.run_scan(machines, transport=t, local_root=local_root)
    assert rep["machines"]["down"]["error"].startswith("ssh:")
    assert rep["machines"]["up"]["identical"] == ["same"]
    assert rep["ok"] is False


def test_run_scan_powershell_gets_ps_script(local_root):
    make_skill(local_root, "same")
    sha = sha_of(local_root / "same" / "SKILL.md")
    machines = [{"name": "win", "host": "w", "root": "C:/s", "kind": "powershell"}]
    t = FakeTransport({"win": (f"same|{sha}|same\r\nDONE\r\n").encode("utf-16")})
    rep = sf.run_scan(machines, transport=t, local_root=local_root)
    assert rep["machines"]["win"]["identical"] == ["same"]
    assert "Get-FileHash" in t.scripts["win"]


def test_run_scan_rejects_unknown_kind(local_root):
    make_skill(local_root, "same")
    machines = [{"name": "odd", "host": "h", "root": "/s", "kind": "telnet"}]
    rep = sf.run_scan(machines, transport=FakeTransport({}), local_root=local_root)
    assert "unsupported kind" in rep["machines"]["odd"]["error"]
