"""Red-first tests for the dependency-currency memo (``pm.currency_memo``).

``venv_is_current`` is a pure function of a set of files plus its arguments.
The memo caches its verdict on disk, keyed on stat-facts of every input the
probe reads. It may answer only when the arguments match and nothing changed;
every other case must fall back to the real probe.
"""
import json
import os
from pathlib import Path

import pytest

from pm import client, install, packages
from pm.currency_memo import MEMO_NAME
from pm.environments import runtime_facts_path
from pm.plugin_inputs import Members


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (root / "uv.lock").write_text("lock\n", encoding="utf-8")
    return root


@pytest.fixture
def member():
    home = Path(os.environ["HERMES_HOME"])
    plug = home / "plugins" / "mempkg"
    plug.mkdir(parents=True)
    (plug / "pyproject.toml").write_text("[project]\nname = 'mempkg'\n", encoding="utf-8")
    (plug / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")
    return plug


@pytest.fixture
def stubbed_probe(monkeypatch):
    """Point the direct probe at a facts record stamped STAMP1; verdict True."""
    monkeypatch.setattr(packages.Venv, "expected_stamp",
                        lambda self, extras, *, plugin_dirs=None: "STAMP1")
    monkeypatch.setattr(install, "_carried", lambda package, recorded: list(recorded))
    monkeypatch.setattr(install, "_runtime_state_matches",
                        lambda fact, stamp, *, project_root=None: True)


@pytest.fixture
def seeded(project, member, stubbed_probe):
    """A valid memo for (extras=['e1'], plugins=Members([member])), verdict True."""
    return _seed(project, member)


def _seed(project, member):
    facts = runtime_facts_path(project)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(json.dumps(
        {"schema": 1, "packages": {"venv": {"stamp": "STAMP1", "extras": ["e1"]}}}),
        encoding="utf-8")
    assert install.venv_is_current(extras=["e1"], plugins=Members([member]),
                                   project_root=project) is True
    assert (facts.parent / MEMO_NAME).is_file()
    return member


def _armed(monkeypatch):
    """Both probe branches raise if reached, proving the memo did not answer."""
    calls = []

    def armed(*args, **kwargs):
        calls.append(1)
        raise AssertionError("the full probe ran; the memo should have answered")

    monkeypatch.setattr(install, "venv_is_current", armed)
    monkeypatch.setattr(client, "_request", armed)
    return calls


def _memo(project):
    return runtime_facts_path(project).parent / MEMO_NAME


def _bump(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns + 1_000_000_000, st.st_mtime_ns + 1_000_000_000))


class TestMemoAnswers:
    def test_client_answers_from_memo_without_probing(self, seeded, project, monkeypatch):
        calls = _armed(monkeypatch)
        assert client.venv_is_current(extras=["e1"], plugins=Members([seeded]),
                                      project_root=project) is True
        assert not calls

    def test_false_verdict_is_cached_too(self, project, member, monkeypatch, stubbed_probe):
        facts = runtime_facts_path(project)
        facts.parent.mkdir(parents=True, exist_ok=True)
        facts.write_text(json.dumps(
            {"schema": 1, "packages": {"venv": {"stamp": "STAMP1", "extras": ["e1"]}}}),
            encoding="utf-8")
        monkeypatch.setattr(install, "_runtime_state_matches",
                            lambda fact, stamp, *, project_root=None: False)
        assert install.venv_is_current(extras=["e1"], plugins=Members([member]),
                                       project_root=project) is False
        calls = _armed(monkeypatch)
        assert client.venv_is_current(extras=["e1"], plugins=Members([member]),
                                      project_root=project) is False
        assert not calls


class TestEveryChangeFallsBack:
    def _assert_probe_ran(self, seeded, project, monkeypatch):
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([seeded]),
                                   project_root=project)

    def test_uv_lock_touch(self, seeded, project, monkeypatch):
        _bump(project / "uv.lock")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_pyproject_touch(self, seeded, project, monkeypatch):
        _bump(project / "pyproject.toml")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_member_edit(self, seeded, project, monkeypatch):
        (seeded / "pkg.py").write_text("VALUE = 2\n", encoding="utf-8")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_new_profile_dir(self, seeded, project, monkeypatch):
        profiles = Path(os.environ["HERMES_HOME"]) / "profiles"
        profiles.mkdir(exist_ok=True)
        (profiles / "p-abcd1234").mkdir()
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_home_config_appears(self, seeded, project, monkeypatch):
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
            "plugins:\n  disabled: []\n", encoding="utf-8")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_pm_source_touch(self, seeded, project, monkeypatch):
        # Drift guard: any edit inside the pm/ package invalidates the memo.
        _bump(Path(install.__file__).with_name("workspace.py"))
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_corrupt_memo(self, seeded, project, monkeypatch):
        _memo(project).write_text("{not json", encoding="utf-8")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_missing_memo(self, seeded, project, monkeypatch):
        _memo(project).unlink()
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_off_switch(self, seeded, project, monkeypatch):
        monkeypatch.setenv("HERMES_PM_CURRENCY_MEMO", "0")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_different_extras_key(self, seeded, project, monkeypatch):
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e2"], plugins=Members([seeded]),
                                   project_root=project)

    def test_git_state_appears(self, seeded, project, monkeypatch):
        # The running version derives from git and gates member fitness; a
        # checkout under the same walked files must still invalidate the memo.
        git = project / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_semantic_input_touch(self, seeded, project, monkeypatch):
        from pm import paths

        _bump(paths.repo_root() / "hermes_constants.py")
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_chmod_moves_ctime(self, seeded, project, monkeypatch):
        (project / "uv.lock").chmod(0o600)
        self._assert_probe_ran(seeded, project, monkeypatch)

    def test_symlinked_plugin_retarget(self, project, member, stubbed_probe, tmp_path,
                                       monkeypatch):
        import os as _os

        home = Path(_os.environ["HERMES_HOME"])
        real_a = tmp_path / "plug_a"
        real_a.mkdir()
        (real_a / "plugin.toml").write_text("name: lnk\n", encoding="utf-8")
        real_b = tmp_path / "plug_b"
        real_b.mkdir()
        (real_b / "plugin.toml").write_text("name: lnk\n", encoding="utf-8")
        # Same directory times, so only the link itself proves the change.
        st = real_a.stat()
        _os.utime(real_b, ns=(st.st_atime_ns, st.st_mtime_ns))
        (home / "plugins" / "lnk").symlink_to(real_a)
        seeded_dir = _seed(project, member)
        _armed(monkeypatch)
        (home / "plugins" / "lnk").unlink()
        (home / "plugins" / "lnk").symlink_to(real_b)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([seeded_dir]),
                                   project_root=project)

    def test_symlinked_plugin_top_level_edit(self, project, member, stubbed_probe, tmp_path,
                                             monkeypatch):
        import os as _os

        home = Path(_os.environ["HERMES_HOME"])
        real = tmp_path / "plug_real"
        real.mkdir()
        (real / "plugin.toml").write_text("name: lnk\n", encoding="utf-8")
        (home / "plugins" / "lnk").symlink_to(real)
        seeded_dir = _seed(project, member)
        _armed(monkeypatch)
        # In-place, same-size edit: moves the file's mtime, not the dir's.
        (real / "plugin.toml").write_text("name: lnx\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([seeded_dir]),
                                   project_root=project)


class TestHermesIdentityInputs:
    """The running version (install stamp first, git second) gates member fitness.

    ``running_hermes_version`` reads ``install-stamp.json`` when present and git
    otherwise, so both are probe inputs for *every* root -- including project
    roots, whose memo otherwise records only the project's own git facts.
    """

    @staticmethod
    def _repo(monkeypatch, tmp_path, *, worktree=False):
        repo = tmp_path / "hre"
        repo.mkdir()
        main = tmp_path / "main"
        heads = main / ".git" / "refs" / "heads"
        heads.mkdir(parents=True)
        (main / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (heads / "main").write_text("0" * 40 + "\n", encoding="utf-8")
        if worktree:
            admin = main / ".git" / "worktrees" / "wt"
            admin.mkdir(parents=True)
            (admin / "commondir").write_text("../../../\n", encoding="utf-8")
            (admin / "HEAD").write_text("f" * 40 + "\n", encoding="utf-8")
            (repo / ".git").write_text(f"gitdir: {admin}\n", encoding="utf-8")
        else:
            (repo / ".git").mkdir()
            (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (repo / ".git" / "refs" / "heads").mkdir(parents=True)
            (repo / ".git" / "refs" / "heads" / "main").write_text("0" * 40 + "\n",
                                                                   encoding="utf-8")
        (repo / "hermes_constants.py").write_text("", encoding="utf-8")
        (repo / "hermes_yaml.py").write_text("", encoding="utf-8")
        (repo / "hermes_cli").mkdir()
        (repo / "hermes_cli" / "plugins_manifest.py").write_text("", encoding="utf-8")
        (repo / "hermes_cli" / "version_info.py").write_text("", encoding="utf-8")
        from pm import paths

        monkeypatch.setattr(paths, "repo_root", lambda: repo)
        return repo

    def test_install_stamp_edit(self, project, member, stubbed_probe, monkeypatch, tmp_path):
        repo = self._repo(monkeypatch, tmp_path)
        (repo / "install-stamp.json").write_text('{"v": 1}\n', encoding="utf-8")
        _seed(project, member)
        (repo / "install-stamp.json").write_text('{"v": 2}\n', encoding="utf-8")
        _bump(repo / "install-stamp.json")
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([member]),
                                   project_root=project)

    def test_hermes_retag_on_project_memo(self, project, member, stubbed_probe, monkeypatch,
                                          tmp_path):
        self._repo(monkeypatch, tmp_path)
        _seed(project, member)
        tags = tmp_path / "hre" / ".git" / "refs" / "tags"
        tags.mkdir(parents=True)
        (tags / "v9").write_text("0" * 40 + "\n", encoding="utf-8")
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([member]),
                                   project_root=project)

    def test_shallow_flip(self, project, member, stubbed_probe, monkeypatch, tmp_path):
        repo = self._repo(monkeypatch, tmp_path)
        _seed(project, member)
        (repo / ".git" / "shallow").write_text("0" * 40 + "\n", encoding="utf-8")
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([member]),
                                   project_root=project)

    def test_worktree_checkout_moves_own_head(self, project, member, stubbed_probe, monkeypatch,
                                              tmp_path):
        self._repo(monkeypatch, tmp_path, worktree=True)
        _seed(project, member)
        admin = tmp_path / "main" / ".git" / "worktrees" / "wt"
        (admin / "HEAD").write_text("e" * 40 + "\n", encoding="utf-8")
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([member]),
                                   project_root=project)

    def test_poisoned_git_head_keeps_writer_alive(self, project, member, stubbed_probe,
                                                  monkeypatch, tmp_path):
        repo = self._repo(monkeypatch, tmp_path)
        (repo / ".git" / "HEAD").write_bytes(b"\xff\xfe not utf-8")
        from pm import currency_memo

        currency_memo.write(project, extras=["e1"],
                            plugins_encoded={"kind": "members", "paths": [str(member)]},
                            facts_paths=[], member_sources={}, homes=[],
                            environment=project / "venv", current=True)
        assert _memo(project).is_file()

    def test_machine_flip(self, seeded, project, monkeypatch):
        memo = _memo(project)
        data = json.loads(memo.read_text(encoding="utf-8"))
        data.setdefault("sig", {})["machine"] = ["_plan9", "z80"]
        memo.write_text(json.dumps(data), encoding="utf-8")
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([seeded]),
                                   project_root=project)


class TestMemberSymlinkSpelling:
    """``members_stamp`` hashes symlink *bytes* (``os.readlink``), not just the target.

    A top-level link is already caught by the plugin-entry link lstat; a link
    deeper inside a member is seen only by the walk, which must therefore
    record the spelling itself.
    """

    def test_respelled_link_same_target(self, seeded, project, monkeypatch):
        sub = seeded / "sub"
        sub.mkdir()
        (sub / "t.py").write_text("1\n", encoding="utf-8")
        (sub / "l.py").symlink_to("t.py")
        _seed(project, seeded)  # record the link as written
        (sub / "l.py").unlink()
        (sub / "l.py").symlink_to("./t.py")  # same target stats, different spelling
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([seeded]),
                                   project_root=project)

    def test_broken_link_retarget(self, seeded, project, monkeypatch):
        sub = seeded / "sub"
        sub.mkdir()
        (sub / "l.py").symlink_to("missing_a")  # broken: stat is None on both sides
        _seed(project, seeded)
        (sub / "l.py").unlink()
        (sub / "l.py").symlink_to("missing_b")
        _armed(monkeypatch)
        with pytest.raises(AssertionError):
            client.venv_is_current(extras=["e1"], plugins=Members([seeded]),
                                   project_root=project)
