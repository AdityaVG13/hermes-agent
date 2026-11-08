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
