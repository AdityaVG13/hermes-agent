"""verify: the project's KNOWN gate as a first-class verb.

Council ruling (2026-11-06): a verify that GUESSES the command is a false
green. Discovery must ride agent.coding_context.detect_project_facts (the
single source of truth behind the workspace snapshot's "Verify:" line);
a project with no known gate gets an honest refusal pointing at terminal.
"""

import textwrap

from tools.verify_tool import _handle_verify


def _make_project(tmp_path, makefile: str = ""):
    if makefile:
        (tmp_path / "Makefile").write_text(textwrap.dedent(makefile))
    return tmp_path


class TestRefusesWhenNoGate:
    def test_no_gate_refuses_and_names_terminal(self, tmp_path):
        _make_project(tmp_path)
        result = _handle_verify({"path": str(tmp_path)})
        assert "error" in result
        assert "terminal" in result["error"]
        # never guesses a default command
        assert "command" not in result

    def test_missing_root_is_an_error(self, tmp_path):
        result = _handle_verify({"path": str(tmp_path / "nope")})
        assert "error" in result


class TestRunsDiscoveredGate:
    def test_makefile_check_pass(self, tmp_path):
        _make_project(tmp_path, """
            check:
            \t@echo ALL GREEN
        """)
        result = _handle_verify({"path": str(tmp_path)})
        assert result["command"] == "make check"
        assert result["passed"] is True
        assert result["exit_code"] == 0
        assert "ALL GREEN" in result["output_tail"]
        assert result["commands_discovered"] == ["make check"]

    def test_makefile_check_fail_is_structured(self, tmp_path):
        _make_project(tmp_path, """
            check:
            \t@echo boom on stdout
            \t@echo boom on stderr >&2
            \t@exit 3
        """)
        result = _handle_verify({"path": str(tmp_path)})
        assert result["passed"] is False
        # GNU make exits 2 on recipe failure (it does not propagate the command's status)
        assert result["exit_code"] != 0
        assert "boom" in result["output_tail"]
        assert result["stdout_bytes"] > 0
        assert result["stderr_bytes"] > 0

    def test_command_index_selects_second_gate(self, tmp_path):
        _make_project(tmp_path, """
            test:
            \t@echo ran test
            lint:
            \t@echo ran lint
        """)
        result = _handle_verify({"path": str(tmp_path), "command_index": 1})
        assert result["command"] == "make lint"
        assert "ran lint" in result["output_tail"]

    def test_command_index_out_of_range_errors(self, tmp_path):
        _make_project(tmp_path, """
            check:
            \t@echo ok
        """)
        result = _handle_verify({"path": str(tmp_path), "command_index": 5})
        assert "error" in result


class TestSessionCwd:
    def test_default_path_uses_session_cwd_not_process_cwd(self, tmp_path, monkeypatch):
        # A gate lives in the SESSION's dir; the process cwd has no gate. If verify
        # resolved the default against the process cwd it would refuse here.
        gated = tmp_path / "gated"
        gated.mkdir()
        (gated / "Makefile").write_text("check:\n\t@echo session gate\n")
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        from tools.terminal_tool import get_session_cwd
        monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda k: str(gated))
        result = _handle_verify({}, task_id="sess-1")
        assert result.get("command") == "make check"
        assert "session gate" in result.get("output_tail", "")

    def test_falls_back_to_process_cwd_when_unrecorded(self, tmp_path, monkeypatch):
        gated = tmp_path / "gated"
        gated.mkdir()
        (gated / "Makefile").write_text("check:\n\t@echo fallback gate\n")
        monkeypatch.chdir(gated)
        monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda k: None)
        result = _handle_verify({}, task_id="sess-2")
        assert result.get("passed") is True


class TestTimeoutKillsGroup:
    def test_timeout_kills_grandchildren(self, tmp_path):
        # A gate that spawns a background child and outlives the timeout: the
        # whole process GROUP must die, or orphaned test workers linger.
        (tmp_path / "Makefile").write_text(
            "check:\n\t@python3 -c \"import subprocess,time; "
            "p=subprocess.Popen(['sleep','30']); "
            "open('child.pid','w').write(str(p.pid)); time.sleep(60)\"\n"
        )
        result = _handle_verify({"path": str(tmp_path), "timeout": 3})
        assert result.get("timeout") is True
        import os
        pid = int((tmp_path / "child.pid").read_text().strip())
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        assert not alive, f"grandchild {pid} survived the gate timeout"


class TestEnvelope:
    def test_output_is_bounded_with_honest_truncation(self, tmp_path):
        _make_project(tmp_path, """
            check:
            \t@python3 -c "print('x' * 50000)"
        """)
        result = _handle_verify({"path": str(tmp_path)})
        assert result["truncated"] is True
        assert len(result["output_tail"]) <= 6100
        assert result["stdout_bytes"] >= 50000

    def test_default_path_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_project(tmp_path, """
            check:
            \t@echo cwd gate
        """)
        result = _handle_verify({})
        assert result["passed"] is True
