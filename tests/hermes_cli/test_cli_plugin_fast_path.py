"""Fast path for plugin-provided CLI commands: load only the owning plugin.

``hermes quota status --cached`` (the quota widget's poll, spawned every
refresh interval) used to pay the full plugin-discovery sweep — importing all
84 plugin modules, ~600ms — just to attach one argparse subcommand. When the
invoked top-level command is declared by exactly one enabled plugin's
``provides_commands`` manifest field, the CLI loads only that plugin and skips
the sweep; the sweep stays the fallback for every other case (unknown command,
ambiguous ownership, disabled owner, undeclared providers), so behavior
outside the fast path is unchanged.

Invariants pinned here (failure-first: written before the implementation):

1. ``provides_commands`` parses from plugin.yaml into the manifest (and is
   absent-safe for v1 manifests that do not declare it).
2. ``PluginManager.load_cli_plugin_for_command`` loads exactly the single
   owner, leaves ``_discovered`` False (a later full sweep still runs), and
   returns True.
3. No owner / ambiguous owners / gated-out owner -> False with zero plugin
   loads (caller falls back to the full sweep).
4. ``_register_plugin_cli_commands`` takes the fast path when offered and
   calls full ``discover_plugins()`` only when the fast path declines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _manifest(name: str, *, provides_commands: list | None = None, kind: str = "standalone"):
    from hermes_cli.plugins import PluginManifest

    fields = {}
    if provides_commands is not None:
        fields["provides_commands"] = provides_commands
    return PluginManifest(name=name, kind=kind, **fields)


# ── 1. manifest parsing ────────────────────────────────────────────────────


def test_provides_commands_parses_from_plugin_yaml(tmp_path: Path):
    from hermes_cli.plugins_manifest import parse_manifest_file

    plugin_dir = tmp_path / "fastcli"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        "name: fastcli\nversion: 1.0.0\nprovides_commands:\n  - fastcli\n", encoding="utf-8"
    )
    manifest = parse_manifest_file(plugin_dir / "plugin.yaml", plugin_dir, "user", "fastcli")
    assert getattr(manifest, "provides_commands", None) == ["fastcli"]


def test_provides_commands_absent_defaults_empty(tmp_path: Path):
    from hermes_cli.plugins_manifest import parse_manifest_file

    plugin_dir = tmp_path / "plaincli"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text("name: plaincli\nversion: 1.0.0\n", encoding="utf-8")
    manifest = parse_manifest_file(plugin_dir / "plugin.yaml", plugin_dir, "user", "plaincli")
    assert getattr(manifest, "provides_commands", None) == []


# ── 2/3. single-owner routing and fallbacks ────────────────────────────────


def _patched_manager(monkeypatch, manifests, *, disabled=None, enabled=None):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    loaded: list = []

    monkeypatch.setattr(PluginManager, "_collect_directory_manifests", lambda self: list(manifests))
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    monkeypatch.setattr(PluginManager, "_load_plugin", lambda self, m: loaded.append(m))
    monkeypatch.setattr(
        plugins_module, "_get_disabled_plugins", lambda: set(disabled or ())
    )
    monkeypatch.setattr(plugins_module, "_get_enabled_plugins", lambda: enabled)
    return manager, loaded


def test_single_owner_loads_only_that_plugin(monkeypatch):
    owner = _manifest("quota", provides_commands=["quota"])
    heavy = _manifest("heavy-browser", provides_commands=[])
    manager, loaded = _patched_manager(monkeypatch, [owner, heavy], enabled={"quota", "heavy-browser"})

    assert manager.load_cli_plugin_for_command("quota") is True
    assert [m.name for m in loaded] == ["quota"]
    assert manager._discovered is False  # a later full sweep must still run


def test_no_owner_falls_back(monkeypatch):
    heavy = _manifest("heavy-browser", provides_commands=[])
    manager, loaded = _patched_manager(monkeypatch, [heavy], enabled={"heavy-browser"})

    assert manager.load_cli_plugin_for_command("quota") is False
    assert loaded == []


def test_ambiguous_owners_fall_back(monkeypatch):
    a = _manifest("quota-a", provides_commands=["quota"])
    b = _manifest("quota-b", provides_commands=["quota"])
    manager, loaded = _patched_manager(monkeypatch, [a, b], enabled={"quota-a", "quota-b"})

    assert manager.load_cli_plugin_for_command("quota") is False
    assert loaded == []


def test_disabled_owner_falls_back(monkeypatch):
    owner = _manifest("quota", provides_commands=["quota"])
    manager, loaded = _patched_manager(monkeypatch, [owner], disabled={"quota"}, enabled=set())

    assert manager.load_cli_plugin_for_command("quota") is False
    assert loaded == []


def test_already_discovered_declines(monkeypatch):
    owner = _manifest("quota", provides_commands=["quota"])
    manager, loaded = _patched_manager(monkeypatch, [owner], enabled={"quota"})
    manager._discovered = True

    assert manager.load_cli_plugin_for_command("quota") is False
    assert loaded == []


# ── 4. CLI wiring ──────────────────────────────────────────────────────────


def _parser_choices() -> set:
    import argparse

    from hermes_cli.main import _build_cli_parser

    _parser, _subparsers = _build_cli_parser()
    return set(_subparsers.choices)


def test_register_plugin_cli_commands_uses_fast_path(monkeypatch):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.main import _register_plugin_cli_commands
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    full_sweeps = []
    monkeypatch.setattr(
        plugins_module, "discover_plugins", lambda *a, **k: full_sweeps.append(a)
    )
    monkeypatch.setattr(PluginManager, "load_cli_plugin_for_command", lambda self, cmd: True)
    manager._cli_commands["fastcli"] = {
        "name": "fastcli",
        "help": "fast cli",
        "description": "",
        "setup_fn": lambda p: None,
        "handler_fn": lambda args: 0,
    }
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.main._first_positional_argv", lambda: "fastcli")

    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    _register_plugin_cli_commands(subparsers)

    assert full_sweeps == []  # the sweep never ran
    assert "fastcli" in subparsers.choices


def test_register_plugin_cli_commands_falls_back_to_sweep(monkeypatch):
    from hermes_cli import plugins as plugins_module
    from hermes_cli.main import _register_plugin_cli_commands
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    full_sweeps = []
    monkeypatch.setattr(
        plugins_module, "discover_plugins", lambda *a, **k: full_sweeps.append(a)
    )
    monkeypatch.setattr(PluginManager, "load_cli_plugin_for_command", lambda self, cmd: False)
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.main._first_positional_argv", lambda: "fastcli")

    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    _register_plugin_cli_commands(subparsers)

    assert len(full_sweeps) == 1  # existing behavior preserved
