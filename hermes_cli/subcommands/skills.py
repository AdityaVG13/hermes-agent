"""``hermes skills`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag, add_yes_flag


def _flag(parser, *names, help, **kw):
    parser.add_argument(*names, action="store_true", help=help, **kw)


# Registry sources, then provider filters (GitHub taps stored under source="github").
_SOURCE_CHOICES = [
    "all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub", "browse-sh",
    "nvidia", "openai", "anthropic", "huggingface", "voltagent", "gstack", "minimax"]


def build_skills_parser(subparsers, *, cmd_skills: Callable) -> None:
    """Attach the ``skills`` subcommand to ``subparsers``."""
    skills_parser = subparsers.add_parser(
        "skills", help="Search, install, configure, and manage skills",
        description="Search, install, inspect, audit, configure, and manage skills from skills.sh, well-known agent skill endpoints, GitHub, ClawHub, and other registries.",
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_trust = skills_subparsers.add_parser("trust",
        help="Trust a project so its repo-local skills (./.hermes/skills, ./.agents/skills) load")
    skills_trust.add_argument("path", nargs="?", default=None,
        help="Project root to trust (default: enclosing git checkout of cwd)")

    skills_untrust = skills_subparsers.add_parser(
        "untrust", help="Revoke project-skill trust for a repo")
    skills_untrust.add_argument("path", nargs="?", default=None,
        help="Project root to untrust (default: enclosing git checkout of cwd)")

    skills_browse = skills_subparsers.add_parser(
        "browse", help="Browse all available skills (paginated)")
    skills_browse.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    skills_browse.add_argument(
        "--size", type=int, default=20, help="Results per page (default: 20)")
    skills_browse.add_argument("--source", default="all", choices=_SOURCE_CHOICES,
        help="Filter by source or provider (e.g. nvidia, openai) (default: all)")

    skills_search = skills_subparsers.add_parser("search", help="Search skill registries")
    skills_search.add_argument("query", help="Search query")
    skills_search.add_argument("--source", default="all", choices=_SOURCE_CHOICES,
        help="Filter by source or provider (e.g. nvidia, openai)")
    skills_search.add_argument("--limit", type=int, default=25, help="Max results")
    add_json_flag(
        skills_search, "Output JSON instead of a table (full identifiers, scripting-friendly)")

    skills_install = skills_subparsers.add_parser("install", help="Install a skill")
    skills_install.add_argument("identifier",
        help="Skill identifier (e.g. openai/skills/skill-creator) or a direct HTTP(S) URL to a SKILL.md file",
    )
    skills_install.add_argument("--category", default="", help="Category folder to install into")
    skills_install.add_argument("--name", default="",
        help="Override the skill name (useful when installing from a URL whose SKILL.md has no `name:` frontmatter)",
    )
    _flag(skills_install, "--force", help="Install despite blocked scan verdict")
    add_yes_flag(skills_install, "Skip confirmation prompt (needed in TUI mode)")

    skills_inspect = skills_subparsers.add_parser(
        "inspect", help="Preview a skill without installing")
    skills_inspect.add_argument("identifier", help="Skill identifier")

    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.add_argument("--source", default="all", choices=["all", "hub", "builtin", "local"])
    _flag(skills_list, "--enabled-only",
        help="Hide disabled skills. Use with -p <profile> to see exactly "
        "which skills will load for that profile.")

    skills_check = skills_subparsers.add_parser(
        "check", help="Check installed hub skills for updates")
    skills_check.add_argument("name", nargs="?", help="Specific skill to check (default: all)")

    skills_health = skills_subparsers.add_parser(
        "health", help="Diagnose the skills library for hidden, orphaned, or stale state",
        description="Read-only diagnosis: same-name duplicates that make skills "
            "uninvocable, hub lock entries whose directory is truly missing, dead "
            "symlinks, husk directories, and a catalog cache that disagrees with the "
            "library. Exits non-zero when findings exist.")
    add_json_flag(skills_health, "Output JSON instead of text (scripting-friendly)")

    skills_fleet = skills_subparsers.add_parser(
        "fleet", help="Compare this machine's skills library against fleet machines",
        description="Cross-machine drift scan. Machines come from the "
            "skills.fleet.machines config block (name/host/root/kind); probes are "
            "loader-faithful (unlimited depth, frontmatter-name keyed, symlink-aware) "
            "and read-only. Exits non-zero when drift, diverged twins, or unreachable "
            "machines exist.")
    fleet_sub = skills_fleet.add_subparsers(dest="fleet_action")
    fleet_scan = fleet_sub.add_parser(
        "scan", help="Probe every machine and classify drift vs this machine")
    add_json_flag(fleet_scan, "Output JSON instead of text (scripting-friendly)")
    fleet_scan.add_argument("--only", nargs="+", metavar="NAME",
        help="Restrict the scan to these machine names from the config block")
    fleet_push = fleet_sub.add_parser(
        "push", help="Plan (and with --yes, apply) pushing skills to fleet machines",
        description="Dry-run by default. --stale pushes only provably-safe updates: "
            "skills missing on a machine, and copies whose SKILL.md hash matches a "
            "RECORDED prior Mac generation (the fleet manifest baseline) — i.e. the "
            "satellite is behind. Diverged-unknown, machine-owned, twin, symlinked, "
            "and policy-absent skills are refused. Writes stage-verify before swap, "
            "back the old copy up outside the skills root, and re-probe afterwards.")
    fleet_push.add_argument("--stale", action="store_true",
        help="Select missing + provably-behind skills on every machine")
    fleet_push.add_argument("--names", nargs="+", metavar="SKILL",
        help="Push these local skills (still refuses diverged/owned)")
    fleet_push.add_argument("--machines", nargs="+", metavar="NAME",
        help="Restrict the push to these machine names")
    fleet_push.add_argument("--yes", action="store_true",
        help="Apply the plan (default: dry-run)")
    add_json_flag(fleet_push, "Output JSON instead of text (scripting-friendly)")
    fleet_manifest = fleet_sub.add_parser(
        "manifest", help="Re-snapshot this machine's skills as the push baseline",
        description="Records per-skill per-file hashes into the fleet manifest "
            "(skills.fleet.baseline_manifest, default ~/.hermes/skills.fleet-manifest.json). "
            "This is the evidence store behind 'push --stale': a satellite copy "
            "matching a recorded generation is provably behind. Atomic write; "
            "prior generation kept at <path>.prev. `push --yes` refreshes it "
            "automatically after verified writes.")
    fleet_manifest.add_argument("--path", metavar="FILE",
        help="Manifest path (default: skills.fleet.baseline_manifest or ~/.hermes/skills.fleet-manifest.json)")
    add_json_flag(fleet_manifest, "Output JSON instead of text (scripting-friendly)")

    skills_update = skills_subparsers.add_parser("update", help="Update installed hub skills")
    skills_update.add_argument(
        "name", nargs="?", help="Specific skill to update (default: all outdated skills)")
    _flag(skills_update, "--force",
        help="Overwrite skills you have edited locally (they are skipped by default)")

    skills_audit = skills_subparsers.add_parser("audit", help="Re-scan installed hub skills")
    skills_audit.add_argument("name", nargs="?", help="Specific skill to audit (default: all)")
    _flag(skills_audit, "--deep", help="Run AST-level analysis on Python files (opt-in diagnostic)")

    skills_uninstall = skills_subparsers.add_parser(
        "uninstall", help="Remove a hub-installed skill")
    skills_uninstall.add_argument("name", help="Skill name to remove")
    add_yes_flag(skills_uninstall)

    skills_reset = skills_subparsers.add_parser("reset",
        help="Reset a bundled skill — clears 'user-modified' tracking so updates work again",
        description="Clear a bundled skill's entry from the sync manifest (~/.hermes/skills/.bundled_manifest) "
            "so future 'hermes update' runs stop marking it as user-modified. Pass --restore to also "
            "replace the current copy with the bundled version.")
    skills_reset.add_argument("name", help="Skill name to reset (e.g. google-workspace)")
    _flag(skills_reset, "--restore",
        help="Also delete the current copy and re-copy the bundled version")
    add_yes_flag(skills_reset, "Skip confirmation prompt when using --restore")

    skills_list_modified = skills_subparsers.add_parser(
        "list-modified", help="List bundled skills you've edited (which `hermes update` keeps)",
        description="Show the bundled skills whose local copy differs from the version last "
            "synced, i.e. the ones `hermes update` reports as user-modified and skips. "
            "Use `hermes skills diff <name>` to see changes and `hermes skills reset "
            "<name>` to resume updates.")
    add_json_flag(skills_list_modified, "Output the list as JSON")

    skills_diff = skills_subparsers.add_parser(
        "diff", help="Show how your copy of a bundled skill differs from the stock version",
        description="Print a unified diff between your local copy of a bundled skill and the "
            "current bundled (stock) version, so you can confirm what changed before "
            "running `hermes skills reset`.")
    skills_diff.add_argument("name", help="Skill name to diff (e.g. google-workspace)")

    skills_opt_out = skills_subparsers.add_parser(
        "opt-out", help="Stop bundled skills from being seeded into this profile",
        description="Write the .no-bundled-skills marker so the installer, "
            "`hermes update`, and any direct sync stop seeding bundled skills "
            "into the active profile. By default nothing already on disk is "
            "touched. Pass --remove to ALSO delete bundled skills that are "
            "unmodified (user-edited and hub/local skills are never removed).")
    _flag(skills_opt_out, "--remove", help="Also delete already-present unmodified bundled skills")
    add_yes_flag(skills_opt_out, "Skip confirmation prompt when using --remove")

    skills_opt_in = skills_subparsers.add_parser(
        "opt-in", help="Re-enable bundled-skill seeding (undo opt-out)",
        description="Remove the .no-bundled-skills marker so bundled skills are seeded "
            "again on the next `hermes update`. Pass --sync to re-seed now.")
    _flag(skills_opt_in, "--sync",
        help="Re-seed bundled skills immediately instead of waiting for update")

    skills_repair_official = skills_subparsers.add_parser(
        "repair-official", help="Backfill or restore official optional skills from repo source",
        description="Repair official optional skill provenance. By default, only backfills "
            "hub metadata for exact matches. Pass --restore to replace missing or "
            "mutated active copies from optional-skills/, moving existing copies to "
            "a restore backup first. Use name 'all' to repair every optional skill.")
    skills_repair_official.add_argument(
        "name", help="Official optional skill folder/frontmatter name, or 'all'")
    _flag(skills_repair_official, "--restore",
        help="Restore from official optional source, backing up existing matching copies")
    add_yes_flag(skills_repair_official, "Skip confirmation prompt when using --restore")

    skills_publish = skills_subparsers.add_parser("publish", help="Publish a skill to a registry")
    skills_publish.add_argument("skill_path", help="Path to skill directory")
    skills_publish.add_argument(
        "--to", default="github", choices=["github", "clawhub"], help="Target registry")
    skills_publish.add_argument(
        "--repo", default="", help="Target GitHub repo (e.g. openai/skills)")

    skills_snapshot = skills_subparsers.add_parser(
        "snapshot", help="Export/import skill configurations")
    snapshot_subparsers = skills_snapshot.add_subparsers(dest="snapshot_action")
    snap_export = snapshot_subparsers.add_parser("export", help="Export installed skills to a file")
    snap_export.add_argument("output", help="Output JSON file path (use - for stdout)")
    snap_import = snapshot_subparsers.add_parser(
        "import", help="Import and install skills from a file")
    snap_import.add_argument("input", help="Input JSON file path")
    _flag(snap_import, "--force", help="Force install despite caution verdict")

    skills_tap = skills_subparsers.add_parser("tap", help="Manage skill sources")
    tap_subparsers = skills_tap.add_subparsers(dest="tap_action")
    tap_subparsers.add_parser("list", help="List configured taps")
    tap_add = tap_subparsers.add_parser("add", help="Add a GitHub repo as skill source")
    tap_add.add_argument("repo", help="GitHub repo (e.g. owner/repo)")
    tap_rm = tap_subparsers.add_parser("remove", help="Remove a tap")
    tap_rm.add_argument("name", help="Tap name to remove")

    # config sub-action: interactive enable/disable
    skills_subparsers.add_parser(
        "config", help="Interactive skill configuration — enable/disable individual skills")

    skills_parser.set_defaults(func=cmd_skills)
