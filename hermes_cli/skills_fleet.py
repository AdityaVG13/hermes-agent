"""``hermes skills fleet scan`` — cross-machine skills drift classification.

The 2026-11-03 fleet campaign reconciled Mac/Spark/Windows skill libraries by
hand; every failure class it hit is a probe-design constraint encoded here:

- **unlimited depth** — depth-capped probes missed ``mlops/models/*`` twins;
- **frontmatter-name keying** — dir names differ from declared names
  (``audiocraft`` vs ``audiocraft-audio-generation``), so dir-name keying
  reports present skills as missing;
- **symlink-following** — the loader walks with ``followlinks=True``; a probe
  that does not follow them sees a different library than the machine serves;
- **leaf rule** — a directory holding ``SKILL.md`` is a skill; ``SKILL.md``
  nested beneath one (fixtures) is never a skill;
- **encoding reality** — PowerShell ``>`` redirects emit UTF-16 LE, PS 5.1
  ``Remove-Item`` semantics differ from SSH POSIX sh, and AppleDouble ``._``
  litter rides along any Mac→Windows copy.

Read-only by construction: probes emit ``name|sha256|relpath`` rows and nothing
else; nothing on any machine is written outside the transport's scratch files.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

# The one rule set every side consumes. Mirrors ``agent.skill_utils``'s
# EXCLUDED_SKILL_DIRS plus the walker's leaf/followlinks semantics — the
# contract test in tests/hermes_cli/test_skills_fleet.py fails if the loader's
# prune list ever grows past this copy.
PROBE_RULES: dict[str, Any] = {
    "max_depth": None,          # unlimited: category nesting is data, not policy
    "follow_symlinks": True,    # os.walk(followlinks=True) on the loader side
    "prunes": (
        ".git", ".github", ".hub", ".archive", ".curator_backups", ".locks",
        ".hermes-dist", ".venv", "venv", "node_modules", "site-packages",
        "__pycache__", ".tox", ".nox", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", ".skillopt", ".skillloop-snapshots", "_sources", "cache",
    ),
}

_DONE = "DONE"


@dataclass(frozen=True)
class FleetRow:
    name: str      # frontmatter name (falls back to dir basename)
    sha: str       # sha256 hex of SKILL.md (lowercase)
    relpath: str   # skill dir relative to the machine's skills root, posix slashes
    symlink: bool


# ── local scan ───────────────────────────────────────────────────────────────

def _frontmatter_name(skill_md: str | os.PathLike) -> tuple[str, bool]:
    """(name, is_declared). Declared frontmatter name wins; dir basename is the
    fallback for SKILL.md files too malformed to carry one."""
    try:
        raw = open(skill_md, "rb").read()
    except OSError:
        return os.path.basename(os.path.dirname(skill_md)), False
    text = raw.decode("utf-8", errors="replace")
    try:
        from agent.skill_utils import parse_frontmatter
        fm, _body = parse_frontmatter(text)
        name = str(fm.get("name") or "").strip()
    except Exception:
        m = re.search(r"^name:\s*(\S.*)$", text, re.MULTILINE)
        name = m.group(1).strip().strip("\"'") if m else ""
    return (name, True) if name else (os.path.basename(os.path.dirname(str(skill_md))), False)


def _sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def local_rows(root) -> list[FleetRow]:
    """Loader-faithful rows for a local skills root: same walk semantics as the
    catalog scanner (unlimited depth, followlinks, prune list, SKILL.md dirs
    are leaves). Broken symlink dirs yield no row and no crash."""
    root = os.fspath(root)
    prunes = set(PROBE_RULES["prunes"])
    rows: list[FleetRow] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=PROBE_RULES["follow_symlinks"]):
        dirnames[:] = [d for d in dirnames if d not in prunes]
        if "SKILL.md" not in filenames:
            continue
        skill_md = os.path.join(dirpath, "SKILL.md")
        name, _declared = _frontmatter_name(skill_md)
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rows.append(FleetRow(name, _sha256(skill_md), rel, os.path.islink(dirpath)))
        dirnames[:] = []  # a dir holding SKILL.md is a leaf
    rows.sort(key=lambda r: r.name)
    return rows


# ── remote probe scripts (generated, never inline over ssh) ─────────────────

def render_bash_probe() -> str:
    """POSIX probe: takes <skills-root> <out-file>, writes name|sha|rel rows +
    DONE. Implements every PROBE_RULES member the hard way — no ``-maxdepth``
    (that cap is the fleet's original blind spot) and ``find -L`` for symlink
    following. The leaf rule is enforced by skipping any SKILL.md with a
    SKILL.md-bearing ancestor below the root."""
    names = " -o ".join(f"-name {d}" for d in PROBE_RULES["prunes"])
    return f"""#!/bin/sh
# usage: probe.sh <skills-root> <out-file>
ROOT="$1"; OUT="$2"
: > "$OUT" || exit 9
find -L "$ROOT" \\( {names} \\) -prune -o -type f -name SKILL.md -print 2>/dev/null | while read -r f; do
  d=$(dirname "$f")
  skip=""
  p="$d"
  while [ "$p" != "$ROOT" ] && [ "$p" != "/" ]; do
    p=$(dirname "$p")
    if [ "$p" != "$ROOT" ] && [ -f "$p/SKILL.md" ]; then skip=1; break; fi
  done
  [ -n "$skip" ] && continue
  nm=$(grep -m1 '^name:' "$f" | cut -d: -f2- | tr -d ' \\t"' | tr -d '\\r')
  [ -n "$nm" ] || nm=$(basename "$d")
  echo "$nm|$(sha256sum "$f" | head -c64)|${{d#"$ROOT"/}}" >> "$OUT"
done
echo "{_DONE}" >> "$OUT"
"""


_PS_PRUNES = ", ".join(f"'{d}'" for d in PROBE_RULES["prunes"])


def render_ps_probe() -> str:
    """Windows PowerShell 5.1 probe: -Root/-Out params, same row format. Manual
    recursion (``Get-ChildItem -Recurse`` cannot be told to follow reparse
    points on 5.1 and its ``-Depth`` caps are the Windows-side blind spot);
    reparse points are resolved into the visited-set key so link cycles
    terminate."""
    return f"""param([string]$Root, [string]$Out)
Set-Content -Path $Out -Value ""
$script:Prunes = @({_PS_PRUNES})
# Cycle protection is owned by the PARENT loop: it resolves each child's key
# (reparse points resolve to their target) and marks it before recursing. Walk
# itself never consults the set — a child pre-marked by its parent would
# otherwise look "already visited" and the whole walk would no-op.
$script:Seen = @{{}}
function Walk([string]$Dir) {{
  $skillMd = Join-Path $Dir 'SKILL.md'
  if (Test-Path -LiteralPath $skillMd -PathType Leaf) {{
    $rel = $Dir.Substring($Root.Length).TrimStart('\\', '/').Replace('\\', '/')
    $nm = ''
    $m = Select-String -LiteralPath $skillMd -Pattern '^name:\\s*(.+)$' | Select-Object -First 1
    if ($m) {{ $nm = $m.Matches.Groups[1].Value.Trim().Trim('"').Trim("'") }}
    if (-not $nm) {{ $nm = Split-Path -Leaf $Dir }}
    $h = (Get-FileHash -Algorithm SHA256 -LiteralPath $skillMd).Hash.ToLower()
    Add-Content -Path $Out -Value "$nm|$h|$rel"
    return
  }}
  foreach ($e in Get-ChildItem -LiteralPath $Dir -Force -ErrorAction SilentlyContinue) {{
    if (-not $e.PSIsContainer) {{ continue }}
    if ($script:Prunes -contains $e.Name) {{ continue }}
    $key = $e.FullName
    if ($e.Attributes -band [IO.FileAttributes]::ReparsePoint) {{
      if ($e.Target) {{ $key = [string]($e.Target | Select-Object -First 1) }}
    }}
    if ($script:Seen.ContainsKey($key)) {{ continue }}
    $script:Seen[$key] = $true
    Walk $e.FullName
  }}
}}
$Root = (Get-Item -LiteralPath $Root).FullName.TrimEnd('\\')
Walk $Root
Add-Content -Path $Out -Value "{_DONE}"
"""


def parse_probe_output(raw: bytes, require_sentinel: bool = True) -> list[FleetRow]:
    """Rows from either probe. UTF-16 LE BOM (PS ``>`` redirect), UTF-8 BOM, or
    plain UTF-8; CRLF tolerated; everything after DONE ignored; malformed
    lines skipped rather than fatal (partial output still classifies). A
    stream with no DONE sentinel means the probe died — that is a machine
    failure, never a library with zero skills."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    rows: list[FleetRow] = []
    saw_sentinel = False
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        if line == _DONE:
            saw_sentinel = True
            break
        parts = line.split("|")
        if len(parts) != 3 or not parts[0] or not parts[1]:
            continue
        rows.append(FleetRow(parts[0], parts[1].strip().lower(), parts[2], False))
    if require_sentinel and not saw_sentinel:
        raise ValueError("probe output missing DONE sentinel (probe died before finishing)")
    return rows


# ── classification ───────────────────────────────────────────────────────────

def _twins(rows: Iterable[FleetRow]) -> dict[str, dict[str, list[str]]]:
    by_name: dict[str, list[FleetRow]] = {}
    for r in rows:
        by_name.setdefault(r.name, []).append(r)
    benign: dict[str, list[str]] = {}
    diverged: dict[str, list[str]] = {}
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        paths = sorted(r.relpath for r in group)
        if len({r.sha for r in group}) == 1:
            benign[name] = paths
        else:
            diverged[name] = paths
    return {"benign": benign, "diverged": diverged}


def classify(local: list[FleetRow], machines: dict[str, list[FleetRow]],
             allowed_missing: Optional[dict[str, set[str]]] = None,
             ) -> dict[str, Any]:
    """Local rows vs each machine's rows → per-machine drift classes plus
    per-side twin detection. Twins whose copies disagree are CRITICAL (the
    hidden-skill class); benign twins and extras are informational."""
    allowed_missing = allowed_missing or {}
    critical: list[str] = []
    twins: dict[str, Any] = {"local": _twins(local)}
    for name, paths in twins["local"]["diverged"].items():
        critical.append(f"local: diverged twin '{name}' at {', '.join(paths)}")

    local_by_name: dict[str, list[FleetRow]] = {}
    for r in local:
        local_by_name.setdefault(r.name, []).append(r)
    local_names = set(local_by_name)

    out: dict[str, Any] = {"machines": {}, "twins": twins, "critical": critical}
    any_finding = bool(critical)
    for machine, rows in machines.items():
        m_twins = _twins(rows)
        twins[machine] = m_twins
        for name, paths in m_twins["diverged"].items():
            critical.append(f"{machine}: diverged twin '{name}' at {', '.join(paths)}")

        m_by_name: dict[str, list[FleetRow]] = {}
        for r in rows:
            m_by_name.setdefault(r.name, []).append(r)
        m_names = set(m_by_name)

        identical, diverged, relocated = [], [], {}
        for name in sorted(local_names & m_names):
            l_shas = {r.sha for r in local_by_name[name]}
            m_shas = {r.sha for r in m_by_name[name]}
            if l_shas & m_shas:
                identical.append(name)
            else:
                diverged.append(name)
            if name not in m_twins["benign"] and name not in m_twins["diverged"] \
                    and name not in twins["local"]["benign"] and name not in twins["local"]["diverged"]:
                l_path = local_by_name[name][0].relpath
                m_path = m_by_name[name][0].relpath
                if l_path != m_path:
                    relocated[name] = m_path

        allow = set(allowed_missing.get(machine, ()))
        missing = sorted(n for n in local_names - m_names if n not in allow)
        allowed = sorted(n for n in local_names - m_names if n in allow)
        extras = sorted(m_names - local_names)
        out["machines"][machine] = {
            "identical": identical, "diverged": diverged, "missing": missing,
            "allowed_missing": allowed, "extra": extras, "relocated": relocated,
            "counts": {"rows": len(rows), "names": len(m_names)},
        }
        if diverged or missing:
            any_finding = True

    out["ok"] = not any_finding and not critical
    return out


# ── orchestration ────────────────────────────────────────────────────────────

Transport = Callable[[dict, str], bytes]
_SSH_TIMEOUT = 600


def _remote_path(root: str) -> str:
    """A root for the remote sh command line that keeps tilde expansion: a
    leading ``~/`` becomes ``"$HOME"/rest`` (expands even quoted, safe against
    spaces); anything else is single-quoted verbatim."""
    import shlex
    if root.startswith("~/"):
        return '"$HOME"/' + shlex.quote(root[2:])
    return shlex.quote(root)


def _ssh_transport(machine: dict, script: str) -> bytes:
    """Ship the generated script as a file (never inline over ssh — quoting
    corruption ate two fleet probes), run it, pull the rows back."""
    import shlex
    import subprocess
    import tempfile
    host = machine["host"]
    kind = machine["kind"]
    if kind == "bash":
        local_ext, remote_script, remote_out = ".sh", "/tmp/hermes-fleet-probe.sh", "/tmp/hermes-fleet-probe.out"
        run = f"sh {remote_script} {_remote_path(machine['root'])} {remote_out}"
    else:
        local_ext, remote_script = ".ps1", "AppData/Local/Temp/hermes-fleet-probe.ps1"
        remote_out = "AppData/Local/Temp/hermes-fleet-probe.out"
        run = (f"powershell -NoProfile -ExecutionPolicy Bypass -File {remote_script}"
               f" -Root {shlex.quote(machine['root'])} -Out {remote_out}")

    def _ssh(cmd: str, timeout: int = _SSH_TIMEOUT) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", "-n", host, cmd], capture_output=True,
                              timeout=timeout, check=False)

    def _scp(src: str, dst: str) -> subprocess.CompletedProcess:
        return subprocess.run(["scp", "-q", src, dst], capture_output=True,
                              timeout=120, check=False)

    with tempfile.NamedTemporaryFile("w", suffix=local_ext, delete=False,
                                     encoding="utf-8", newline="\n") as fh:
        fh.write(script)
        local_script = fh.name
    import pathlib
    local_out = local_script + ".out"
    try:
        r = _scp(local_script, f"{host}:{remote_script}")
        if r.returncode != 0:
            raise RuntimeError(f"ssh: scp script to {host} failed: {r.stderr.decode(errors='replace').strip()}")
        r = _ssh(run)
        if r.returncode != 0:
            raise RuntimeError(f"ssh: probe on {host} failed: {r.stderr.decode(errors='replace').strip()[:400]}")
        r = _scp(f"{host}:{remote_out}", local_out)
        if r.returncode != 0:
            raise RuntimeError(f"ssh: scp rows from {host} failed: {r.stderr.decode(errors='replace').strip()}")
        return pathlib.Path(local_out).read_bytes()
    finally:
        pathlib.Path(local_script).unlink(missing_ok=True)
        pathlib.Path(local_out).unlink(missing_ok=True)
        _ssh(f"rm -f {remote_script} {remote_out}", timeout=60)


def run_scan(machines: list[dict], transport: Optional[Transport] = None,
             local_root=None) -> dict[str, Any]:
    """Probe every machine, classify against the local root. A machine whose
    transport fails is reported (``error``) and excluded from classification;
    the others still classify."""
    if local_root is None:
        from tools import skills_tool as st
        local_root = st.SKILLS_DIR
    transport = transport or _ssh_transport
    local = local_rows(local_root)

    rows_by_machine: dict[str, list[FleetRow]] = {}
    errors: dict[str, str] = {}
    for machine in machines:
        name, kind = machine.get("name", "?"), machine.get("kind", "")
        if kind not in ("bash", "powershell"):
            errors[name] = f"unsupported kind {kind!r} (expected bash or powershell)"
            continue
        script = render_bash_probe() if kind == "bash" else render_ps_probe()
        try:
            rows_by_machine[name] = parse_probe_output(transport(machine, script))
        except Exception as exc:  # transport failures must not sink the fleet
            errors[name] = str(exc) or exc.__class__.__name__

    allowed = {m["name"]: set(m.get("allow_missing", ())) for m in machines if "name" in m}
    report = classify(local, rows_by_machine, allowed_missing=allowed)
    for name, err in errors.items():
        report["machines"][name] = {"error": err}
    report["ok"] = bool(report["ok"]) and not errors
    report["local_counts"] = {"rows": len(local), "names": len({r.name for r in local})}
    return report


def _fleet_machines(only: Optional[list[str]] = None) -> list[dict]:
    from hermes_cli.config import cfg_get, load_config
    machines = cfg_get(load_config(), "skills", "fleet", "machines") or []
    if not isinstance(machines, list):
        return []
    if only:
        machines = [m for m in machines if isinstance(m, dict) and m.get("name") in only]
    return [m for m in machines if isinstance(m, dict)]


def do_fleet_scan(as_json: bool = False, only: Optional[list[str]] = None,
                  console: Optional[Any] = None) -> Optional[bool]:
    """CLI entry: ``hermes skills fleet scan``. Prints the report; returns
    False (exit 1) when any finding or machine error exists."""
    machines = _fleet_machines(only)
    if not machines:
        hint = ("No fleet machines configured. Add to config.yaml:\n"
                "  skills:\n"
                "    fleet:\n"
                "      machines:\n"
                "        - name: spark\n"
                "          host: spark-ts          # ssh alias\n"
                "          root: ~/.hermes/profiles/spark/skills\n"
                "          kind: bash\n"
                "        - name: windows\n"
                "          host: desktop-pe75qag\n"
                "          root: AppData/Local/hermes/skills\n"
                "          kind: powershell\n")
        (console or _console()).print(hint)
        return None

    report = run_scan(machines)
    if as_json:
        import json
        print(json.dumps(report, indent=1))
    else:
        _render(report, console or _console())
    return None if report["ok"] else False


def _render(report: dict, c) -> None:
    lc = report["local_counts"]
    c.print(f"[bold]Skills fleet[/] — local {lc['names']} names, "
            f"{len(report['machines'])} machine(s)\n")
    if report["critical"]:
        c.print("[bold red]CRITICAL — diverged twins (hidden-skill class):[/]")
        for line in report["critical"][:20]:
            c.print(f"  {line}")
        if len(report["critical"]) > 20:
            c.print(f"  [dim]… {len(report['critical']) - 20} more[/]")
        c.print()
    for name, m in report["machines"].items():
        if "error" in m:
            c.print(f"[bold red]{name}: unreachable[/] — {m['error'][:200]}")
            continue
        counts = m["counts"]
        c.print(f"[bold]{name}[/] — {counts['names']} names ({counts['rows']} rows): "
                f"{len(m['identical'])} identical, "
                f"[yellow]{len(m['diverged'])} diverged[/], "
                f"{len(m['missing'])} missing"
                + (f" (+{len(m['allowed_missing'])} allowed)" if m["allowed_missing"] else "")
                + f", {len(m['extra'])} extra"
                + (f", {len(m['relocated'])} relocated" if m["relocated"] else ""))
        for label, style in (("diverged", "yellow"), ("missing", "yellow")):
            if m[label]:
                c.print(f"  [{style}]{label} ({len(m[label])}):[/] " + ", ".join(m[label][:15])
                        + (" …" if len(m[label]) > 15 else ""))
        tw = report["twins"].get(name, {})
        for kind in ("diverged", "benign"):
            if tw.get(kind):
                pairs = "; ".join(f"{n} @ {', '.join(p)}" for n, p in list(tw[kind].items())[:8])
                c.print(f"  [{'bold red' if kind == 'diverged' else 'dim'}]"
                        f"{kind} twins ({len(tw[kind])}):[/] {pairs}")
    c.print("[green]fleet consistent[/]" if report["ok"] else
            "[bold red]fleet drift detected[/]")


def _console():
    from rich.console import Console
    return Console()
