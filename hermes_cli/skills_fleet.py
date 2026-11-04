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


def _iter_skill_dirs(root):
    """Yield (dirpath, rel, islink) for every loader-visible skill dir. This is
    THE walk: local_rows (scan/push) and build_manifest (baseline) share it so
    the evidence store can never disagree with what the tools compare."""
    root = os.fspath(root)
    prunes = set(PROBE_RULES["prunes"])
    for dirpath, dirnames, filenames in os.walk(root, followlinks=PROBE_RULES["follow_symlinks"]):
        dirnames[:] = [d for d in dirnames if d not in prunes]
        if "SKILL.md" not in filenames:
            continue
        skill_md = os.path.join(dirpath, "SKILL.md")
        name, _declared = _frontmatter_name(skill_md)
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        yield dirpath, name, rel, os.path.islink(dirpath)
        dirnames[:] = []  # a dir holding SKILL.md is a leaf


def local_rows(root) -> list[FleetRow]:
    """Loader-faithful rows for a local skills root: same walk semantics as the
    catalog scanner (unlimited depth, followlinks, prune list, SKILL.md dirs
    are leaves). Broken symlink dirs yield no row and no crash."""
    rows = [FleetRow(name, _sha256(os.path.join(d, "SKILL.md")), rel, islink)
            for d, name, rel, islink in _iter_skill_dirs(root)]
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

Transport = Callable[..., bytes]
_SSH_TIMEOUT = 600


def _remote_path(root: str) -> str:
    """A root for the remote sh command line that keeps tilde expansion: a
    leading ``~/`` becomes ``"$HOME"/rest`` (expands even quoted, safe against
    spaces); anything else is single-quoted verbatim."""
    import shlex
    if root.startswith("~/"):
        return '"$HOME"/' + shlex.quote(root[2:])
    return shlex.quote(root)


def _ssh_transport(machine: dict, script: str, *, payload: Optional[bytes] = None,
                   entries: Optional[bytes] = None, phase: str = "scan") -> bytes:
    """Ship the generated script as a file (never inline over ssh — quoting
    corruption ate two fleet probes), run it, pull the rows back.

    ``phase="write"`` additionally ships ``payload`` (the bundle tgz) and
    ``aux["entries"]`` (the per-skill manifest) and returns the writer's
    status stream instead of probe rows."""
    import shlex
    import subprocess
    import tempfile
    import pathlib
    host = machine["host"]
    kind = machine["kind"]
    bundle_remote = entries_remote = ""
    reads_stdout = False
    if phase == "write":
        if kind == "bash":
            script_remote = "/tmp/hermes-fleet-writer.sh"
            entries_remote, bundle_remote = "/tmp/hermes-fleet-entries.txt", "/tmp/hermes-fleet-bundle.tgz"
            out_remote = "/tmp/hermes-fleet-write.out"
            run = (f"sh {script_remote} {_remote_path(machine['root'])} {bundle_remote} "
                   f"{entries_remote} {out_remote}")
        else:
            script_remote = "AppData/Local/Temp/hermes-fleet-writer.ps1"
            entries_remote = "AppData/Local/Temp/hermes-fleet-entries.txt"
            bundle_remote = "AppData/Local/Temp/hermes-fleet-bundle.tgz"
            out_remote = "AppData/Local/Temp/hermes-fleet-write.out"
            run = (f"powershell -NoProfile -ExecutionPolicy Bypass -File {script_remote}"
                   f" -Root {shlex.quote(machine['root'])} -Bundle {bundle_remote}"
                   f" -Entries {entries_remote} -Out {out_remote}")
    elif phase == "read":
        reads_stdout = True
        if kind == "bash":
            script_remote = "/tmp/hermes-fleet-reader.sh"
            entries_remote = "/tmp/hermes-fleet-read-entries.txt"
            run = (f"ROOT={_remote_path(machine['root'])} ENTRIES={entries_remote}"
                   f" sh {script_remote}")
        else:
            script_remote = "AppData/Local/Temp/hermes-fleet-reader.ps1"
            entries_remote = "AppData/Local/Temp/hermes-fleet-read-entries.txt"
            run = (f"powershell -NoProfile -ExecutionPolicy Bypass -File {script_remote}"
                   f" -Root {shlex.quote(machine['root'])} -Entries {entries_remote}")
        out_remote = script_remote  # unused; bodies return on stdout
    elif kind == "bash":
        script_remote, out_remote = "/tmp/hermes-fleet-probe.sh", "/tmp/hermes-fleet-probe.out"
        run = f"sh {script_remote} {_remote_path(machine['root'])} {out_remote}"
    else:
        script_remote = "AppData/Local/Temp/hermes-fleet-probe.ps1"
        out_remote = "AppData/Local/Temp/hermes-fleet-probe.out"
        run = (f"powershell -NoProfile -ExecutionPolicy Bypass -File {script_remote}"
               f" -Root {shlex.quote(machine['root'])} -Out {out_remote}")

    def _ssh(cmd: str, timeout: int = _SSH_TIMEOUT) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", "-n", host, cmd], capture_output=True,
                              timeout=timeout, check=False)

    def _scp(src: str, dst: str) -> subprocess.CompletedProcess:
        return subprocess.run(["scp", "-q", src, dst], capture_output=True,
                              timeout=120, check=False)

    ext = ".sh" if kind == "bash" else ".ps1"
    with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False,
                                     encoding="utf-8", newline="\n") as fh:
        fh.write(script)
        local_script = fh.name
    local_out = local_script + ".out"
    files: list[tuple[str, str]] = [(local_script, f"{host}:{script_remote}")]
    local_bundle = local_entries = None
    if payload is not None:
        import pathlib as _p
        local_bundle = local_script + ".tgz"
        _p.Path(local_bundle).write_bytes(payload)
        files.append((local_bundle, f"{host}:{bundle_remote}"))
    if entries is not None:
        import pathlib as _p
        local_entries = local_script + ".entries"
        _p.Path(local_entries).write_bytes(entries)
        files.append((local_entries, f"{host}:{entries_remote}"))
    try:
        for src, dst in files:
            r = _scp(src, dst)
            if r.returncode != 0:
                raise RuntimeError(f"ssh: scp to {host} failed: {r.stderr.decode(errors='replace').strip()}")
        r = _ssh(run)
        if r.returncode != 0:
            raise RuntimeError(f"ssh: {phase} on {host} failed: {r.stderr.decode(errors='replace').strip()[:400]}")
        if reads_stdout:
            return r.stdout
        r = _scp(f"{host}:{out_remote}", local_out)
        if r.returncode != 0:
            raise RuntimeError(f"ssh: scp output from {host} failed: {r.stderr.decode(errors='replace').strip()}")
        import pathlib
        return pathlib.Path(local_out).read_bytes()
    finally:
        import pathlib
        pathlib.Path(local_script).unlink(missing_ok=True)
        pathlib.Path(local_out).unlink(missing_ok=True)
        pathlib.Path(local_bundle).unlink(missing_ok=True) if local_bundle else None
        pathlib.Path(local_entries).unlink(missing_ok=True) if local_entries else None
        if kind == "bash":
            _ssh(f"rm -f {script_remote} {out_remote}"
                 + (f" {bundle_remote}" if payload is not None else "")
                 + (f" {entries_remote}" if entries is not None else ""),
                 timeout=60)
        else:  # Windows ssh lands in cmd.exe — no rm; use PowerShell
            remote_files = [script_remote]
            if not reads_stdout:
                remote_files.append(out_remote)
            if payload is not None:
                remote_files.append(bundle_remote)
            if entries is not None:
                remote_files.append(entries_remote)
            ps_list = ",".join(f"'{f}'" for f in remote_files)
            _ssh('powershell -NoProfile -Command "Remove-Item -Force '
                 f'-ErrorAction SilentlyContinue {ps_list}"',
                 timeout=60)


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

    rows_by_machine, errors = _probe_machines(machines, transport)
    allowed = {m["name"]: set(m.get("allow_missing", ())) for m in machines if "name" in m}
    report = classify(local, rows_by_machine, allowed_missing=allowed)
    for name, err in errors.items():
        report["machines"][name] = {"error": err}
    report["ok"] = bool(report["ok"]) and not errors
    report["local_counts"] = {"rows": len(local), "names": len({r.name for r in local})}
    return report


def _probe_machines(machines: list[dict], transport: Transport
                    ) -> tuple[dict[str, list[FleetRow]], dict[str, str]]:
    """Run the read-only probe on every machine. Returns (rows, errors); a
    machine in ``errors`` is absent from ``rows`` and vice versa."""
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
    return rows_by_machine, errors


# ── push: plan, bundle, write protocol ───────────────────────────────────────

def _unsafe_root(root: str) -> bool:
    """A machine root that resolves to a filesystem root, the home dir itself,
    or emptiness — writing there is the rmtree-the-world hazard, refused."""
    return str(root).strip().rstrip("/") in ("", ".", "/", "~")


def _is_category_target(rel: str, sat_paths: set[str]) -> bool:
    """True when ``rel`` would land on a satellite dir that already holds
    other skills — creating there nests skills under a future leaf."""
    return any(p.startswith(rel + "/") for p in sat_paths)


def plan_push(local_root, rows_by_machine: dict[str, Optional[list[FleetRow]]],
              machines: list[dict], baseline: dict[str, str],
              selector: tuple[str, list[str]],
              generations: Optional[dict[str, set[str]]] = None) -> dict[str, Any]:
    """Decide per machine what a push would do, refusing everything risky.

    ``--stale`` semantics are evidence-based: a satellite copy may be REPLACED
    only when its sha matches a RECORDED prior Mac generation — ``generations``
    (name → set of shas, normally every generation the fleet manifest kept)
    or, absent that, the single ``baseline`` snapshot. Anything else that
    differs is diverged-unknown and refused; owned names are refused outright;
    creates refuse category-bucket targets and policy-absent names.
    """
    gens = generations if generations is not None else {
        n: {s} for n, s in baseline.items() if s}
    local = local_rows(local_root)
    refusals: list[str] = []
    local_twins = _twins(local)
    if local_twins["diverged"]:
        names = ", ".join(sorted(local_twins["diverged"]))
        refusals.append(f"local: diverged twin(s) {names} — push refused")

    by_name: dict[str, list[FleetRow]] = {}
    for r in local:
        by_name.setdefault(r.name, []).append(r)
    twin_names = set(local_twins["benign"]) | set(local_twins["diverged"])
    symlink_names = {r.name for r in local if r.symlink}

    mode, names = selector
    if mode == "names":
        universe = list(dict.fromkeys(names))
        for n in universe:
            if n not in by_name:
                refusals.append(f"{n}: not local")
    else:
        universe = sorted(by_name)

    plan: dict[str, Any] = {"machines": {}, "refusals": refusals}
    errors_present = False
    for machine in machines:
        mname = machine.get("name", "?")
        root = str(machine.get("root", ""))
        if _unsafe_root(root):
            refusals.append(f"{mname}: unsafe root {root!r}")
            plan["machines"][mname] = {"error": "unsafe root"}
            errors_present = True
            continue
        rows = rows_by_machine.get(mname)
        if rows is None:
            refusals.append(f"{mname}: machine unreachable")
            plan["machines"][mname] = {"error": "probe failed"}
            errors_present = True
            continue

        owned = set(machine.get("owned", ()))
        allow_missing = set(machine.get("allow_missing", ()))
        sat_paths = {r.relpath for r in rows}
        sat_by_name: dict[str, list[FleetRow]] = {}
        for r in rows:
            sat_by_name.setdefault(r.name, []).append(r)

        out: dict[str, Any] = {"create": [], "replace": [], "skip_identical": [],
                               "refused": {}, "targets": {}, "new_shas": {},
                               "replace_reasons": {}}
        for n in universe:
            group = by_name.get(n)
            if not group:
                continue  # already refused as not-local
            lrow = sorted(group, key=lambda r: r.relpath)[0]
            if n in twin_names:
                out["refused"][n] = "twin (multiple local copies)"
                refusals.append(f"{mname}: {n} twin on local side — refusing")
                continue
            if n in symlink_names:
                out["refused"][n] = "symlinked skill dir"
                refusals.append(f"{mname}: {n} symlinked skill dir — refusing")
                continue
            sat_rows = sat_by_name.get(n)
            if not sat_rows:
                if n in allow_missing:
                    out["refused"][n] = "absent by policy (allow_missing)"
                    refusals.append(f"{mname}: {n} absent by policy")
                    continue
                if _is_category_target(lrow.relpath, sat_paths):
                    out["refused"][n] = f"target {lrow.relpath} is a satellite category"
                    refusals.append(f"{mname}: {n} target {lrow.relpath} is a category — refusing")
                    continue
                out["create"].append(n)
                out["targets"][n] = lrow.relpath
                out["new_shas"][n] = lrow.sha
                continue
            sat = sorted(sat_rows, key=lambda r: r.relpath)[0]
            if sat.sha == lrow.sha:
                out["skip_identical"].append(n)
                continue
            if n in owned:
                out["refused"][n] = "owned by machine"
                refusals.append(f"{mname}: {n} owned by machine — refusing")
                continue
            if sat.sha in gens.get(n, set()):
                out["replace"].append(n)  # provably behind a recorded Mac generation
                out["targets"][n] = sat.relpath  # replace in place, wherever it lives
                out["new_shas"][n] = lrow.sha
                out["replace_reasons"][n] = "behind (recorded generation)"
            else:
                out["refused"][n] = "diverged (matches no recorded Mac generation)"
                refusals.append(f"{mname}: {n} diverged — refusing")
        plan["machines"][mname] = out

    plan["ok"] = not refusals and not errors_present
    return plan


_JUNK_FILENAMES = {".DS_Store"}


def _is_junk_member(name: str) -> bool:
    return name in _JUNK_FILENAMES or name.startswith("._")


def build_bundle(skills: dict[str, str | os.PathLike]) -> bytes:
    """One tgz for the whole push: top-level member ``<name>/`` per skill.
    Symlinks anywhere in a skill dir are refused (hub policy — a link would
    copy its target and leak it to the agent); AppleDouble and Finder litter
    never ships."""
    import io
    import tarfile
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for name, path in skills.items():
            path = os.fspath(path)
            for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
                for d in dirnames:
                    if os.path.islink(os.path.join(dirpath, d)):
                        raise ValueError(f"{name}: contains symlink {d!r} — refusing to bundle")
                for fn in filenames:
                    if os.path.islink(os.path.join(dirpath, fn)):
                        raise ValueError(f"{name}: contains symlink {fn!r} — refusing to bundle")
                    if _is_junk_member(fn):
                        continue
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, path).replace(os.sep, "/")
                    tf.add(full, arcname=f"{name}/{rel}")
    return buffer.getvalue()


def render_bash_writer() -> str:
    """POSIX write side. Args: <skills-root> <bundle.tgz> <entries-file>
    <status-out>. Entries: ``name|relpath|newsha``. Per entry: extract once,
    verify the STAGED SKILL.md hash before touching the target (a corrupt
    transfer must never reach the library), refuse category buckets and
    non-dirs, back the old copy up OUTSIDE the skills root, then swap.
    Status lines ``OK|REFUSED|FAILED <name> [reason]`` + DONE."""
    return f"""#!/bin/sh
# usage: writer.sh <skills-root> <bundle.tgz> <entries> <status-out>
ROOT="$1"; BUNDLE="$2"; ENTRIES="$3"; OUT="$4"
: > "$OUT" || exit 9
TS=$(date -u +%Y%m%d-%H%M%S)
BK="$(dirname "$ROOT")/skills-fleet-backup-$TS"
STAGE="$(dirname "$ROOT")/.fleet-stage-$$"
cleanup() {{ rm -rf "$STAGE"; }}
trap cleanup EXIT INT TERM
mkdir -p "$STAGE" || {{ echo "FAILED setup stage-dir" >> "$OUT"; echo DONE >> "$OUT"; exit 0; }}
tar -xzf "$BUNDLE" -C "$STAGE" 2>>"$OUT" || {{ echo "FAILED setup extract" >> "$OUT"; echo DONE >> "$OUT"; exit 0; }}
printf '%s' "" > /dev/null
while IFS='|' read -r nm rel newsha; do
  [ -n "$nm" ] || continue
  TARGET="$ROOT/$rel"
  if [ -e "$TARGET" ] && [ ! -d "$TARGET" ]; then
    echo "REFUSED $nm target-not-a-dir" >> "$OUT"; continue
  fi
  if [ -d "$TARGET" ] && [ ! -f "$TARGET/SKILL.md" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
    echo "REFUSED $nm category-target" >> "$OUT"; continue
  fi
  if [ ! -f "$STAGE/$nm/SKILL.md" ]; then
    echo "FAILED $nm missing-in-bundle" >> "$OUT"; continue
  fi
  got=$(sha256sum "$STAGE/$nm/SKILL.md" | head -c64)
  if [ "$got" != "$newsha" ]; then
    echo "FAILED $nm stage-hash-mismatch" >> "$OUT"; continue
  fi
  if [ -d "$TARGET" ]; then
    mkdir -p "$BK" && mv "$TARGET" "$BK/$nm" || {{ echo "FAILED $nm backup" >> "$OUT"; continue; }}
  fi
  mkdir -p "$(dirname "$TARGET")" && mv "$STAGE/$nm" "$TARGET" \\
    && echo "OK $nm" >> "$OUT" || echo "FAILED $nm swap" >> "$OUT"
done < "$ENTRIES"
echo "{_DONE}" >> "$OUT"
"""


def render_ps_writer() -> str:
    """Windows write side, PS 5.1 + tar.exe (bsdtar on Win10+). Same protocol
    and guards as the bash writer."""
    return f"""param([string]$Root, [string]$Bundle, [string]$Entries, [string]$Out)
Set-Content -Path $Out -Value ""
$Root = (Get-Item -LiteralPath $Root).FullName.TrimEnd('\\')
$Ts = Get-Date -AsUTC -Format "yyyyMMdd-HHmmss"
$Bk = Join-Path (Split-Path -Parent $Root) "skills-fleet-backup-$Ts"
$Stage = Join-Path (Split-Path -Parent $Root) ".fleet-stage-$PID"
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
try {{
  tar -xzf $Bundle -C $Stage 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) {{ Add-Content $Out "FAILED setup extract"; Add-Content $Out "{_DONE}"; return }}
  foreach ($line in Get-Content -LiteralPath $Entries) {{
    if (-not $line.Trim()) {{ continue }}
    $parts = $line.Split('|')
    if ($parts.Count -lt 3) {{ continue }}
    $nm = $parts[0]; $rel = $parts[1]; $newsha = $parts[2]
    $target = Join-Path $Root ($rel -replace '/', '\\')
    if ((Test-Path -LiteralPath $target) -and -not (Test-Path -LiteralPath $target -PathType Container)) {{
      Add-Content $Out "REFUSED $nm target-not-a-dir"; continue }}
    if ((Test-Path -LiteralPath $target -PathType Container) -and -not (Test-Path -LiteralPath (Join-Path $target 'SKILL.md'))) {{
      $kids = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue)
      if ($kids.Count -gt 0) {{ Add-Content $Out "REFUSED $nm category-target"; continue }}
    }}
    $staged = Join-Path $Stage ($nm + '\\SKILL.md')
    if (-not (Test-Path -LiteralPath $staged -PathType Leaf)) {{
      Add-Content $Out "FAILED $nm missing-in-bundle"; continue }}
    $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $staged).Hash.ToLower()
    if ($got -ne $newsha) {{ Add-Content $Out "FAILED $nm stage-hash-mismatch"; continue }}
    if (Test-Path -LiteralPath $target) {{
      New-Item -ItemType Directory -Force -Path $Bk | Out-Null
      Move-Item -LiteralPath $target (Join-Path $Bk $nm) -Force
    }}
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    try {{
      Move-Item -LiteralPath (Join-Path $Stage $nm) $target -Force
      Add-Content $Out "OK $nm"
    }} catch {{ Add-Content $Out "FAILED $nm swap" }}
  }}
}} finally {{
  Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
}}
Add-Content -Path $Out -Value "{_DONE}"
"""


def parse_write_output(raw: bytes, expected: Optional[set] = None) -> dict[str, tuple[str, str]]:
    """``OK|REFUSED|FAILED <name> [reason…]`` lines → ``{name: (status, reason)}``.
    With ``expected`` (the planned skill names), a status line whose second
    token is NOT a planned name is keyed under ``""`` — a GLOBAL setup failure
    the caller must fail on. Missing DONE sentinel = writer died → ValueError
    (never a silent no-op)."""
    text = raw.decode("utf-8", errors="replace")
    out: dict[str, tuple[str, str]] = {}
    saw_sentinel = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == _DONE:
            saw_sentinel = True
            break
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0] in ("OK", "REFUSED", "FAILED"):
            if expected is not None and parts[1] not in expected:
                reason = " ".join(parts[1:])
                out[""] = (parts[0], reason)
                continue
            out.setdefault(parts[1], (parts[0], parts[2] if len(parts) > 2 else ""))
    if not saw_sentinel:
        raise ValueError("writer output missing DONE sentinel (writer died before finishing)")
    return out


def render_bash_reader() -> str:
    """Read-phase script: marker-delimited SKILL.md bodies for the names the
    plan refused as diverged — the lossless proof fetches evidence, never
    trusts the plan. ``$ROOT``/``$ENTRIES`` come from the transport."""
    return r'''
while IFS='|' read -r n rel || [ -n "$n" ]; do
  [ -n "$n" ] || continue
  echo "===BODY==$n"
  cat "$ROOT/$rel/SKILL.md" 2>/dev/null
  echo
done < "$ENTRIES"
echo "DONE"
'''


def render_ps_reader() -> str:
    return """param([string]$Root, [string]$Entries)
foreach ($line in Get-Content -LiteralPath $Entries) {
    if (-not $line.Trim()) { continue }
    $n, $rel = $line.Split('|', 2)
    Write-Output ("===BODY==" + $n)
    try { Get-Content -Raw -LiteralPath (Join-Path $Root ($rel -replace '/', '\\')) } catch {}
    Write-Output ""
}
Write-Output "DONE"
"""


def parse_body_output(raw: bytes) -> dict[str, str]:
    """``===BODY==name`` delimited bodies → {name: body}. Markers delimit;
    there is NO text sentinel — bodies are arbitrary markdown that may
    legitimately contain a bare ``DONE`` line."""
    out: dict[str, str] = {}
    cur, buf = None, []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("===BODY=="):
            if cur is not None:
                out[cur] = "\n".join(buf).replace("\r", "")
            cur, buf = line[len("===BODY=="):].strip(), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).replace("\r", "")
    return out


def _lossless_pass(report: dict[str, Any], machines: list[dict],
                   local_paths: dict[str, str], transport) -> None:
    """Upgrade ``diverged`` refusals the satellite BODY disproves: if every
    line of the satellite's SKILL.md exists in Mac's copy, replacement cannot
    lose satellite data — move it to replace with reason ``lossless-subset``.
    One unique satellite line keeps it refused."""
    for machine in machines:
        mname, kind = machine.get("name", "?"), machine.get("kind", "")
        m = report["machines"].get(mname)
        if not m or "error" in m:
            continue
        diverged = [n for n, why in m["refused"].items()
                    if "matches no recorded Mac generation" in why]
        if not diverged:
            continue
        rows = {r.name: r for r in (report.get("_rows", {}).get(mname) or [])}
        wanted = [n for n in diverged if n in rows]
        if not wanted:
            continue
        entries = "\n".join(f"{n}|{rows[n].relpath}" for n in wanted) + "\n"
        script = render_ps_reader() if kind == "powershell" else render_bash_reader()
        try:
            raw = transport(machine, script, phase="read", entries=entries.encode())
        except Exception as exc:
            report["refusals"].append(f"{mname}: lossless read failed: {exc}")
            continue
        bodies = parse_body_output(raw)
        for n in diverged:
            body = bodies.get(n, "")
            if not body.strip():
                continue
            try:
                mac_text = open(os.path.join(local_paths[n], "SKILL.md"),
                                encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            sat_lines = set(body.rstrip("\n").split("\n"))
            mac_lines = set(mac_text.replace("\r\n", "\n").rstrip("\n").split("\n"))
            if sat_lines - mac_lines:
                m["refused"][n] = "diverged (satellite holds unique lines)"
                continue
            del m["refused"][n]
            report["refusals"] = [r for r in report["refusals"]
                                  if r != f"{mname}: {n} diverged — refusing"]
            m["replace"].append(n)
            m["targets"][n] = rows[n].relpath
            m["new_shas"][n] = sf_sha_of_file(os.path.join(local_paths[n], "SKILL.md"))
            m["replace_reasons"][n] = "lossless-subset"
    report["ok"] = not report["refusals"] and not any(
        "error" in m for m in report["machines"].values())


def sf_sha_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_push(machines: list[dict], local_root=None, transport=None,
             baseline: Optional[dict[str, str]] = None,
             selector: tuple[str, list[str]] = ("stale", []),
             apply: bool = False,
             generations: Optional[dict[str, set[str]]] = None,
             lossless: bool = False) -> dict[str, Any]:
    """Scan → plan → (apply: bundle + write + post-verify). Without ``apply``
    the transport only ever sees read-only probe traffic."""
    if local_root is None:
        from tools import skills_tool as st
        local_root = st.SKILLS_DIR
    transport = transport or _ssh_transport
    local = local_rows(local_root)
    local_paths: dict[str, str] = {}
    for r in local:
        local_paths.setdefault(r.name, os.path.join(os.fspath(local_root), r.relpath))

    rows_by_machine, errors = _probe_machines(machines, transport)
    rows_in: dict[str, Optional[list[FleetRow]]] = dict(rows_by_machine)
    for name, err in errors.items():
        rows_in[name] = None
    plan = plan_push(local_root, rows_in, machines, baseline or {}, selector,
                     generations=generations)
    report: dict[str, Any] = {**plan, "applied": False}
    if lossless:
        report["_rows"] = rows_by_machine
        _lossless_pass(report, machines, local_paths, transport)
        report.pop("_rows", None)

    if not apply:
        report["ok"] = bool(report.get("ok"))
        return report
    report["applied"] = True

    for machine in machines:
        mname, kind = machine.get("name", "?"), machine.get("kind", "")
        m = report["machines"].get(mname, {})
        todo = list(m.get("create", [])) + list(m.get("replace", []))
        if not todo or "error" in m:
            m.setdefault("applied", [])
            m.setdefault("verified", [])
            m.setdefault("remote_refused", [])
            m.setdefault("remote_failed", [])
            continue
        try:
            bundle = build_bundle({n: local_paths[n] for n in todo})
        except ValueError as exc:
            report["refusals"].append(f"{mname}: {exc}")
            m["applied"], m["verified"] = [], []
            m["remote_refused"], m["remote_failed"] = [], []
            report["ok"] = False
            continue
        entries = "".join(f"{n}|{m['targets'][n]}|{m['new_shas'][n]}\n" for n in todo)
        writer = render_bash_writer() if kind == "bash" else render_ps_writer()
        try:
            raw = transport(machine, writer, payload=bundle, phase="write",
                            entries=entries.encode())
            statuses = parse_write_output(raw, expected=set(todo))
        except Exception as exc:
            report["refusals"].append(f"{mname}: write transport failed: {exc}")
            m["applied"], m["verified"] = [], []
            m["remote_refused"], m["remote_failed"] = [], []
            report["ok"] = False
            continue
        m["applied"] = [n for n in todo if statuses.get(n, ("", ""))[0] == "OK"]
        m["remote_refused"] = [n for n in todo if statuses.get(n, ("", ""))[0] == "REFUSED"]
        m["remote_failed"] = [n for n in todo if statuses.get(n, ("", ""))[0] == "FAILED"]
        if "" in statuses:  # global setup failure: nothing ran, the whole write died
            m["remote_failed"] = list(todo)
            m["applied"] = []
            m["verified"] = []
            report["refusals"].append(
                f"{mname}: writer failed globally ({statuses[''][1] or 'setup'}) — no skill written")
            report["ok"] = False
            continue

        # post-verify: re-probe and require every applied name to now match Mac
        probe = render_bash_probe() if kind == "bash" else render_ps_probe()
        try:
            vrows = parse_probe_output(transport(machine, probe, phase="verify"))
        except Exception as exc:
            report["refusals"].append(f"{mname}: verify probe failed: {exc}")
            m["verified"] = []
            report["ok"] = False
            continue
        vsha = {r.name: r.sha for r in vrows}
        m["verified"] = [n for n in m["applied"] if vsha.get(n) == m["new_shas"][n]]
        for n in m["applied"]:
            if n not in m["verified"]:
                report["refusals"].append(f"{mname}: {n} post-verify mismatch")
        if m["remote_refused"] or m["remote_failed"] or len(m["verified"]) != len(m["applied"]):
            report["ok"] = False
    return report


def _load_baseline(path=None) -> dict[str, str]:
    """name → recorded SKILL.md sha from the fleet manifest (the ``--stale``
    behind-test's evidence store). Missing/unreadable → empty (stale degrades
    to creates-only)."""
    import json
    if path is None:
        path = _baseline_path()
    try:
        data = json.loads(open(path, encoding="utf-8").read())
    except (OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    skills = data.get("skills") or {}
    if isinstance(skills, dict):
        items = list(skills.items())
    else:
        items = []
        for e in skills:  # tolerate both [{name:…}, …] and [[name, {…}], …]
            if isinstance(e, dict):
                items.append((e.get("name"), e))
            elif isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[1], dict):
                items.append((e[0], e[1]))
    for name, entry in items:
        if isinstance(entry, dict):
            sha = (entry.get("files") or {}).get("SKILL.md")
            if name and sha:
                out[name] = sha
    return out


def _load_generations(path=None) -> dict[str, set[str]]:
    """name → every recorded SKILL.md sha across the manifest's kept
    generations (current + ``.prev``). The honest behind-test evidence: a
    satellite matching ANY recorded generation is provably behind, not
    diverged-unknown."""
    import json
    if path is None:
        path = _baseline_path()
    gens: dict[str, set[str]] = {}
    for candidate in (path, str(path) + ".prev"):
        try:
            data = json.loads(open(candidate, encoding="utf-8").read())
        except (OSError, ValueError):
            continue
        skills = data.get("skills") or {}
        if isinstance(skills, dict):
            items = list(skills.items())
        else:
            items = []
            for e in skills:
                if isinstance(e, dict):
                    items.append((e.get("name"), e))
                elif isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[1], dict):
                    items.append((e[0], e[1]))
        for name, entry in items:
            if isinstance(entry, dict):
                sha = (entry.get("files") or {}).get("SKILL.md")
                if name and sha:
                    gens.setdefault(name, set()).add(sha)
    return gens


def _baseline_path() -> str:
    from hermes_cli.config import cfg_get, load_config
    return cfg_get(load_config(), "skills", "fleet", "baseline_manifest") \
        or os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
                        "skills.fleet-manifest.json")


def build_manifest(local_root) -> dict[str, Any]:
    """Snapshot THIS machine's skills as the recorded generation: per-skill
    per-file sha256s, loader-faithful (same walk as scan/push). This file is
    the evidence store `push --stale` testifies against — a satellite hash
    matching a recorded generation is provably behind, anything else is
    diverged-unknown and refused. An empty/missing root REFUSES rather than
    write a baseline that would brand every satellite copy diverged."""
    import datetime
    utc = datetime.UTC
    skills: list[list[Any]] = []
    for dirpath, name, rel, islink in _iter_skill_dirs(local_root):
        files: dict[str, str] = {}
        for fdir, _dirs, fns in os.walk(dirpath, followlinks=False):
            for fn in fns:
                if _is_junk_member(fn):
                    continue
                full = os.path.join(fdir, fn)
                frel = os.path.relpath(full, dirpath).replace(os.sep, "/")
                try:
                    files[frel] = _sha256(full)
                except OSError:
                    continue
        files = dict(sorted(files.items()))
        skills.append([name, {"relpath": rel, "symlink": islink, "files": files}])
    if not skills:
        raise ValueError(f"refusing to snapshot: no loader-visible skills under {local_root}")
    skills.sort(key=lambda pair: pair[0])
    return {
        "generator": "hermes skills fleet manifest",
        "generated_at": datetime.datetime.now(utc).isoformat(timespec="seconds"),
        "root": os.fspath(local_root),
        "skills": skills,
    }


def write_manifest(path, manifest: dict[str, Any]) -> None:
    """Atomic (tmp + rename) with the prior generation kept at ``<path>.prev``
    — the baseline is the only evidence of 'behind', so its history is never
    silently discarded."""
    import json
    import tempfile
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.exists(path):
        prev = path + ".prev"
        try:
            os.replace(path, prev)
        except OSError:
            pass
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".",
                               prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def refresh_baseline(local_root=None, path=None) -> int:
    """Regenerate the baseline from current disk; returns the skill count."""
    if local_root is None:
        from tools import skills_tool as st
        local_root = st.SKILLS_DIR
    manifest = build_manifest(local_root)
    write_manifest(path or _baseline_path(), manifest)
    return len(manifest["skills"])


def do_fleet_manifest(path: Optional[str] = None, as_json: bool = False,
                      console: Optional[Any] = None) -> Optional[bool]:
    """CLI entry: ``hermes skills fleet manifest`` — record this machine's
    current generation as the push baseline."""
    from tools import skills_tool as st
    target = path or _baseline_path()
    manifest = build_manifest(st.SKILLS_DIR)
    write_manifest(target, manifest)
    n = len(manifest["skills"])
    files = sum(len((e or {}).get("files", {})) for _name, e in manifest["skills"])
    if as_json:
        import json
        print(json.dumps({"path": target, "skills": n, "files": files,
                          "generated_at": manifest["generated_at"]}, indent=1))
    else:
        (console or _console()).print(
            f"Baseline refreshed: [bold]{n}[/] skills, {files} files hashed → {target}\n"
            f"(prior generation kept at {target}.prev)")
    return True


def do_fleet_push(as_json: bool = False, stale: bool = False,
                  names: Optional[list[str]] = None, machines_filter: Optional[list[str]] = None,
                  apply: bool = False, lossless: bool = False,
                  console: Optional[Any] = None) -> Optional[bool]:
    """CLI entry: ``hermes skills fleet push``. Dry-run by default; writes
    only with ``--yes``. Returns False on any refusal or failed entry."""
    if not stale and not names:
        (console or _console()).print("Specify a selector: --stale or --names N [N…]")
        return None
    machines = _fleet_machines(machines_filter)
    if not machines:
        (console or _console()).print("No fleet machines configured (skills.fleet.machines).")
        return None
    baseline = _load_baseline()
    report = run_push(machines, baseline=baseline,
                      selector=("stale", []) if stale else ("names", list(names or [])),
                      apply=apply, generations=_load_generations(),
                      lossless=lossless)
    if apply and report.get("applied") and any(
            m.get("verified") for m in report["machines"].values()
            if isinstance(m, dict)):
        # Record the just-pushed generation as the new baseline, so future Mac
        # edits make satellite copies provably-behind instead of unknown.
        try:
            n = refresh_baseline()
            report["baseline_refreshed"] = n
        except ValueError as exc:
            report["refusals"].append(f"baseline refresh refused: {exc}")
    if as_json:
        import json
        print(json.dumps(report, indent=1))
    else:
        _render_push(report, console or _console(), apply)
    return None if report.get("ok") else False


def _render_push(report: dict, c, apply: bool) -> None:
    head = "Skills fleet push — APPLYING" if apply else "Skills fleet push — DRY RUN (add --yes to apply)"
    c.print(f"[bold]{head}[/]\n")
    for r in report.get("refusals", [])[:25]:
        c.print(f"  [yellow]refused[/] {r}")
    if len(report.get("refusals", [])) > 25:
        c.print(f"  [dim]… {len(report['refusals']) - 25} more refusals[/]")
    for name, m in report.get("machines", {}).items():
        if "error" in m:
            c.print(f"[bold red]{name}: {m['error']}[/]")
            continue
        c.print(f"[bold]{name}[/]: create {len(m.get('create', []))}, "
                f"replace {len(m.get('replace', []))}, "
                f"identical {len(m.get('skip_identical', []))}, "
                f"refused {len(m.get('refused', {}))}")
        if apply:
            c.print(f"  applied {len(m.get('applied', []))}, verified {len(m.get('verified', []))}, "
                    f"remote-refused {len(m.get('remote_refused', []))}, "
                    f"remote-failed {len(m.get('remote_failed', []))}")
    c.print("[green]plan clean[/]" if report.get("ok") else "[bold red]refusals/failures present — nothing unsafe was written[/]")


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
