"""Disk memo for the dependency-currency probe (``venv_is_current``).

The probe re-derives "does the recorded dependency environment still match this
tree?" from scratch on every launch — by design, so a half-finished update or a
swapped checkout can never boot. But it is a pure function of a set of files
plus its arguments, so the verdict can be cached: this module records it
together with stat-facts of every input the probe read, and answers from the
memo only when the arguments match and every recorded fact is unchanged.
Anything else — a missing or corrupt memo, a touched file, a new entry in a
plugin home or the profiles root, an edit inside the pm/ package itself (which
guards against the checker's own code drifting from the walk rules recorded
here) — falls back to the full probe. ``HERMES_PM_CURRENCY_MEMO=0`` disables
both halves.

Only the standard library at module level: the reader runs in fresh processes
before any dependency from the probed environment is importable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MEMO_VERSION = 1
MEMO_NAME = "currency-memo.json"
DISABLE_ENV = "HERMES_PM_CURRENCY_MEMO"

# Mirrors pm.workspace._MEMBER_EXCLUDE plus its ".egg-info" rule so the walk
# covers exactly the files members_stamp hashes (plus directory structure,
# which the stamp also sees through os.walk). The pm/-tree facts in every
# signature invalidate all memos when the checker's own code changes.
_EXCLUDED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})

# Files under the project root that feed the stamp (upstream's ACTIVATION_INPUTS).
_REPO_INPUTS = ("uv.lock", "pyproject.toml")

_UNREADABLE = {"__unreadable__": True}


def disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("0", "false", "off")


def memo_path(project_root: Path) -> Path:
    from pm.environments import install_state_dir

    return install_state_dir(Path(project_root)) / MEMO_NAME


def _fact(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size, os.path.islink(path)]


def _walk(root: Path) -> dict:
    """relpath -> fact for every file under root, pruning the member excludes."""
    files: dict[str, object] = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _EXCLUDED_DIRS and not d.endswith(".egg-info"))
        rel = os.path.relpath(directory, root)
        # Directory names are structure too: a pruned or symlinked subdirectory
        # must not slip past just because it holds no hashed files.
        files[rel + "/"] = list(dirs)
        for name in names:
            files[os.path.join(rel, name)] = _fact(Path(directory) / name)
    return files


def _tree(root: Path) -> dict:
    return {"fact": _fact(root), "files": _walk(root)}


def _entries(dir_path: Path) -> dict:
    """name -> shallow facts for a directory of entries (plugin homes, profiles).

    Catches entries appearing or vanishing, and edits to their top-level files
    (manifests, pyprojects, profile identity markers). An unreadable directory
    never compares equal to anything, including a recorded empty listing.
    """
    out: dict[str, object] = {}
    try:
        names = sorted(os.listdir(dir_path))
    except FileNotFoundError:
        return {}
    except OSError:
        return dict(_UNREADABLE)
    for name in names:
        path = dir_path / name
        if path.is_dir() and not path.is_symlink():
            try:
                top = sorted(os.listdir(path))
            except OSError:
                out[name] = dict(_UNREADABLE)
                continue
            out[name] = {
                "dirs": sorted(n for n in top if (path / n).is_dir()),
                "files": {n: _fact(path / n) for n in top if (path / n).is_file()},
            }
        else:
            out[name] = {"file": _fact(path)}
    return out


def signature(project_root: Path, *, facts_paths, member_sources: dict, homes,
              environment: Path) -> dict:
    """Stat-facts of every input the probe reads. Writer side; needs warm pm."""
    from pm import paths
    from pm.environments import dependency_home_root
    from pm.extras import declined_extras_path

    root = Path(project_root)
    declined = declined_extras_path(root)
    lockfile = paths.lockfile_path()
    return {
        "pm": _tree(Path(__file__).resolve().parent),
        "repo": {name: _fact(root / name) for name in _REPO_INPUTS},
        "lockfile": {"path": str(lockfile), "fact": _fact(lockfile)},
        "facts": {str(path): _fact(Path(path)) for path in facts_paths},
        "declined": {"path": str(declined), "fact": _fact(declined)},
        "profiles": _entries(dependency_home_root() / "profiles"),
        "repo_plugins": _entries(paths.repo_root() / "plugins"),
        "homes": [{"path": str(home), "config": _fact(Path(home) / "config.yaml"),
                   "plugins": _entries(Path(home) / "plugins")} for home in homes],
        "members": {"identities": sorted(str(k) for k in member_sources),
                    "sources": {str(source): _tree(Path(source))
                                for source in sorted(set(member_sources.values()), key=str)}},
        "env": {"path": str(environment), "pyvenv": _fact(Path(environment) / "pyvenv.cfg")},
    }


def write(project_root: Path, *, extras, plugins_encoded, facts_paths, member_sources: dict,
          homes, environment: Path, current: bool) -> None:
    if disabled():
        return
    sig = signature(project_root, facts_paths=facts_paths, member_sources=member_sources,
                    homes=homes, environment=environment)
    payload = {"version": MEMO_VERSION, "current": bool(current),
               "key": {"extras": list(extras or []), "plugins": plugins_encoded},
               "sig": sig}
    path = memo_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read(project_root: Path, *, extras, plugins_encoded):
    """The recorded verdict, or None whenever anything at all is in doubt."""
    if disabled():
        return None
    try:
        memo = json.loads(memo_path(project_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(memo, dict) or memo.get("version") != MEMO_VERSION:
        return None
    if not isinstance(memo.get("current"), bool):
        return None
    if memo.get("key") != {"extras": list(extras or []), "plugins": plugins_encoded}:
        return None
    sig = memo.get("sig")
    if not isinstance(sig, dict):
        return None
    try:
        return memo["current"] if _unchanged(Path(project_root), sig) else None
    except Exception:
        return None


def _unchanged(root: Path, sig: dict) -> bool:
    from pm import paths
    from pm.environments import dependency_home_root

    if _tree(Path(__file__).resolve().parent) != sig.get("pm"):
        return False
    if {name: _fact(root / name) for name in _REPO_INPUTS} != sig.get("repo"):
        return False
    facts = sig.get("facts")
    if not isinstance(facts, dict) or {path: _fact(Path(path)) for path in facts} != facts:
        return False
    lockfile = sig.get("lockfile")
    if not isinstance(lockfile, dict) or _fact(Path(lockfile["path"])) != lockfile.get("fact"):
        return False
    declined = sig.get("declined")
    if not isinstance(declined, dict) or _fact(Path(declined["path"])) != declined.get("fact"):
        return False
    if _entries(dependency_home_root() / "profiles") != sig.get("profiles"):
        return False
    if _entries(paths.repo_root() / "plugins") != sig.get("repo_plugins"):
        return False
    homes = sig.get("homes")
    if not isinstance(homes, list):
        return False
    for home in homes:
        if (not isinstance(home, dict)
                or _fact(Path(home["path"]) / "config.yaml") != home.get("config")
                or _entries(Path(home["path"]) / "plugins") != home.get("plugins")):
            return False
    members = sig.get("members")
    # Staged selections carry identity→source mappings, but those identities
    # are pinned by the memo key (plugin_inputs.encode) and resolved ones map
    # each dir to itself, so walking the recorded sources is complete here.
    sources = members.get("sources") if isinstance(members, dict) else None
    if not isinstance(sources, dict):
        return False
    if {source: _tree(Path(source)) for source in sources} != sources:
        return False
    env = sig.get("env")
    if not isinstance(env, dict) or _fact(Path(env["path"]) / "pyvenv.cfg") != env.get("pyvenv"):
        return False
    return True
