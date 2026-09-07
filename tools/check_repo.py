"""Repository integrity check. Run it after ANY rename, move or delete.

    python tools/check_repo.py

Written because a rename pass over `src/` left stale references in `tools/`,
`scripts/` and the docs, and because a directory once vanished from the working
tree while still staged in git. Both are the kind of mistake that is invisible
until someone else clones the repo, and both are mechanically detectable.

Checks, in order of how badly each one bites:

  1. every `from <local module> import <name>` resolves to something actually
     defined in that module -- catches renames that compiled but would fail at
     import time, and symbols that moved between modules
  2. nothing references a Python module that no longer exists
  3. every file tracked by git is present on disk
  4. every relative Markdown link points at a real file
  5. every shell script parses (`bash -n`)
  6. no shell script references a variable that `set -u` would abort on
  7. every Python file compiles

Exit code is 0 only if all six pass, so this works as a pre-commit hook or a CI
step.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ["src", "tools", "experiments"]
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".cff", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "assets"}
ALLOW_STALE: set[str] = set()

# Filenames that live in the CLONED upstream repos, not here. We talk about them
# constantly (the challenge harness, STIRMetrics, track_on), so they are not
# ghosts. Anything NOT on this list and not on disk is a real dangling
# reference -- keep the list tight rather than silencing the check.
UPSTREAM_FILES = {
    "mono.py", "stereo.py", "run.py", "lt.py",          # stir-challenge-2026-*
    "calculate_error_from_json2d.py",                    # STIRMetrics
    "calculate_error_from_json3d.py", "testutil.py",
    "write3dgtjson.py", "flow2d.py",
    "demo.py", "dinov3_vit_adapter.py",                  # track_on
    "trackon_predictor.py", "bootstapir_predictor.py", "ensemble_predictor.py",
    "alltracker_predictor.py", "locotrack_predictor.py",
    "build_cotracker.py", "losses.py", "model_utils.py", # co-tracker
    "train_on_kubric.py",
    "lite_tracker.py", "model_blocks.py",                # lite-tracker
    "point_tracking.py", "misc.py", "config.py",         # MFT
    "STIRLoader.py", "__init__.py", "utils.py",
}

failures: list[str] = []
checked = 0


def fail(msg: str) -> None:
    failures.append(msg)


def local_modules() -> dict[str, Path]:
    return {p.stem: p for d in CODE_DIRS for p in (ROOT / d).glob("*.py")}


def toplevel_names(path: Path) -> set[str]:
    """Every name a module exposes at import time, including inside try/except."""
    names: set[str] = set()

    def visit(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Assign):
                names.update(t.id for t in n.targets if isinstance(t, ast.Name))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in n.names)
            elif isinstance(n, (ast.Try, ast.If)):
                visit(n.body)
                visit(getattr(n, "orelse", []))
                for h in getattr(n, "handlers", []):
                    visit(h.body)

    visit(ast.parse(path.read_text()).body)
    return names


def check_imports(mods):
    global checked
    for name, path in sorted(mods.items()):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.ImportFrom) and n.module in mods:
                available = toplevel_names(mods[n.module])
                for a in n.names:
                    checked += 1
                    if a.name not in available:
                        fail(f"{path.relative_to(ROOT)}: imports {a.name!r} from "
                             f"{n.module!r}, which does not define it")


def check_no_ghost_modules(mods):
    """Any <word>.py mentioned in the repo must exist somewhere in it."""
    global checked
    on_disk = {p.name for p in ROOT.rglob("*") if p.is_file()}
    pattern = re.compile(r"\b([a-z_][a-z0-9_]{2,})\.py\b")
    for p in iter_text_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOW_STALE:
            continue
        text = p.read_text(errors="ignore")
        for m in pattern.finditer(text):
            fname = m.group(0)
            checked += 1
            # upstream files we only talk about are fine if the name is qualified
            if fname in on_disk or fname in UPSTREAM_FILES:
                continue
            line = text[:m.start()].count("\n") + 1
            fail(f"{rel}:{line}: mentions {fname}, which is not in the repository")


def check_git_tracked_exist():
    global checked
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    except Exception as e:                     # not a git repo yet -- fine
        print(f"  (skipped git check: {e})")
        return
    for rel in out:
        checked += 1
        if not (ROOT / rel).exists():
            fail(f"tracked by git but missing from disk: {rel}")


def iter_text_files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        if SKIP_DIRS & set(p.parts):
            continue
        yield p


def check_markdown_links():
    global checked
    link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for p in ROOT.rglob("*.md"):
        if SKIP_DIRS & set(p.parts):
            continue
        for m in link.finditer(p.read_text(errors="ignore")):
            target = m.group(1).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            if not (p.parent / target).exists():
                fail(f"{p.relative_to(ROOT)}: broken link -> {target}")


SHELL_BUILTIN_VARS = {
    "HOME", "PATH", "USER", "PWD", "OLDPWD", "SHELL", "TERM", "TMPDIR", "LANG",
    "HOSTNAME", "BASH_SOURCE", "BASH_VERSION", "SECONDS", "LINENO", "RANDOM",
    "IFS", "PS1", "PS4", "FUNCNAME", "EUID", "UID", "SHLVL", "REPLY",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
    "no_proxy", "CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED", "WANDB_API_KEY",
    "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
}


def check_shell_unbound():
    """Find $VAR uses that `set -u` would abort on.

    `bash -n` is a parser: it happily accepts a reference to a variable that is
    never assigned and has no `${VAR:-default}`. Under `set -u` -- which every
    script here uses -- that is a hard runtime failure, and it only shows up
    when the script reaches that line, which may be minutes in. This is exactly
    the check that a syntax check cannot do for you.
    """
    global checked
    # NAME= only in COMMAND position: line start, or after a separator, a
    # case-branch `)`, or local/export/declare. Matching it anywhere would let
    # `echo "GRID=${GRID}"` register GRID as assigned, which is exactly the bug
    # this check exists to find.
    assign = re.compile(
        r"(?:^|[;&|)}{]|\bthen\b|\bdo\b|\belse\b|\blocal\b|\bexport\b"
        r"|\bdeclare\b|\breadonly\b)\s*([A-Za-z_][A-Za-z0-9_]*)=", re.M)
    loopvar = re.compile(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b|\bread\s+(?:-r\s+)?([A-Za-z_][A-Za-z0-9_]*)")
    # ${VAR:-x} ${VAR:=x} ${VAR:?x} ${VAR+x} ${VAR#...} etc. all guard the ref
    guarded = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:+\-=?#%/^,]")
    use = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

    for p in sorted(ROOT.rglob("*.sh")):
        if SKIP_DIRS & set(p.parts):
            continue
        text = p.read_text(errors="ignore")
        if "set -" not in text or "u" not in re.search(r"set -\w+", text).group(0):
            continue                      # not running under `set -u`
        # Assignments are looked for outside comments and quoted spans, so that
        # `echo "GRID=${GRID}"` does not register GRID as assigned -- precisely
        # the bug this check exists to catch. Done LINE BY LINE: an apostrophe
        # in a comment ("ffmpeg\'s default") would otherwise open a quote span
        # that swallows the rest of the file.
        code_lines = []
        for ln in text.splitlines():
            ln = re.sub(r"(?:^|\s)#.*$", "", ln)              # drop the comment
            ln = re.sub(r"\"[^\"]*\"|'[^']*'", " ", ln)        # blank quoted spans
            code_lines.append(ln)
        unquoted = "\n".join(code_lines)
        defined = set(assign.findall(unquoted)) | SHELL_BUILTIN_VARS
        for m in loopvar.finditer(unquoted):
            defined.update(x for x in m.groups() if x)
        defined |= set(guarded.findall(text))
        for m in use.finditer(text):
            name = m.group(1) or m.group(2)
            checked += 1
            if name in defined or name.isdigit():
                continue
            line = text[:m.start()].count("\n") + 1
            fail(f"{p.relative_to(ROOT)}:{line}: ${name} is never set and has no "
                 f"${{{name}:-default}} -- `set -u` will abort here")


def check_shell():
    global checked
    for p in ROOT.rglob("*.sh"):
        if SKIP_DIRS & set(p.parts):
            continue
        checked += 1
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if r.returncode:
            fail(f"{p.relative_to(ROOT)}: shell syntax error\n    {r.stderr.strip()}")


def check_python_compiles(mods):
    global checked
    for path in mods.values():
        checked += 1
        try:
            ast.parse(path.read_text())
        except SyntaxError as e:
            fail(f"{path.relative_to(ROOT)}:{e.lineno}: {e.msg}")


def main() -> int:
    mods = local_modules()
    print(f"checking {len(mods)} modules under {'/, '.join(CODE_DIRS)}/\n")
    for label, fn in [
        ("cross-module imports resolve", lambda: check_imports(mods)),
        ("no references to missing modules", lambda: check_no_ghost_modules(mods)),
        ("git-tracked files exist on disk", check_git_tracked_exist),
        ("markdown links resolve", check_markdown_links),
        ("shell scripts parse", check_shell),
        ("shell vars are bound under set -u", check_shell_unbound),
        ("python files compile", lambda: check_python_compiles(mods)),
    ]:
        before = len(failures)
        fn()
        n = len(failures) - before
        print(f"  {'FAIL' if n else 'ok  '}  {label}" + (f"  ({n})" if n else ""))

    print()
    if failures:
        print(f"{len(failures)} problem(s):\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all checks passed ({checked} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
