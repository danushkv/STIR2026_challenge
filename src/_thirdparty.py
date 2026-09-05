"""Locate the cloned upstream repositories.

WHY THIS EXISTS: every module that needs an upstream repo used to find it by
walking one directory up from itself -- `Path(__file__).parent.parent / "X"`.
That worked only because the code lived at `<clones>/files/`, one level under
the directory holding the clones. This repository is one level deeper, so the
hop no longer lands anywhere and the failure is a bare ModuleNotFoundError from
inside the upstream package, which says nothing about the real cause.

Resolution order, first hit wins:
  1. the repo-specific environment variable  (e.g. $LITETRACKER_ROOT)
  2. $THIRDPARTY_ROOT/<dirname>              (what the scripts export)
  3. ../<dirname> and ../../<dirname> relative to this file

None of this changes what gets imported when a clone is found at the old
relative location -- it only adds ways to say where it is.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def find_repo(dirnames, env_var, marker, what):
    """Return the first existing clone root, or raise with everything tried.

    dirnames: directory name(s) the clone may have, in preference order
    env_var:  variable naming the clone root directly
    marker:   a path INSIDE the clone that must exist, to reject empty dirs
    """
    if isinstance(dirnames, str):
        dirnames = [dirnames]
    cands = []
    if os.environ.get(env_var):
        cands.append(Path(os.environ[env_var]))
    for d in dirnames:
        if os.environ.get("THIRDPARTY_ROOT"):
            cands.append(Path(os.environ["THIRDPARTY_ROOT"]) / d)
        cands += [_HERE.parent / d, _HERE.parent.parent / d, _HERE / d]
    for c in cands:
        if (c / marker).exists():
            return c
    raise ImportError(
        f"{what} not found. Set {env_var} to the cloned repo, or THIRDPARTY_ROOT "
        f"to the directory holding the clones (see docs/ENVIRONMENTS.md).\nTried:\n  "
        + "\n  ".join(str(c) for c in cands)
    )


def add_to_syspath(*paths):
    """Prepend paths to sys.path, preserving the given order, skipping dupes."""
    for p in reversed([str(p) for p in paths]):
        if p not in sys.path:
            sys.path.insert(0, p)
