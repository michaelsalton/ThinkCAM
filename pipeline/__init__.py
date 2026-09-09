"""Offline event-pipeline tools: raw recording -> frames -> tracks -> COLMAP.

Every tool here is a module, run from the repo root as

    python -m pipeline.<tool> --help

rather than as a loose script, so that the cross-imports between them
(capture_probe -> accumulate_frames, frame_metrics -> view_sparse, ...) and
thinkcam.probe_worker's import of accumulate_frames all resolve the same way:
by package name, off the repo root. The GUI in thinkcam/ imports from here;
nothing here imports from thinkcam except constants.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def child_env():
    """Environment for a `python -m pipeline.x` subprocess.

    The tools re-invoke each other out of process (a determinism check needs
    two runs with different history; the GUI reads capture_probe's stdout as a
    progress line). The child keeps the parent's cwd, since the caller's
    relative --input path has to keep resolving, so this pins the repo root on
    PYTHONPATH rather than leaving `-m` to find the package via that cwd.
    """
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + prior if prior else "")
    return env
