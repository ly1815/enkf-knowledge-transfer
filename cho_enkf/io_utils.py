"""
io_utils.py
===========
Centralised I/O helpers.

Call set_dirs(pkl_dir, fig_dir) at the top of each script to configure
the output directories for that script.  All save/load/fig_path calls
then use those directories by default.
"""

import pickle
from pathlib import Path

# Mutable module-level directories — set by each script via set_dirs()
_PKL_DIR: Path = None
_FIG_DIR: Path = None


def set_dirs(pkl_dir, fig_dir):
    """Set the pickle and figure output directories for the current script."""
    global _PKL_DIR, _FIG_DIR
    _PKL_DIR = Path(pkl_dir)
    _FIG_DIR = Path(fig_dir)
    _PKL_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_dirs():
    """No-op kept for backward compatibility. Dirs are created by set_dirs()."""
    if _PKL_DIR is not None:
        _PKL_DIR.mkdir(parents=True, exist_ok=True)
    if _FIG_DIR is not None:
        _FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Pickle helpers ──────────────────────────────────────────────────────────

def save_pkl(item, fname: str, subdir: Path = None):
    """Save *item* to _PKL_DIR / fname (or subdir / fname)."""
    folder = Path(subdir) if subdir is not None else _PKL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / fname
    with open(path, 'wb') as fh:
        pickle.dump(item, fh)
    print(f"Saved: {path}")


def load_pkl(fname: str, subdir: Path = None):
    """Load and return the object stored at _PKL_DIR / fname (or subdir / fname)."""
    folder = Path(subdir) if subdir is not None else _PKL_DIR
    path = folder / fname
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ─── Figure path helper ──────────────────────────────────────────────────────

def fig_path(fname: str, subdir: Path = None) -> Path:
    """Return the full Path for a figure file, creating the parent dir if needed."""
    folder = Path(subdir) if subdir is not None else _FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / fname
