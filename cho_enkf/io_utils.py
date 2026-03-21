"""
io_utils.py
===========
Centralised I/O helpers.  All pkl and figure paths are constructed from
the directories defined in config.py so that changing RUN_NAME is the only
thing needed to version a new experiment.
"""

import pickle
from pathlib import Path

from cho_enkf.config import PKL_DIR, FIG_DIR, SENS_PKL_DIR, SENS_FIG_DIR


def ensure_dirs():
    """Create all output directories if they do not already exist."""
    for d in (PKL_DIR, FIG_DIR, SENS_PKL_DIR, SENS_FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ─── Pickle helpers ──────────────────────────────────────────────────────────

def save_pkl(item, fname: str, subdir: Path = None):
    """Save *item* to PKL_DIR / fname (or subdir / fname)."""
    folder = subdir if subdir is not None else PKL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / fname
    with open(path, 'wb') as fh:
        pickle.dump(item, fh)
    print(f"Saved: {path}")


def load_pkl(fname: str, subdir: Path = None):
    """Load and return the object stored at PKL_DIR / fname (or subdir / fname)."""
    folder = subdir if subdir is not None else PKL_DIR
    path = folder / fname
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# Convenience wrappers for the sensitivity sub-folder
def save_sens_pkl(item, fname: str):
    save_pkl(item, fname, subdir=SENS_PKL_DIR)


def load_sens_pkl(fname: str):
    return load_pkl(fname, subdir=SENS_PKL_DIR)


# ─── Figure path helper ──────────────────────────────────────────────────────

def fig_path(fname: str, subdir: Path = None) -> Path:
    """Return the full Path for a figure file, creating the parent dir if needed."""
    folder = subdir if subdir is not None else FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / fname


def sens_fig_path(fname: str) -> Path:
    return fig_path(fname, subdir=SENS_FIG_DIR)
