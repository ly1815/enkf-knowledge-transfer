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


def has_results(pkl_dir=None):
    """Return True if the pkl directory exists and contains at least one .pkl file.

    Used to auto-detect whether to load from disk or run from scratch:
      LOAD_FROM_PKL = has_results()
    """
    folder = Path(pkl_dir) if pkl_dir is not None else _PKL_DIR
    return folder is not None and folder.exists() and any(folder.glob("*.pkl"))


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


# ─── Run notes ───────────────────────────────────────────────────────────────

def init_run_notes(results_dir):
    """Create a run_notes.txt in results_dir if it does not already exist."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    notes_file = results_dir / "run_notes.txt"
    if not notes_file.exists():
        notes_file.write_text(
            f"Run: {results_dir.name}\n"
            "=" * 40 + "\n\n"
            "Scripts run:\n"
            "  [ ] 01_ensemble_tuning\n"
            "  [ ] 02_longterm_pred\n"
            "  [ ] 03_irregular\n"
            "  [ ] 04_sensitivity\n"
            "  [ ] 05_comparisons\n\n"
            "Changes vs previous run:\n"
            "  - \n\n"
            "Notes:\n"
            "  - \n"
        )
        print(f"Created: {notes_file}")


# ─── Figure path helper ──────────────────────────────────────────────────────

def fig_path(fname: str, subdir: Path = None) -> Path:
    """Return the full Path for a figure file, creating the parent dir if needed."""
    folder = Path(subdir) if subdir is not None else _FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / fname
