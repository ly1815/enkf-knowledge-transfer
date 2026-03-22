"""
run_all.py
==========
Runs the full pipeline in order. Each script auto-detects whether to load
from existing pkl files or run from scratch based on the current RUN_NAME.

Run from project root:
    poetry run python scripts/run_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Use non-interactive backend so plt.show() is a no-op in subprocesses
# (figures are still saved to disk)
BATCH_ENV = {**os.environ, "MPLBACKEND": "Agg"}

SCRIPTS = [
    "scripts/01_ensemble_tuning.py",
    "scripts/02_longterm_pred.py",
    "scripts/03_irregular.py",
    "scripts/04_sensitivity.py",
    "scripts/05_comparisons.py",
]

def run_script(script):
    print("\n" + "=" * 60)
    print(f"Running: {script}")
    print("=" * 60)
    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT, env=BATCH_ENV)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"\nFinished: {script} ({elapsed / 60:.1f} min)")

for script in SCRIPTS:
    run_script(script)

print("\n" + "=" * 60)
print("All scripts complete.")
print("=" * 60)
