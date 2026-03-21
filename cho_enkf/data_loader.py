"""
data_loader.py
==============
Load CHO cell culture datasets from Excel files.
"""

from pathlib import Path
import pandas as pd


def load_datasets(data_dir: Path, dataset_files: list) -> dict:
    """
    Load all datasets from Excel files.

    Parameters
    ----------
    data_dir : Path
        Directory containing the .xlsx files.
    dataset_files : list[str]
        File names (without path) to load.

    Returns
    -------
    datasets : dict
        {dataset_name: {"schedule": df, "feed": df, "exp_meas": df}}
    """
    datasets = {}
    for filename in dataset_files:
        file_path = data_dir / filename
        name = file_path.stem
        try:
            datasets[name] = {
                "schedule": pd.read_excel(file_path, sheet_name="schedule"),
                "feed":     pd.read_excel(file_path, sheet_name="feed"),
                "exp_meas": pd.read_excel(file_path, sheet_name="exp_meas"),
            }
            print(f"Loaded: {name}")
        except Exception as exc:
            print(f"Error loading {name}: {exc}")
    return datasets
