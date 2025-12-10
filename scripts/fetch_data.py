#!/usr/bin/env python3
"""
fetch datasets from kaggle and save to ../data/
"""

import os
import subprocess
from pathlib import Path

# create data directory
data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(exist_ok=True)

# dataset identifiers from kaggle
datasets = {
    "housing": "yasserh/housing-prices-dataset",
    "spotify": "maharshipandya/-spotify-tracks-dataset",
    "students": "nikhil7280/student-performance-multiple-linear-regression"
}

print("downloading datasets to:", data_dir.absolute())

for name, dataset_id in datasets.items():
    print(f"\nfetching {name}...")
    try:
        # download and unzip to data directory
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(data_dir), "--unzip"],
            check=True
        )
        print(f"✓ {name} downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"failed to download {name}: {e}")
    except FileNotFoundError:
        print("kaggle CLI not found. install with: pip install kaggle")
        break

print("\ndone! check", data_dir.absolute())