"""Fetch a pre-generated AMLSim sample dataset (no Java simulator required).

Usage:
    python -m src.data_pipeline.download_amlsim --dataset 20K_cycle200
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

AMLSIM_RAW_BASE = "https://raw.githubusercontent.com/IBM/AMLSim/master/sample"
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download(dataset: str) -> Path:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = DATA_RAW_DIR / f"{dataset}.tgz"
    url = f"{AMLSIM_RAW_BASE}/{dataset}.tgz"

    if not tgz_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tgz_path)
    else:
        print(f"Already downloaded: {tgz_path}")

    extract_dir = DATA_RAW_DIR / dataset
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir} ...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(DATA_RAW_DIR)
    else:
        print(f"Already extracted: {extract_dir}")

    return extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="20K_cycle200",
        choices=["20K_cycle200", "20K_fanin200", "20K_fanin200cycle200"],
    )
    args = parser.parse_args()
    path = download(args.dataset)
    print(f"Ready at: {path}")
