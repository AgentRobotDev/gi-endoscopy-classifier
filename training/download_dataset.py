"""Downloads and extracts the Kvasir-v2 GI endoscopy dataset (~400 MB).

If automatic download fails, get the zip manually from one of:
  • https://datasets.simula.no/kvasir/  (look for "Download" button)
  • https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset-for-classification
Then run:
  python download_dataset.py --zip ~/Downloads/kvasir-v2.zip
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
import urllib3

# Primary URL — update here if Simula moves the file again
DATASET_URL = "https://datasets.simula.no/downloads/kvasir-v2.zip"

DOWNLOAD_HELP = """
Automatic download failed. Get the zip manually from one of these sources:

  1. Simula Research Lab (official):
       https://datasets.simula.no/kvasir/
       → click the Download button for "Kvasir-v2"

  2. Kaggle mirror (requires free account):
       https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset-for-classification
       → download the zip, or use:  kaggle datasets download meetnagadia/kvasir-dataset-for-classification

Then extract or run:
  python download_dataset.py --zip /path/to/kvasir-v2.zip --dest ../data
"""


def _stream(url: str, dest_path: Path, verify: bool) -> None:
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("  (SSL verification disabled — conda certificate workaround)")

    with requests.get(url, stream=True, timeout=120, verify=verify) as r:
        if r.status_code == 404:
            raise FileNotFoundError(f"404 — file not found at {url}")
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    bar = "#" * (pct // 2)
                    print(f"\r  [{bar:<50}] {pct}%", end="", flush=True)
    print()


def extract(zip_path: Path, dest: Path) -> Path:
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Detect top-level directory name inside the zip
        top_dirs = {p.split("/")[0] for p in zf.namelist() if "/" in p}
        zf.extractall(dest)

    # Return the extracted root directory
    for candidate in ["kvasir-v2", "kvasir_v2", "Kvasir-v2"] + list(top_dirs):
        p = dest / candidate
        if p.exists() and p.is_dir():
            # Normalise to kvasir-v2
            if p.name != "kvasir-v2":
                p.rename(dest / "kvasir-v2")
            return dest / "kvasir-v2"

    raise RuntimeError(f"Could not find extracted dataset directory inside {dest}")


def download(dest: Path, url: str, zip_path: Path | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    extract_path = dest / "kvasir-v2"

    if extract_path.exists():
        print(f"Dataset already at {extract_path}")
        return

    # -- Pre-downloaded zip provided --
    if zip_path is not None:
        if not zip_path.exists():
            sys.exit(f"ERROR: zip not found at {zip_path}")
        out = extract(zip_path, dest)
        print(f"Done — dataset ready at {out}")
        print("Classes:", sorted(p.name for p in out.iterdir() if p.is_dir()))
        return

    # -- Try automatic download --
    tmp = dest / "kvasir-v2.zip"
    if not tmp.exists():
        print(f"Downloading from {url}")
        try:
            _stream(url, tmp, verify=True)
        except requests.exceptions.SSLError:
            print("SSL error — retrying without certificate verification ...")
            tmp.unlink(missing_ok=True)
            try:
                _stream(url, tmp, verify=False)
            except FileNotFoundError:
                tmp.unlink(missing_ok=True)
                print(DOWNLOAD_HELP)
                sys.exit(1)
        except FileNotFoundError:
            tmp.unlink(missing_ok=True)
            print(DOWNLOAD_HELP)
            sys.exit(1)
    else:
        print(f"Archive found at {tmp}, skipping download.")

    out = extract(tmp, dest)
    print(f"Done — dataset ready at {out}")
    print("Classes:", sorted(p.name for p in out.iterdir() if p.is_dir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="../data",
                        help="Directory to save/extract dataset (default: ../data)")
    parser.add_argument("--url", default=DATASET_URL,
                        help="Override the download URL")
    parser.add_argument("--zip", dest="zip_path", default=None,
                        help="Path to a manually downloaded kvasir-v2.zip")
    args = parser.parse_args()
    download(Path(args.dest), args.url, Path(args.zip_path) if args.zip_path else None)
