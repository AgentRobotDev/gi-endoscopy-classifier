"""Copy N sample images per class into app/static/samples/ for the demo UI."""

import argparse
import random
import shutil
from pathlib import Path


def export(data_dir: Path, app_static_dir: Path, n_per_class: int, seed: int) -> None:
    src = data_dir / "kvasir-v2"
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found at {src}. Run download_dataset.py first.")

    dest = app_static_dir / "samples"
    dest.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    total = 0

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg"))
        if not images:
            continue

        selected = rng.sample(images, min(n_per_class, len(images)))
        out_dir = dest / class_dir.name
        out_dir.mkdir(exist_ok=True)

        for img in selected:
            shutil.copy2(img, out_dir / img.name)

        print(f"  {class_dir.name}: {len(selected)} images")
        total += len(selected)

    print(f"\nExported {total} images to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--app-static-dir", default="../app/static")
    parser.add_argument("--n-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export(Path(args.data_dir), Path(args.app_static_dir), args.n_per_class, args.seed)
