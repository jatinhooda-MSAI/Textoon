#!/usr/bin/env python3
"""
One-time prep: zip -> extracted 128s -> 64x64 HDF5 + sanity grid.
"""
import argparse
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def find_image_root(raw_dir: Path, sample_filename: str) -> Path:
    """Zips often wrap contents in a top-level folder; locate where images actually live."""
    if (raw_dir / sample_filename).exists():
        return raw_dir
    for sub in raw_dir.rglob("*"):
        if sub.is_dir() and (sub / sample_filename).exists():
            return sub
    raise FileNotFoundError(f"Could not locate {sample_filename} under {raw_dir}")


def main(args):
    base = Path(args.base_dir)
    zip_path = base / args.zip_name
    raw_dir = base / "images_128_raw"
    h5_path = base / "data_64.h5"
    sanity_path = base / "sanity_grid.png"
    metadata_path = base / "metadata.csv"

    # 1. Extract (idempotent — skip if already extracted)
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"Extracting {zip_path} -> {raw_dir}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    else:
        print(f"{raw_dir} already populated — skipping extract.")

    # 2. Load metadata to define the canonical filename order
    df = pd.read_csv(metadata_path)
    filenames = df["filename"].tolist()
    n = len(filenames)
    print(f"{n} files in metadata.csv")

    image_root = find_image_root(raw_dir, filenames[0])
    print(f"Image root: {image_root}")

    # 3. Resize + pack into HDF5
    # chunks=(1, 64, 64, 3) -> each image is one chunk -> O(1) random access
    # no compression: 48k * 64 * 64 * 3 ~= 590 MB, fits in page cache after first epoch
    missing = []
    with h5py.File(h5_path, "w") as h5:
        images_ds = h5.create_dataset(
            "images", shape=(n, 64, 64, 3), dtype="uint8", chunks=(1, 64, 64, 3),
        )
        names_ds = h5.create_dataset(
            "filenames", shape=(n,), dtype=h5py.string_dtype(encoding="utf-8"),
        )

        for i, fn in enumerate(tqdm(filenames, desc="resize+pack")):
            p = image_root / fn
            if not p.exists():
                missing.append(fn)
                continue
            try:
                img = Image.open(p).convert("RGB").resize((64, 64), Image.LANCZOS)
                images_ds[i] = np.asarray(img, dtype=np.uint8)
                names_ds[i] = fn
            except Exception as e:
                print(f"  failed {fn}: {e}")
                missing.append(fn)

    if missing:
        print(f"WARNING: {len(missing)} missing/unreadable. First 5: {missing[:5]}")
        # write a sidecar so the Dataset class can mask these out later
        pd.Series(missing).to_csv(base / "missing_files.csv", index=False, header=["filename"])

    # 4. Sanity grid — random 16 images, fixed seed for reproducibility
    print("Writing sanity grid...")
    with h5py.File(h5_path, "r") as h5:
        rng = np.random.RandomState(0)
        valid_idxs = [i for i in range(n) if h5["filenames"][i]]  # skip missing
        idxs = rng.choice(valid_idxs, size=16, replace=False)
        grid = np.zeros((4 * 64, 4 * 64, 3), dtype=np.uint8)
        for k, idx in enumerate(idxs):
            r, c = k // 4, k % 4
            grid[r*64:(r+1)*64, c*64:(c+1)*64] = h5["images"][idx]

    Image.fromarray(grid).save(sanity_path)
    print(f"\nHDF5: {h5_path} ({h5_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Sanity grid: {sanity_path}")
    print(f"  -> scp this back and eyeball it before going further")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="e.g. /projects/p12345/cartoon_data")
    ap.add_argument("--zip_name", default="images_128.zip")
    main(ap.parse_args())