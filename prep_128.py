import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE = Path("/projects/e32706/kij6504/")
RAW = BASE / "images_128_raw"
H5 = BASE / "data_128.h5"

df = pd.read_csv(BASE / "metadata.csv")
filenames = df["filename"].tolist()

# Find the actual image dir (zips often nest)
sample = filenames[0]
if (RAW / sample).exists():
    img_root = RAW
else:
    img_root = next(p for p in RAW.rglob("*") if p.is_dir() and (p / sample).exists())
print(f"Image root: {img_root}")

n = len(filenames)
with h5py.File(H5, "w") as h5:
    imgs = h5.create_dataset("images", shape=(n, 128, 128, 3), dtype="uint8",
                             chunks=(1, 128, 128, 3))
    names = h5.create_dataset("filenames", shape=(n,),
                              dtype=h5py.string_dtype(encoding="utf-8"))
    for i, fn in enumerate(tqdm(filenames)):
        p = img_root / fn
        img = Image.open(p).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.LANCZOS)
        imgs[i] = np.asarray(img, dtype=np.uint8)
        names[i] = fn

print(f"Wrote {H5} ({H5.stat().st_size / 1e6:.0f} MB)")