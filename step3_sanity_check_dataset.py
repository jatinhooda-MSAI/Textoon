from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from dataset import CartoonDataset

BASE = Path(".")

ds = CartoonDataset(
    h5_path=BASE / "data_64.h5",
    metadata_csv=BASE / "metadata.csv",
    attr_cols_json=BASE / "attr_cols.json",
    hflip=False,
)

print(f"Dataset size: {len(ds)}")
print(f"Num attributes: {len(ds.attr_cols)}")

img, attrs = ds[0]
print(f"Image: {tuple(img.shape)} {img.dtype} [{img.min():.2f}, {img.max():.2f}]")
print(f"Attrs: {tuple(attrs.shape)} sum={attrs.sum().item()}")

loader = DataLoader(ds, batch_size=32, num_workers=4,
                    shuffle=True, persistent_workers=True)
batch_img, batch_attrs = next(iter(loader))
print(f"Batch: {tuple(batch_img.shape)}, {tuple(batch_attrs.shape)}")

rng = np.random.RandomState(42)
idxs = rng.choice(len(ds), size=16, replace=False)

cell_w, cell_h = 64, 124
grid = Image.new("RGB", (20 * cell_w, 20 * cell_h), (255, 255, 255))
draw = ImageDraw.Draw(grid)
font = ImageFont.load_default()

for k, idx in enumerate(idxs):
    img, attrs = ds[idx]
    arr = ((img + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    r, c = k*5 // 4, k*5 % 4
    grid.paste(Image.fromarray(arr), (c * cell_w, r * cell_h))

    active = [n for n, v in zip(ds.attr_cols, attrs.tolist()) if v > 0.5]
    label = f"{ds.filenames[idx][:18]}\n" + ", ".join(active[:4])
    if len(active) > 4:
        label += f" +{len(active) - 4}"
    draw.text((c * cell_w + 2, r * cell_h + 66), label, fill=(0, 0, 0), font=font)

out = BASE / "sanity_dataset.png"
grid.save(out)
print(f"Wrote {out}")