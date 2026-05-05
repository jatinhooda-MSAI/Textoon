import sys
import json
from pathlib import Path
import torch
from torchvision.utils import save_image

sys.path.insert(0, "/projects/e32706/kij6504")
from model import build_model

BASE = Path("/projects/e32706/kij6504")
RUN_DIR = Path("/projects/e32706/kij6504/diffusion/runs/run_64")
CKPT = RUN_DIR / "checkpoints" / "final.pt"
OUT = RUN_DIR / "eval"
OUT.mkdir(exist_ok=True)

# Load attribute vocabulary
attr_cols = json.load(open(BASE / "attr_cols.json"))
attr_to_idx = {name: i for i, name in enumerate(attr_cols)}

# Pick attributes to toggle. Use names that exist in YOUR attr_cols.
# Visually salient ones with enough training examples are best.
TOGGLE_ATTRS = [
    "smile",
    "blonde_hair",
    "open_mouth",
    "blush",
    "twintails",
    "hat",
]

# Verify they exist in your vocabulary
TOGGLE_ATTRS = [a for a in TOGGLE_ATTRS if a in attr_to_idx]
print(f"Toggling: {TOGGLE_ATTRS}")

# Base attributes (always on): something that produces a coherent character
BASE_ATTRS = ["1girl", "long_hair", "blue_eyes"]
BASE_ATTRS = [a for a in BASE_ATTRS if a in attr_to_idx]

# Build model and load EMA weights
unet, diffusion = build_model(num_attrs=len(attr_cols), image_size=64)
ckpt = torch.load(CKPT, map_location="cuda")
diffusion.load_state_dict(ckpt["ema"])
diffusion = diffusion.cuda().eval()
print(f"Loaded checkpoint at step {ckpt['step']}")

# Build base attribute vector
def make_attrs(active_names):
    v = torch.zeros(len(attr_cols))
    for n in active_names:
        v[attr_to_idx[n]] = 1.0
    return v

base_vec = make_attrs(BASE_ATTRS)

# For each toggle attribute, build (off, on) pair using SAME noise
torch.manual_seed(42)
n_seeds = 4  # generate 4 different characters per attribute

# Pre-generate fixed noise — used for ALL samples so toggling shows controlled change
# Sampling currently re-randomizes noise inside sample_with_attrs; we need to fix it.
# Easiest: sample with a fixed seed before each call.

rows = []
labels = []
for attr_name in TOGGLE_ATTRS:
    off_vec = base_vec.clone()
    on_vec = base_vec.clone()
    on_vec[attr_to_idx[attr_name]] = 1.0

    # Stack n_seeds copies so we get diverse characters per attr
    off_batch = off_vec.unsqueeze(0).repeat(n_seeds, 1).cuda()
    on_batch = on_vec.unsqueeze(0).repeat(n_seeds, 1).cuda()

    # Use same starting noise for both halves of each pair
    torch.manual_seed(42)
    off_samples = diffusion.sample_with_attrs(off_batch, guidance_scale=5.0)

    torch.manual_seed(42)  # same seed → same starting noise
    on_samples = diffusion.sample_with_attrs(on_batch, guidance_scale=5.0)

    # Interleave: off_0, on_0, off_1, on_1, ...
    interleaved = torch.stack([off_samples, on_samples], dim=1).flatten(0, 1)
    rows.append(interleaved)
    labels.append(attr_name)
    print(f"  generated {attr_name}")

# Stack all rows into one big grid
grid = torch.cat(rows, dim=0)
save_image(grid, OUT / "toggle_grid.png", nrow=n_seeds * 2)

# Also save individual per-attribute images so you can use them in the writeup
for attr_name, row in zip(labels, rows):
    save_image(row, OUT / f"toggle_{attr_name}.png", nrow=n_seeds * 2)

print(f"\nWrote {OUT}/toggle_grid.png")
print("Each row is one attribute. Within a row, columns alternate OFF/ON for 4 different characters.")