import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

sys.path.insert(0, "../")  # for dataset.py
from step2_dataset import CartoonDataset
from model import build_model

BASE = Path("../")
OUT = Path("./overfit_out")
OUT.mkdir(exist_ok=True)

# Tiny fixed subset
full = CartoonDataset(
    h5_path=BASE / "data_64.h5",
    metadata_csv=BASE / "train.csv",
    attr_cols_json=BASE / "attr_cols.json",
    hflip=False,
)
ds = Subset(full, list(range(32)))
loader = DataLoader(ds, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)

# Build model with CFG dropout DISABLED for this test
unet, diffusion = build_model(num_attrs=91, image_size=64)
unet.cfg_dropout_prob = 0.0
diffusion = diffusion.cuda()

opt = torch.optim.AdamW(diffusion.parameters(), lr=2e-4)

# Save the 32 ground-truth images so we can compare against samples
gt_imgs = torch.stack([full[i][0] for i in range(32)])
gt_attrs = torch.stack([full[i][1] for i in range(32)])
save_image((gt_imgs + 1) / 2, OUT / "ground_truth.png", nrow=8)

print("Training...")
diffusion.train()
step = 0
target_steps = 2000
while step < target_steps:
    for img, attrs in loader:
        img = img.cuda()
        attrs = attrs.cuda()
        loss = diffusion(img, attrs=attrs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"  step {step}: loss {loss.item():.4f}")
        if step >= target_steps:
            break

print("\nSampling 8 images conditioned on first 8 ground-truth attrs...")
diffusion.eval()
samples = diffusion.sample_with_attrs(gt_attrs[:8].cuda(), guidance_scale=1.0)
save_image(samples, OUT / "samples.png", nrow=8)
print(f"Wrote {OUT}/ground_truth.png and {OUT}/samples.png")