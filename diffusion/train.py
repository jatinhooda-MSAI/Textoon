import sys
import time
from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, "../")
from step2_dataset import CartoonDataset
from model import build_model

# ---- Config ----
BASE = Path("../")
RUN_DIR = Path("./runs/run_64")
CKPT_DIR = RUN_DIR / "checkpoints"
SAMP_DIR = RUN_DIR / "samples"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
SAMP_DIR.mkdir(exist_ok=True)

TOTAL_STEPS = 400_000
BATCH_SIZE = 256
NUM_WORKERS = 8
LR = 2e-4
WARMUP_STEPS = 2000
EMA_DECAY = 0.9999
EMA_START = 1000
CKPT_EVERY = 10_000
SAMPLE_EVERY = 5_000
LOG_EVERY = 100

# ---- Data ----
train_ds = CartoonDataset(
    h5_path=BASE / "data_64.h5",
    metadata_csv=BASE / "train.csv",
    attr_cols_json=BASE / "attr_cols.json",
    hflip=True,
)
loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    shuffle=True, pin_memory=True, drop_last=True, persistent_workers=True,
)
print(f"Dataset: {len(train_ds)} | Batch: {BATCH_SIZE} | Steps/epoch: {len(loader)}", flush=True)

# Fixed sampling conditions (so progression is comparable across checkpoints)
fixed_attrs = torch.stack([train_ds[i][1] for i in range(8)]).cuda()
fixed_gt = torch.stack([train_ds[i][0] for i in range(8)]).cuda()
save_image((fixed_gt + 1) / 2, SAMP_DIR / "fixed_ground_truth.png", nrow=8)

# ---- Model ----
unet, diffusion = build_model(num_attrs=91, image_size=64)
diffusion = diffusion.cuda()
ema_diffusion = copy.deepcopy(diffusion).eval()
for p in ema_diffusion.parameters():
    p.requires_grad_(False)
diffusion = torch.compile(diffusion, mode="reduce-overhead")
for p in ema_diffusion.parameters():
    p.requires_grad_(False)

print(f"Params: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M", flush=True)

# ---- Optimizer + LR schedule ----
opt = torch.optim.AdamW(diffusion.parameters(), lr=LR, weight_decay=0.0)

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

@torch.no_grad()
def ema_update(decay):
    for p_ema, p in zip(ema_diffusion.parameters(), diffusion.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1 - decay)
    for b_ema, b in zip(ema_diffusion.buffers(), diffusion.buffers()):
        b_ema.copy_(b)

# ---- Training loop ----
print("Starting training...", flush=True)
diffusion.train()
step = 0
loss_accum, t_last = 0.0, time.time()

while step < TOTAL_STEPS:
    for img, attrs in loader:
        img = img.cuda(non_blocking=True)
        attrs = attrs.cuda(non_blocking=True)

        for g in opt.param_groups:
            g["lr"] = get_lr(step)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = diffusion(img, attrs=attrs)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        opt.step()

        if step >= EMA_START:
            ema_update(EMA_DECAY)
        else:
            ema_update(0.0)  # just copy weights until EMA starts

        loss_accum += loss.item()
        step += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t_last
            print(f"step {step:6d} | loss {loss_accum/LOG_EVERY:.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | "
                  f"{LOG_EVERY/dt:.1f} steps/s", flush=True)
            loss_accum, t_last = 0.0, time.time()

        if step % SAMPLE_EVERY == 0:
            ema_diffusion.eval()
            samples = ema_diffusion.sample_with_attrs(fixed_attrs, guidance_scale=5.0)
            save_image(samples, SAMP_DIR / f"step_{step:06d}.png", nrow=8)
            print(f"  -> wrote samples step_{step:06d}.png", flush=True)

        if step % CKPT_EVERY == 0:
            torch.save({
                "step": step,
                "model": diffusion.state_dict(),
                "ema": ema_diffusion.state_dict(),
                "opt": opt.state_dict(),
            }, CKPT_DIR / f"step_{step:06d}.pt")
            print(f"  -> wrote checkpoint step_{step:06d}.pt", flush=True)

        if step >= TOTAL_STEPS:
            break

# Final save
torch.save({
    "step": step, "model": diffusion.state_dict(),
    "ema": ema_diffusion.state_dict(), "opt": opt.state_dict(),
}, CKPT_DIR / "final.pt")
print("Training complete.", flush=True)