import sys
import json
from pathlib import Path
import torch
from torchvision.utils import save_image

sys.path.insert(0, "/projects/e32706/kij6504")
from model import AttrConditionedUnet, AttrGaussianDiffusion

BASE = Path("/projects/e32706/kij6504")
RUN_DIR = Path("/projects/e32706/kij6504/diffusion128/runs/run_128")
CKPT = RUN_DIR / "checkpoints" / "final.pt"
OUT = RUN_DIR / "eval"

attr_cols = json.load(open(BASE / "attr_cols.json"))
attr_to_idx = {name: i for i, name in enumerate(attr_cols)}

unet = AttrConditionedUnet(
    num_attrs=91,
    cfg_dropout_prob=0.1,
    dim=96,                       # match what you ended up training with
    dim_mults=(1, 2, 4, 8),       # match what you trained with
    channels=3,
    flash_attn=False,              # match training config
)
diffusion = AttrGaussianDiffusion(
    unet,
    image_size=128,               # was 64
    timesteps=1000,
    beta_schedule="cosine",
    objective="pred_noise",
    auto_normalize=False,
).cuda()
ckpt = torch.load(CKPT, map_location="cuda")
diffusion.load_state_dict(ckpt["ema"])
diffusion.eval()

# Fixed attributes for interpolation
# ATTRS = ["1girl", "long_hair", "blue_hair", "smile"]
ATTRS = ["1girl", "long_hair", "blonde_hair", "smile", "blue_eyes"]
ATTRS = [a for a in ATTRS if a in attr_to_idx]
v = torch.zeros(len(attr_cols))
for n in ATTRS:
    v[attr_to_idx[n]] = 1.0

# We need a custom sampler that takes pre-specified noise (since the default
# randomizes inside the function). Quick inline version:
@torch.no_grad()
def sample_from_noise(diffusion, noise, attrs, guidance_scale=5.0):
    img = noise.clone()
    zero_attrs = torch.zeros_like(attrs)
    for t_int in reversed(range(diffusion.num_timesteps)):
        t = torch.full((img.shape[0],), t_int, device=img.device, dtype=torch.long)
        eps_cond = diffusion.model(img, t, x_self_cond=None, attrs=attrs)
        eps_uncond = diffusion.model(img, t, x_self_cond=None, attrs=zero_attrs)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        a_t = diffusion.alphas_cumprod[t_int]
        a_prev = diffusion.alphas_cumprod_prev[t_int]
        b_t = 1 - a_t / a_prev
        x0 = ((img - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        coef_x0 = a_prev.sqrt() * b_t / (1 - a_t)
        coef_xt = (1 - b_t).sqrt() * (1 - a_prev) / (1 - a_t)
        mean = coef_x0 * x0 + coef_xt * img
        if t_int > 0:
            var = b_t * (1 - a_prev) / (1 - a_t)
            img = mean + var.sqrt() * torch.randn_like(img)
        else:
            img = mean
    return (img + 1) * 0.5

# Two endpoint noises
torch.manual_seed(0)
noise_a = torch.randn(1, 3, 128, 128, device="cuda")
torch.manual_seed(1)
noise_b = torch.randn(1, 3, 128, 128, device="cuda")

# Interpolate (slerp would be more correct but lerp is fine for visualization)
n_frames = 8
alphas = torch.linspace(0, 1, n_frames)
noises = torch.cat([(1 - a) * noise_a + a * noise_b for a in alphas], dim=0)
attrs = v.unsqueeze(0).repeat(n_frames, 1).cuda()

samples = sample_from_noise(diffusion, noises, attrs, guidance_scale=5.0)
save_image(samples, OUT / "interpolation.png", nrow=n_frames)
print(f"Wrote {OUT}/interpolation.png")