import torch
from model import build_model

unet, diffusion = build_model(num_attrs=91, image_size=64)
diffusion = diffusion.cuda()

x = torch.randn(4, 3, 64, 64).cuda()
attrs = torch.randint(0, 2, (4, 91)).float().cuda()

# Training forward pass
diffusion.train()
loss = diffusion(x, attrs=attrs)
print(f"Training loss: {loss.item():.4f}")
print(f"Params: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")

# Sampling
diffusion.eval()
samples = diffusion.sample_with_attrs(attrs[:2], guidance_scale=5.0)
print(f"Samples: {tuple(samples.shape)} [{samples.min():.2f}, {samples.max():.2f}]")