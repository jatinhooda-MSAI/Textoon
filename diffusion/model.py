import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


class AttrConditionedUnet(Unet):
    def __init__(self, num_attrs, cfg_dropout_prob=0.1, **unet_kwargs):
        super().__init__(**unet_kwargs)
        self.num_attrs = num_attrs
        self.cfg_dropout_prob = cfg_dropout_prob

        time_dim = self.time_mlp[-1].out_features
        self.attr_mlp = nn.Sequential(
            nn.Linear(num_attrs, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, time, x_self_cond=None, attrs=None):
        if self.training and attrs is not None and self.cfg_dropout_prob > 0:
            mask = (torch.rand(attrs.shape[0], 1, device=attrs.device)
                    > self.cfg_dropout_prob).float()
            attrs = attrs * mask

        t_emb = self.time_mlp(time)
        if attrs is not None:
            t_emb = t_emb + self.attr_mlp(attrs)

        return self._forward_with_emb(x, t_emb, x_self_cond)

    def _forward_with_emb(self, x, t, x_self_cond=None):
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class AttrGaussianDiffusion(GaussianDiffusion):
    """GaussianDiffusion that passes `attrs` through to the U-Net."""

    def p_losses(self, x_start, t, attrs=None, noise=None, offset_noise_strength=None):
        import torch.nn.functional as F
        noise = noise if noise is not None else torch.randn_like(x_start)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Pass attrs through to the model
        model_out = self.model(x, t, x_self_cond=None, attrs=attrs)

        target = noise  # objective='pred_noise'
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])

        # lucidrains weights losses by SNR; replicate that
        loss = loss * self.loss_weight.gather(0, t)
        return loss.mean()

    def forward(self, img, attrs=None, *args, **kwargs):
        b, c, h, w, device = *img.shape, img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.p_losses(img, t, attrs=attrs, *args, **kwargs)

    @torch.no_grad()
    def sample_with_attrs(self, attrs, guidance_scale=5.0):
        """Classifier-free guided sampling. attrs: (B, num_attrs)."""
        batch_size = attrs.shape[0]
        device = next(self.model.parameters()).device
        shape = (batch_size, self.channels, self.image_size, self.image_size)

        img = torch.randn(shape, device=device)
        zero_attrs = torch.zeros_like(attrs)

        for t_int in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)

            # Conditional and unconditional predictions
            eps_cond = self.model(img, t, x_self_cond=None, attrs=attrs)
            eps_uncond = self.model(img, t, x_self_cond=None, attrs=zero_attrs)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # Standard DDPM denoising step using lucidrains' precomputed buffers
            alpha_t = self.alphas_cumprod[t_int]
            alpha_t_prev = self.alphas_cumprod_prev[t_int]
            beta_t = 1 - alpha_t / alpha_t_prev

            x0_pred = (img - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            mean = (alpha_t_prev.sqrt() * beta_t / (1 - alpha_t)) * x0_pred + \
                   ((1 - alpha_t_prev).sqrt() * (1 - beta_t / (1 - alpha_t))) * img

            if t_int > 0:
                noise = torch.randn_like(img)
                var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
                img = mean + var.sqrt() * noise
            else:
                img = mean

        return self.unnormalize(img)


def build_model(num_attrs=91, image_size=64):
    unet = AttrConditionedUnet(
        num_attrs=num_attrs,
        cfg_dropout_prob=0.1,
        dim=64,
        dim_mults=(1, 2, 4, 4),
        channels=3,
        flash_attn=False,
    )
    diffusion = AttrGaussianDiffusion(
        unet,
        image_size=image_size,
        timesteps=1000,
        beta_schedule="cosine",
        objective="pred_noise",
    )
    return unet, diffusion