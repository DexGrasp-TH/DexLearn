import os
import numpy as np
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
import math

from .diff_mlp import MLP

# constants

# ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def jacobian_trace(log_prob_type, dx, dy):
    if log_prob_type == "accurate_cont":
        # time consuming
        jacobian_mat = jacobian_matrix(dy, dx)
        return jacobian_mat.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    elif log_prob_type == "estimate":
        # quick
        return approx_jacobian_trace(dy, dx)
    else:
        return 0


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):  # change this with ODEFunc.beta
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.0
    for i in range(z.shape[1]):
        sum_diag += (
            torch.autograd.grad(f[:, i].sum(), z, retain_graph=True)[0]
            .contiguous()[:, i]
            .contiguous()
        )

    return sum_diag.contiguous()


class ODEFunc(nn.Module):
    def __init__(self, diffusion, cond, calculate_log_prob=False):
        super().__init__()
        self.diffusion = diffusion
        self.cond = cond
        self.calculate_log_prob = calculate_log_prob

    def angle(self, t, s=0.008):
        return (t + s) / (1 + s) * math.pi * 0.5

    def beta(self, angle, s=0.008):  # change this with cosine_beta_schedule
        return np.pi / (1 + s) * angle.sin() / angle.cos()

    def forward(self, t, z):
        # print(t.item())
        t = t[None].repeat(z[0].shape[0])
        if self.calculate_log_prob:
            with torch.set_grad_enabled(True):
                z[0].requires_grad_(True)
                dz_dt = self._forward(t, z[0])
                dlogp_z_dt = -trace_df_dz(dz_dt.reshape(z[0].shape[0], -1), z[0]).view(
                    -1
                )
            return (dz_dt, dlogp_z_dt)
        else:
            return (self._forward(t, z[0]), torch.zeros_like(z[1]))

    def _forward(self, t, z):
        pred_noise = self.diffusion.model_predictions(
            z, self.cond, t, exact_t=True
        ).pred_noise
        angle = self.angle(t[:, None, None])
        beta = self.beta(angle)
        return (
            -0.5 * beta * z
            + 0.5 * beta * pred_noise / (1 - angle.cos().square()).sqrt()
        )


class SinusoidalPosEmb(nn.Module):
    # compute sinusoidal positional embeddings
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor):
        """
        x: torch.Tensor, shape (B,)
        return emb: torch.Tensor, shape (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * self.theta
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPWrapper(MLP):
    def __init__(self, channels, feature_dim, *args, **kwargs):
        self.channels = channels
        input_dim = channels + feature_dim
        super().__init__(input_dim=input_dim, *args, **kwargs)
        self.embedding = SinusoidalPosEmb(feature_dim)

    def forward(self, x, t, cond):
        t = self.embedding(t)
        return super().forward(torch.cat([x, cond + t], dim=-1))


class GaussianDiffusion1D(nn.Module):
    def __init__(self, model, config, cond_fn=lambda x, t, cond: cond):
        super().__init__()
        self.config = config
        self.model = model
        self.cond_fn = cond_fn
        if config.scheduler_type == "DDPMScheduler":
            self.scheduler = DDPMScheduler(**config.scheduler)
        elif config.scheduler_type == "DDIMScheduler":
            self.scheduler = DDIMScheduler(**config.scheduler)
        elif config.scheduler_type == "EulerAncestralDiscreteScheduler":
            self.scheduler = EulerAncestralDiscreteScheduler(**config.scheduler)
        elif config.scheduler_type == "EulerDiscreteScheduler":
            self.scheduler = EulerDiscreteScheduler(**config.scheduler)
        print(self.scheduler)
        self.timesteps = config.scheduler.num_train_timesteps
        # self.scheduler.set_timesteps(self.timesteps, device=config.device)
        self.inference_timesteps = config.num_inference_timesteps
        self.prediction_type = config.scheduler.prediction_type
        if self.config.loss_type == "l1":
            self.diff_loss = nn.SmoothL1Loss(reduction="mean")
        elif self.config.loss_type == "l2":
            self.diff_loss = nn.MSELoss(reduction="mean")
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)

    def calculate_loss(self, x, cond):
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=x.device, dtype=torch.long
        )
        noise = torch.randn_like(x)
        noised_x = self.scheduler.add_noise(x, noise, t)
        cond = self.cond_fn(noised_x, t / self.timesteps, cond)
        pred = self.model(noised_x, t / self.timesteps, cond=cond)
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x
        elif self.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x, noise, t)
        else:
            raise NotImplementedError()

        loss = self.diff_loss(pred, target)

        return loss

    def sample(self, cond):
        x = torch.randn(cond.shape[0], self.model.channels, device=cond.device)
        log_prob = (-x.square() / 2 - np.log(2 * np.pi) / 2).sum(1)
        self.scheduler.set_timesteps(self.inference_timesteps, device=cond.device)

        # with torch.no_grad():
        #     for t in self.scheduler.timesteps:
        #         t_pad = torch.full((x.shape[0],), t.item(), device=x.device, dtype=torch.long)
        #         cond_now = self.cond_fn(x, t_pad / self.timesteps, cond)
        #         model_output = self.model(x, t_pad / self.timesteps, cond=cond_now)
        #         x = self.scheduler.step(model_output, t, x).prev_sample
        need_log_prob = self.config.log_prob_type is not None
        last_t = self.timesteps
        with torch.set_grad_enabled(need_log_prob):
            for t in self.scheduler.timesteps:
                dx = torch.zeros_like(x)
                dx.requires_grad_(need_log_prob)
                x += dx
                dt = torch.full(
                    (x.shape[0], 1),
                    (last_t - t.item()) / self.timesteps,
                    device=x.device,
                    dtype=torch.float,
                )
                last_t = t.item()
                t_pad = torch.full(
                    (x.shape[0],), t.item(), device=x.device, dtype=torch.long
                )
                cond_now = self.cond_fn(x, t_pad / self.timesteps, cond)
                model_output = self.model(x, t_pad / self.timesteps, cond=cond_now)
                alpha_prod = self.scheduler.alphas_cumprod.to(x.device)[t_pad][:, None]
                betas = self.scheduler.betas.to(x.device)[t_pad][:, None]
                if self.prediction_type == "epsilon":
                    noise = model_output
                elif self.prediction_type == "v_prediction":
                    noise = (
                        model_output * alpha_prod.sqrt() + x * (1 - alpha_prod).sqrt()
                    )
                score = -1 / (1 - alpha_prod).sqrt() * noise
                beta = betas * self.timesteps
                if self.config.ode:
                    dy = (-0.5 * beta * x - score * beta / 2) * dt
                else:
                    dy = (
                        -0.5 * beta * x - score * beta
                    ) * dt + beta.sqrt() * torch.randn_like(x) * dt.sqrt()
                log_prob -= (
                    jacobian_trace(self.config.log_prob_type, dx, -dy / dt) * dt[:, 0]
                )
                x = x - dy
                x = x.detach()
                log_prob = log_prob.detach()

        if not need_log_prob:
            log_prob *= 0
        return x, log_prob

    def log_prob(self, x, cond):
        raise NotImplementedError()
