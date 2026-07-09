# Copyright (c) 2026 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ODE->SDE conversion for flow matching, following FlowTTS-GRPO (arXiv:2606.23190).

The deterministic probability-flow ODE ``dx_t = v_theta(x_t, t) dt`` is converted
into an SDE that preserves the marginal distributions while injecting Gaussian
exploration noise (paper Eq.6/7):

    x_mean   = x_t + [v + sigma_t^2 / (2 * (1 - t)) * (-x_t + t * v)] * dt
    x_{t+dt} = x_mean + sigma_t * sqrt(dt) * eps,    eps ~ N(0, I)
    sigma_t  = a * sqrt((1 - t) / t)

so each stochastic transition is a diagonal Gaussian with closed-form
log-density (paper Eq.11), which is what makes the GRPO likelihood ratio
computable for the flow-matching module.

CosyVoice3 uses a cosine t scheduler (t_span = 1 - cos(u * pi / 2)), hence dt
differs per step: all formulas below take the *actual* per-step dt. sigma_t
diverges at t=0, so the stochastic window must start at step >= 1.
"""

import dataclasses
import math
from typing import Optional

import torch


def make_t_span(n_timesteps: int, t_scheduler: str = 'cosine',
                device: torch.device = torch.device('cpu'),
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Replicates ConditionalCFM.forward's t discretization. Shape (n_timesteps + 1,)."""
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
    if t_scheduler == 'cosine':
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    return t_span


def sde_sigma(t: float, noise_level: float, t_clamp: float = 1e-4) -> float:
    """sigma_t = a * sqrt((1 - t) / t), clamped away from t=0/1 for stability."""
    t = min(max(t, t_clamp), 1.0 - t_clamp)
    return noise_level * math.sqrt((1.0 - t) / t)


def sde_mean(x: torch.Tensor, v: torch.Tensor, t: float, dt: float, sigma_t: float,
             t_clamp: float = 1e-4) -> torch.Tensor:
    """Mean of the SDE transition kernel (paper Eq.7 drift term)."""
    one_minus_t = max(1.0 - t, t_clamp)
    drift = v + (sigma_t ** 2) / (2.0 * one_minus_t) * (-x + t * v)
    return x + drift * dt


def ode_step(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """Plain Euler ODE update, identical to ConditionalCFM.solve_euler."""
    return x + dt * v


def gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, std: float,
                     mask: Optional[torch.Tensor] = None,
                     reduction: str = 'mean') -> torch.Tensor:
    """Log-density of a diagonal Gaussian N(mean, std^2 I), reduced over feature dims.

    Args:
        x/mean: (B, C, T)
        std: scalar standard deviation sigma_t * sqrt(dt)
        mask: optional (B, 1, T) validity mask; padded frames are excluded
        reduction: 'mean' averages the per-element log-prob over valid elements
            (as in the official Flow-GRPO implementation, keeps PPO ratios in a
            numerically sane range), 'sum' matches the paper formula literally.
    Returns:
        (B,) log-prob per sample.
    """
    logp = -((x - mean) ** 2) / (2.0 * std ** 2) - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    if mask is not None:
        logp = logp * mask
        denom = mask.sum(dim=(1, 2)) * x.shape[1]
    else:
        denom = torch.full((x.shape[0],), float(x.shape[1] * x.shape[2]), device=x.device, dtype=x.dtype)
    logp = logp.sum(dim=(1, 2))
    if reduction == 'mean':
        logp = logp / denom.clamp(min=1.0)
    elif reduction != 'sum':
        raise ValueError(f'unknown reduction {reduction}')
    return logp


def sample_window_start(start_min: int, start_max: int,
                        generator: Optional[torch.Generator] = None) -> int:
    """Uniformly sample the first SDE step index S_min in [start_min, start_max].

    Step indices are 0-based Euler steps; the paper samples S_min in [1, 3] so the
    window never touches step 0 where t=0 makes sigma_t diverge.
    """
    assert start_min >= 1, 'stochastic window must not include step 0 (t=0 => sigma_t=inf)'
    assert start_max >= start_min
    return int(torch.randint(start_min, start_max + 1, (1,), generator=generator).item())


@dataclasses.dataclass
class Transition:
    """One stochastic (SDE) transition recorded during rollout.

    Tensors have the group dimension first: (G, C, T). ``logprob_old`` is the
    behavior-policy log-density recorded at rollout time (theta_old).
    """
    step: int
    t: float
    dt: float
    sigma: float
    x_t: torch.Tensor
    x_next: torch.Tensor
    logprob_old: torch.Tensor

    @property
    def std(self) -> float:
        return self.sigma * math.sqrt(self.dt)

    def to(self, device: torch.device) -> 'Transition':
        return Transition(self.step, self.t, self.dt, self.sigma,
                          self.x_t.to(device), self.x_next.to(device), self.logprob_old.to(device))
