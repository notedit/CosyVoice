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
"""GRPO objective pieces for FlowTTS-GRPO (arXiv:2606.23190)."""

from typing import Optional, Tuple

import torch


def group_advantages(rewards: torch.Tensor, std_eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-relative advantages (paper Eq.9).

    Args:
        rewards: (num_groups, G) terminal rewards, one row per prompt group.
        std_eps: groups whose reward std is below this are marked invalid and
            must be dropped from the batch (paper: discard zero-std groups).
    Returns:
        advantages: (num_groups, G), zero for invalid groups.
        valid: (num_groups,) bool mask of groups with non-degenerate rewards.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    valid = std.squeeze(1) > std_eps
    adv = (rewards - mean) / std.clamp(min=std_eps)
    adv = adv * valid.unsqueeze(1)
    return adv, valid


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor,
                  advantages: torch.Tensor, clip_eps: float = 0.2) -> Tuple[torch.Tensor, dict]:
    """Clipped surrogate objective (paper Eq.10, minus the KL term).

    Args:
        logp_new: (N,) log-prob of stored transitions under current policy.
        logp_old: (N,) behavior-policy log-prob recorded at rollout time.
        advantages: (N,) terminal-reward advantages broadcast to each window step.
        clip_eps: PPO clip range epsilon.
    Returns:
        scalar loss (to minimize) and a dict of diagnostics.
    """
    log_ratio = logp_new - logp_old
    ratio = log_ratio.exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
        stats = {
            'ratio_mean': ratio.mean().item(),
            'ratio_max': ratio.max().item(),
            'clip_frac': clip_frac.item(),
        }
    return loss, stats


def gaussian_mean_kl(mean_new: torch.Tensor, mean_ref: torch.Tensor, std: float,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """KL(pi_theta || pi_ref) between two Gaussian transition kernels with equal std.

    For N(mu1, s^2 I) vs N(mu2, s^2 I) the KL reduces to ||mu1 - mu2||^2 / (2 s^2).
    Returned as the per-element mean over valid positions so its scale matches the
    'mean'-reduced log-probs used in the surrogate loss.
    """
    kl = (mean_new - mean_ref) ** 2 / (2.0 * std ** 2)
    if mask is not None:
        kl = kl * mask
        denom = (mask.sum() * mean_new.shape[1]).clamp(min=1.0)
    else:
        denom = torch.tensor(float(mean_new.numel()), device=mean_new.device)
    return kl.sum() / denom
