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
"""CPU unit tests for the SDE sampler and policy math (no pretrained model needed)."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flow_grpo.sde_solver import (  # noqa: E402
    Transition,
    gaussian_logprob,
    make_t_span,
    ode_step,
    sample_window_start,
    sde_mean,
    sde_sigma,
)
from flow_grpo.policy import FlowGRPOPolicy  # noqa: E402


class DummyEstimator(torch.nn.Module):
    """Tiny stand-in for the DiT: velocity depends on x, mu, cond, spks and t."""

    def __init__(self, mel_dim=80):
        super().__init__()
        self.proj = torch.nn.Conv1d(mel_dim * 3, mel_dim, kernel_size=1)
        self.t_proj = torch.nn.Linear(1, mel_dim)

    def forward(self, x, mask, mu, t, spks, cond, streaming=False):
        h = torch.cat([x, mu, cond], dim=1)
        out = self.proj(h) + self.t_proj(t.reshape(-1, 1)).unsqueeze(-1) + spks.unsqueeze(-1)
        return out * mask


class DummyFlow(torch.nn.Module):
    """Mimics the parts of CausalMaskedDiffWithDiT that FlowGRPOPolicy touches."""

    def __init__(self, mel_dim=80, vocab_size=100, token_mel_ratio=2):
        super().__init__()
        self.output_size = mel_dim
        self.token_mel_ratio = token_mel_ratio
        self.input_embedding = torch.nn.Embedding(vocab_size, mel_dim)
        self.spk_embed_affine_layer = torch.nn.Linear(192, mel_dim)
        self.pre_lookahead_layer = torch.nn.Identity()
        self.decoder = torch.nn.Module()
        self.decoder.t_scheduler = 'cosine'
        self.decoder.estimator = DummyEstimator(mel_dim)


def make_policy(noise_level=0.5, **kwargs):
    torch.manual_seed(0)
    flow = DummyFlow()
    defaults = dict(n_timesteps=10, noise_level=noise_level, window_size=2,
                    window_start_min=1, window_start_max=3)
    defaults.update(kwargs)
    return FlowGRPOPolicy(flow, **defaults)


def make_conditions(policy, num_tokens=12, prompt_tokens=4):
    token = torch.randint(0, 100, (1, num_tokens))
    prompt_token = torch.randint(0, 100, (1, prompt_tokens))
    prompt_feat = torch.randn(1, prompt_tokens * 2, 80)
    embedding = torch.randn(1, 192)
    return policy.prepare_conditions(token, prompt_token, prompt_feat, embedding)


def test_t_span_matches_cfm():
    """Must replicate ConditionalCFM's cosine discretization exactly."""
    t_span = make_t_span(10, 'cosine')
    ref = torch.linspace(0, 1, 11)
    ref = 1 - torch.cos(ref * 0.5 * torch.pi)
    assert torch.allclose(t_span, ref)
    assert t_span[0] == 0.0 and abs(t_span[-1] - 1.0) < 1e-6


def test_sigma_schedule():
    assert sde_sigma(0.5, 0.5) == pytest.approx(0.5 * math.sqrt(1.0))
    assert sde_sigma(0.2, 0.5) == pytest.approx(0.5 * math.sqrt(0.8 / 0.2))
    # clamped near the endpoints instead of diverging
    assert math.isfinite(sde_sigma(0.0, 0.5)) and math.isfinite(sde_sigma(1.0, 0.5))


def test_sde_zero_noise_reduces_to_ode():
    """With a=0 the SDE drift correction vanishes and the update equals Euler ODE."""
    x = torch.randn(2, 80, 30)
    v = torch.randn(2, 80, 30)
    t, dt = 0.3, 0.1
    mean = sde_mean(x, v, t, dt, sigma_t=0.0)
    assert torch.allclose(mean, ode_step(x, v, dt), atol=1e-6)


def test_rollout_zero_noise_matches_deterministic_solver():
    """With a=0 there is no stochastic action: the rollout degenerates to plain
    Euler ODE from the same z0 and records no transitions."""
    policy = make_policy(noise_level=0.0)
    cond = make_conditions(policy)
    gen = torch.Generator().manual_seed(42)
    mel, transitions, _ = policy.rollout(cond, group_size=1, generator=gen, window_start=2)
    assert len(transitions) == 0

    # replay: identical initial noise, pure ODE all the way
    gen = torch.Generator().manual_seed(42)
    group = policy._expand_group(cond, 1)
    t_span = make_t_span(policy.n_timesteps, 'cosine')
    x = torch.randn(group['mu'].shape, generator=gen)
    with torch.no_grad():
        for step in range(policy.n_timesteps):
            t, dt = float(t_span[step]), float(t_span[step + 1] - t_span[step])
            v = policy._velocity(x, group, t)
            x = ode_step(x, v, dt)
    assert torch.allclose(mel, x.float(), atol=1e-5)


def test_gaussian_logprob_matches_torch_distribution():
    torch.manual_seed(1)
    mean = torch.randn(3, 80, 20)
    std = 0.37
    x = mean + std * torch.randn_like(mean)
    ref = torch.distributions.Normal(mean, std).log_prob(x).sum(dim=(1, 2))
    ours = gaussian_logprob(x, mean, std, reduction='sum')
    assert torch.allclose(ours, ref, atol=1e-4)
    ours_mean = gaussian_logprob(x, mean, std, reduction='mean')
    assert torch.allclose(ours_mean, ref / (80 * 20), atol=1e-6)


def test_gaussian_logprob_respects_mask():
    mean = torch.zeros(1, 4, 10)
    x = torch.ones(1, 4, 10)
    mask = torch.zeros(1, 1, 10)
    mask[:, :, :5] = 1.0
    full = gaussian_logprob(x, mean, 1.0, reduction='sum')
    masked = gaussian_logprob(x, mean, 1.0, mask=mask, reduction='sum')
    assert torch.allclose(masked, full / 2, atol=1e-5)


def test_recompute_logprob_matches_rollout():
    """Same params => transition_logprob must reproduce the rollout-time logp_old."""
    policy = make_policy()
    cond = make_conditions(policy)
    gen = torch.Generator().manual_seed(7)
    _, transitions, _ = policy.rollout(cond, group_size=4, generator=gen)
    for tr in transitions:
        logp, mean = policy.transition_logprob(cond, tr, group_size=4)
        assert torch.allclose(logp, tr.logprob_old, atol=1e-5)
        assert mean.shape == tr.x_t.shape


def test_recompute_logprob_differentiable():
    policy = make_policy()
    cond = make_conditions(policy)
    _, transitions, _ = policy.rollout(cond, group_size=2)
    logp, _ = policy.transition_logprob(cond, transitions[0], group_size=2)
    loss = logp.sum()
    loss.backward()
    grads = [p.grad for p in policy.flow.decoder.estimator.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


def test_window_start_bounds():
    starts = {sample_window_start(1, 3) for _ in range(200)}
    assert starts == {1, 2, 3}
    with pytest.raises(AssertionError):
        sample_window_start(0, 3)


def test_rollout_window_placement():
    policy = make_policy()
    cond = make_conditions(policy)
    _, transitions, start = policy.rollout(cond, group_size=2, window_start=3)
    assert start == 3
    assert [tr.step for tr in transitions] == [3, 4]
    for tr in transitions:
        assert tr.t > 0.0 and math.isfinite(tr.sigma) and tr.sigma > 0.0


def test_transition_std_positive():
    tr = Transition(step=1, t=0.1, dt=0.05, sigma=1.5,
                    x_t=torch.zeros(1, 2, 3), x_next=torch.zeros(1, 2, 3),
                    logprob_old=torch.zeros(1))
    assert tr.std == pytest.approx(1.5 * math.sqrt(0.05))
