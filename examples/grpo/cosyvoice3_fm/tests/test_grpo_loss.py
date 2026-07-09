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
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flow_grpo.grpo_loss import gaussian_mean_kl, group_advantages, ppo_clip_loss  # noqa: E402
from flow_grpo.lora import (  # noqa: E402
    inject_lora,
    lora_disabled,
    lora_parameters,
    lora_state_dict,
    load_lora_state_dict,
    merge_lora,
)


def test_group_advantages_normalization():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    adv, valid = group_advantages(rewards)
    assert valid.tolist() == [True, False]
    assert adv[1].abs().sum() == 0.0                      # degenerate group zeroed
    assert adv[0].mean().abs() < 1e-6                     # centered
    assert adv[0].std() == pytest.approx(1.0, abs=1e-5)   # unit std (torch std, ddof=1)
    # affine-invariance: shifting/scaling rewards leaves advantages unchanged
    adv2, _ = group_advantages(rewards * 3.0 + 7.0)
    assert torch.allclose(adv, adv2, atol=1e-5)


def test_ppo_clip_zero_advantage_zero_gradient():
    logp_new = torch.randn(8, requires_grad=True)
    logp_old = logp_new.detach() + 0.05 * torch.randn(8)
    loss, _ = ppo_clip_loss(logp_new, logp_old, torch.zeros(8))
    loss.backward()
    assert torch.allclose(logp_new.grad, torch.zeros(8))


def test_ppo_clip_direction_and_clipping():
    # positive advantage, ratio already above 1+eps -> clipped branch has no gradient
    logp_old = torch.zeros(1)
    logp_new = torch.tensor([1.0], requires_grad=True)  # ratio e ~ 2.72 > 1.2
    loss, stats = ppo_clip_loss(logp_new, logp_old, torch.ones(1), clip_eps=0.2)
    # min(ratio*A, clip(ratio)*A) = 1.2 -> constant w.r.t. theta
    assert loss.item() == pytest.approx(-1.2, abs=1e-4)
    loss.backward()
    assert torch.allclose(logp_new.grad, torch.zeros(1))
    assert stats['clip_frac'] == 1.0

    # in-range ratio -> gradient pushes logp up for positive advantage
    logp_new = torch.tensor([0.05], requires_grad=True)
    loss, _ = ppo_clip_loss(logp_new, logp_old, torch.ones(1), clip_eps=0.2)
    loss.backward()
    assert logp_new.grad.item() < 0.0  # minimizing loss increases logp


def test_gaussian_mean_kl():
    mean_new = torch.ones(2, 4, 5)
    mean_ref = torch.zeros(2, 4, 5)
    kl = gaussian_mean_kl(mean_new, mean_ref, std=2.0)
    assert kl.item() == pytest.approx(1.0 / (2 * 4.0), abs=1e-6)
    assert gaussian_mean_kl(mean_ref, mean_ref, std=2.0).item() == 0.0


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_q = torch.nn.Linear(8, 8)
        self.attn.to_k = torch.nn.Linear(8, 8)
        self.other = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.attn.to_q(x) + self.attn.to_k(x) + self.other(x)


def test_lora_inject_toggle_merge_roundtrip():
    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(3, 8)
    base_out = model(x)

    replaced = inject_lora(model, rank=2, alpha=4)
    assert sorted(replaced) == ['attn.to_k', 'attn.to_q']  # 'other' not targeted
    assert len(lora_parameters(model)) == 4

    # B init to zero => output unchanged right after injection
    assert torch.allclose(model(x), base_out, atol=1e-6)

    # make adapters non-trivial
    for p in lora_parameters(model):
        torch.nn.init.normal_(p, std=0.5)
    adapted_out = model(x)
    assert not torch.allclose(adapted_out, base_out, atol=1e-3)

    # reference policy: adapters disabled == base model
    with lora_disabled(model):
        assert torch.allclose(model(x), base_out, atol=1e-6)
    assert torch.allclose(model(x), adapted_out, atol=1e-6)  # re-enabled afterwards

    # save/load roundtrip
    state = lora_state_dict(model)
    model2 = Tiny()
    model2.load_state_dict(Tiny().state_dict(), strict=True)
    model2.load_state_dict(model.attn.to_q.base.state_dict(), strict=False)
    model2 = Tiny()
    inject_lora(model2, rank=2, alpha=4)
    # copy base weights so outputs are comparable
    model2.load_state_dict(model.state_dict(), strict=True)
    load_lora_state_dict(model2, state)
    assert torch.allclose(model2(x), adapted_out, atol=1e-6)

    # merge folds adapters into plain Linear, keeping outputs and stock keys
    merge_lora(model)
    assert isinstance(model.attn.to_q, torch.nn.Linear)
    assert torch.allclose(model(x), adapted_out, atol=1e-5)
    assert set(model.state_dict().keys()) == set(Tiny().state_dict().keys())
