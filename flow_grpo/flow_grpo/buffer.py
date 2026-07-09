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
"""Rollout storage for one GRPO iteration."""

import dataclasses
from typing import Dict, List, Optional

import torch

from .sde_solver import Transition


@dataclasses.dataclass
class GroupRollout:
    """Everything produced for one prompt group (G samples sharing conditions)."""
    prompt: dict                                # the source data item (text / prompt_wav / ...)
    conditions: Dict[str, torch.Tensor]         # FM conditioning, batch dim 1
    transitions: List[Transition]               # ws stochastic transitions, tensors (G, 80, T)
    window_start: int
    group_size: int
    wavs: Optional[List[torch.Tensor]] = None   # G generated waveforms (1, n) at 24 kHz
    reward_components: Optional[Dict[str, List[float]]] = None
    rewards: Optional[torch.Tensor] = None      # (G,) composed rewards
    advantages: Optional[torch.Tensor] = None   # (G,)

    def detach_(self):
        for tr in self.transitions:
            tr.x_t = tr.x_t.detach()
            tr.x_next = tr.x_next.detach()
            tr.logprob_old = tr.logprob_old.detach()
        return self
