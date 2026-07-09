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
"""Multi-objective reward composition (paper Sec 3.3, weighted scheme).

    R = lambda_ss * R_ss / std_g(R_ss) + lambda_asr * R_asr / std_g(R_asr)
        + lambda_mos * R_mos / std_g(R_mos)

where std_g is the standard deviation *within the group* of G samples. The
std-normalization makes the lambdas meaningful across objectives with very
different scales; the paper's best config is lambda = (1.0, 1.0, 0.4).

Individual reward backends live in speaker_sim.py / asr.py / dnsmos.py and are
lazily initialized so this package imports without GPU dependencies.
"""

from typing import Dict, List, Optional

import torch


class RewardComposer:

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 std_eps: float = 1e-6):
        self.weights = weights if weights is not None else {'ss': 1.0, 'asr': 1.0, 'mos': 0.4}
        self.std_eps = std_eps

    def compose(self, components: Dict[str, List[float]]) -> torch.Tensor:
        """Combines per-sample component rewards of ONE group into (G,) rewards.

        Components whose within-group std is ~0 contribute a constant, which the
        group-relative advantage cancels anyway; we skip the normalization there
        to avoid dividing by ~0.
        """
        keys = [k for k in self.weights if k in components]
        assert keys, f'no known reward components in {list(components)}'
        length = len(components[keys[0]])
        total = torch.zeros(length, dtype=torch.float32)
        for key in keys:
            values = torch.tensor(components[key], dtype=torch.float32)
            assert len(values) == length, f'component {key} has inconsistent group size'
            std = values.std()
            if std > self.std_eps:
                values = values / std
            total = total + self.weights[key] * values
        return total


__all__ = ['RewardComposer']
