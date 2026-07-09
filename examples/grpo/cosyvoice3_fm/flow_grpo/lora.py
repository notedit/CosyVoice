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
"""Minimal self-contained LoRA for the CosyVoice3 DiT estimator.

Deliberately dependency-free (no peft): the DiT is not a HuggingFace model, and
we need exact control over adapter toggling (reference policy = adapters off)
and merging back into a stock ``flow.pt`` checkpoint.

Paper config (arXiv:2606.23190): rank=32, alpha=64, targeting every attention
q/k/v/out projection and both feed-forward linears of each DiT block, which for
the CosyVoice3 DiT (dim=1024, depth=22, ff_mult=2) yields ~10.09M trainable
parameters (2.78% of the FM module) - matching the paper.
"""

import contextlib
import math
import re
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Matches DiT module paths: transformer_blocks.N.attn.to_{q,k,v}, attn.to_out.0,
# ff.ff.0.0 (project_in Linear) and ff.ff.2 (project_out Linear).
DEFAULT_TARGET_PATTERNS = [
    r'\.attn\.to_q$',
    r'\.attn\.to_k$',
    r'\.attn\.to_v$',
    r'\.attn\.to_out\.0$',
    r'\.ff\.ff\.0\.0$',
    r'\.ff\.ff\.2$',
]


class LoRALinear(nn.Module):
    """nn.Linear wrapped with a low-rank residual: y = Wx + b + (alpha/r) * B A x."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True
        self.merged = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled and not self.merged:
            out = out + F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


def _iter_lora_layers(model: nn.Module) -> Iterator[Tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def inject_lora(model: nn.Module, rank: int = 32, alpha: float = 64.0, dropout: float = 0.0,
                target_patterns: Iterable[str] = tuple(DEFAULT_TARGET_PATTERNS)) -> List[str]:
    """Replaces matching nn.Linear submodules with LoRALinear in place.

    Returns the list of replaced qualified module names.
    """
    patterns = [re.compile(p) for p in target_patterns]
    replaced = []
    # snapshot first: we mutate the module tree while iterating
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and any(p.search('.' + name) for p in patterns)]
    for name, module in targets:
        parent = model
        *path, leaf = name.split('.')
        for part in path:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        lora = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        if leaf.isdigit():
            parent[int(leaf)] = lora
        else:
            setattr(parent, leaf, lora)
        replaced.append(name)
    return replaced


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for _, layer in _iter_lora_layers(model):
        params += [layer.lora_A, layer.lora_B]
    return params


def set_lora_enabled(model: nn.Module, enabled: bool):
    for _, layer in _iter_lora_layers(model):
        layer.enabled = enabled


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    """Context manager exposing the frozen base model (the GRPO reference policy)."""
    set_lora_enabled(model, False)
    try:
        yield model
    finally:
        set_lora_enabled(model, True)


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Folds every adapter into its base weight and restores plain nn.Linear modules,
    so the resulting state_dict is key-compatible with the stock flow checkpoint."""
    layers = list(_iter_lora_layers(model))
    for name, layer in layers:
        layer.merge()
        parent = model
        *path, leaf = name.split('.')
        for part in path:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        if leaf.isdigit():
            parent[int(leaf)] = layer.base
        else:
            setattr(parent, leaf, layer.base)
    return model


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, layer in _iter_lora_layers(model):
        state[f'{name}.lora_A'] = layer.lora_A.detach().cpu()
        state[f'{name}.lora_B'] = layer.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    consumed = 0
    for name, layer in _iter_lora_layers(model):
        layer.lora_A.data.copy_(state[f'{name}.lora_A'].to(layer.lora_A.device))
        layer.lora_B.data.copy_(state[f'{name}.lora_B'].to(layer.lora_B.device))
        consumed += 2
    if consumed != len(state):
        raise ValueError(f'LoRA state mismatch: model consumed {consumed} tensors, checkpoint has {len(state)}')
