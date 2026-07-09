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

from rewards import RewardComposer  # noqa: E402
from rewards.asr import error_rate, normalize_en, normalize_zh  # noqa: E402


def test_composer_weighted_std_normalization():
    composer = RewardComposer(weights={'ss': 1.0, 'asr': 1.0, 'mos': 0.4})
    components = {
        'ss': [0.7, 0.8, 0.9],
        'asr': [1.0, 0.5, 0.0],
        'mos': [3.0, 3.5, 4.0],
    }
    total = composer.compose(components)
    expected = torch.zeros(3)
    for key, lam in composer.weights.items():
        vals = torch.tensor(components[key])
        expected += lam * vals / vals.std()
    assert torch.allclose(total, expected, atol=1e-5)


def test_composer_constant_component_skips_normalization():
    composer = RewardComposer(weights={'ss': 1.0, 'asr': 1.0})
    components = {'ss': [0.5, 0.5, 0.5], 'asr': [0.2, 0.4, 0.6]}
    total = composer.compose(components)
    assert torch.isfinite(total).all()
    # constant component shifts all samples equally -> zero advantage contribution
    centered = total - total.mean()
    asr = torch.tensor(components['asr'])
    centered_asr = (asr / asr.std()) - (asr / asr.std()).mean()
    assert torch.allclose(centered, centered_asr, atol=1e-5)


def test_composer_ignores_unknown_and_missing():
    composer = RewardComposer(weights={'ss': 1.0, 'asr': 1.0, 'mos': 0.4})
    total = composer.compose({'ss': [0.1, 0.2], 'extra': [9.0, 9.0]})
    assert total.shape == (2,)


def test_error_rate_zh_cer():
    assert error_rate('今天天气很好', '今天天气很好') == 0.0
    assert error_rate('今天天气很好。', '今天天气很好') == 0.0     # punctuation stripped
    assert error_rate('今天天气很好', '今天天气很差') == pytest.approx(1 / 6)
    assert error_rate('今天', '今天天天') == pytest.approx(1.0)     # 2 insertions / 2 ref chars


def test_error_rate_en_wer():
    assert error_rate('hello world', 'Hello, world!') == 0.0
    assert error_rate('hello world', 'hello there world') == pytest.approx(0.5)
    assert normalize_en('Hello, World!') == ['hello', 'world']
    assert normalize_zh('你好, 世界！') == ['你', '好', '世', '界']
