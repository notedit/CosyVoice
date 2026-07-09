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
"""Perceptual-quality reward R_mos: DNSMOS P.835 OVRL score (paper Sec 3.3).

Runs the Microsoft DNS-Challenge ``sig_bak_ovr.onnx`` model on 16 kHz audio and
returns the polynomial-fitted OVRL MOS, following the reference
``dnsmos_local.py`` implementation (9.01 s sliding windows, 1 s hop, averaged).

Model download (user side):
    wget https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
"""

import logging
from typing import List

import numpy as np
import torch

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01  # seconds, fixed by the onnx model


class DNSMOSReward:

    def __init__(self, onnx_path: str, device: str = 'cuda'):
        self.onnx_path = onnx_path
        self.device = device
        self._session = None
        # non-personalized polynomial mappings from dnsmos_local.py
        self._p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self._p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self._p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def _lazy_init(self):
        if self._session is None:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in str(self.device) \
                else ['CPUExecutionProvider']
            self._session = ort.InferenceSession(self.onnx_path, providers=providers)
            logging.info(f'loaded DNSMOS model {self.onnx_path}')
        return self._session

    def score(self, wav_16k: torch.Tensor) -> float:
        """P.835 OVRL MOS of a mono 16 kHz waveform."""
        session = self._lazy_init()
        audio = wav_16k.squeeze().cpu().numpy().astype('float32')
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = np.concatenate([audio, audio])
        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        ovr_scores = []
        for i in range(num_hops):
            seg = audio[int(i * SAMPLING_RATE): int((i + INPUT_LENGTH) * SAMPLING_RATE)]
            if len(seg) < len_samples:
                continue
            raw = session.run(None, {'input_1': seg[np.newaxis, :]})[0][0]
            ovr_scores.append(float(self._p_ovr(raw[2])))
        return float(np.mean(ovr_scores)) if ovr_scores else 0.0

    def __call__(self, wavs_16k: List[torch.Tensor]) -> List[float]:
        scores = []
        for wav in wavs_16k:
            try:
                scores.append(self.score(wav))
            except Exception as e:
                logging.warning(f'DNSMOS failed, assigning 0 reward: {e}')
                scores.append(0.0)
        return scores
