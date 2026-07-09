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
"""Intelligibility reward R_asr = 1 - CER (zh, Paraformer) / 1 - WER (en, Whisper-large-v3).

Matches the paper's setup: Paraformer transcribes Chinese, Whisper-large-v3
transcribes English; the error rate against the target text is mapped to
[0, 1] via max(0, 1 - ER).
"""

import logging
import re
import string
from typing import List

import torch


def _levenshtein(ref: List[str], hyp: List[str]) -> int:
    if len(ref) == 0:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


_ZH_CHARS = re.compile(r'[一-鿿]')
_PUNCT = set(string.punctuation) | set('，。！？；：“”‘’、《》（）【】…—·「」『』')


def contains_chinese(text: str) -> bool:
    return bool(_ZH_CHARS.search(text))


def normalize_zh(text: str) -> List[str]:
    """Character-level units: strip punctuation/whitespace, lowercase latin."""
    return [c.lower() for c in text if not c.isspace() and c not in _PUNCT]


def normalize_en(text: str) -> List[str]:
    """Word-level units: lowercase, strip punctuation."""
    text = ''.join(c if c not in _PUNCT else ' ' for c in text.lower())
    return text.split()


def error_rate(ref_text: str, hyp_text: str) -> float:
    """CER for Chinese references, WER otherwise."""
    if contains_chinese(ref_text):
        ref, hyp = normalize_zh(ref_text), normalize_zh(hyp_text)
    else:
        ref, hyp = normalize_en(ref_text), normalize_en(hyp_text)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / len(ref)


class ASRReward:
    """Callable: (list of 16 kHz mono wav tensors, target text) -> list of 1 - ER scores."""

    def __init__(self,
                 zh_model_id: str = 'paraformer-zh',
                 en_model_id: str = 'openai/whisper-large-v3',
                 device: str = 'cuda'):
        self.zh_model_id = zh_model_id
        self.en_model_id = en_model_id
        self.device = device
        self._paraformer = None
        self._whisper = None

    def _lazy_zh(self):
        if self._paraformer is None:
            from funasr import AutoModel
            self._paraformer = AutoModel(model=self.zh_model_id, disable_update=True, device=self.device)
            logging.info(f'loaded ASR model {self.zh_model_id}')
        return self._paraformer

    def _lazy_en(self):
        if self._whisper is None:
            from transformers import pipeline
            self._whisper = pipeline('automatic-speech-recognition', model=self.en_model_id,
                                     torch_dtype=torch.float16 if 'cuda' in str(self.device) else torch.float32,
                                     device=self.device)
            logging.info(f'loaded ASR model {self.en_model_id}')
        return self._whisper

    def transcribe(self, wav_16k: torch.Tensor, zh: bool) -> str:
        wav = wav_16k.squeeze().cpu()
        if zh:
            res = self._lazy_zh().generate(input=wav.numpy(), disable_pbar=True)
            return res[0]['text'] if res else ''
        res = self._lazy_en()({'raw': wav.numpy(), 'sampling_rate': 16000},
                              generate_kwargs={'language': 'english'})
        return res['text']

    def __call__(self, wavs_16k: List[torch.Tensor], target_text: str) -> List[float]:
        zh = contains_chinese(target_text)
        scores = []
        for wav in wavs_16k:
            try:
                hyp = self.transcribe(wav, zh)
            except Exception as e:  # a single undecodable sample should not kill the run
                logging.warning(f'ASR failed, assigning 0 reward: {e}')
                scores.append(0.0)
                continue
            scores.append(max(0.0, 1.0 - error_rate(target_text, hyp)))
        return scores
