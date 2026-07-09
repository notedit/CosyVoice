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
"""Speaker-similarity reward R_ss: ERes2Net cosine similarity in [0, 1].

Paper (arXiv:2606.23190): cosine similarity between ERes2Net embeddings of the
generated audio and the prompt audio. We use the 3D-Speaker ERes2Net checkpoint
from ModelScope; the exact checkpoint is configurable since the paper does not
pin a version.
"""

import logging
from typing import List

import torch
import torch.nn.functional as F


class SpeakerSimilarityReward:
    """Callable: (list of 16 kHz mono float tensors, prompt wav) -> list of scores."""

    def __init__(self,
                 model_id: str = 'iic/speech_eres2net_sv_zh-cn_16k-common',
                 device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def _lazy_init(self):
        if self._pipeline is not None:
            return
        from modelscope.pipelines import pipeline
        self._pipeline = pipeline(task='speaker-verification', model=self.model_id, device=self.device)
        logging.info(f'loaded speaker verification model {self.model_id}')

    def embed(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Extracts a speaker embedding from a (1, n) or (n,) 16 kHz waveform."""
        self._lazy_init()
        wav = wav_16k.squeeze().cpu().numpy()
        # 3D-Speaker pipelines expose compute_embedding via save_dict/model internals;
        # the public API accepts raw numpy input and returns the embedding on request.
        embedding = self._pipeline([wav], output_emb=True)['embs'][0]
        return torch.from_numpy(embedding).flatten()

    def __call__(self, wavs_16k: List[torch.Tensor], prompt_wav_16k: torch.Tensor) -> List[float]:
        prompt_emb = F.normalize(self.embed(prompt_wav_16k), dim=0)
        scores = []
        for wav in wavs_16k:
            emb = F.normalize(self.embed(wav), dim=0)
            # raw cosine similarity, directly comparable to the paper's SS2 metric;
            # group advantages are affine-invariant so no [0, 1] remap is needed
            scores.append(float(torch.dot(prompt_emb, emb).item()))
        return scores
