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
"""CosyVoice3 rollout engine for FlowTTS-GRPO.

Pipeline per prompt (paper Fig.1): frontend extracts prompt tokens / mel /
speaker embedding; the FROZEN LLM autoregressively generates semantic tokens
ONCE; the FM policy then draws G stochastic samples via the windowed SDE, all
sharing the same conditions, so group reward differences are attributable to
the FM actions alone; HiFT (frozen) turns mels into waveforms for the rewards.
"""

import logging
import os
from typing import Dict, Optional

import torch
import torchaudio

from hyperpyyaml import load_hyperpyyaml

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav

from flow_grpo.buffer import GroupRollout
from flow_grpo.policy import FlowGRPOPolicy

# FSQ silent/breath tokens, mirrored from CosyVoice3Model (cosyvoice/cli/model.py)
SILENT_TOKENS = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
MAX_SILENT_TOKENS = 5
REWARD_SAMPLE_RATE = 16000


def load_cosyvoice3(model_dir: str, device: torch.device, fp16_llm: bool = False):
    """Loads frontend + llm/flow/hift from a Fun-CosyVoice3-0.5B directory,
    mirroring cosyvoice.cli.cosyvoice.CosyVoice3.__init__ but keeping the flow
    model trainable (no jit/trt/vllm)."""
    hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    if not os.path.exists(hyper_yaml_path):
        raise ValueError(f'{hyper_yaml_path} not found!')
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                 configs['feat_extractor'],
                                 os.path.join(model_dir, 'campplus.onnx'),
                                 os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
                                 os.path.join(model_dir, 'spk2info.pt'),
                                 configs['allowed_special'])
    llm, flow, hift = configs['llm'], configs['flow'], configs['hift']
    llm.load_state_dict(torch.load(os.path.join(model_dir, 'llm.pt'), map_location='cpu', weights_only=True), strict=True)
    flow.load_state_dict(torch.load(os.path.join(model_dir, 'flow.pt'), map_location='cpu', weights_only=True), strict=True)
    hift.load_state_dict({k.replace('generator.', ''): v
                          for k, v in torch.load(os.path.join(model_dir, 'hift.pt'), map_location='cpu',
                                                 weights_only=True).items()},
                         strict=True)
    llm.to(device).eval()
    if fp16_llm:
        llm.half()
    flow.to(device).eval()   # eval the whole run: dropout off keeps the policy density well-defined
    hift.to(device).eval()
    sample_rate = configs['sample_rate']
    del configs
    return frontend, llm, flow, hift, sample_rate


class TTSRollout:

    def __init__(self,
                 model_dir: str,
                 device: torch.device,
                 policy: FlowGRPOPolicy,
                 llm=None, hift=None, frontend=None, sample_rate: int = 24000):
        self.device = device
        self.policy = policy
        self.llm = llm
        self.hift = hift
        self.frontend = frontend
        self.sample_rate = sample_rate
        self.model_dir = model_dir

    @torch.no_grad()
    def generate_tokens(self, text: str, prompt_text: str, prompt_wav: str) -> Optional[Dict]:
        """Runs frontend + frozen LLM once; returns model inputs + generated tokens."""
        text = self.frontend.text_normalize(text, split=False)
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        model_input = self.frontend.frontend_zero_shot(text, prompt_text, prompt_wav, self.sample_rate, '')
        speech_tokens, num_silent = [], 0
        token_generator = self.llm.inference(
            text=model_input['text'].to(self.device),
            text_len=torch.tensor([model_input['text'].shape[1]], dtype=torch.int32).to(self.device),
            prompt_text=model_input['prompt_text'].to(self.device),
            prompt_text_len=torch.tensor([model_input['prompt_text'].shape[1]], dtype=torch.int32).to(self.device),
            prompt_speech_token=model_input['llm_prompt_speech_token'].to(self.device),
            prompt_speech_token_len=torch.tensor([model_input['llm_prompt_speech_token'].shape[1]],
                                                 dtype=torch.int32).to(self.device),
            embedding=model_input['llm_embedding'].to(self.device))
        for token in token_generator:
            if token in SILENT_TOKENS:
                num_silent += 1
                if num_silent > MAX_SILENT_TOKENS:
                    continue
            else:
                num_silent = 0
            speech_tokens.append(token)
        if len(speech_tokens) == 0:
            logging.warning(f'LLM generated no tokens for text "{text[:40]}...", skipping prompt')
            return None
        model_input['normalized_text'] = text
        model_input['speech_token'] = torch.tensor([speech_tokens], dtype=torch.int32, device=self.device)
        return model_input

    @torch.no_grad()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """(1, 80, T) generated-region mel -> (1, n) waveform at 24 kHz."""
        wav, _ = self.hift.inference(speech_feat=mel, finalize=True)
        return wav

    def rollout_group(self, item: dict, group_size: int,
                      generator: Optional[torch.Generator] = None) -> Optional[GroupRollout]:
        """item: {'text', 'prompt_text', 'prompt_wav'} -> GroupRollout with G wavs."""
        model_input = self.generate_tokens(item['text'], item['prompt_text'], item['prompt_wav'])
        if model_input is None:
            return None
        conditions = self.policy.prepare_conditions(
            token=model_input['speech_token'],
            prompt_token=model_input['flow_prompt_speech_token'].to(self.device),
            prompt_feat=model_input['prompt_speech_feat'].to(self.device),
            embedding=model_input['flow_embedding'].to(self.device))
        mel, transitions, window_start = self.policy.rollout(conditions, group_size, generator)

        prompt_mel_len = int(conditions['prompt_mel_len'])
        wavs = []
        for g in range(group_size):
            wav = self.vocode(mel[g:g + 1, :, prompt_mel_len:])
            wavs.append(wav.cpu())
        return GroupRollout(prompt={**item, 'normalized_text': model_input['normalized_text']},
                            conditions=conditions,
                            transitions=transitions,
                            window_start=window_start,
                            group_size=group_size,
                            wavs=wavs)

    def resample_for_reward(self, wav_24k: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.resample(wav_24k, self.sample_rate, REWARD_SAMPLE_RATE)

    def load_prompt_wav_16k(self, prompt_wav: str) -> torch.Tensor:
        return load_wav(prompt_wav, REWARD_SAMPLE_RATE)
