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
"""Merges a trained LoRA checkpoint into a stock CosyVoice3 flow.pt.

The output file is key-compatible with the original checkpoint, so it can be
dropped into the pretrained model directory (or passed around) and loaded by
the standard CosyVoice3 class - CFG, streaming and TRT export keep working.

    python export_merged.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
        --lora_ckpt exp/fm_grpo/lora_last.pt --output flow_grpo.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from hyperpyyaml import load_hyperpyyaml  # noqa: E402

from flow_grpo.lora import inject_lora, load_lora_state_dict, merge_lora  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--lora_ckpt', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = get_args()
    with open(os.path.join(args.model_dir, 'cosyvoice3.yaml')) as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(args.model_dir, 'CosyVoice-BlankEN'),
                                                 'llm': None, 'hift': None})
    flow = configs['flow']
    flow.load_state_dict(torch.load(os.path.join(args.model_dir, 'flow.pt'),
                                    map_location='cpu', weights_only=True), strict=True)

    ckpt = torch.load(args.lora_ckpt, map_location='cpu', weights_only=True)
    lora_cfg = ckpt.get('config', {}).get('lora', {'rank': 32, 'alpha': 64})
    inject_lora(flow.decoder.estimator, rank=lora_cfg['rank'], alpha=lora_cfg['alpha'])
    load_lora_state_dict(flow.decoder.estimator, ckpt['lora'])
    merge_lora(flow.decoder.estimator)

    state = flow.state_dict()
    # sanity: merged model must be key-compatible with the stock checkpoint
    original_keys = set(torch.load(os.path.join(args.model_dir, 'flow.pt'),
                                   map_location='cpu', weights_only=True).keys())
    merged_keys = set(state.keys())
    assert merged_keys == original_keys, \
        f'key mismatch after merge: +{merged_keys - original_keys} -{original_keys - merged_keys}'
    torch.save(state, args.output)
    print(f'merged flow checkpoint (step {ckpt.get("step", "?")}) saved to {args.output}')


if __name__ == '__main__':
    main()
