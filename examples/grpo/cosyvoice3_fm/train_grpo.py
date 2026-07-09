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
"""FlowTTS-GRPO trainer for the CosyVoice3 flow-matching module (arXiv:2606.23190).

Only the DiT estimator's LoRA adapters are trained; LLM / speech tokenizer /
vocoder stay frozen. Single GPU:

    python train_grpo.py --config conf/grpo.yaml --train_data data/train.jsonl \
        --model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B --output_dir exp/fm_grpo

Multi GPU (each rank rolls out its own prompts, LoRA grads are all-reduced):

    torchrun --nproc_per_node 8 train_grpo.py --config conf/grpo.yaml ...
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from flow_grpo.grpo_loss import gaussian_mean_kl, group_advantages, ppo_clip_loss  # noqa: E402
from flow_grpo.lora import inject_lora, lora_disabled, lora_parameters, lora_state_dict, load_lora_state_dict  # noqa: E402
from flow_grpo.policy import FlowGRPOPolicy  # noqa: E402
from rewards import RewardComposer  # noqa: E402
from rewards.asr import ASRReward  # noqa: E402
from rewards.dnsmos import DNSMOSReward  # noqa: E402
from rewards.speaker_sim import SpeakerSimilarityReward  # noqa: E402
from rollout import TTSRollout, load_cosyvoice3  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--train_data', required=True, help='jsonl with text/prompt_text/prompt_wav per line')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--resume', default='', help='resume from a lora checkpoint (.pt)')
    parser.add_argument('--seed', type=int, default=1986)
    return parser.parse_args()


def init_distributed():
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group(backend='nccl')
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    else:
        rank, world_size = 0, 1
    device = torch.device(f'cuda:{os.environ.get("LOCAL_RANK", 0)}' if torch.cuda.is_available() else 'cpu')
    return rank, world_size, device


def build_rewards(cfg, device):
    reward_device = cfg['rewards'].get('device', 'same')
    reward_device = str(device) if reward_device == 'same' else reward_device
    return {
        'ss': SpeakerSimilarityReward(cfg['rewards']['speaker_model'], device=reward_device),
        'asr': ASRReward(cfg['rewards']['zh_asr_model'], cfg['rewards']['en_asr_model'], device=reward_device),
        'mos': DNSMOSReward(cfg['rewards']['dnsmos_onnx'], device=reward_device),
    }


def compute_group_rewards(group, rollout_engine, reward_fns, weights, composer):
    """Fills group.reward_components / group.rewards in place."""
    wavs_16k = [rollout_engine.resample_for_reward(w) for w in group.wavs]
    prompt_wav_16k = rollout_engine.load_prompt_wav_16k(group.prompt['prompt_wav'])
    components = {}
    if weights.get('ss', 0) != 0:
        components['ss'] = reward_fns['ss'](wavs_16k, prompt_wav_16k)
    if weights.get('asr', 0) != 0:
        components['asr'] = reward_fns['asr'](wavs_16k, group.prompt['normalized_text'])
    if weights.get('mos', 0) != 0:
        components['mos'] = reward_fns['mos'](wavs_16k)
    group.reward_components = components
    group.rewards = composer.compose(components)
    return group


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    rank, world_size, device = init_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    scfg, gcfg, lcfg = cfg['sampling'], cfg['grpo'], cfg['lora']
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. models: everything frozen except LoRA on the DiT estimator
    frontend, llm, flow, hift, sample_rate = load_cosyvoice3(args.model_dir, device, cfg.get('fp16_llm', False))
    for p in list(llm.parameters()) + list(flow.parameters()) + list(hift.parameters()):
        p.requires_grad_(False)
    replaced = inject_lora(flow.decoder.estimator, rank=lcfg['rank'], alpha=lcfg['alpha'],
                           dropout=lcfg.get('dropout', 0.0))
    flow.decoder.estimator.to(device)
    params = lora_parameters(flow.decoder.estimator)
    for p in params:
        p.requires_grad_(True)
    num_trainable = sum(p.numel() for p in params)
    if rank == 0:
        logging.info(f'LoRA injected into {len(replaced)} linears, trainable params: {num_trainable / 1e6:.2f}M')
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        load_lora_state_dict(flow.decoder.estimator, ckpt['lora'])
        start_step = ckpt.get('step', 0)
        if rank == 0:
            logging.info(f'resumed LoRA from {args.resume} at step {start_step}')
    else:
        start_step = 0

    policy = FlowGRPOPolicy(flow,
                            n_timesteps=scfg['n_timesteps'],
                            noise_level=scfg['noise_level'],
                            window_size=scfg['window_size'],
                            window_start_min=scfg['window_start_min'],
                            window_start_max=scfg['window_start_max'],
                            logprob_reduction=scfg.get('logprob_reduction', 'mean'))
    rollout_engine = TTSRollout(args.model_dir, device, policy,
                                llm=llm, hift=hift, frontend=frontend, sample_rate=sample_rate)
    reward_fns = build_rewards(cfg, device)
    composer = RewardComposer(weights=cfg['rewards']['weights'], std_eps=gcfg.get('std_eps', 1e-6))

    optimizer = torch.optim.AdamW(params, lr=gcfg['lr'])
    total_steps = gcfg['total_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_steps))
    for _ in range(start_step):
        scheduler.step()

    # 2. data, sharded by rank
    with open(args.train_data) as f:
        data = [json.loads(line) for line in f if line.strip()]
    data = data[rank::world_size]
    assert len(data) > 0, 'no training prompts for this rank'
    if rank == 0:
        logging.info(f'{len(data)} prompts per rank, world size {world_size}')

    metrics_path = os.path.join(args.output_dir, f'metrics_rank{rank}.jsonl')
    group_size = gcfg['group_size']
    prompts_per_iter = gcfg['prompts_per_iter']
    updates_per_iter = gcfg['updates_per_iter']
    step, iteration = start_step, 0

    while step < total_steps:
        iteration += 1
        tic = time.time()

        # ---- rollout phase (theta_old) ----
        groups = []
        while len(groups) < prompts_per_iter:
            item = random.choice(data)
            try:
                group = rollout_engine.rollout_group(item, group_size)
            except Exception as e:
                logging.warning(f'rollout failed for "{item["text"][:40]}...": {e}')
                continue
            if group is None:
                continue
            group = compute_group_rewards(group, rollout_engine, reward_fns, composer.weights, composer)
            groups.append(group.detach_())

        rewards = torch.stack([g.rewards for g in groups]).to(device)          # (num_groups, G)
        advantages, valid = group_advantages(rewards, gcfg.get('std_eps', 1e-6))
        kept = [i for i in range(len(groups)) if valid[i]]
        dropped = len(groups) - len(kept)
        for i in kept:
            groups[i].advantages = advantages[i]

        # ---- update phase: exactly updates_per_iter optimizer steps on EVERY rank
        # (ranks may keep different group counts; the loop structure must stay
        # symmetric or the all_reduce calls below deadlock) ----
        iter_stats = []
        if len(kept) > 0:
            order = kept * ((updates_per_iter + len(kept) - 1) // len(kept))
            random.shuffle(order)
            chunks = [order[i::updates_per_iter] for i in range(updates_per_iter)]
        else:
            chunks = [[] for _ in range(updates_per_iter)]
        for chunk in chunks:
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            num_terms = sum(len(groups[i].transitions) for i in chunk)
            stats_acc = {'loss': 0.0, 'kl': 0.0, 'ratio_mean': 0.0, 'clip_frac': 0.0}
            for i in chunk:
                g = groups[i]
                for tr in g.transitions:
                    logp_new, mean_new = policy.transition_logprob(g.conditions, tr, g.group_size)
                    with lora_disabled(flow.decoder.estimator), torch.no_grad():
                        _, mean_ref = policy.transition_logprob(g.conditions, tr, g.group_size)
                    loss_clip, stats = ppo_clip_loss(logp_new, tr.logprob_old, g.advantages,
                                                     gcfg['clip_eps'])
                    mask = g.conditions['mask'].repeat(g.group_size, 1, 1)
                    kl = gaussian_mean_kl(mean_new, mean_ref, tr.std, mask)
                    loss = (loss_clip + gcfg['kl_beta'] * kl) / num_terms
                    loss.backward()
                    stats_acc['loss'] += loss.item()
                    stats_acc['kl'] += kl.item() / num_terms
                    stats_acc['ratio_mean'] += stats['ratio_mean'] / num_terms
                    stats_acc['clip_frac'] += stats['clip_frac'] / num_terms
            if world_size > 1:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(params, gcfg['grad_clip'])
            optimizer.step()
            scheduler.step()
            step += 1
            if chunk:
                iter_stats.append(stats_acc)

        # ---- logging / checkpointing ----
        record = {
            'iteration': iteration,
            'step': step,
            'lr': scheduler.get_last_lr()[0],
            'reward_mean': rewards.mean().item(),
            'dropped_groups': dropped,
            'time': round(time.time() - tic, 1),
        }
        for key in ('ss', 'asr', 'mos'):
            values = [v for g in groups for v in (g.reward_components or {}).get(key, [])]
            if values:
                record[f'reward_{key}'] = sum(values) / len(values)
        if iter_stats:
            for key in iter_stats[0]:
                record[key] = sum(s[key] for s in iter_stats) / len(iter_stats)
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        if rank == 0:
            logging.info(json.dumps(record))
            if iteration % cfg['logging'].get('save_every', 50) == 0 or step >= total_steps:
                ckpt_path = os.path.join(args.output_dir, f'lora_step{step}.pt')
                torch.save({'lora': lora_state_dict(flow.decoder.estimator),
                            'step': step, 'config': cfg}, ckpt_path)
                torch.save({'lora': lora_state_dict(flow.decoder.estimator),
                            'step': step, 'config': cfg}, os.path.join(args.output_dir, 'lora_last.pt'))
                logging.info(f'saved {ckpt_path}')

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
