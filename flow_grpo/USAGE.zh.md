# FlowTTS-GRPO 使用说明（中文）

本目录是 **FlowTTS-GRPO**（[arXiv:2606.23190](https://arxiv.org/abs/2606.23190)）在
CosyVoice3 上的复现：只用在线 GRPO 微调 flow-matching（FM）模块，LLM / speech
tokenizer / vocoder 全部冻结。算法原理见 [README.md](README.md)，本文只讲怎么用。

目录：
[环境安装](#1-环境安装) · [模型下载](#2-模型与资源下载) · [数据准备](#3-数据准备) ·
[小规模验证](#4-小规模验证冒烟测试) · [正式训练](#5-正式训练) ·
[导出模型](#6-合并-lora-并导出) · [评测](#7-评测) · [常见问题](#8-常见问题)

## 1. 环境安装

推荐使用独立的 Python 3.10 环境（与仓库其他部分隔离）。以 uv 为例，在仓库根目录：

```bash
uv venv venv_fm_grpo --python 3.10
git submodule update --init --recursive   # third_party/Matcha-TTS 是必需的

VIRTUAL_ENV=$PWD/venv_fm_grpo uv pip install \
    --index-strategy unsafe-best-match \
    -r requirements.txt -r flow_grpo/requirements.txt
VIRTUAL_ENV=$PWD/venv_fm_grpo uv pip install "setuptools<81"
```

三个已知的安装坑（上面的命令已经处理了）：

| 问题 | 处理 |
|---|---|
| uv 多 index 解析失败（`protobuf==4.25` 不在 onnxruntime 的 Azure index 上） | 加 `--index-strategy unsafe-best-match` |
| `openai-whisper` 构建失败：`No module named 'pkg_resources'` | 构建约束 `setuptools<80`（写入文件后加 `--build-constraints <file>`），或先 `uv pip install "setuptools<80"` 再 `--no-build-isolation` |
| `lightning` 运行时报 `No module named 'pkg_resources'` | venv 内安装 `setuptools<81` |

modelscope 的 pipeline（ERes2Net 说话人模型）还需要 `addict simplejson sortedcontainers`，
且与 `datasets>=4` 不兼容——如果要用 `fetch_cv3_eval.py` 拉数据，安装 `datasets==3.0.1`。

安装完成后自检：

```bash
cd flow_grpo
PYTHONPATH=..:../third_party/Matcha-TTS ../venv_fm_grpo/bin/python -m pytest tests -q
# 期望：21 passed（纯 CPU，不需要模型和 GPU）
```

以下命令均假设**在 `flow_grpo/` 目录内**执行，并已激活环境
（`source ../venv_fm_grpo/bin/activate`，或像上面那样直接用 venv 里的 python）。
`run.sh` 会自动设置 `PYTHONPATH=..:../third_party/Matcha-TTS`；单独运行某个脚本时
需要手动带上这个 `PYTHONPATH`。

## 2. 模型与资源下载

```bash
bash run.sh -1 -1
```

做两件事：

- 从 ModelScope 下载 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（约 9GB）到
  `../pretrained_models/Fun-CosyVoice3-0.5B`。模型目录没有 `spk2info.pt` 是正常的，
  zero-shot 路径用不到它。
- 下载 DNSMOS P.835 的 onnx（`models/sig_bak_ovr.onnx`，约 1MB）。

另外两个奖励模型在**首次训练/评测时自动下载**（也可以提前跑一次触发缓存）：

- ERes2Net 说话人模型：`iic/speech_eres2net_sv_zh-cn_16k-common`（ModelScope）
- Paraformer 中文 ASR：`paraformer-zh`（FunASR）；英文用 `openai/whisper-large-v3`
  （HuggingFace，只在合成英文文本时才加载）

## 3. 数据准备

### 3.1 数据格式

训练数据是 jsonl，每行一个 prompt：

```json
{"text": "要合成的文本", "prompt_text": "参考音频的转写", "prompt_wav": "/abs/path/ref.wav"}
```

要求很宽松：`prompt_wav` 是克隆的参考音频（**3–30 秒**，超过 30 秒会被 speech
tokenizer 拒绝）；`text` 是让模型合成的内容，可以和音频无关。GRPO 是在线 RL，
不需要目标音频。

### 3.2 数据来源

| 场景 | 数据 | 说明 |
|---|---|---|
| 管线验证（几十条） | CV3-Eval | `fetch_cv3_eval.py` 一条命令搞定，见下 |
| 小规模训练（几千条） | AISHELL-3（OpenSLR SLR93） | 218 个说话人、~19GB，metadata 可用 `SparkAudio/voxbox` 的 jsonl |
| 复现论文数字 | WenetSpeech4TTS Premium（zh）+ LibriTTS-960（en） | 论文抽 ~40k prompts；重说话人多样性，不必全量下载 |

快速拉验证数据（从 HF `yuekai/CV3-Eval` 取 zero-shot 中文子集）：

```bash
python fetch_cv3_eval.py --split zero_shot_zh --num_samples 32
# 生成 data/raw.jsonl + data/prompt_wavs/*.wav
```

### 3.3 过滤、切分与难例增强

```bash
bash run.sh 0 1
```

- stage 0（`prepare_data.py`）：校验音频存在/可读/时长合规，切出 train/val。
  小数据集记得把 `--val_size`（默认 200）改小。
- stage 1（`make_hard_cases.py`）：按论文 Sec 3.4 做 LWR/SMR/GSR 重复增强，合成
  绕口令式难例（默认 20000 条，小规模验证给 `--num_samples 8` 就够）。产出
  `data/train_all.jsonl` = 正常 + 难例。

## 4. 小规模验证（冒烟测试）

`conf/grpo_smoke.yaml` 是缩小版配置（2 prompts/iter × G=4 × 2 iterations，共 4 个
优化步），单 GPU 几分钟跑完，用来确认整条链路（frontend → LLM 生成 → SDE rollout →
三路奖励 → PPO 更新 → 存 ckpt）没有问题：

```bash
PYTHONPATH=..:../third_party/Matcha-TTS python train_grpo.py \
    --config conf/grpo_smoke.yaml \
    --model_dir ../pretrained_models/Fun-CosyVoice3-0.5B \
    --train_data data/train_all.jsonl \
    --output_dir exp/fm_grpo_smoke
```

验证要点：

- 启动日志出现 `LoRA injected into 132 linears, trainable params: 10.09M`
  （和论文一致；数字不对说明模型或 LoRA 配置有问题）；
- `exp/fm_grpo_smoke/metrics_rank0.jsonl` 每次迭代追加一行，`reward_ss` /
  `reward_asr` / `reward_mos` 都是有限值；
- 产出 `lora_last.pt`（只含 LoRA 权重，~40MB 量级）。

## 5. 正式训练

```bash
bash run.sh 2 2        # 默认 torchrun 8 卡，用 conf/grpo.yaml
```

论文设定：512 样本/迭代 = 8 GPU × 8 prompts × G=8，共 10k 优化步。改卡数时在
`run.sh` 里调 `num_gpus`，并按 GPU 数等比调 `grpo.prompts_per_iter`（它是**每卡**的
prompt 数）。恢复训练用 `--resume exp/fm_grpo/lora_step*.pt`。

`conf/grpo.yaml` 中标 `[paper]` 的值来自论文，不建议动；标 `[default]` 的（PPO clip
ε、KL 权重 β 等）论文未给，可调。

**监控**：`tail -f exp/fm_grpo/metrics_rank0.jsonl`，关键字段：

| 字段 | 含义 | 期望行为 |
|---|---|---|
| `reward_mean` | 组归一化后的总奖励 | 随训练上升 |
| `reward_ss` / `reward_asr` / `reward_mos` | 三路原始奖励均值 | ss、mos 上升，asr 基本持平（论文结论）|
| `ratio_mean` | PPO 重要性比率 | 接近 1，偏离过大说明 rollout 与更新脱节 |
| `clip_frac` | 被 clip 的比例 | 长期 >0.5 时考虑调小 lr 或 clip_eps |
| `kl` | 与参考策略（LoRA 关闭）的 KL | 缓慢增长，爆炸则加大 `kl_beta` |
| `dropped_groups` | 组内奖励方差为 0 被丢弃的组 | 偶尔非零正常，持续偏高说明奖励无区分度 |

## 6. 合并 LoRA 并导出

```bash
bash run.sh 3 3
# 即：python export_merged.py --model_dir ../pretrained_models/Fun-CosyVoice3-0.5B \
#        --lora_ckpt exp/fm_grpo/lora_last.pt --output exp/fm_grpo/flow_grpo.pt
```

把 LoRA 权重合并回**标准格式的 `flow.pt`**（脚本会断言 key 与原始 checkpoint 完全
一致）。合并后的文件可以直接替换预训练目录里的 `flow.pt`，用原版 `CosyVoice3` 类
加载，CFG / 流式 / TRT 路径全部不受影响。

## 7. 评测

### 7.1 测试数据

与训练同格式的 jsonl。中文条目报 CER，英文条目报 WER（按 `text` 是否含汉字自动
判断）。来源建议：

- **CV3-Eval**（HF `yuekai/CV3-Eval`）：`fetch_cv3_eval.py --split zero_shot_zh` 直接生成；
- **Seed-TTS-Eval**（[BytedanceSpeech/seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)）：
  论文主基准，其 meta 文件转成本格式即可。

注意测试集要和训练集分开（例如 `fetch_cv3_eval.py` 换一个 split，或用
`data/val.jsonl`）。

### 7.2 跑评测：baseline vs GRPO 对比

```bash
# baseline（原始 flow.pt）
python evaluate.py --model_dir ../pretrained_models/Fun-CosyVoice3-0.5B \
    --test_data data/test_zh.jsonl --output_dir exp/eval_baseline

# GRPO（合并后的 flow checkpoint）
python evaluate.py --model_dir ../pretrained_models/Fun-CosyVoice3-0.5B \
    --flow_ckpt exp/fm_grpo/flow_grpo.pt \
    --test_data data/test_zh.jsonl --output_dir exp/eval_grpo
```

即 `bash run.sh 4 4`。有用的开关：

- `--skip_synthesis`：复用 `output_dir/wavs/` 里已合成的音频，只重算指标；
- `--fp16`：合成用半精度；
- `--config`：奖励/评测模型 ID 从这里读（默认 `conf/grpo.yaml`）。

合成是断点续跑的（已存在的 wav 会跳过），中断后重跑同一命令即可。

### 7.3 输出与指标含义

每个 `--output_dir` 下产出：

- `wavs/NNNNNN.wav`：按测试集行号命名的合成音频；
- `per_item.jsonl`：逐条结果 `{idx, text, hyp(ASR转写), err, ss, dnsmos_p835_ovrl}`，
  用于定位坏例；
- `summary.json`：汇总指标。

| summary 字段 | 含义 | 对应论文指标 |
|---|---|---|
| `cer_zh` | 中文字符错误率（Paraformer 转写，越低越好） | CER |
| `wer_en` | 英文词错误率（Whisper-large-v3 转写） | WER |
| `ss_eres2net` | 与 prompt 音频的 ERes2Net 余弦相似度（越高越好） | **SS2** |
| `dnsmos_p835_ovrl` | DNSMOS P.835 OVRL 感知质量分（越高越好） | DNSMOS |

对比两个 `summary.json`，期望方向（论文 Seed-TTS-Eval-zh 上的参考值）：

| 指标 | Baseline | + FM-GRPO |
|---|---|---|
| SS2 (ERes2Net) | 0.830 | **0.859** ↑ |
| DNSMOS OVRL | 3.353 | **3.536** ↑ |
| CER | 1.20 | 1.26（基本持平）|

### 7.4 与论文严格对齐（SS1 / UTMOS）

论文的 **SS1 用 WavLM 嵌入**，本脚本报的是 ERes2Net（= 论文 SS2，也是训练奖励）。
要得到可与论文/其他系统直接对比的 SS1、UTMOS 数字，用官方
[seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) 工具箱对
`output_dir/wavs/` 里保存的音频离线打分即可（该工具箱依赖较重，建议放在单独环境）。

另外注意：ERes2Net 同时是训练奖励和评测指标，存在 reward hacking 的可能——结论
以独立的 SS1/主观听感为准，`ss_eres2net` 大涨但 SS1 不动就要警惕。

## 8. 常见问题

- **`ModuleNotFoundError: matcha`**：没初始化 submodule 或没带
  `PYTHONPATH=..:../third_party/Matcha-TTS`。
- **modelscope 报 `No module named 'addict'` 或 datasets 导入错误**：
  `uv pip install addict simplejson sortedcontainers "datasets==3.0.1"`。
- **模型下载 404**：ModelScope ID 必须带 `-2512` 后缀
  （`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`）。
- **ffmpeg 警告**：无害，音频加载走 torchaudio 后端。
- **rollout 偶发 warning「LLM generated no tokens」**：单条 prompt 失败会自动跳过
  换下一条，属正常容错；大面积出现时检查数据的 `prompt_text` 与音频是否匹配。
- **显存**：单卡冒烟测试约需 10GB+（0.5B LLM fp16 + FM + HiFT + 奖励模型同卡）；
  紧张时把 `rewards.device` 指到另一张卡，或调小 `group_size`。
