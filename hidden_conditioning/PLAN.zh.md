# CosyVoice3 Hidden-State Conditioning PoC — 实施计划

冻结 LM，把其最后一层 hidden state 经零初始化 projector 注入 flow 的 token embedding，
与纯 token 基线做受控微调对比，两周内回答一个决策问题：

> **LM hidden 能否在不伤可懂度（CER/WER）与音色（SIM）的前提下，提升韵律/自然度/表现力？**

不碰训练基建、不碰流式、不动 LM。全部工作是微调 + 评测。

---

## 1. 背景与假设

- **信息瓶颈**：FSQ speech token 是 25 Hz × log2(6561) ≈ 12.7 bit/token ≈ 317 bit/s 的硬瓶颈。
  LM 最后层 hidden（896 维连续）保留了量化前的句级语义、韵律意图和 token 分布信息。
- **先例**：TorToiSe → IndexTTS 一系把 AR latent 喂给解码器；IndexTTS-2 的消融（去掉 GPT
  latent 则稳定性/表现力下降）证明信息有用。注意：那是**从头训练**的证据，本 PoC 验证的是
  "微调期追加"能兑现多少——这正是要回答的问题。
- **为什么冻结 LM 的 PoC 特别干净**：projector 末层零初始化 ⇒ 第 0 步与官方 `flow.pt`
  逐位一致，训练只会在 hidden 通路真正降低 FM loss 时偏离基线，归因链路无歧义。
  且 LM 冻结共享 ⇒ 评测可做**同 token 配对对比**，消掉采样方差。

**三种结局都有决策价值**：
1. **Go**：主观收益确认、护栏未破 → 进入流式集成 / 更宽注入通道的正式版。
2. **容量受限**：val loss 分离但主观无感 → 80 维 mu 通道是瓶颈，评估加宽注入（改 DiT 接口）。
3. **No-go**：val loss 不分离（对齐已排查）→ hidden 在 FM 工作点上冗余，归档结论。

---

## 2. 技术设计

### 2.1 目标模型

**CosyVoice3（Fun-CosyVoice3-0.5B）**，理由：
- 本 fork 的评测基建（`flow_grpo/evaluate.py`、AISHELL-3 基线数字、8 卡评测脚本）全部对准它；
- `examples/libritts/cosyvoice3/run.sh` stage 5 已循环 `llm flow hifigan` 并支持
  `--checkpoint $pretrained_model_dir/flow.pt` 继续训练，`train_conf` 已是 SFT 配置
  （lr=1e-5, constantlr, warmup 2500, accum_grad 2, amp）；
- `cosyvoice/bin/train.py` 加载 checkpoint 用 `strict=False`（train.py:138），新增 projector
  的缺失键不阻塞加载。

⚠️ run.sh stage 5 有一句过期提示 "We only support llm traning for now"——**冒烟日必须先验证
CV3 flow 监督训练路径可跑**；若有坑，整套方案原样退到 CosyVoice2
（`examples/libritts/cosyvoice2`，flow 训练路径更成熟，注入点同构，见 §2.3）。

### 2.2 Hidden 提取（训练态，teacher forcing）

新增独立 util（建议 `hidden_conditioning/extract.py`，或挂在 LM 类上），**不要复用**
`CosyVoice3LM.forward` 的 `prepare_lm_input_target`——它有 50% 概率走 5:15 bistream
交错分支（llm.py:318），与离线推理布局不符。

按**推理布局**构造 unistream 序列，一次 no-grad 前向：

```
lm_input = [ sos_emb,
             embed_tokens(instruct_token),   # batch 自带 instruct_token 字段
             embed_tokens(text_token),
             task_id_emb,
             speech_embedding(speech_token) ]
hidden = llm(lm_input).hidden_states[-1]     # Qwen2Encoder 本来就返回它（llm.py:240）
```

**对齐规则（唯一权威，全项目引用此处）**：
设 P = 1 + T_instruct + T_text + 1（sos 到 task_id 的 prefix 长度，0-based 下 task_id
位于 P−1）。预测 `speech_token[i]` 的 hidden 在输出位置 **(P−1)+i**，即：

```
hidden_for_flow = lm_output[:, P-1 : P-1+T_speech, :]   # 从 task_id 位置起，长 T_speech
```

依据：unistream 的 `lm_target = [IGNORE]*(1+T_instr+T_text) + speech + [eos]`，
logits 位置 j 训练 target[j]，target 中 speech[0] 的下标恰为 P−1（task_id 输入位置）。

dtype：LM 以 bf16 no-grad 前向，输出 cast 到 flow 的计算 dtype。

### 2.3 注入（flow 侧）

`CausalMaskedDiffWithDiT`（CV2 则为 `CausalMaskedDiffWithXvec`，注入点同构）：

```python
# __init__，由 config 开关 use_llm_hidden 控制
self.hidden_proj = nn.Sequential(nn.LayerNorm(896), nn.Linear(896, input_size))
nn.init.zeros_(self.hidden_proj[1].weight)
nn.init.zeros_(self.hidden_proj[1].bias)

# forward / inference，flow.py:343 一处（CV3；CV2 在 flow.py:209/254）
token = self.input_embedding(torch.clamp(token, min=0)) * mask
token = token + self.hidden_proj(llm_hidden) * mask * hidden_keep   # 新增
```

- **hidden dropout**：训练时按样本以 p=0.1~0.2 独立把 `hidden_keep` 置 0（在 CFG 之外）。
  两个作用：保留纯 token 回退能力（S2⁻ 消融合法），缓解 exposure bias。
- **CFG 自动兼容**：注入发生在 mu 生成之前。训练 `training_cfg_rate` 按样本置零
  mu/spks/cond（flow_matching.py:185-189）时 hidden 已折入 mu；推理 uncond 分支只给
  `mu_in[0]` 赋值（flow_matching.py:105），同理。**无需单独设计。**
- **DiT/TRT estimator 接口（x, mask, mu, t, spks, cond）完全不变**，TRT 导出不受影响。

### 2.4 训练循环集成

- flow 训练 batch 已自带 `text_token / instruct_token / speech_token / speech_feat /
  embedding`（processor.py:395-414），**on-the-fly 提取 hidden，parquet 格式零改动**。
  代码先例：`SpeechTokenExtractor` 已在 forward 内在线抽 token（flow.py:325）。
- 冻结 LM（~1GB bf16）挂在 flow module 上但 `requires_grad_(False)`，不进 optimizer；
  DDP 下确认不参与梯度同步（放 buffer 外持有，或核对 find_unused_parameters 行为）。
- 开销预估：LM 一次 no-grad 前向 vs 22 层 DiT 反向，step 时间增幅应 <20%，冒烟日实测。

### 2.5 推理链路

- **生成段**：解码循环里 `y_pred`（llm.py:539-549）就是预测当前 token 的 hidden，
  随 token 一起 yield，**零额外算力**。
- **prompt 段**：首步 `forward_one_step` 已对全 prefix
  `[sos, text, task_id, prompt_speech]` 前向，输出按同一对齐规则切片
  `[P-1 : P-1+T_prompt]` 即为 prompt token 对应的 hidden。
- `cli/model.py` 的 llm→flow 传值处增加 hidden 流转（tts 与 token2wav 路径）。
- **PoC 只支持 HF eager 路径**：vllm 分支（`inference_wrapper`）拿不到 hidden，
  记录为生产化 gap（需 vllm hidden 输出支持，或后续蒸馏回纯 token），不阻塞。

### 2.6 单元测试（合入前必须绿；沿用 `flow_grpo/tests/` 的 CPU 测试传统）

1. **对齐自检**：hidden 切片过 `llm_decoder` 的 argmax 对 teacher token 的命中率
   ≈ LM teacher-forcing acc（~70%+）；错位一格会跌到接近乱猜——这是 off-by-one 的探雷器。
2. **零初始化等价**：`use_llm_hidden=True` 且 projector 为初始权重时，flow 输出与
   baseline **逐位一致**（forward 与 inference 各测一次）。
3. **长度一致**：hidden 长度 == token 长度；prompt+生成拼接后仍一致。

---

## 3. 实验矩阵与数据

### 3.1 系统矩阵

| 系统 | 说明 | 作用 |
|---|---|---|
| S0 | 官方 `flow.pt`，不动 | 绝对质量锚点 |
| S1 | token-only，同数据微调 | **控制臂**（吸收"微调数据分布"效应）|
| S2 | token+hidden 微调 | 实验臂 |
| S2⁻ | S2 推理时 hidden 置零 | 免费消融：分离"训练正则化" vs "推理时信息"|

S1/S2 严格同数据、同步数、同超参、同 seed。**没有 S1 的对比无效**——
不许拿 S2 直接对 S0 下结论。

### 3.2 数据配方

三条选择原则（重要性高于具体数据集）：
1. **原生采样率 ≥24 kHz**（24k mel 监督，16k 上采样高频为空，两臂同变闷，绝对听感失真）；
2. **转写质量不对称地只伤实验臂**（S1 根本不用文本；S2 的 teacher-forcing hidden 完全依赖
   文本正确性，ASR 噪声 ⇒ 系统性低估收益）；
3. **韵律丰富度是命门**（假设是韵律收益，纯朗读语料可能训了也测不出）。

| 档位 | 数据 | 量 | 用途 |
|---|---|---|---|
| 冒烟 | AISHELL-3（85h, 44.1k, 人工转写）+ LibriTTS train-clean-100 | ~185h | 跑通双臂、L0 判读 |
| 主实验 zh | Emilia-zh 高分子集（24k, DNSMOS 头部 + **Paraformer 复核转写 CER<5% 双门控**）+ AISHELL-3 | 400–600h | 韵律丰富度主来源 |
| 主实验 en | LibriTTS-R 585h（富余加 Emilia-en 100–200h）| ~600h | recipe 现成 |
| 备选 | WenetSpeech4TTS Premium（945h 但 **16k**，带宽警告）| — | 仅补充 |
| 探针 | ESD（zh+en 各 10 人 × 5 情感）| — | **不进训练**，评测文本/prompt 来源 |

总量 ~1000h、zh:en ≈ 1:1。规模不是瓶颈，不必贪多。
新语料只需写 prepare 脚本产出 `wav.scp/text/utt2spk/spk2utt/instruct`
（instruct 统一 `"You are a helpful assistant.<|endofprompt|>"`），
stage 1–3（campplus / speech_tokenizer_v3 / parquet）全部复用现成工具。

### 3.3 训练配置

- 8 GPU torch_ddp + amp；`train_conf` 现值：lr 1e-5, constantlr, warmup 2500, accum_grad 2。
- 冒烟：双臂各 ~10k step；主实验：双臂各 50–100k step（2–3 天墙钟）。
- 每 ~2k step 用 200 条 dev 子集快评（CER+SS2+DNSMOS，套用 `run_eval_8gpu_ckpts.sh` 模式）。

---

## 4. 评测协议

### L0 — 训练期（免费，最早信号，D5 决策点）

- 固定 held-out 集上 S1 vs S2 的 **val loss 分离度**：零初始化保证起点重合，
  分离度即"hidden 提供的可用信息量"。
- projector 输出范数监控：恒零 = 死通路，**先查对齐再谈其他**。
- **L0 不分离 ⇒ 不进主实验**，回 §2.6 的对齐测试排查。

### L1 — 客观自动评测

| 工具 | 指标 | 说明 |
|---|---|---|
| seed-tts-eval 官方 | test-zh CER（Paraformer）、test-en WER（Whisper-large-v3）、test-hard、SS1（WavLM-SV）| 与 CosyVoice3 论文可比 |
| `flow_grpo/evaluate.py` | SS2（ERes2Net）、DNSMOS P.835 | 现成，双 SIM 口径都留 |
| UTMOS | 自然度 proxy | 比 DNSMOS 对韵律敏感，便宜 |
| 自建表现力集 | 100–200 条情感/疑问/强调/长复句（可借 ESD 文本域）| **防假阴性关键**：seed-tts-eval 偏朗读，对假设不敏感 |

**配对设计（本方案独有红利，必须用）**：每条测试文本 LM 只采样一次，缓存 token+hidden，
四系统用完全相同的 token 解码。报告逐条 paired delta + bootstrap CI（或 Wilcoxon），
不报裸点数——非配对下 CER 差 0.1 以内就是噪声。

### L2 — 主观评测（主终点）

- **CMOS/AB 强制选择**：分层抽 30–50 条（普通 zh/en、难例、表现力、长句）× 5–10 人；
  同 token 配对使该样本量足以显著。复用 `LISTENING.md` HTML 听音索引模式。
- **SMOS 单独做**：直接检验"客观 SIM 微降但主观相似度升"假设——没有 SMOS 这条预警无法落地。

### L3 — 诊断探针（各 10 分钟量级）

1. **错配 hidden**：正确 token + 换一句话的 hidden 喂 S2 ⇒ 韵律应可听地改变；
   毫无变化 = flow 推理时忽略 hidden，L1/L2 差异全是训练正则化效应。
2. **采样温度应力**：调高 top_k/温度，S2 是否比 S1 退化更快 ⇒ exposure bias 定量，
   决定是否加 hidden 噪声/更高 dropout。
3. **S2⁻ vs S1**：分离正则化收益 vs 推理时信息收益。

### 判读标准（双向防"单指标骗"）

- **主终点**：表现力集 paired CMOS ≥ +0.2，或 AB 偏好 ≥60%（p<0.05）。
- **护栏**（相对 S1）：CER 恶化 ≤ 0.1 绝对值；SS1/SS2 下降 ≤ 0.01；DNSMOS 持平。
- SIM 降 0.01 内且 SMOS 升 ⇒ 按韵律变化解读；**SIM 降 >0.02 不豁免**——那是真实音色漂移，
  回查 hidden 音色信息与 spk conditioning 打架。

---

## 5. 任务分解与排期（10 个工作日，1–2 人）

| # | 任务 | 工作量 | 依赖 |
|---|---|---|---|
| T1 | hidden 提取 util + 对齐/等价/长度单测 | 1.5d | — |
| T2 | flow 注入 + projector + config 开关 | 0.5d | — |
| T3 | 训练循环集成（冻结 LM、dropout、DDP 细节）| 1d | T1, T2 |
| T4 | 推理链路 + 配对缓存评测模式 | 1d | T1, T2 |
| T5 | 数据准备（冒烟档即刻；Emilia 子集下载+双门控并行）| 1–2d | 并行 |
| T6 | 评测扩展（seed-tts-eval 接入、UTMOS、表现力集、paired 统计）| 1.5d | 并行 |
| T7 | 冒烟双臂 + L0 判读 ← **第一个决策点** | 1d | T3, T5 冒烟档 |
| T8 | 主实验双臂训练（挂机 + 2k step 快评）| 2–3d 墙钟 | T7 通过 |
| T9 | 完整 L1–L3 评测 + 听音 | 2d | T4, T6, T8 |
| T10 | 结论备忘录 | 0.5d | T9 |

**关键路径**：T1→T3→T7→T8→T9。T5/T6 与编码并行。
日历示意：D1–2 T1/T2/T5 起步 → D3 T3/T6 → D4 T4 + 冒烟启动 → **D5 L0 判读** +
主数据就绪 → D6–8 主训练 → D9–10 评测听音 → D11–12 缓冲 + 结论。

---

## 6. 风险清单

| 风险 | 等级 | 对策 |
|---|---|---|
| 对齐 off-by-one（不报错、静默吃掉收益、误判无效）| **高** | §2.2 唯一权威规则 + argmax 命中率单测（T1 内完成）|
| CV3 flow 监督训练路径未充分验证（run.sh 过期提示）| 中 | 冒烟日首件事验证；坏则整套退 CV2 |
| 转写噪声不对称伤实验臂 | 中 | Emilia 双门控；冒烟档全人工转写 |
| Exposure bias（训练真值 hidden vs 推理采样 hidden）| 中 | hidden dropout 10–20%；L3 温度应力探针；恶化再加噪声 |
| 音色纠缠导致 SIM 大降 | 中 | SS 护栏 −0.02 硬线 + SMOS 对照 |
| 16k 带宽（WS4TTS）压暗两臂 | 低 | 主配方不用；仅备选并知情 |
| 小数据漂移（两臂同降）| 低 | S0 锚点常驻评测矩阵 |
| vllm/TRT 生产路径 gap | 低（PoC 外）| 记录：vllm 需 hidden 输出或蒸馏；TRT 不受影响 |

---

## 7. 交付物

1. S1/S2 checkpoint + 训练曲线（含 L0 分离度图）；
2. 评测汇总表：4 系统 × 全指标 + 置信区间（配对统计）；
3. 听音对比页（LISTENING.md 模式）+ CMOS/SMOS 原始打分；
4. **决策备忘录**：Go / 容量受限 / No-go 三选一 + 证据链 + 后续路线
   （Go ⇒ 流式集成与更宽注入通道评估；No-go ⇒ 归档，结论本身有价值）。

---

## 附录 A：关键代码草图

### A.1 提取（训练态，batch 内 on-the-fly）

```python
@torch.no_grad()
def extract_teacher_hidden(lm, batch, device):
    """返回与 speech_token 等长对齐的 hidden: (B, T_speech, 896)。
    对齐规则见 PLAN §2.2：从 task_id 输出位置起切 T_speech 长。"""
    text_emb = lm.llm.model.model.embed_tokens(batch['text_token'].to(device))
    instr_emb = lm.llm.model.model.embed_tokens(batch['instruct_token'].to(device))
    speech_emb = lm.speech_embedding(batch['speech_token'].to(device))
    sos = lm.speech_embedding.weight[lm.sos].reshape(1, 1, -1)      # CosyVoice3LM
    task = lm.speech_embedding.weight[lm.task_id].reshape(1, 1, -1)
    # 逐样本按真实长度拼 [sos, instr, text, task, speech]，pad 后一次前向
    # （拼接/pad 逻辑参照 llm.py pad_unpad_sequence，恒走 unistream）
    lm_output, _ = lm.llm(lm_input, lm_input_len)
    # 切片：P-1 = 1 + T_instr[i] + T_text[i]，长度 T_speech[i]
    return sliced_hidden  # cast 到 flow dtype
```

### A.2 注入（flow forward，训练与推理同一处）

```python
token = self.input_embedding(torch.clamp(token, min=0)) * mask
if self.use_llm_hidden:
    keep = (torch.rand(b, 1, 1, device=device) > self.hidden_dropout).float() \
           if self.training else 1.0
    token = token + self.hidden_proj(llm_hidden) * mask * keep
```

### A.3 推理捕获（llm 解码循环）

```python
y_pred, cache = self.llm.forward_one_step(lm_input, masks=..., cache=cache)
hidden_i = y_pred[:, -1]            # 预测本步 token 的 hidden，随 top_ids 一起返回
# 首步时 y_pred 含全 prefix 输出：prompt 段 hidden = y_pred[:, P-1 : P-1+T_prompt]
```

## 附录 B：本仓库相关锚点

- LM hidden 已是现成输出：`cosyvoice/llm/llm.py:240`（`hidden_states[-1]`）
- 解码循环逐步 hidden：`cosyvoice/llm/llm.py:539-549`
- 注入点：`cosyvoice/flow/flow.py:343`（CV3）/ `flow.py:209,254`（CV2）
- CFG 训练置零：`cosyvoice/flow/flow_matching.py:185-189`；推理 uncond：`flow_matching.py:105`
- flow batch 字段齐备：`cosyvoice/dataset/processor.py:395-414`
- checkpoint 宽松加载：`cosyvoice/bin/train.py:138`（`strict=False`）
- 数据管线模板：`examples/libritts/cosyvoice3/run.sh` stage 0–3；训练 stage 5
- 评测复用：`flow_grpo/evaluate.py`、`flow_grpo/exp/run_eval_8gpu_ckpts.sh`、
  `flow_grpo/exp/LISTENING.md`、`flow_grpo/make_hard_cases.py`
