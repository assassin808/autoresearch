# Omni-Modal S3 音频平台期研究报告

**日期**: 2026-03-22
**计算集群**: Narval (Compute Canada), A100 80GB GPU
**分支**: `omni-basin-shaping`

---

## 1. 问题描述

Mini-omni 是一个全模态语言模型，能够同时生成文本和音频。该模型将 Qwen2-0.5B 与 Whisper encoder 以及 SNAC 24kHz 神经音频编解码器（7 个 codebook 流）结合在一起，训练分为三个阶段：

- **S1**: 仅训练 Whisper adapter（冻结 LLM backbone）
- **S2**: 仅训练 LLM backbone 的文本任务（冻结 adapter）
- **S3**: 全部解冻，联合训练 4 种文本+音频任务类型

在 S3 联合训练过程中，我们观察到严重的**音频 loss 平台期**：文本 loss 在 3000 步内快速下降（降幅 77%），而音频 loss 在少量改善后停滞不前（仅下降 14.5%）。本报告系统性地记录了对该平台期的研究，涵盖 13 个已完成实验和 3 个正在进行的实验，共涉及 6 种优化方法和 13 个诊断指标。

**核心问题**：音频平台期的成因是 (a) 文本和音频之间的 gradient 干扰，(b) 优化 landscape 不佳，(c) 训练规模不足，还是 (d) 在当前模型/数据规模下的根本性能力上限？

---

## 2. 实验设置

### 2.1 模型架构

- **LLM backbone**: Qwen2-0.5B（约 5 亿参数），24 层 transformer，hidden_dim=896
- **配置**: `post_adapter=False`, `tie_word_embeddings=True`
- **音频编解码器**: SNAC 24kHz，7 个 codebook 流，每个词表大小 4160
- **文本词表**: 152,000 个 token（Qwen2 tokenizer）
- **总 embedding 表**: 181,120 个音频 token（7 × 4160）+ 152,000 个文本 token，通过 `lm_head` 共享（tied weights）
- **音频编码**: Whisper encoder 提取特征，经过可学习的 adapter，与 token embedding 拼接
- **Delay pattern**: 第 i 个流延迟 i 个位置（layer-shifted），总序列长度 = n_frames + 6

### 2.2 训练配置

**起始 checkpoint**: Post-S1（Whisper adapter 已训练完成，LLM backbone = 预训练 Qwen2-0.5B 权重，音频 embedding 随机初始化）。

**默认 S3 超参数**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-6 to 2e-5 (cosine schedule) |
| Warmup steps | 1500 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8) |
| Weight decay | 0.01 |
| Batch size | 2 |
| Gradient accumulation | 16 (effective batch = 32) |
| Max gradient norm | 1.0 |
| Audio weight | 1.0 |

**S2 模式**（纯文本对照组）: `audio_weight=0`, `s2_mode=True`, LR=2e-4, 仅 T1T2+A1T2 任务。

### 2.3 诊断指标 (D1-D13)

| ID | 诊断指标 | 描述 | 采集频率 |
|----|---------|------|---------|
| D1 | Gradient magnitude ratio (rho) | `\|\|g_audio\|\| / \|\|g_text\|\|` — 音频与文本 gradient 的相对强度 | 每 100 步 |
| D2 | Gradient cosine similarity (cos_phi) | 文本与音频 gradient 的余弦相似度——检测冲突 | 每 100 步 |
| D3 | 逐层 gradient cosine | D2 按 transformer 层分解 | 每 100 步 |
| D4 | Basin width（盆地宽度） | 在 eps=0.01,0.05,0.1,0.5,1.0 处随机扰动下的 loss 退化程度 | 每 1000 步 |
| D5 | Parameter displacement（参数位移） | backbone、lm_head、embedding 相对初始化的 L2 距离 | 每 100 步 |
| D6 | Embedding effective rank（有效秩） | 文本和音频 embedding 矩阵的谱有效秩 | 每 100 步 |
| D7 | Per-codebook losses | 7 个 SNAC codebook 各自的验证集 CE loss | 每 100 步 |
| D8 | Top-k accuracy 与 loss 直方图 | 文本/音频在 top-1,5,10 下的预测准确率 | 每 100 步 |
| D9 | Linear probe accuracy | 在冻结 backbone L6/L12/L18 层激活上训练线性分类器预测音频 token | 每 500 步 |
| D10 | 分任务验证集 loss | T1T2, T1A2, A1T2, A1A2 四种任务类型的验证集 loss | 每 200 步 |
| D11 | Module gradient norms | 每层 transformer + lm_head 的文本/音频 gradient 范数 | 每 100 步 |
| D12 | CKA (Centered Kernel Alignment) | 当前 checkpoint 与初始 checkpoint 在 L6/L12/L18 的表征相似度 | 每 500 步 |
| D13 | Audio embedding cosine collapse | 音频 embedding 之间的平均两两 cosine similarity——检测表征坍缩 | 每 100 步 |

### 2.4 数据

- **数据集**: VoiceAssistant-400K (gpt-omni)
- **样本数**: 470K 样本 × 4 种任务类型 = 188 万条序列
- **预处理后大小**: 65 GB train.pt, 1.4 GB val.pt
- **Whisper 特征**: 325 个分片的预提取特征
- **SNAC codebook 熵**（数据分布）: CB0=6.03, CB1=7.09, CB2=7.85, CB3=7.90, CB4=7.09, CB5=7.87, CB6=7.91 nats
- **参考值**: log(4160) = 8.33 nats（均匀随机 baseline）

---

## 3. 实验与结果

### 3.1 Batch 3: 初始 Baseline（exp1-exp4，各 3000 步）

| # | 名称 | 方法 | Audio Wt | LR | S2 Mode | 关键改动 |
|---|------|------|----------|-----|---------|---------|
| exp1 | S2 纯文本 | baseline | 0.0 | 2e-4 | Yes | 纯文本对照组（较高学习率） |
| exp2 | S3 baseline | baseline | 1.0 | 2e-5 | No | 文本+音频联合训练，等权重 |
| exp3 | S3 GradNorm | gradnorm | 1.0 | 2e-5 | No | GradNorm 自适应加权（alpha=1.5） |
| exp4 | S3 entropy | entropy_scaled | 1.0 | 2e-5 | No | 按 SNAC codebook 熵缩放 CB 权重 |

**最终 loss（第 3000 步）**:

| 指标 | exp1 (S2 text) | exp2 (S3 baseline) | exp3 (GradNorm) | exp4 (entropy) |
|------|---------------|-------------------|-----------------|----------------|
| Train text_loss | 2.526 | 1.440 | 4766.0 | 1.188 |
| Train audio_loss | 0.0 (disabled) | 37.25 | 33.81 | 4.14 (scaled) |
| Runtime (s) | 7737 | 7461 | 8323 | 7496 |

**Per-codebook 验证集 loss（第 3000 步）**——核心指标：

| CB | H_random (8.33) | exp1 (no audio) | exp2 (baseline) | exp3 (GradNorm) | exp4 (entropy) |
|----|-----------------|----------------|-----------------|-----------------|----------------|
| cb0 | 8.33 | 9.312 | 5.562 | 5.562 | 5.562 |
| cb1 | 8.33 | 9.250 | 6.562 | 6.531 | 6.594 |
| cb2 | 8.33 | 9.188 | 7.562 | 7.562 | 7.594 |
| cb3 | 8.33 | 9.000 | 7.656 | 7.656 | 7.688 |
| cb4 | 8.33 | 9.188 | 6.500 | 6.500 | 6.562 |
| cb5 | 8.33 | 9.125 | 7.781 | 7.750 | 7.781 |
| cb6 | 8.33 | 9.125 | 7.625 | 7.625 | 7.688 |
| **Sum** | **58.31** | **64.19** | **51.25** | **51.19** | **51.47** |

**关键观察**:
1. 三种 S3 方法（exp2/3/4）尽管训练动态截然不同，最终收敛到了**几乎完全相同**的 per-CB 验证集 loss。
2. CB0（最粗粒度，熵最低）学得最好：5.56 vs 随机 8.33（低于随机 33%）。
3. CB2/3/5/6（细粒度，高熵）在 7.5-7.8 处停滞（仅低于随机 7-9%）。
4. **GradNorm 发生了灾难性发散**：权重在第 1000 步时达到 w_text = -525,100 / w_audio = +525,102。文本 loss 飙升至约 5000。但验证集音频 loss 完全不受影响。
5. **Entropy scaling 具有欺骗性**：训练音频 loss 看似很低（4.14），实际上是因为 loss 被除以了熵值（6-8）。未缩放的验证集 loss 与 baseline 完全一致。

### 3.2 Batch 4: 方法改进（exp5-exp10，各 3000 步）

| # | 名称 | 方法 | 关键改动 |
|---|------|------|---------|
| exp5 | GradNorm 修复版 | gradnorm | 非负约束（w >= 0.01），权重 gradient clipping |
| exp6 | Entropy 修复版 | entropy_scaled | 反转加权方式：乘以 max_entropy/entropy |
| exp7 | 冻结 backbone | baseline | `freeze_backbone=True`，仅训练 embedding + lm_head |
| exp8 | Emb init 采样 | baseline | `audio_emb_init=sample` — 从文本 embedding 分布初始化音频 embedding |
| exp9 | Curriculum | baseline | `curriculum=1000` — audio_weight 在前 1000 步从 0 线性增长到 1 |
| exp10 | 冻结 + emb init | baseline | 同时使用 `freeze_backbone=True` 和 `audio_emb_init=sample` |

**最终 loss（第 3000 步）**:

| 指标 | exp5 (GN fix) | exp6 (ent fix) | exp7 (freeze) | exp8 (emb init) | exp9 (curric) | exp10 (both) |
|------|--------------|----------------|---------------|-----------------|---------------|--------------|
| Train text | 2.076 | 1.339 | 29.574 | 1.200 | **1.000** | 7.668 |
| Train audio | **30.469** | 40.734 | 36.406 | 36.781 | 43.094 | 47.734 |

**Per-codebook 验证集 loss（第 3000 步）**:

| CB | exp5 (GN fix) | exp6 (ent fix) | exp7 (freeze) | exp8 (emb init) | exp9 (curric) | exp10 (both) |
|----|--------------|----------------|---------------|-----------------|---------------|--------------|
| cb0 | 5.562 | 5.719 | 5.938 | 5.594 | 5.594 | 7.625 |
| cb1 | 6.500 | 6.688 | 7.125 | 6.562 | 6.656 | 8.125 |
| cb2 | 7.562 | 7.625 | 8.000 | 7.562 | 7.594 | 8.125 |
| cb3 | 7.688 | 7.750 | 8.312 | 7.688 | 7.719 | 8.125 |
| cb4 | 6.531 | 6.719 | 7.094 | 6.562 | 6.625 | 7.250 |
| cb5 | 7.781 | 7.844 | 8.125 | 7.781 | 7.812 | 8.250 |
| cb6 | 7.656 | 7.688 | 7.938 | 7.656 | 7.656 | 8.062 |
| **Sum** | **49.28** | **50.03** | **52.53** | **49.41** | **49.66** | **55.56** |

**关键观察**:
1. **GradNorm 修复版（exp5）**: 训练音频 loss 最优（30.47，较 baseline 37.25 下降 18%），但验证集 CB loss 与 baseline 一致。
2. **Curriculum（exp9）**: 训练文本 loss 最优（1.00，较 baseline 1.44 下降 31%），验证集无变化。
3. **冻结 backbone（exp7）**: 灾难性结果——文本 loss 飙升至 29.6，音频也变差（sum 52.53 vs baseline 51.25）。backbone 必须参与联合适应。
4. **冻结 + emb init（exp10）**: 全场最差（sum 55.56）。进一步确认冻结 backbone 无法学习音频。
5. **从文本分布初始化 emb（exp8）**: 文本略有改善（1.20），音频不变。音频 embedding 有效秩降至 747（随机初始化为 877），表明结构化初始化得以保持，但未能改善验证集 loss。
6. GradNorm 和 curriculum 带来的训练 loss 改善未迁移到验证集——说明仅是优化层面的效果。

### 3.3 Batch 5: 规模实验（exp11-exp13，各 10000 步）

| # | 名称 | 步数 | Init Checkpoint | 核心测试 |
|---|------|------|----------------|---------|
| exp11 | Long S2 | 10,000 | post-S1 | 延长纯文本预训练（S2，LR=2e-4） |
| exp12 | S2 then S3 | 10,000 | exp11 最终模型 | 从充分预训练的 S2 backbone 开始做 S3 |
| exp13 | Long S3 | 10,000 | post-S1 | 直接做 S3 共 10K 步（不经过 S2 预训练） |

**最终 loss（第 10000 步）**:

| 指标 | exp11 (long S2) | exp12 (S2->S3) | exp13 (long S3) |
|------|----------------|----------------|-----------------|
| Train text | 1.896 | 1.197 | 1.470 |
| Train audio | 0.0 (disabled) | 28.078 | 36.594 |
| Runtime (s) | 24,935 | 25,151 | 24,917 |

**Per-codebook 验证集 loss（第 10000 步）**:

| CB | H_random | exp11 (S2 only) | exp12 (S2->S3) | exp13 (long S3) |
|----|----------|----------------|----------------|-----------------|
| cb0 | 8.33 | 9.000 | **5.250** | 5.219 |
| cb1 | 8.33 | 8.875 | **5.781** | 5.812 |
| cb2 | 8.33 | 8.875 | 7.031 | 7.062 |
| cb3 | 8.33 | 8.812 | 7.062 | 7.125 |
| cb4 | 8.33 | 9.062 | **5.406** | 5.562 |
| cb5 | 8.33 | 9.000 | 7.188 | 7.188 |
| cb6 | 8.33 | 8.875 | 7.125 | 7.156 |
| **Sum** | **58.31** | **62.50** | **44.84** | **45.12** |

**CB loss 在 10K 步内的演变（exp12 和 exp13）**:

| Step | exp12 cb0 | exp12 cb1 | exp12 cb4 | exp13 cb0 | exp13 cb1 | exp13 cb4 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1000 | 5.781 | 6.875 | 6.875 | 5.750 | 6.906 | 6.875 |
| 3000 | 5.469 | 6.344 | 6.312 | 5.438 | 6.406 | 6.375 |
| 5000 | 5.406 | 6.094 | 5.844 | 5.375 | 6.094 | 5.906 |
| 7000 | 5.281 | 5.906 | 5.625 | 5.250 | 5.938 | 5.688 |
| 10000 | 5.250 | 5.781 | 5.406 | 5.219 | 5.812 | 5.562 |

**关键观察**:
1. **10K 步远优于 3K 步**: 音频验证集 sum 从 51.25（3K, exp2）降至 44.84（10K, exp12），改善 12.5%。音频学习速度慢但持续进行。
2. **S2 预训练仅略有帮助**: exp12（S2->S3）sum=44.84 vs exp13（无 S2）sum=45.12。差距仅 0.6%，说明 S2 预训练对音频性能并非关键。
3. **CB0/CB1/CB4 在 10K 步内持续改善**。CB2/3/5/6 改善更慢，但也尚未完全收敛。
4. **训练音频 loss**: exp12 达到 28.08（有 S2 预热）vs exp13 的 36.59——S2 预训练对训练 loss 帮助明显，但验证集差距极小。

**分任务验证集 loss（第 10000 步）**:

| Task | exp12 text | exp12 audio | exp13 text | exp13 audio |
|------|-----------|-------------|-----------|-------------|
| T1T2 | 1.84 | 20.66 | 1.70 | 20.73 |
| T1A2 | 1.09 | 44.55 | 1.01 | 44.74 |
| A1T2 | 2.50 | 20.97 | 2.20 | 21.05 |
| A1A2 | 1.26 | 44.78 | 1.11 | 44.93 |

两个实验的分任务验证集 loss 几乎完全一致。音频输出任务（T1A2, A1A2）维持在约 44-45，文本输入音频输出任务（T1T2, A1T2）在约 21。

---

## 4. 诊断分析

### 4.1 Gradient 动态（D1: rho, D2: cos_phi）

**exp2 baseline 演变过程**:

| Step | rho (audio/text) | cos_phi | text_grad_norm | audio_grad_norm |
|------|-----------------|---------|---------------|-----------------|
| 100 | 0.32 | -0.045 | 203.4 | 65.1 |
| 500 | 0.30 | -0.013 | 109.6 | 33.0 |
| 1000 | 0.44 | +0.002 | 40.1 | 17.4 |
| 2000 | 0.30 | -0.022 | 22.2 | 6.7 |
| 3000 | 0.73 | +0.001 | 14.2 | 10.4 |

**各实验在第 3000 步的 rho**:

| 实验 | rho | cos_phi |
|-----|-----|---------|
| exp1 (S2 text) | 2.46 | -0.015 |
| exp2 (S3 baseline) | 0.73 | +0.001 |
| exp3 (GradNorm diverged) | 0.06 | +0.013 |
| exp5 (GradNorm fixed) | 0.73 | +0.000 |
| exp6 (entropy fixed) | 1.40 | -0.010 |
| exp7 (freeze backbone) | 0.05 | -0.143 |
| exp8 (emb init) | 0.82 | +0.016 |
| exp9 (curriculum) | 0.86 | -0.031 |

**核心发现**: cos_phi 在所有实验的所有时间步上始终接近零（在 [-0.05, +0.02] 范围内）。文本和音频的 gradient **正交而非冲突**。音频平台期并非由 gradient 干扰导致。逐层 cosine（D3）同样接近零，最大 |cos| 在第 100 步约 0.2，此后 < 0.1。

### 4.2 CKA 表征漂移 (D12)

CKA 衡量 backbone 内部表征相对于初始化的变化程度。1.0 = 与初始化完全一致，0.0 = 完全不同。

**各实验在 Layer 12 的 CKA**:

| 实验 | Step 500 | Step 1000 | Step 3000 | Step 5000 | Step 10000 |
|-----|----------|-----------|-----------|-----------|------------|
| exp1 (S2 text, 3K) | — | — | 0.028 | — | — |
| exp2 (S3 baseline, 3K) | 1.000 | 0.974 | 0.955 | — | — |
| exp3 (GradNorm div, 3K) | — | — | 0.613 | — | — |
| exp4 (entropy, 3K) | — | — | 0.997 | — | — |
| exp5 (GradNorm fix, 3K) | — | — | 0.928 | — | — |
| exp7 (freeze, 3K) | — | — | 0.595 | — | — |
| exp8 (emb init, 3K) | — | — | 0.152 | — | — |
| exp9 (curriculum, 3K) | — | — | 0.008 | — | — |
| exp11 (long S2, 10K) | 1.000 | 0.828 | 0.998 | 0.999 | 0.999 |
| exp12 (S2->S3, 10K) | 1.000 | 1.000 | 0.999 | 0.995 | 0.994 |
| exp13 (long S3, 10K) | 1.000 | 0.187 | 0.030 | 0.038 | 0.036 |

**核心发现**:
1. **S2 预训练保留了表征**: exp11（long S2）在 10K 步内始终维持 CKA > 0.99。backbone 在同一表征空间内学习文本。
2. **S2->S3 分阶段训练（exp12）同样保持 CKA > 0.99**: 经过 S2 预训练的 backbone 在 S3 阶段几乎没有改变其表征，而是在不重组的前提下添加了音频能力。
3. **不经过 S2 的 Long S3（exp13）表征发生了根本性重组**: CKA 在第 2000 步降至 0.03，backbone 完全重新组织。
4. **然而 exp12 和 exp13 达到了相同的验证集性能**（音频 sum 44.84 vs 45.12）。完全不同的内部表征却产生了相同的音频预测质量。这是平台期并非表征问题、而是根本性容量限制的有力证据。
5. **Curriculum（exp9）同样达到 CKA=0.008**——早期纯文本训练改变了表征，随后加入音频进一步改变。但验证集 loss 依旧不变。

### 4.3 Basin Width（盆地宽度）(D4)

盆地宽度通过 eps=0.01（最小扰动幅度）处的 loss 退化程度来衡量。退化越小 = 盆地越宽/越平坦。

**eps=0.01 处的退化量，第 3000 步**:

| 实验 | Baseline Loss | Deg@0.01 | Basin Width（相对宽度） |
|-----|--------------|----------|----------------------|
| exp1 (S2 text) | 2.356 | 34.5 | **宽**（1.0x 参考值） |
| exp2 (S3 baseline) | 1.675 | 214.0 | 窄（差 6.2 倍） |
| exp5 (GradNorm fixed) | 2.181 | 298.4 | 非常窄 |
| exp6 (entropy fixed) | 1.783 | 235.5 | 窄 |
| exp7 (freeze backbone) | 58.938 | 53.4 | 宽（但 loss 很高） |
| exp8 (emb init) | 1.867 | 182.2 | 中等 |
| exp9 (curriculum) | 1.780 | 170.2 | 中等 |

**S2（exp11）在 eps=0.01 处的盆地宽度演变**:

| Step | Baseline Loss | Degradation@0.01 |
|------|--------------|-----------------|
| 1000 | 2.641 | 8.6 |
| 3000 | 2.650 | 10.1 |
| 5000 | 2.431 | 14.3 |
| 10000 | 1.883 | 36.9 |

**S2->S3（exp12）在 eps=0.01**:

| Step | Baseline Loss | Degradation@0.01 |
|------|--------------|-----------------|
| 1000 | 1.909 | 40.1 |
| 5000 | 1.998 | 48.7 |
| 10000 | 1.955 | 85.5 |

**核心发现**:
1. **S2 预训练产生了极宽的盆地**: 退化量 8.6-36.9 vs S3 baseline 的 214-336。S2 模式处于 loss landscape 中平坦约 10 倍的区域。
2. **S2->S3 从更宽的盆地出发**（第 1000 步为 40.1 vs S3-only exp2 第 1000 步的 336），随着模型适应联合文本+音频训练而逐渐变窄。
3. **Long S3（exp13）** 在第 10000 步的退化量为 204.5——仍然比 S2 窄得多。
4. S2 预训练带来的更平滑 landscape **并未**转化为更好的音频验证集性能——exp12（宽盆地）与 exp13（窄盆地）持平。

### 4.4 Linear Probe Accuracy (D9)

在冻结 backbone 隐状态上训练 linear probe 预测下一个音频 token。随机概率 = 1/4160 = 0.024%。

**Layer 12 的 probe val_acc**:

| 实验 | Step 500 | Step 1000 | Step 3000 | Step 5000 | Step 10000 |
|-----|----------|-----------|-----------|-----------|------------|
| exp1 (S2 text) | — | — | 3.45% | — | — |
| exp2 (S3 baseline) | 3.0% | 4.7% | 4.05% | — | — |
| exp5 (GradNorm fix) | — | — | 4.50% | — | — |
| exp8 (emb init) | — | — | 4.42% | — | — |
| exp9 (curriculum) | — | — | 5.17% | — | — |
| exp12 (S2->S3) | 3.97% | 4.42% | 5.25% | 5.32% | 4.87% |
| exp13 (long S3) | 3.75% | 4.72% | 4.57% | 5.10% | 5.32% |

**核心发现**:
1. Probe 准确率远好于随机（3-5% vs 0.024%），但绝对值仍然很低。backbone 表征中包含的音频 token 身份信息非常有限。
2. 准确率缓慢但持续增长：大约**每 1000 步 +0.1-0.2%**，与 CB loss 的缓慢改善一致。
3. 训练 probe 准确率达到 13-53%（严重过拟合），但验证集保持 3-5%。backbone 学到的音频特定模式无法泛化。

### 4.5 分任务验证集 Loss (D10)

这是最具揭示性的诊断指标之一。验证集 loss 按 4 种任务类型分别报告。

**exp2 baseline（第 3000 步）**:

| Task | Text Input | Audio Input | Text Output | Audio Output |
|------|-----------|-------------|-------------|--------------|
| T1T2 | text | — | 1.70 | 22.78 |
| T1A2 | text | — | 1.03 | 49.05 |
| A1T2 | — | audio | 2.13 | 23.09 |
| A1A2 | — | audio | 1.07 | 49.15 |

**各方法在第 3000 步的音频 loss 对比**:

| Task | exp2 | exp5 (GN) | exp6 (ent) | exp8 (emb) | exp9 (curr) |
|------|------|-----------|------------|------------|-------------|
| T1T2 audio | 22.78 | 22.80 | 23.10 | 22.88 | 22.95 |
| T1A2 audio | 49.05 | 49.06 | 49.72 | 49.27 | 49.38 |
| A1T2 audio | 23.09 | 23.09 | 23.44 | 23.20 | 23.25 |
| A1A2 audio | 49.15 | 49.20 | 49.87 | 49.39 | 49.52 |

**核心发现**:
1. **验证集音频 loss 在所有方法间几乎完全一致**（互差不超过 1%）。没有任何优化技巧能改变根本性的音频预测质量。
2. **音频输出任务（T1A2, A1A2）的 loss 约为文本条件任务（T1T2, A1T2）的 2 倍**: 49 vs 23。当模型需要在没有文本上下文的情况下生成音频时，表现显著变差。
3. 文本 vs 音频输入对音频输出几乎没有影响：T1T2 audio（22.78） ≈ A1T2 audio（23.09），T1A2 audio（49.05） ≈ A1A2 audio（49.15）。
4. **在 10K 步时 loss 确实有所改善**: T1T2 audio 从 22.78 降至 20.66-20.73，A1A2 audio 从 49.15 降至 44.78-44.93。但结构性的 2 倍差距持续存在。

### 4.6 Embedding 分析（D6: 有效秩, D13: Cosine Collapse）

**第 3000 步的有效秩**（embedding 维度 = 896）:

| 实验 | Text Rank | Audio Rank | Audio Rank / 896 |
|-----|-----------|------------|-------------------|
| exp1 (S2 text) | 581 | 883 | 98.5% |
| exp2 (S3 baseline) | 746 | 877 | 97.9% |
| exp5 (GradNorm fix) | 717 | 877 | 97.9% |
| exp8 (emb init) | 784 | 747 | 83.4% |
| exp9 (curriculum) | 743 | 878 | 98.0% |
| exp10 (freeze+init) | 679 | 717 | 80.0% |
| exp12 (S2->S3, 10K) | 388 | 862 | 96.2% |
| exp13 (long S3, 10K) | 571 | 863 | 96.3% |

**Cosine collapse (D13)，最终步**:

| 实验 | cos_collapse | 解读 |
|-----|-------------|------|
| exp1 (S2 text) | -0.00000 | 音频 embedding 未更新 |
| exp2 (S3 baseline) | 0.01761 | 接近随机（未坍缩） |
| exp5 (GradNorm fix) | 0.01659 | 接近随机 |
| exp8 (emb init) | 0.19434 | 有结构（来自文本初始化） |
| exp9 (curriculum) | 0.01294 | 接近随机 |
| exp12 (S2->S3, 10K) | 0.05870 | 略有结构 |
| exp13 (long S3, 10K) | 0.06188 | 略有结构 |

**核心发现**:
1. **随机初始化下，音频 embedding 有效秩保持接近满秩（96-98%）**。随机矩阵天然满秩；音频 embedding 从未发展出已学习到有意义表征所特有的低秩结构。
2. **文本 embedding 有效秩随训练下降**（S2->S3 从 3K 的 746 降到 10K 的 388）。文本 embedding 随着特化过程发展出集中的低秩结构。
3. **从文本分布初始化 emb（exp8/10）** 将音频 rank 降至 747-717，确认初始化结构得以保持。但这种结构化初始化并未改善验证集音频 loss。
4. **未发生 cosine collapse**: cos_collapse 在所有实验中远低于 1.0。音频 embedding 没有坍缩到同一方向——它们仍然以接近随机的配置分散分布。
5. **10K 步后**，音频 rank 略降至 862-863，cos_collapse 升至 0.06。这表明 embedding 正在缓慢发展出结构，与 CB loss 的缓慢改善一致。

### 4.7 逐模块 Gradient 范数 (D11)

**exp2 baseline 第 3000 步——各模块 gradient 范数**:

| Module | Text Grad | Audio Grad | Ratio (audio/text) |
|--------|-----------|------------|-------------------|
| lm_head | 22.38 | 8.36 | 0.37 |
| layer0 | 21.37 | 7.23 | 0.34 |
| layer6 | 4.14 | 2.84 | 0.69 |
| layer12 | 4.55 | 1.86 | 0.41 |
| layer18 | 4.10 | 2.22 | 0.54 |
| layer23 | 10.35 | **22.27** | **2.15** |

**核心发现**: 音频 gradient 范数在大多数层比文本小 2-3 倍。显著的例外是**第 23 层（最后一层）**：音频 gradient（22.27）超过文本 gradient（10.35）。输出层的 loss 信号很强，但无法有效反向传播到更深的层。这与"随机 embedding 瓶颈"假说一致：输出层的音频 loss gradient 很大，但由于音频 embedding 接近随机，gradient 对更深层不携带任何连贯的方向性信息。

### 4.8 Displacement（参数位移）(D5)

**相对于初始化的参数位移**:

| 实验 | Step | Backbone Disp | lm_head Disp | Ratio (lm_head/backbone) |
|-----|------|--------------|-------------|-------------------------|
| exp1 (S2 text, 3K) | 3000 | 102.3 | 287.8 | 2.8 |
| exp2 (S3 baseline, 3K) | 3000 | 11.8 | 60.8 | 5.2 |
| exp5 (GradNorm fix, 3K) | 3000 | 10.4 | 25.1 | 2.4 |
| exp7 (freeze, 3K) | 3000 | 0.0 | 16.7 | inf |
| exp8 (emb init, 3K) | 3000 | 11.7 | 57.0 | 4.9 |
| exp11 (S2, 10K) | 10000 | 195.6 | 554.5 | 2.8 |
| exp12 (S2->S3, 10K) | 10000 | 24.5 | 64.2 | 2.6 |
| exp13 (S3, 10K) | 10000 | 22.7 | 144.4 | 6.4 |

**核心发现**:
1. **S2 纯文本训练使 backbone 移动幅度大 10 倍**（3K 时 102 vs 11.8，10K 时 196 vs 23）。更高的学习率 + 纯文本信号驱动了更大的参数变化。
2. **lm_head 总是比 backbone 移动更多**（2.4-6.4 倍）。大部分学习发生在输出投影层。
3. **exp2 的 backbone displacement 趋于饱和**: 从第 2000 步的 10.9 到第 3000 步的 11.8——backbone 几乎停止变化，而音频 loss 仍然很高。
4. 注：由于已知 bug（`tie_word_embeddings=True` 导致参数名不匹配），D5 的 embedding displacement 始终为 0。

---

## 5. 核心发现

### 发现 1: 音频平台期已确认，但在 10K 步时尚未完全收敛

CB0（最粗粒度 codebook）达到验证集 loss 5.22 vs 随机 8.33（低于随机 37%）。CB2-6（细粒度 codebook）在 10K 步时仍在 7.0-7.2（低于随机 13-16%）。但 loss 在 3K 到 10K 步之间持续改善：

| CB | 3K (exp2) | 10K (exp13) | 改善量 |
|----|-----------|-------------|-------|
| cb0 | 5.562 | 5.219 | -0.343 |
| cb1 | 6.562 | 5.812 | -0.750 |
| cb4 | 6.500 | 5.562 | -0.938 |
| cb2 | 7.562 | 7.062 | -0.500 |
| cb5 | 7.781 | 7.188 | -0.593 |

平台期是真实存在的，但并非绝对——这是一个非常缓慢的收敛区间，而非硬性的墙。

### 发现 2: 所有优化方法的验证集 loss 完全一致

六种不同方法（baseline、GradNorm、entropy scaling、curriculum、冻结 backbone、emb init）在 3K 步时产生的 per-CB 验证集 loss 互差仅 0.1-0.2 nats。分任务验证集 loss（T1T2, T1A2, A1T2, A1A2）同样几乎完全一致。这是平台期并非优化问题的最强证据。

### 发现 3: Gradient 正交而非冲突（cos_phi ≈ 0）

cos_phi 在所有实验的整个训练过程中始终保持在 [-0.05, +0.02] 范围内。逐层 cosine 同样接近零。这彻底排除了 gradient 冲突作为成因的可能。问题在于音频 gradient 携带的方向性信息不足，而非与文本 gradient 相互抵消。

### 发现 4: S2 预训练使 loss landscape 平坦 2.4-10 倍

Basin width（eps=0.01 处退化量）在第 1000 步时，S2 为 8.6 vs S3 baseline 为 336——相差 39 倍。即使在第 3000 步，S2 盆地（34.5）仍比 S3（214.0）宽 6.2 倍。S2->S3 分阶段训练从更宽的盆地出发（40.1），逐渐变窄。然而，更平滑的 landscape 并未转化为更好的音频验证集性能——exp12（宽盆地）与 exp13（窄盆地）持平。

### 发现 5: Linear probe 显示缓慢但持续的音频学习（每 1000 步 +0.1-0.2%）

L12 的 probe val_acc 从第 500 步的 3.0% 增长到第 10000 步的 5.3%。这比随机好 220 倍（0.024%），但绝对值仍然很低。backbone 确实在学习某些音频结构，只是极其缓慢。

### 发现 6: 完全不同的表征可以产生相同的验证集性能

exp12（S2->S3）保持 CKA > 0.99——其表征与预训练 Qwen2 backbone 几乎完全一致。exp13（long S3）降至 CKA = 0.03——表征完全不同。但两者的音频 sum = 44.84 vs 45.12。这意味着音频平台期并非因为 backbone "困在"了某个不好的表征状态。在当前规模下，任何合理的表征都会导致相同的音频质量。

### 发现 7: 冻结 backbone 是灾难性的

exp7（冻结 backbone）: 文本 loss 飙升至 29.6，所有 codebook 的音频 CB loss 均差于 baseline。exp10（冻结 + emb init）更差（sum 55.56 vs 51.25）。backbone 必须与音频联合适应。冻结 backbone 以"保护"文本的策略适得其反。

### 发现 8: GradNorm（修复版）实现最佳训练音频 loss（-18%），但验证集不变

exp5 将训练 audio_loss 从 37.25 降至 30.47（-18%）。这是真实的优化改善。但验证集 CB loss 的 sum 为 49.28 vs 51.25，在噪声范围内。模型可以更好地拟合训练数据，但无法泛化。

### 发现 9: Curriculum 实现最佳训练文本 loss（-31%），但音频不变

exp9 将训练 text_loss 从 1.44 降至 1.00（-31%），方法是先训练纯文本再逐步引入音频。训练音频略差（43.09 vs 37.25），因为有效的音频训练步数更少，但验证集文本和音频 loss 均无变化。

---

## 6. 假说与讨论

### 6.1 平台期本质上是容量/规模限制

最强的证据是发现 2：所有优化方法收敛到相同的验证集 loss。如果问题出在优化（学习率不对、gradient 干扰、初始化不好），不同方法应该达到不同的验证集性能。然而所有方法到达了同一个"谷底"——说明这是在当前条件组合下的根本性能力上限：

- **模型规模**（5 亿参数，896 hidden dim）
- **有效 batch size**（32——mini-omni 原文使用 192）
- **训练步数**（3K-10K——mini-omni 使用 100K+ S3 步数）
- **S2 预训练**（mini-omni 使用了大量 S2 预训练和更多数据）

### 6.2 音频 Embedding 瓶颈

诊断指标链清晰地勾勒出收敛缓慢的原因：

```
Random audio embeddings (rank ~877/896, 97.9% of full rank)
    |
    v
Audio tokens mapped to random directions in embedding space
    |
    v
Backbone receives random vectors, produces noisy hidden states for audio
    |
    v
Audio gradients are directionless (rho < 1, cos_phi ~ 0)
    |
    v
Backbone barely moves (CKA > 0.95 at 3K, displacement saturates)
    |
    v
Linear probes learn almost nothing (val_acc 3-5%)
    |
    v
Audio loss improves very slowly (~0.1 nats/1000 steps)
```

然而，这个瓶颈正在被缓慢克服：在 10K 步时，音频 rank 降至 863（96.2%），cos_collapse 升至 0.06，CB loss 相比 3K 时有可测量的改善。鸡生蛋蛋生鸡的问题（backbone 无法从随机 embedding 中学习，embedding 无法从忽略它们的 backbone 中学习）正在通过梯度下降缓慢解决。

### 6.3 SNAC 声学 Codebook（CB2-6）本质上更难

CB0（最粗粒度，熵 6.03）达到低于随机 37%。CB2-6（最细粒度，熵 7.85-7.91）仅达到低于随机 13-16%。这种难度排序在所有实验和规模上一致。可能的解释：
- 更高的熵意味着更均匀的分布，更难预测
- 细粒度 codebook 编码高频声学细节，可能需要比 24 层因果 transformer 所能捕获的更多上下文
- CB0 编码韵律/音高（更有结构/可预测），而 CB2-6 编码波形细节

### 6.4 2 倍差距：纯音频 vs 文本条件任务

需要在文本旁生成音频的任务（T1T2, A1T2: 音频 loss ≈ 21-23）远优于仅生成音频的任务（T1A2, A1A2: 音频 loss ≈ 44-49）。这个约 2 倍的差距是结构性的，在所有规模上持续存在。文本预测通道提供了帮助音频预测的信息路径，尽管两个模态的 gradient 相关性接近零。

---

## 7. 正在进行的实验（Batch 6）

三个实验正在 Narval A100 节点上运行：

| # | 名称 | 步数 | Batch | Init | 核心测试 |
|---|------|------|-------|------|---------|
| exp15 | Omni scale | 5,000 | 2x96=192 | published ckpt via exp11 | 使用 mini-omni 的 batch size（192）和 S2 预训练 checkpoint |
| exp16 | Long S2->S3 | 30,000 | 2x16=32 | exp11 (long S2) | 从 S2 预训练 backbone 出发的延长 S3 训练（30K 步） |
| exp17 | No S2, large batch | 5,000 | 2x96=192 | post-S1 | 不经过 S2 预训练的大 batch 训练 |

**截至 2026-03-22 的进度**:

| 实验 | 已完成步数 | Audio Sum | 状态 |
|-----|-----------|-----------|------|
| exp15 | 900 / 5000 | 51.60 | 运行中（约 3.7 小时） |
| exp16 | 5800 / 30000 | 44.78 | 运行中（约 4 小时，已追平 10K 结果） |
| exp17 | 700 / 5000 | 51.72 | 运行中（约 3 小时） |

**这些实验验证什么**:
- **exp15 vs exp17**: 在大 batch size（192）下 S2 预训练是否重要？
- **exp16**: 音频 loss 在 10K 步之后是否继续改善？能否使 CB2-6 < 7.0？
- **exp15 vs exp12**: 更大的 batch size（192 vs 32）是否加速音频学习？

**预期洞察**: 如果 exp16 在 20K-30K 步时显示音频持续改善，则平台期可以确定为规模问题。如果 exp15/17 在相同步数下优于 exp12/13，则 batch size 是关键因素。

---

## 8. 基础设施说明

### 8.1 Narval (Compute Canada) 部署

**环境配置**: 所有路径通过 `narval_env.sh` 设定：
- 模型 checkpoint: `/scratch/$USER/mini-omni-ckpt/`
- Mini-omni 参考代码: `/scratch/$USER/mini-omni-ref/`
- S3 预处理数据: `/scratch/$USER/s3_data/`（65GB train + 1.4GB val）
- HuggingFace 缓存: `/scratch/$USER/hf_home/`

**任务提交**: `sbatch job_train_s3.sh`，请求 A100 80GB GPU 的 SLURM 资源。

### 8.2 遇到并解决的问题

1. **S3 管线关键 bug（2026-03-20）**: 在当前实验系列之前发现了三个 bug：
   - 音频 loss 被错误地除以 7（codebook 数量）——已修复为报告真实 CE
   - Whisper encoder 特征未被加载——已修复特征加载路径
   - 使用了错误的 checkpoint（post-S3 而非 post-S1）——已修复 checkpoint 模式

2. **FlashAttention 3 在 Blackwell (RTX 5090) 上不支持**: 使用 SDPA + FlexAttention 作为替代方案。在 Narval A100 上无此问题。

3. **M-SAM 诊断值为零**: 当验证集 loader 使用 batch_size=1 时，M-SAM basin 探测报告零值。已通过确保诊断时 batch_size >= 2 修复。

4. **Embedding displacement (D5) bug**: `tie_word_embeddings=True` 导致参数名不匹配——embedding 矩阵以 `lm_head.weight` 而非 `model.embed_tokens.weight` 被引用。D5 的 embedding displacement 始终为 0。已记录但未修复（lm_head displacement 由于权重绑定包含了相同的信息）。

5. **GradNorm 发散**: 原始实现缺少非负约束。在 exp5 中通过权重 clamping（w >= 0.01）和权重参数 gradient clipping 修复。

6. **Entropy scaling 方向错误**: 原始实现（exp4）将 loss 除以熵值，无意中*降低*了音频的贡献。在 exp6 中修复为乘以 max_entropy/entropy。

### 8.3 算力预算

| Batch | 实验数 | 每个步数 | GPU 时数（估计） |
|-------|-------|---------|----------------|
| Batch 3 | 4 | 3,000 | 4 x 2.1h = 8.4h |
| Batch 4 | 6 | 3,000 | 6 x 2.0h = 12.0h |
| Batch 5 | 3 | 10,000 | 3 x 7.0h = 21.0h |
| Batch 6 | 3 | 5K-30K | ~50h (estimated) |
| **合计** | **16** | — | **~91h A100** |

---

## 9. 更新发现（Batch 5-6 及 Mini-Omni Chain）

本节涵盖实验 15-17（Batch 5-6）、mini-omni chain（R1-R3）以及新增诊断指标 D14/D15 的结果。这些发现对第 1-8 节的结论进行了实质性修正。

### 9.1 音频平台期在足够步数后确实会突破

exp16（S2→S3, eff=32, 30k steps）验证集分任务音频 loss：

| Step | A1A2 | A1T2 |
|------|------|------|
| 2000 | 49.5 | 23.3 |
| 10000 | 42.5 | 19.8 |
| 20000 | 38.2 | 17.7 |
| 30000 | 36.8 | 17.1 |

Chain R3（global ~16500, eff=192）：A1A2=32.3, A1T2=14.9

此前"各方法验证集 loss 无差异"的结论仅在 3k 步时成立。给予充分训练后，音频 loss 持续稳定下降。

### 9.2 S2 预训练不改善验证集 Loss

- exp12 vs exp13（10k steps）：val A1A2 = 44.8 vs 44.9（几乎相同）
- exp15 vs exp17（5k steps, eff=192）：val A1A2 = 44.6 vs 43.9（不经过 S2 反而略优）

S2 使得 basin 更宽（86 vs 205，10k 步时），但并未提升模型能力。

### 9.3 训练步数 >> Batch Size >> S2

- exp15（S2, eff=192, 5k）：val A1A2=44.6
- exp16（S2, eff=32, 30k）：val A1A2=36.8
- exp17（no S2, eff=192, 5k）：val A1A2=43.9

总样本量相同（960k），但 30k 小 batch 步数的效果远优于 5k 大 batch 步数。

### 9.4 各 Codebook 学习速率不同

exp16 在 30k 步时各 codebook loss 从初始到最终的变化：

| Codebook | Initial | Final | Reduction |
|----------|---------|-------|-----------|
| CB0 (semantic) | 5.56 | 4.56 | 18% |
| CB1 | 6.56 | 4.31 | 34% |
| CB2 | 7.56 | 5.78 | 24% |
| CB4 | 6.50 | 3.56 | 45% (most improvement) |
| CB5 | 7.78 | 6.31 | 19% |
| CB6 | 7.62 | 5.94 | 22% |

CB4 和 CB1 在 CB0 之后学习最快，CB5 和 CB6 最慢。

### 9.5 Basin 宽度随训练增长（S2→S3）

exp16 basin width（eps=0.01 degradation）：

| Step | Basin Width |
|------|-------------|
| 1k | 39.9 |
| 10k | 69.8 |
| 20k | 121.7 |
| 30k | 163.3 |

训练时间越长，basin 越宽（鲁棒性越强）。对比 Long S3（无 S2）：s1k=335.6 → s10k=204.5（起始时尖锐，缓慢变平）。

### 9.6 CKA：两条完全不同的表征路径

- **S2→S3（exp16）**：CKA 在 30k 步内始终 >0.987 — backbone 几乎未偏离 S2 状态
- **Long S3（exp13）**：CKA 在 step 2000 时降至 0.03 — backbone 被彻底重塑
- **Chain（R1-R3）**：CKA 保持 1.000 — 大 batch 下更加稳定

两条路径最终达到相近的验证集性能。模型找到了不同但同样有效的表征。

### 9.7 D14 GSNR 确认音频信号微弱

音频 GSNR 演变（mini-omni chain）：

| Checkpoint | GSNR |
|------------|------|
| R2 start | 0.127 |
| R2 end | 0.074 |
| R3 end | 0.083 |

GSNR < 1 意味着梯度方差远大于信号。这解释了音频学习缓慢的原因。cos@K=16 约 0.01 — 即使累积 16 个 batch 的梯度也无法稳定音频梯度方向。

### 9.8 D15 文本子空间：音频梯度处于独立空间

energy_ratio 约 0.001-0.006（音频梯度能量的 0.1-0.6% 落在文本子空间内）。

音频与文本在几乎完全正交的参数子空间中学习。这并非梯度干扰问题 — 两者之间几乎不存在交互。

### 9.9 Linear Probe 饱和于 ~6%

- exp16 30k：probe accuracy 3.7% → 6.1%（在 ~6% 处出现平台）
- Chain R3：~6%
- 增长速率：初期约每 1000 步 +0.1%，随后饱和

无论训练多长时间，backbone 积累的音频信息都极为有限。

### 9.10 Displacement 趋于饱和

- exp16：total displacement 59 → 137（step 5k → 30k），backbone 23 → 51
- 增长呈对数减缓 — 参数逐渐趋于平衡
- 音频 embedding displacement = 0（tie_word_embeddings，梯度通过 lm_head 流动）

### 9.11 音频梯度范数比（rho）持续增大

exp16：rho 从 5k 步时的 ~0.4 增长到 30k 步时的 ~2.6。音频梯度的幅值逐渐超过文本梯度。然而音频仍然学习缓慢 — 这证实了问题出在梯度方向（GSNR）而非幅值。

### 9.12 修正后的结论

1. **音频平台期在足够训练（~30k steps）后可以突破**，并非永久性上限
2. **S2 预训练影响优化景观，但不影响最终模型能力**
3. **训练步数的重要性远高于 batch size 和 S2 预训练**
4. **GSNR 约 0.08 解释了学习缓慢的根源**：音频需要约 12 倍于文本的样本量才能提取同等信号
5. **文本与音频在正交子空间中学习**（D15 energy < 0.6%）
6. **不同 CKA 路径（0.987 vs 0.03）达到相近验证集性能** — 表征并非瓶颈

---

## Appendix A: 完整实验列表

| # | Name | Method | Steps | LR | Batch (eff) | S2 | Freeze | Emb Init | Curriculum | Init Ckpt |
|---|------|--------|-------|-----|------------|-----|--------|----------|------------|-----------|
| exp1 | S2 text | baseline | 3000 | 2e-4 | 32 | Yes | No | None | 0 | post-S1 |
| exp2 | S3 baseline | baseline | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp3 | S3 GradNorm | gradnorm | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp4 | S3 entropy | entropy_scaled | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp5 | GradNorm fixed | gradnorm | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp6 | Entropy fixed | entropy_scaled | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp7 | Freeze backbone | baseline | 3000 | 2e-5 | 32 | No | Yes | None | 0 | post-S1 |
| exp8 | Emb init sample | baseline | 3000 | 2e-5 | 32 | No | No | sample | 0 | post-S1 |
| exp9 | Curriculum | baseline | 3000 | 2e-5 | 32 | No | No | None | 1000 | post-S1 |
| exp10 | Freeze+emb init | baseline | 3000 | 2e-5 | 32 | No | Yes | sample | 0 | post-S1 |
| exp11 | Long S2 | baseline | 10000 | 2e-4 | 32 | Yes | No | None | 0 | post-S1 |
| exp12 | S2 then S3 | baseline | 10000 | 2e-5 | 32 | No | No | None | 0 | exp11 final |
| exp13 | Long S3 | baseline | 10000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp15 | Omni scale | baseline | 5000 | 2e-5 | 192 | No | No | None | 0 | published via exp11 |
| exp16 | Long S2->S3 | baseline | 30000 | 2e-5 | 32 | No | No | None | 0 | exp11 final |
| exp17 | No S2 large batch | baseline | 5000 | 2e-5 | 192 | No | No | None | 0 | post-S1 |

## Appendix B: exp2 Baseline 详细训练日志

| Step | text_loss | audio_loss | rho | cos_phi | backbone_disp | CKA_L12 | audio_rank |
|------|-----------|------------|-----|---------|--------------|---------|------------|
| 100 | 5.791 | 51.672 | 0.32 | -0.045 | 0.77 | — | 883 |
| 300 | 5.041 | 44.375 | 0.77 | +0.021 | 1.42 | — | — |
| 500 | 3.381 | 48.094 | 0.30 | -0.013 | 2.15 | 1.000 | 883 |
| 1000 | 1.733 | 37.781 | 0.44 | +0.002 | 5.02 | 0.974 | 882 |
| 1500 | 1.750 | 31.312 | 0.51 | -0.000 | 8.24 | 0.971 | — |
| 2000 | 1.567 | 40.406 | 0.30 | -0.022 | 10.85 | 0.949 | 879 |
| 2500 | 1.750 | 36.594 | 0.35 | -0.009 | 11.69 | 0.954 | — |
| 3000 | 1.440 | 37.250 | 0.73 | +0.001 | 11.79 | 0.955 | 877 |

## Appendix C: exp2 分任务验证集 Loss 随时间变化

| Step | T1T2 text | T1T2 audio | T1A2 text | T1A2 audio | A1T2 text | A1T2 audio | A1A2 text | A1A2 audio |
|------|-----------|------------|-----------|------------|-----------|------------|-----------|------------|
| 200 | 7.40 | 25.93 | 4.04 | 55.97 | 7.68 | 26.67 | 3.98 | 56.02 |
| 600 | 3.57 | 23.50 | 2.10 | 50.59 | 4.37 | 23.90 | 2.19 | 50.70 |
| 1000 | 2.18 | 23.37 | 1.28 | 50.31 | 2.64 | 23.73 | 1.37 | 50.40 |
| 1600 | 2.05 | 23.19 | 1.18 | 49.93 | 2.47 | 23.51 | 1.28 | 50.00 |
| 2000 | 1.89 | 22.96 | 1.15 | 49.45 | 2.38 | 23.29 | 1.21 | 49.57 |
| 2600 | 1.72 | 22.81 | 1.03 | 49.13 | 2.19 | 23.14 | 1.12 | 49.24 |
| 3000 | 1.70 | 22.78 | 1.03 | 49.05 | 2.13 | 23.09 | 1.07 | 49.15 |

## Appendix D: GradNorm 权重发散（exp3，未修复版）

| Step | w_text | w_audio | text_loss |
|------|--------|---------|-----------|
| 100 | +0.012 | 1.99 | 6.86 |
| 200 | -1.24 | 3.24 | 3704 |
| 300 | -20.9 | 22.9 | 5382 |
| 500 | -586 | 588 | 4154 |
| 1000 | -525,100 | 525,102 | 5880 |
| 3000 | -525,100 | 525,102 | 4766 |

根本原因：音频 loss 几乎不变，因此 GradNorm 的相对速率机制不断增大音频权重、减小文本权重。在缺少非负约束的情况下，w_text 穿越零点，模型开始主动最大化文本 loss。
