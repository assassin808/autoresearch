# Audio Plateau 深度分析 / Deep Analysis of the Audio Plateau

**日期**: 2026-03-19
**分支**: `omni-basin-shaping`
**基于**: 7 个训练实验 + 2 个观察实验 + 8 轮文献综述

---

## 一、实验矩阵

| 实验 | 起点 | 训 audio? | 训 text? | LR | 目的 |
|------|------|----------|---------|-----|------|
| **s3_adam** | post-S3 ckpt | Yes | Yes | 2e-5 | S3 baseline |
| **s3_lambda3** | post-S3 ckpt | Yes (3×) | Yes | 2e-5 | 放大 audio 梯度 |
| **s3_gradproj** | post-S3 ckpt | Yes (投影) | Yes | 2e-5 | 去除对抗分量 |
| **s3_cbweight** | post-S3 ckpt | Yes (100:10:1) | Yes | 2e-5 | 语义 CB 加权 |
| **s3_msam** | post-S3 ckpt | Yes (M-SAM) | Yes | 2e-5 | 寻找平坦区域 |
| **s3_skip_s2** | post-S1 (随机 audio emb) | Yes | Yes | 2e-4 | 跳过 S2 |
| **s2_textonly** | post-S3 ckpt | No | Yes | 2e-4 | 模拟 authentic S2 |
| obs_1 | 随机 audio emb | Yes | Yes | 2e-4 | S2 观察 (omni) |
| obs_2 | 随机 audio emb | No | Yes | 2e-4 | S2 观察 (text-only) |

---

## 二、核心发现

---

### 发现 1：Audio 有自己的收敛时钟，不受 text 梯度大小影响

skip_s2 的 txt_gn 是 s3_adam 的 5-30 倍（50-344 vs 11-15），但 step 500 之后两者的 audio CB loss 下降速度几乎一样：

| 区间 | s3_adam cb0 变化 | skip_s2 cb0 变化 | s3_adam aud_gn | skip_s2 aud_gn |
|------|----------------|-----------------|---------------|----------------|
| 100→500 | -0.375 | **-1.969** | 3.0 | 8.5 |
| 500→1000 | -0.156 | -0.156 | 2.7 | 3.4 |
| 1000→1500 | -0.156 | -0.125 | 2.9 | 3.2 |
| 1500→2000 | -0.125 | -0.188 | 3.0 | 2.9 |
| 2000→2500 | -0.062 | +0.000 | 2.9 | 2.7 |
| 2500→3000 | +0.000 | -0.031 | 2.8 | 2.9 |

Step 500 之后，无论 text 梯度大 10 倍还是 1 倍，audio 的 CB loss 下降速度都是每 500 步 ~0.15，到 step 2000+ 都 stall。Audio 收敛有自己的节奏。

---

### 发现 2：Audio plateau 时梯度没有变小——是梯度效率在指数衰减

s3_adam 全程数据：

| 区间 | aud_gn (平均) | cb0 每步下降 | 归一化效率 (×1e4) |
|------|-------------|------------|-----------------|
| 100→500 | 3.0 | -0.00094 | **3.13** |
| 500→1000 | 2.7 | -0.00031 | 1.15 |
| 1000→1500 | 2.9 | -0.00031 | 1.08 |
| 1500→2000 | 3.0 | -0.00025 | 0.83 |
| 2000→2500 | 2.9 | -0.00013 | 0.43 |
| 2500→3000 | 2.8 | 0 | **0.00** |

- grad_norm 只变了 7%（3.0→2.8），效率从 3.13 降到 0。
- 意味着：**不是梯度不够大，是不同 sample 的梯度方向在互相抵消。**
- Displacement 一直在涨（11.4→38.2），参数确实在动——但在做无用功。

skip_s2 呈现完全相同的模式——前 200 步大幅下降（aud_gn=13 时 cb0 降 1.78），之后 aud_gn 降到 ~3 就跟 s3_adam 一样慢一样停。

---

### 发现 3：Text 梯度正常响应 LR，audio 梯度完全不响应

s3_adam 中 LR 在 step 1500 达到峰值 2e-5：

| 指标 | step 100 | step 1500 (peak LR) | step 3000 |
|------|----------|---------------------|-----------|
| txt_gn | 11.4 | **15.6** (+37%) | 12.4 |
| aud_gn | 3.3 | 3.1 (-6%) | 2.8 |

Text 梯度随 LR 上升明显增大（正常的 loss landscape 行为），audio 梯度对 LR 变化无感。这说明 **audio loss landscape 在 plateau 附近是平坦的**——不管 LR 怎么调，梯度信号都差不多。

---

### 发现 4：S2 是必要的——稳定 backbone，给 audio 腾空间

| 证据 | s3_adam (有 S2) | skip_s2 (无 S2) |
|------|----------------|-----------------|
| txt_gn | 11-15 (稳定) | **50-344** (爆炸) |
| ρ (audio/text) | 0.18-0.29 | **0.01-0.07** (audio 被淹没) |
| cb0 起点 | 5.28 | 8.06 (接近随机 8.3) |
| cb0 终点 | **4.41** | 5.59 |

S2 的作用是让 LLM 先适应 omni 数据格式，降低 text 梯度范数（344→11），这样 audio 梯度才有足够的相对份额参与优化。没有 S2，audio 的 ρ 崩溃到 0.01，被 text 完全淹没。

---

### 发现 5：Text-only 训练主动破坏 audio 表征

s2_only 是唯一一个 **audio grad_norm 持续上升** 同时 **CB val loss 持续恶化** 的实验：

| 区间 | aud_gn | cb0 变化 | cb4 变化 | cb6 变化 |
|------|--------|---------|---------|---------|
| 100→500 | 6.2 | -0.031 | +0.000 | +0.016 |
| 500→1000 | 6.4 | -0.094 | +0.078 | +0.047 |
| 1000→1500 | 7.1 | **+0.031** | **+0.062** | **+0.094** |
| 1500→2000 | 7.9 | **+0.031** | **+0.062** | +0.016 |
| 2000→2500 | 8.4 | **+0.062** | **+0.078** | +0.031 |
| 2500→3000 | 8.6 | **+0.031** | +0.000 | +0.000 |

audio grad_norm 从 6.2 涨到 8.6（+39%）。为什么没训 audio 但 audio 梯度在变大？因为 text-only 训练通过共享参数（wte + backbone）把模型推离 audio 的好区域，audio loss surface 变陡——模型离 audio 最优越来越远，梯度自然越来越大。但这些梯度方向跟 text 优化方向冲突（cos φ < 0），即使之后切到 S3 也难以利用。

**这是唯一一个 grad_norm 涨的实验。** 其他所有实验 aud_gn 要么恒定（s3_adam: ~2.8）要么在衰减（skip_s2: 13→3）。

---

### 发现 6：S2 的 tradeoff——帮助 backbone 但损害 audio embedding

S2 同时做了两件事：

| 效果 | 具体 |
|------|------|
| **帮助** | 稳定 backbone（txt_gn 344→11），提升 text embedding rank（598→765），给 audio 腾出梯度空间 |
| **损害** | 通过共享 embedding 把 audio 表征推离最优（s2_only 中所有 CB loss 上升） |

净效果是帮助：s3_adam 终点 cb0=4.41 远好于 skip_s2 的 5.59。S2 对 backbone 的稳定化收益 > 对 audio embedding 的损害。

---

### 发现 7：梯度干扰集中在 Layer 0（共享 embedding），其他层基本正交

所有实验的 per-layer cos φ 一致显示：

| 层 | cos φ 范围 | 占总干扰比重 |
|----|-----------|------------|
| **Layer 0** (embedding) | -0.13 ~ -0.29 | **70%+** |
| Layer 1-22 (transformer) | ±0.03 | ~0 |
| Layer 23 (output) | 轻微负值 | ~5% |

干扰的物理来源是 **wte 共享 embedding table**——152K text token 和 29K audio token 在同一个 (181120, 896) 矩阵中。Transformer backbone 层的 text 和 audio 梯度几乎正交。

---

### 发现 8：Audio embedding rank 的完整生命周期

逐 100 步的完整轨迹：

**skip_s2（从随机开始）**：
```
step  100: 883.3
step  500: 881.5    (缓慢下降)
step 1000: 873.2    (线性下降 ~-1.0/100步)
step 1500: 857.6    (加速 ~-3.1/100步)
step 2000: 841.7    (继续 ~-3.2/100步)
step 2500: 834.7    (开始减速 ~-1.4/100步)
step 3000: 833.4    (趋稳? ~-0.3/100步)
```

**s3_adam（从已收敛 checkpoint 开始）**：
```
step  100: 635.9
step 1000: 636.5    (纹丝不动)
step 2000: 637.4    (纹丝不动)
step 3000: 638.3    (纹丝不动)
```

**s2_only（text-only 训练）**：
```
audio rank: 635.7 → 635.9  (3000 步完全不变)
text rank:  765.2 → 655.3  (持续线性下降，-3.7/100步)
```

关键观察：

1. **skip_s2 的 audio rank 在 3000 步内持续线性下降（883→833），没有稳定**。最后 500 步有减速趋势但还在降。如果外推，可能还需几千步才能到达 s3_adam 的 636。
2. **s3_adam 的 audio rank 636 完全锁死**——说明在 published checkpoint 训练过程中 rank 已经收敛到这个值。
3. **s2_only 的 text rank 在持续下降（765→655）但 audio rank 完全不动**——text-only 训练只重塑 text embedding，不影响 audio embedding 的结构（但通过 backbone 影响 audio 预测质量）。
4. **skip_s2 的 text rank 剧烈震荡（531-826）**——未适应的 backbone 在 omni 数据上极不稳定，text embedding 也被反复拉扯。

---

### 发现 9：Rank 压缩和 loss 下降是脱耦的

skip_s2 中，audio rank 在持续下降（883→833，embedding 在重组），但 CB loss 在 step 500 后下降极慢：

| 区间 | audio rank 变化 | cb0 变化 |
|------|----------------|---------|
| 100→500 | 883→881 (-2) | **-1.969** (大幅下降) |
| 500→1000 | 881→873 (-8) | -0.156 (慢) |
| 1000→2000 | 873→842 (-31) | -0.313 (慢) |
| 2000→3000 | 842→833 (-9) | -0.031 (几乎停) |

Step 1000→2000 期间 rank 下降了 31 个有效维度，但 cb0 只降了 0.313。**Embedding 在重组但这些重组没有带来更好的 token 预测。** Rank 变化反映的是 embedding 空间在被压缩/正交化，但这不等于学到了更有用的表征。

---

### 发现 10：五种优化策略全部撞同一面墙

| 方法 | 思路 | audio plateau | 否定了什么 |
|------|------|--------------|-----------|
| **baseline** | 标准 AdamW | cb0: 5.28→4.41, step 2200 停 | — |
| **λ=3** | 放大 audio 梯度 3× | 同速 plateau | "audio 梯度太弱" |
| **GradProj** | 去除 audio 梯度中对抗 text 的分量 | 同 plateau 模式 | "text-audio 冲突导致 plateau" |
| **CB weight** (100:10:1) | 聚焦语义 codebook | cb0 更好但细 CB 更差，总量持平 | "codebook 间权重不对" |
| **M-SAM** | Sharpness-aware, 寻找平坦区域 | 位移少 28%，同 loss | "sharp minima 困住了 audio" |

五种覆盖了梯度大小、梯度方向、codebook 权重分配、loss landscape 几何——全部无效。

---

### 发现 11：Basin width 不变——否定 audio-as-GO 假说

Basin probing（ε=0.1 随机扰动后 text loss 恶化程度）：

| step | s3_adam (omni) | obs_2 (text-only) |
|------|---------------|-------------------|
| 1000 | 90.4 | 97.3 |
| 2000 | 92.3 | 96.7 |

差异 <7%，且 omni 训练的 basin **没有变宽**。否定了原始假说 "audio 梯度像 GO-style noise 一样扩宽 text loss basin"。

---

### 发现 12：cos φ 诊断的局限性

s2_only 的 cos φ 随训练变化：
```
step  100: -0.075 (接近正交)
step 1000: +0.003 (正交)
step 1500: -0.199 (开始冲突)
step 2000: -0.296 (冲突加剧)
step 3000: -0.148 (缓和)
```

**注意**：这是一个反事实测量。它在 val set 上分别 backward text_loss 和 audio_loss 得到两组梯度，然后计算夹角。在 s2_only 中 audio loss 根本不参与训练，所以这测量的是 "如果此刻也优化 audio，方向会不会冲突"，不是训练中实际发生的冲突。

不能直接跟 S3 的 cos φ 比较（模型状态、LR、参数空间位置完全不同）。但它跟 CB val loss 上升的趋势一致——text 优化方向确实在跟 audio 的理想方向越来越冲突。

另外，诊断每次只用 val 中第一个同时包含 text+audio mask 的 batch，不同 step 可能取到不同 batch，数值有噪声。**真正的硬证据是 CB val loss 变化方向，而不是 cos φ 的绝对值。**

---

## 三、总结论

### Audio plateau 的根因

**不是 S2 "让 text 饱和挤压 audio"，不是梯度冲突，不是优化器选择。**

Audio plateau 是 **audio token 预测任务本身** 的问题：

1. **梯度效率指数衰减**：grad_norm 恒定但每单位梯度带来的 loss 下降在 3000 步内从 3.13 降到 0。最可能的解释是 SNAC 的 DRI（Discrete Representation Inconsistency）——同一段 audio 编码成不同 token，不同 sample 的梯度方向互相抵消。
2. **Audio embedding rank 锁死在 636**（从已收敛 checkpoint 开始时）：模型的 audio 表达能力被固定，loss 到达这个水平的上限就停了。
3. **Rank 压缩和 loss 下降脱耦**：skip_s2 中 rank 还在降但 loss 已经不动了——embedding 在被压缩但这不等于学到了更有用的表征。

### Text-audio 关系总结

| 结论 | 关键证据 |
|------|---------|
| **Audio 有自己的收敛时钟** | skip_s2 txt_gn 10x 大，但 audio 收敛速度相同 |
| **S2 净效果是帮助 audio** | 无 S2: cb0 终点 5.59; 有 S2: cb0 终点 4.41 |
| **S2 帮助方式是稳定 backbone** | 无 S2: txt_gn 爆炸到 344, ρ→0.01; 有 S2: txt_gn=12, ρ=0.22 |
| **Text-only 训练破坏 audio 表征** | s2_only: 所有 CB val loss 持续上升，aud_gn 持续增大 |
| **唯一耦合点是共享 embedding** | Layer 0 贡献 70%+ 梯度干扰，其他层正交 |
| **Text 梯度正常，audio 梯度异常** | text 响应 LR schedule（+37%），audio 完全不响应 |

### S2 的 tradeoff

| S2 的效果 | 具体 |
|-----------|------|
| 帮助 | 稳定 backbone（txt_gn 344→11），提升 text rank（598→765），ρ 从 0.01 恢复到 0.22 |
| 损害 | text-only 训练推移共享 embedding 损害 audio（s2_only 中 CB loss 上升），text rank 下降（765→655） |
| 净效果 | **帮助**（cb0: 4.41 vs 5.59），backbone 稳定化收益 > embedding 损害 |

---

## 四、已否定的假说

| 假说 | 否定证据 |
|------|---------|
| "Audio 梯度太弱" | aud_gn 恒定 ~2.8，λ=3（3 倍权重）不改善 plateau |
| "Text-audio 梯度冲突导致 plateau" | GradProj 去除对抗分量，同样 plateau |
| "S2 让 text 饱和挤压 audio" | skip_s2 没有 S2 更差（5.59 vs 4.41），S2 是帮助 |
| "Audio-as-GO-noise 扩宽 text basin" | Basin probing 显示宽度不变 |
| "Sharp minima 困住了 audio" | M-SAM 找到更平区域（位移 -28%），loss 不变 |
| "ρ 绝对值太小是问题" | 可以通过 loss weight 调整，不影响 plateau 本身 |
| "Codebook 权重分配不对" | CB weight 100:10:1 只是重分配，总 audio 改善量不变 |

---

## 五、开放问题与假设

### H1: Audio embedding rank 636 是 plateau 的原因之一？

**假设**：audio embedding 在训练早期被压缩到 636 维后锁死，限制了模型的 audio 表达能力。29120 个 audio token 只用 636 个有效维度可能不够。

**但也可能**：636 是 SNAC token 冗余的自然结果（DRI 导致很多 token 功能等价），rank 再高也没用。

**验证方式**：对 audio embedding 加 spectral regularization（惩罚低 rank），强制保持高维度，看 plateau 是否推迟。

### H2: Plateau 时 backbone 中间层是否还在学习有用的 audio 表征？

**观察**：displacement 一直在涨（11.4→38.2），参数在动，但 CB val loss 和 embedding rank 都不动。

**两种可能**：
- A) Backbone 中间层在学到更好的 audio 理解，但 output head / embedding 限制了输出
- B) 参数在做无用功，backbone 也没在学

**验证方式**：在 backbone 中间层（如 layer 12）加 linear probe 预测 audio token，如果 probe accuracy 在 plateau 期间还在涨 → 支持 A；如果也停了 → 支持 B。

### H3: 分离 audio embedding 能否消除 Layer 0 干扰？

**动机**：Layer 0 贡献 70%+ 的梯度干扰，来源是 text 和 audio token 共享 wte。

**实验**：给 audio token 独立 embedding（不在 wte 中），text 和 audio 的输入表征完全分离。保持 backbone 和 output head 共享。

**预期**：cos φ 在 Layer 0 应该消失。但不确定这是否能改善 plateau（因为 plateau 主因可能不是干扰）。

**成本**：低（改几行代码，增加约 26M 参数）。

### H4: 换 codec 是否能打破 plateau？

**文献证据**：Moshi 用 Mimi（有语义层）+ 100:1 权重效果远好于 SNAC。X-Codec 注入语义后 DRI 显著降低。

**我们没有实验数据**。但梯度效率衰减（DRI 导致梯度互相抵消）的解释 strongly implies 更一致的 codec 会推迟或消除 plateau。

### H5: 更长训练能否打破 plateau？

**当前**：3000 步 / ~2.5h。ρ 始终在 Phase 1（0.18-0.29），从未进入 Phase 2（ρ≈1）。

**可能性**：10K-40K 步训练可能让 audio 进入 Phase 2。但考虑到梯度效率已经衰减到 0，更可能的结果是更长训练也不帮忙。

---

## 六、方法论说明

### 诊断是怎么算的

- **ρ(t) = ‖∇L_audio‖ / ‖∇L_text‖**：在 val batch 上分别 backward text_loss 和 audio_loss，计算梯度范数之比
- **cos φ(t)**：同上两组梯度的余弦相似度。+1=完全合作，-1=完全冲突，0=正交
- **CB val losses**：在 val set 上每个 codebook 的独立 cross-entropy loss
- **Embedding rank**：wte 矩阵的 SVD，用归一化奇异值的 entropy 计算有效秩
- **Displacement**：当前参数 θ 到初始 θ_0 的 L2 距离
- **Basin width**：在当前 θ 加随机扰动 ε，测量 text loss 恶化程度

### 诊断的局限

1. **cos φ 是反事实测量**：它测的是 "在当前 θ 下 text 和 audio 目标是否冲突"，不是训练中实际的梯度冲突（训练时用的是加权求和后的梯度）
2. **单 batch 采样**：每次诊断只用 val 中第一个包含 text+audio mask 的 batch，有噪声
3. **CB val losses 的分辨率**：约 0.016（1/64），更细的变化看不到
4. **s3_adam 从 post-S3 checkpoint 开始**：我们观察的是 continued S3 的动态，不是 fresh S3。初始 loss 已经低了
5. **M-SAM 的 rho/cos_phi 数据缺失**：batch_size=1 导致诊断 batch 中 text 和 audio mask 无法同时为 true，梯度指标全为 0

### 不同实验的可比性

- s3_adam, s3_lambda3, s3_gradproj, s3_cbweight, s3_msam：同起点（post-S3 ckpt），同 LR=2e-5，可直接比较
- skip_s2：不同起点（post-S1），不同 LR（2e-4），绝对值不可比但趋势可比
- s2_only：不同训练目标（no audio loss），LR=2e-4，cos φ 和 grad_norm 的绝对值不可直接跟 S3 比
- s3_gradproj：batch_size=1（其他 batch_size=2），val 指标绝对值不同但趋势可比
