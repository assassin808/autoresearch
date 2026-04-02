# Meeting Notes — April 1, 2026

## 会议摘要

### 核心方向：4 个研究任务

#### 任务 1: 定位问题根源 — 逐层 GSNR 分析
- 现在我们只有全局 GSNR=0.08，但不知道是 embedding 层的问题还是中间层的问题
- **需要做逐层 GSNR**：对 wte(embedding)、layer0、layer6、layer12、layer18、layer23、lm_head 分别计算 GSNR
- 如果 embedding/lm_head 的 GSNR 极低但中间层正常 → 瓶颈在 embedding
- 如果所有层 GSNR 都低 → 问题不只在 embedding，可能在数据本身

#### 任务 2: 隔离 TTS adapter 的影响
- 先在 `post_adapter=True` 的配置下训练，看 audio plateau 是否缓解
- 如果加上 TTS adapter 就没问题了 → 确认问题在共享 lm_head
- 如果加了还是有 plateau → 问题更深，在数据或架构层面

#### 任务 3: 单模态 oracle 的聚类与不变量分析
- **纯 audio 模型 oracle**：用预训练的 audio-only 模型（如 SNAC encoder 或 HuBERT），提取 audio token 的表征，做聚类
- **纯 text 模型 oracle**：用 Qwen2-0.5B，提取 text token 的表征，做聚类
- 分析各自的聚类结构（cluster invariants）：
  - 哪些 audio tokens 聚在一起？（语音学上的规律？）
  - 哪些 text tokens 聚在一起？（语义上的规律？）
  - 两种模态的聚类之间有没有**对应关系**（invariant mapping）？

#### 任务 4: Omni 训练中聚类动态的变化
- 在 omni model 训练过程中，跟踪 audio/text 的聚类结构如何变化
- 如果某些 cluster 在 omni 训练中**坍缩 (collapse) 更快** → 这些数据更难学
- 这可以定义 **数据难度 (data difficulty)**：cluster 坍缩速度 = 学习难度
- **Perplexity** 可以作为另一个数据难度指标
- 最终目标：区分"embedding 问题"和"数据本身的问题"

### 研究框架

```
Level 1: 定位瓶颈 (我们已有部分数据)
  → 逐层 GSNR → 是 embedding 还是中间层？
  → 加 TTS adapter → 是共享参数还是架构问题？

Level 2: 理解数据 (新方向)
  → 单模态 oracle 聚类 → audio/text 各自的结构是什么？
  → 跨模态不变量 → audio 和 text 的表征有对应关系吗？

Level 3: 理解训练动态 (新方向)
  → 聚类变化追踪 → 哪些 cluster 在 omni 训练中坍缩？
  → 数据难度定义 → 坍缩速度 = 难度 → 指导训练策略

Level 4: 解决方案
  → Embedding warm-up (基于 Level 1)
  → 数据 curriculum (基于 Level 3: 先训容易的 cluster)
  → 不变量保持 loss (基于 Level 2: 保持跨模态对应关系)
```

### 具体可量化的指标

| 指标 | 含义 | 怎么算 |
|---|---|---|
| **逐层 GSNR** | 每层的梯度信噪比 | 累积 K 个 batch 的逐层梯度, 计算 signal²/variance |
| **Cluster stability** | 聚类在训练中的稳定性 | 每 N 步做 k-means, 计算 cluster assignment 的变化率 |
| **Cross-modal invariant** | audio-text 聚类的对应强度 | Mutual information / alignment score between audio & text clusters |
| **Perplexity per sample** | 单样本的数据难度 | exp(loss) for each training sample |
| **Cluster collapse rate** | 聚类坍缩速度 | 有效聚类数随训练步数的变化 |

### 优先级

1. **立即可做**：逐层 GSNR（我们已有 D11 per-module gradient norms 数据，可以扩展）
2. **需要实现**：TTS adapter 实验（代码已支持 `post_adapter=True`，需要训练权重）
3. **需要设计**：单模态 oracle 聚类分析
4. **长期**：跨模态不变量 + 训练动态追踪

### 关于 Batch Size / Clipping
- 会上提到了 gradient clipping 和 batch size 的影响
- 我们已有 B_crit 理论分析 (audio B_crit≈25, text B_crit≈2)
- 单 GPU 上 B_eff=2 是 wall-clock 最优，但需要验证训练稳定性
