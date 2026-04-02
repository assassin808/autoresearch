# Meeting Notes — April 1, 2026

## 会议摘要

### 核心方向：4 个研究任务

#### 任务 1: 定位问题根源 — 逐层 GSNR 分析
- 现在我们只有全局 GSNR=0.08，但不知道是 embedding 层的问题还是中间层的问题
- **需要做逐层 GSNR**：对 wte(embedding)、layer0、layer6、layer12、layer18、layer23、lm_head 分别计算 GSNR
- 如果 embedding/lm_head 的 GSNR 极低但中间层正常 → 瓶颈在 embedding
- 如果所有层 GSNR 都低 → 问题不只在 embedding，可能在数据本身
- 已有 D6 rank 数据支持 embedding 假说：audio rank 883→847 (几乎不变，仍随机)，text rank 794→273 (大幅特化)

#### 任务 2: Embedding Warm-up 训练策略
- ~~隔离 TTS adapter~~ (mini-omni 发布版不含 TTS adapter，不需要测试)
- 替代方案：**先冻 backbone 只训 audio embedding**，再做全参数 S3 训练
- Phase 0: freeze backbone, train wte[152000:] only (~3k steps)
- Phase 1: unfreeze all, normal S3 training
- 目标：让 audio embedding 先形成结构 (rank 下降)，提升后续训练的 GSNR

#### 任务 3: 单模态 Oracle 的聚类与不变量分析
- **纯 audio 模型 oracle**：用预训练的 audio-only 模型，提取 audio token 的表征，做聚类
- **纯 text 模型 oracle**：用预训练的 text-only 模型，提取 text token 的表征，做聚类
- 分析各自的聚类结构（cluster invariants）：
  - 各模态内部哪些 token 聚在一起？
  - 两种模态的聚类之间有没有**对应关系**（invariant mapping）？
  - 这些不变量能否产生**无监督伪标签 (pseudo labels)**？

#### 任务 4: Omni 训练中的聚类动态与数据难度
- 在 omni model 训练过程中，跟踪 audio/text embedding 的聚类结构如何变化
- 如果某些 cluster 在 omni 训练中**坍缩 (collapse) 更快** → 这些数据更难学
- 这可以定义 **数据难度 (data difficulty)**：cluster 坍缩速度 = 学习难度
- **Epiplexity** 可作为数据难度/复杂度的指标
- 最终目标：区分"embedding 问题"和"数据本身的问题"
  - 如果 embedding warm-up 后 plateau 消失 → 纯 embedding 问题
  - 如果 warm-up 后仍有 plateau → 数据本身的困难，需要 cluster-guided curriculum

### 研究框架

```
Level 1: 定位瓶颈
  → 逐层 GSNR → 是 embedding 还是中间层？
  → D6 rank: audio emb rank 几乎不变 (883→847), 支持 embedding 假说

Level 2: 理解数据
  → 单模态 oracle 聚类 → audio/text 各自的表征结构是什么？
  → 跨模态不变量 → audio 和 text 的聚类有对应关系吗？
  → 无监督伪标签 → 可否从 oracle 聚类中获得 supervision signal？

Level 3: 理解训练动态
  → 聚类变化追踪 → 哪些 cluster 在 omni 训练中坍缩？
  → 数据难度定义 → 坍缩速度 / epiplexity = 难度 → 指导训练策略
  → 即使有完美 embedding，如果数据本身难，训练问题仍存在

Level 4: 解决方案
  → Embedding warm-up (基于 Level 1)
  → 数据 curriculum (基于 Level 3: 先训容易的 cluster)
  → 不变量保持 loss (基于 Level 2: 训练中保持跨模态对应关系)
```

### 具体可量化的指标

| 指标 | 含义 | 怎么算 |
|---|---|---|
| **逐层 GSNR** | 每层的梯度信噪比 | 累积 K 个 batch 的逐层梯度, 计算 signal²/variance |
| **Embedding rank** | embedding 的有效维度 | SVD → 熵 → exp(H) (已有 D6) |
| **Cluster stability** | 聚类在训练中的稳定性 | 每 N 步做 k-means, 计算 cluster assignment 的变化率 |
| **Cross-modal invariant** | audio-text 聚类的对应强度 | Mutual information / alignment score |
| **Epiplexity** | 数据/表征的复杂度 | 待定义（与聚类结构和学习难度关联） |
| **Cluster collapse rate** | 聚类坍缩速度 | 有效聚类数随训练步数的变化 |

### 优先级

1. **立即可做**：逐层 GSNR 实现 + embedding warm-up 实验（集群 4/7 恢复后）
2. **需要设计**：单模态 oracle 聚类分析框架
3. **需要理论**：epiplexity 定义 + 跨模态不变量形式化
4. **长期**：cluster-guided curriculum training

### 关于 Batch Size / Gradient Clipping
- 会上提到了 gradient clipping 和 batch size 的影响
- 我们已有 GSNR 分析: audio GSNR=0.08, 每步梯度 92% 是噪声
- Gradient clipping (max_norm=1.0) 可能在截断有用信号 → 值得检查
