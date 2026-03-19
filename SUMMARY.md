# 实验总结 / Experiment Summary

**项目**: Omni-Model Training Dynamics (Basin Shaping)
**分支**: `omni-basin-shaping`
**日期**: 2026-03-18
**GPU**: 1×RTX 5090 (32GB)

---

## 一、完成了什么

### 基础设施

1. **逆向工程 mini-omni 训练流程** — 官方未公开训练代码，我们从推理代码+论文中完整还原
2. **发现并修复 SNAC delay pattern** — stream i 延迟 i 个位置。修复前 audio loss = 9.45（接近随机），修复后 = 4.56
3. **发现 `tie_word_embeddings=True`** — 旧代码用了 False（694M params），正确应为 True（532M, wte 和 lm_head 共享）
4. **下载并验证官方 checkpoint** — text loss = 2.11, audio loss = 4.62（校正后指标）
5. **构建 S3 数据管线** — 88K VoiceAssistant-400K → 346K 训练序列（4 种任务类型，含 delay pattern）

### 实验（共 5 个训练实验 + 2 个 observation 实验）

| 实验 | 配置 | 步数 | 时长 | 状态 |
|------|------|------|------|------|
| obs_1 | S2 Adam baseline + D1-D7 诊断 | 3000 | 133min | ✅ 完成 |
| obs_2 | S2 text-only (λ=0) 对照 | 2000 | 89min | ✅ 完成 |
| s3_adam | S3 Adam baseline (从 checkpoint) | 3000 | 133min | ✅ 完成 |
| s3_lambda3 | S3 λ=3（更高 audio 权重） | 3000 | 133min | ✅ 完成 |
| s3_gradproj | S3 梯度投影（去除对抗分量） | 3000 | 348min | ✅ 完成 |

### 文献综述（8 轮自动扫描）

覆盖 30+ 篇论文，涵盖：
- Kimi Attention Residuals, 贾佳亚 MGM-Omni
- Moshi 100:1 codebook weighting, DRI 修复
- CAGrad, SAMO, M-SAM 等梯度方法
- VITA-1.5, MinMo, Speech-Omni-Lite 等 frozen-LLM 方法
- SNAC vs 现代 codec（SpeechTokenizer, X-Codec, CosyVoice2）

---

## 二、核心发现

### 发现 1：ρ(t) 始终停留在 Phase 1

ρ = ‖∇L_audio‖ / ‖∇L_text‖ 在整个训练中维持 0.18-0.29。Audio 梯度始终比 text 弱 4×。模型从未进入我们预测的 Phase 2（ρ≈1，模态竞争区域）。

### 发现 2：梯度干扰在 LR 峰值时最强

cos φ（text-audio 梯度夹角）在步骤 1300-1500 达到峰值 **-0.29**，恰好是 LR 达到最大值（2e-5）的时候。之后随 LR 衰减恢复到 -0.09。干扰集中在 Layer 0 和 Layer 23（70% 负 cos φ）。

### 发现 3：Audio 严重 plateau

| 步数 | Audio loss | 每 200 步改善 |
|------|-----------|-------------|
| 0→200 | 4.56→3.92 | -0.64 |
| 200→600 | 3.92→3.76 | -0.04/200步 |
| 600→1600 | 3.76→3.63 | -0.013/200步 |
| 1600→3000 | 3.63→3.57 | -0.004/200步 |

80% 的改善在前 200 步完成。之后几乎停滞。

### 发现 4：Loss reweighting 无法打破 plateau

λ=3（3倍 audio 权重）：audio 3.57 → 3.52（仅 -1.4%），text 无变化。ρ 轨迹几乎不变。简单放大 audio 梯度不能解决问题。

### 发现 5：梯度投影也无法打破 plateau

GradProj（去除 audio 梯度中与 text 对抗的分量）：同样的 plateau 模式，从 step 1800 开始停滞。梯度范数降低 10×，但 audio loss 改善趋势一致。

### 发现 6：Basin width 不变

Basin probing（ε=0.1 扰动后的 text loss 恶化）在 step 1000 和 2000 完全一致（~90-97）。S3 训练不会扩大 text loss basin。**否定了 "audio-as-GO-noise" 的原始假说。**

### 发现 7：问题在 Codec，不在优化器

文献综述 + 实验共同指向：

**SNAC 是 LM 音频预测的最差选择：**
- 无语义信息（不像 SpeechTokenizer, X-Codec, Mimi）
- 无一致性保证（同一段 audio → 不同 token，DRI 问题）
- 为压缩设计，非为 LM 预测设计

**所有现代 omni-model 要么修复了 codec，要么分离了语音路径：**

| 模型 | 方案 | Text 退化 |
|------|------|----------|
| Mini-omni | 共享 backbone | +4%（我们测量） |
| VITA-1.5 | 冻结 LLM | 0% |
| MinMo | LoRA 微调 | ~0% |
| MGM-Omni（贾佳亚）| 脑-嘴分离 | ~0% |
| Qwen3-Omni | Thinker-Talker MoE | ~0% |

**Moshi 用 100:1 的 semantic vs acoustic 权重**，而我们对 7 个 SNAC stream 等权。

---

## 三、我们的独特贡献

不是 "一个更好的优化器"，而是：

> **首次实证描述了共享 backbone omni-model 中优化器碰壁的原因和位置。**

具体来说：
1. **ρ(t) 诊断**：量化了 text-audio 梯度力量比
2. **cos φ(t) 诊断**：定位了梯度干扰的层（Layer 0, 23）和时间（LR 峰值）
3. **Plateau 曲线**：三种优化方法（baseline, λ=3, grad_proj）产生相同的 plateau，证明瓶颈在 codec/token 层面
4. **Basin probing**：证明 audio 训练不改变 loss landscape 几何

---

## 四、下一步建议（按预期影响排序）

### Tier 1：立即可做
1. **Moshi 风格 CB 权重**（CB1=100, CB2=10, CB3=1）— 零代码复杂度，最大预期影响
2. **M-SAM**（NeurIPS 2025）— Shapley 识别主导模态 + SAM landscape 调制

### Tier 2：中等投入
3. **冻结 LLM + LoRA**（VITA-1.5/MinMo 方案）— 消除梯度冲突
4. **DRI-consistent SNAC 重训**— 修复 codec token 一致性

### Tier 3：需要架构变更
5. **脑-嘴分离**（MGM-Omni）— 独立 SpeechLM
6. **替换 SNAC 为语义 codec**（X-Codec/SpeechTokenizer）

---

## 五、文件清单

| 文件 | 内容 |
|------|------|
| `train_s3.py` | S3 训练 + D1-D7 诊断，支持 baseline/grad_proj/adaptive_lambda |
| `prepare_s3.py` | 数据管线（含 delay pattern） |
| `eval_s3.py` | 校正的评估脚本 |
| `REPORT_s3_dynamics.md` | 完整实验报告 |
| `LITERATURE_NOTES.md` | 8 轮文献综述笔记（30+ 篇论文） |
| `PLAN_optionB.md` | 实验计划 |
| `results/s3_adam/diagnostics.json` | Baseline D1-D7 完整数据 |
| `results/s3_lambda3/diagnostics.json` | λ=3 完整数据 |
| `results/s3_gradproj/diagnostics.json` | GradProj 完整数据 |

所有代码和数据已 push 至 `omni-basin-shaping` 分支。
