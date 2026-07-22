# LongVideoGuard — Resume Project Description

## One-line project title

**LongVideoGuard: Evidence-aware and efficiency-focused VideoQA with Qwen3-VL**

---

## Chinese resume version

### Compact three-bullet version

- 基于 **Qwen3-VL-2B + NExT-QA** 搭建端到端 VideoQA 研究与评测框架，完成视频级 Train/Holdout/Development/Frozen 切分、LoRA 多模态 SFT、检索式选帧、动态路由与反事实评测，避免问题级数据泄漏。
- 在相同 8 帧预算下实现并对比 **Uniform / Scene-aware / CLIP Query-aware** 三种采样策略；通过 Question-only、Black-video、Frame Shuffle/Reverse 和证据移除实验验证模型视觉依赖及时间顺序敏感性。
- 在 128 题 Frozen Set 上发现 **Uniform-8 仅损失 1.56pp 准确率，却比 Uniform-16 减少 47.7% 输入 Token、降低 24.4% 模型推理延迟**；同时通过人工审核识别 46.9% 高风险自动负样本，形成完整数据质量与失败分析闭环。

### More engineering-oriented version

- 设计可复现的多模态训练与评测流水线，验证 `pixel_values_videos`、`video_grid_thw`、assistant-only loss masking、LoRA 梯度/参数更新、checkpoint 保存与重载一致性。
- 实现视频帧检索模块：均匀采样、场景变化采样、CLIP 文本-帧相似度检索，以及 Rule/Qwen Question Router 动态选择工具。
- 建立 Frozen Evaluation、Wilson 置信区间和成对 McNemar 检验；最终推荐 Uniform-8 作为准确率-效率最优工程策略。

---

## English resume version

### Recommended three-bullet version

- Built an end-to-end **VideoQA research and evaluation framework** with Qwen3-VL-2B and NExT-QA, including video-level train/holdout/development/frozen splits, multimodal LoRA SFT, frame retrieval, dynamic routing, and counterfactual testing.
- Implemented and compared **Uniform, Scene-aware, and CLIP Query-aware** frame sampling under the same eight-frame budget; used question-only, black-video, frame-shuffle/reversal, and evidence-removal experiments to diagnose visual reliance and temporal reasoning.
- On a 128-question frozen set, showed that **Uniform-8 reduced input tokens by 47.7% and model inference latency by 24.4% versus Uniform-16, with only a 1.56-point accuracy loss**; also identified 46.9% risky automatically generated hard negatives through manual data auditing.

### Short two-bullet version

- Developed a reproducible Qwen3-VL VideoQA pipeline with multimodal LoRA SFT, frame retrieval, routing, counterfactual evaluation, and leakage-safe frozen testing.
- Achieved a 47.7% input-token reduction and 24.4% lower inference latency using Uniform-8 versus Uniform-16, with only a 1.56pp frozen accuracy drop.

---

## GitHub project summary

LongVideoGuard is a VideoQA project that studies how frame sampling, multimodal fine-tuning, text shortcuts, noisy supervision, and temporal perturbations affect Qwen3-VL. The project includes reproducible data splits, LoRA training checks, CLIP retrieval, dynamic routing, counterfactual testing, frozen evaluation, and a Streamlit demo.

---

## Interview keywords

```text
Qwen3-VL
VideoQA
NExT-QA
Multimodal SFT
LoRA / PEFT
Assistant-only loss
Video frame sampling
CLIP retrieval
Scene change detection
Dynamic routing
Counterfactual evaluation
Text shortcut
Frozen evaluation
McNemar test
Wilson interval
Streamlit
```
