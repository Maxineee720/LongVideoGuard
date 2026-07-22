# LongVideoGuard — Interview Pitch

## 30-second version

我做了一个基于 Qwen3-VL 的 VideoQA 项目 LongVideoGuard。它不只是做一次模型微调，而是完整研究了多模态系统里的数据泄漏、LoRA SFT、文本捷径、视频选帧、动态路由和反事实评测。我最终在 128 题 Frozen Set 上发现，Uniform-8 相比 Uniform-16 减少了 47.7% 输入 Token 和 24.4% 推理延迟，但准确率只下降 1.56 个百分点，所以 Uniform-8 是更好的工程默认策略。

---

## 90-second version

LongVideoGuard 是我围绕 Qwen3-VL 和 NExT-QA 搭建的一个可复现 VideoQA 项目。

我首先做了视频级 Train、Holdout、Development 和 Frozen 切分，避免同一视频的不同问题出现在不同集合里。之后我验证了完整的多模态 LoRA SFT 流程，包括视频张量、assistant-only loss masking、梯度、参数更新以及 adapter 保存和重载。

在小样本训练中，我发现模型虽然能达到 100% 训练准确率，但把视频替换掉以后仍然能答对很多题，说明存在明显文本记忆和语言捷径。因此我没有只继续堆训练，而是转向系统评测。

我实现了 Uniform、Scene-aware 和 CLIP Query-aware 三种 8 帧采样策略，还加入了 Rule-based 和 Qwen Router。开发集上 Query-aware 对 Causal 问题更强，Scene-aware 对 Temporal 更强，但在 128 题 Frozen Set 上，这些开发集优势没有稳定泛化。

最终 Uniform-8 达到 67.97%，Uniform-16 是 69.53%，只差 1.56 个百分点，但 Uniform-8 少了 47.7% 输入 Token，推理延迟降低 24.4%。因此我最终推荐 Uniform-8 作为部署默认值。

---

## Three-minute version

我的项目叫 LongVideoGuard，是一个基于 Qwen3-VL-2B 和 NExT-QA 的 VideoQA 研究与评测系统。

这个项目最开始的目标是做一个多模态微调项目，但我在实验过程中发现，只展示“训练以后准确率提高了”并不能说明模型真正理解视频，所以我把重点转向了训练可靠性、视觉依赖、视频采样和泛化评测。

第一部分是数据和训练。我按视频而不是问题进行 Train、Holdout、Development 和 Frozen 切分，避免同一视频的不同问题造成泄漏。然后我验证了完整的多模态 LoRA SFT 流程，包括 `pixel_values_videos`、`video_grid_thw`、prompt masking、assistant-only loss、梯度是否非零、LoRA 参数是否更新，以及 checkpoint 保存后重载能否复现预测。

在 16 条小样本 overfit 实验里，模型训练集达到 100%，但 Swap-video 以后仍然有 68.75% 的样本保持正确，这说明模型大量依赖问题文本和选项，而不是视频。因此我没有简单把 overfit 当成成功，而是进一步做了更大的 QA LoRA 和 hard-negative 训练。

Hard-negative 阶段虽然提升了结构化拒答能力，但我人工审核了 32 个跨视频负样本，发现 46.9% 存在风险，其中 18.8% 实际上仍然可以回答。这说明随机替换视频并不能保证问题不可回答，所以我把它作为数据质量案例，而不是最终 benchmark。

接下来我研究视频帧预算。我实现了三种采样策略：Uniform、Scene-aware 和 CLIP Query-aware，并在相同 8 帧预算下比较。开发集上 Query-aware 对 Causal 问题最好，Scene-aware 对 Temporal 最好。我还做了 Rule-based 和 Qwen Question Router，根据问题类型动态选择采样工具。Qwen Router 的类别准确率是 75%，最终 VideoQA 是 68.75%，与最好的固定策略持平。

为了验证模型到底有没有看视频，我做了 Question-only、Black-video、Frame Reverse、Frame Shuffle 和 Evidence Removal。正常视频比 Question-only 高 25 个百分点，比黑屏高 29.2 个百分点，证明模型确实使用视觉信息。Temporal 题在 Shuffle 后下降 25 个百分点，但 Reverse 后没有下降，说明模型利用局部时间连续性，但并没有可靠学习方向性顺序。

最后我在 128 题 Frozen Set 上做预注册评测。Uniform-8 是 67.97%，Scene-aware-8 是 66.41%，Uniform-16 是 69.53%，三者差异都不显著。最重要的是，Uniform-8 相比 Uniform-16 减少了 47.7% 输入 Token 和 24.4% 模型推理延迟，但只损失 1.56 个百分点准确率。

所以这个项目最后的核心结论不是“复杂方法一定更好”，而是：在多模态系统里，可靠的数据、受控实验、反事实评测和 Frozen Evaluation，比在开发集上追求漂亮数字更重要。

---

## One-sentence ending

这个项目让我最重要的收获是：多模态模型能答对问题，不代表它真的使用了正确的视觉证据，所以必须通过数据审计、Swap-video、反事实干预和 Frozen Evaluation 才能判断系统是否可靠。
