# LongVideoGuard — Big Tech Interview Q&A

## 1. Project overview

### Q1. 用一句话介绍这个项目。

这是一个基于 Qwen3-VL 的 VideoQA 研究系统，重点研究多模态 SFT、视频帧预算、视觉捷径、动态路由、反事实评测和 Frozen 泛化，而不仅是单次准确率提升。

### Q2. 这个项目最大的亮点是什么？

最大亮点是完整的可靠性闭环：

1. 视频级防泄漏切分；
2. 多模态 LoRA 训练验证；
3. Swap-video 检测文本捷径；
4. 多种选帧策略；
5. 动态 Router；
6. 反事实视觉和时序评测；
7. 预注册 Frozen Evaluation；
8. 最终错误分析。

### Q3. 最终最重要的结果是什么？

Uniform-8 相比 Uniform-16：

- 输入 Token 减少 47.7%；
- 模型推理延迟降低 24.4%；
- 峰值显存降低约 211 MB；
- Frozen 准确率只下降 1.56 个百分点。

---

## 2. Dataset and split design

### Q4. 为什么必须按视频切分，而不是按问题切分？

NExT-QA 同一个视频对应多个问题。若按问题随机切分，同一视频可能同时出现在训练和测试中，模型可以记住视频内容或场景，导致数据泄漏和虚高的泛化结果。

### Q5. 你的四个集合分别做什么？

- Train：LoRA 训练；
- Holdout：checkpoint selection；
- Development pilot：采样、Router 和反事实实验；
- Frozen：所有策略固定后的一次性泛化评测。

### Q6. 为什么 Frozen Set 不能继续调参？

一旦根据 Frozen 结果调整参数，Frozen 就变成新的开发集，最终结果会有选择偏差，无法再代表未见数据泛化。

---

## 3. Qwen3-VL and multimodal inputs

### Q7. Qwen3-VL 的视频输入大致如何进入模型？

视频先由 processor 解码和采样成帧，再经视觉编码器提取视觉 token，经过 projector 映射到语言模型维度，和文本 token 一起进入 decoder-only language model。

### Q8. `pixel_values_videos` 和 `video_grid_thw` 分别是什么？

- `pixel_values_videos`：视频帧经过预处理后的视觉输入张量；
- `video_grid_thw`：描述时间、高度、宽度方向 patch 网格，用于模型恢复视觉 token 的空间和时间结构。

### Q9. 为什么你需要检查 input token 和 video grid？

因为多模态流水线最容易出现“代码能运行但视频没有真正进入模型”的问题。检查这些字段可以确认视觉输入确实被 processor 正确构造。

---

## 4. SFT and LoRA

### Q10. 什么是 assistant-only loss？

只对 assistant 输出部分计算语言模型损失，用户 prompt、视频占位符和系统信息的 label 全部设为 `-100`，避免模型学习复现输入。

### Q11. 你如何验证 loss masking 正确？

我检查了：

- prompt prefix 是否完全一致；
- prompt token label 是否全部为 `-100`；
- assistant target token 是否全部参与监督；
- supervised token 数是否与答案长度一致。

### Q12. 为什么使用 LoRA？

Qwen3-VL-2B 有约 21 亿参数，LoRA 只训练低秩增量矩阵，可以显著降低显存、存储和训练成本，同时保留基础模型能力。

### Q13. 你的 LoRA 训练了多少参数？

约 321 万个可训练参数，占总参数约 0.15%。

### Q14. 如何证明 LoRA 真的在更新？

我同时检查：

- 第一次反向传播的 gradient norm 非零；
- 224/224 个 LoRA tensor 都发生变化；
- 参数 delta L2 非零；
- adapter 保存重载以后预测与训练后一致。

### Q15. 为什么小样本 overfit 不是最终成功？

因为 overfit 只能验证训练管线可学习。我的模型在训练集达到 100%，但 Swap-video 后仍有 68.75% 保持正确，说明它可能记住问题和选项，而不是依赖视频。

---

## 5. Shortcut learning

### Q16. Swap-video 实验是什么？

保持问题和选项不变，但替换成另一段视频。若模型预测仍不变或仍正确，说明它可能主要使用文本先验而不是视觉证据。

### Q17. 你的 Swap-video 结果说明什么？

小样本 SFT 中，正确视频准确率 100%，替换视频后仍有 68.75%，说明文本捷径非常严重，但 31.25% 的预测发生变化，也说明模型并非完全忽略视频。

### Q18. Question-only 为什么还能有 43.75%？

因为多选题存在语言先验、选项分布偏差和常识推断。43.75% 明显高于随机 20%，说明仅凭文本就能解决一部分问题。

---

## 6. Hard negatives

### Q19. 你的 hard negative 怎么构造？

把问题和选项与另一段不相关视频配对，并将目标设成 unanswerable。

### Q20. 为什么这种构造有问题？

问题可能不依赖具体视频，或替换视频中恰好也出现相同动作，因此“跨视频”不等于“不可回答”。

### Q21. 人工审核结果是什么？

32 条样本中：

- 17 条 clearly unanswerable；
- 9 条 possibly answerable；
- 6 条 actually answerable；
- 风险率 46.9%。

### Q22. 你如何改进 hard-negative 生成？

可以：

1. 使用语义和视觉过滤；
2. 检查替换视频是否包含问题实体和动作；
3. 使用教师 VLM 多次验证；
4. 加入人工审核；
5. 构造局部证据缺失，而不是完全随机替换。

---

## 7. Frame sampling

### Q23. Uniform sampling 的优缺点是什么？

优点：

- 简单；
- 成本低；
- 时间覆盖稳定；
- 泛化强。

缺点：

- 可能漏掉短暂关键事件；
- 不考虑问题语义。

### Q24. Scene-aware sampling 如何实现？

先均匀抽取候选帧，计算相邻帧 RGB 直方图变化，优先选择变化较大的帧，同时加入最小时间间隔，避免帧过度集中。

### Q25. Query-aware sampling 如何实现？

用 CLIP：

- 问题经过 text encoder；
- 候选帧经过 image encoder；
- 计算 cosine similarity；
- 使用类似 MMR 的策略，同时考虑相关性、视觉去重和时间分散。

### Q26. 为什么 Query-aware 对 Causal 更强？

因果问题通常依赖特定人物、动作或事件，问题语义可以帮助检索更相关的局部画面。

### Q27. 为什么 Query-aware 对 Temporal 反而较弱？

Query-aware 容易集中在语义相关的局部时间段，降低全局时间覆盖，可能丢失动作前后关系。

### Q28. 为什么最终 Uniform-8 反而最好？

开发集上的复杂采样优势没有稳定泛化。Frozen Set 上 Uniform-8 更稳健、实现更简单，并且与 Uniform-16 只有 1.56pp 差距。

---

## 8. Dynamic routing

### Q29. Router 的输入和输出是什么？

输入只有问题文本，输出问题类别：

- causal；
- temporal；
- descriptive。

再映射到对应采样工具。

### Q30. Rule Router 和 Qwen Router 哪个更好？

- Rule Router 分类 68.75%，VideoQA 66.67%；
- Qwen Router 分类 75%，VideoQA 68.75%。

Qwen 更好，但仍未超过最佳固定策略。

### Q31. 为什么分类更准不一定最终答题更准？

因为：

- Router 分类错误不一定改变最终答案；
- Router 分类正确，也不保证被选方法一定答对；
- 三种采样方法之间预测差异有限。

---

## 9. Counterfactual evaluation

### Q32. 为什么做 Black-video？

用于判断模型是否依赖真实视觉内容。正常视频比黑屏高 29.2pp，证明视觉信息有效。

### Q33. 为什么做 Question-only？

用于估计语言先验和文本捷径。Question-only 43.75%，说明文本偏差不可忽视。

### Q34. Reverse 和 Shuffle 的区别是什么？

- Reverse 保留全部相邻关系，只改变方向；
- Shuffle 打破局部连续性和全局顺序。

### Q35. 为什么 Reverse 不降、Shuffle 降？

模型可能利用局部动作连续性和共现，但没有可靠编码前后方向。Reverse 仍保留相邻结构，而 Shuffle 会破坏连续性。

### Q36. relevant-mask 为什么不如 random-mask 破坏大？

CLIP 高相似帧不一定是决定答案的因果证据；它只衡量语义接近。此外，关键证据可能分布在多个帧中，或被其他帧冗余覆盖。

---

## 10. Frozen evaluation and statistics

### Q37. 为什么用 Wilson interval？

因为准确率是二项比例，样本量只有 128。Wilson interval 比简单正态近似在中小样本上更稳定。

### Q38. 为什么用 McNemar test？

不同策略回答的是同一批问题，是成对二分类结果。McNemar 只关注两个方法不一致的样本，适合比较成对准确率差异。

### Q39. 结果显著吗？

不显著：

- Scene-aware-8 vs Uniform-8：p = 0.791；
- Uniform-8 vs Uniform-16：p = 0.774。

因此不能声称复杂策略显著优于基线。

### Q40. Frozen 结果推翻开发集结论，你怎么看？

这正是 Frozen Evaluation 的价值。开发集只有 48 题，两道题就对应 4.17pp，采样波动很大。Frozen Set 表明开发集优势不稳定，所以工程上应选择更简单稳健的 Uniform-8。

---

## 11. Error analysis

### Q41. 三种方法全部错误说明什么？

32/128 三种都错，其中 26 条三种还输出同一个错误答案。说明多数顽固错误来自模型推理、语言偏差或问题难度，而不是采样。

### Q42. Uniform-16 为什么会救回一些题？

更多帧可以补充关键动作或上下文，尤其对部分因果和时序题有帮助。

### Q43. 为什么 Uniform-16 也会把原来对的题变错？

额外帧可能引入无关动作和人物，增加注意力干扰，因此视觉 token 数与准确率并非单调关系。

### Q44. Majority vote 为什么只提升一点？

三种策略高度相关。虽然有 25 条预测分歧，但多数情况下两个方法会一起犯同样的错误，因此简单投票无法充分利用互补性。

---

## 12. Engineering and deployment

### Q45. Demo 的默认策略是什么？

Uniform-8，因为它在 Frozen 上是最好的 8 帧方法，并且成本最低、链路最简单。

### Q46. 你的延迟是否包含完整预处理？

不完全包含。当前报告的延迟主要是帧已经准备好以后 Qwen 推理时间。Query-aware 的 CLIP 检索和视频解码需要单独计入端到端延迟。

### Q47. 如何降低端到端延迟？

可以：

- 缓存视频帧特征；
- 使用轻量 scene detector；
- 批量 CLIP 编码；
- 异步视频解码；
- 使用 TensorRT / vLLM 类推理优化；
- 固定 Uniform-8 避免额外检索。

### Q48. 如何扩展到开放式问答？

需要：

- 将输出从 A–E 改成自由文本；
- 使用 generation metrics 或 LLM judge；
- 加入答案标准化；
- 防止 hallucination；
- 增加证据定位和引用。

---

## 13. Limitations and future work

### Q49. 这个项目最大的局限是什么？

- Frozen 只有 128 题；
- 使用 2B 模型；
- Query-aware 没有使用答案选项；
- Counterfactual 输入有分布外问题；
- hard negatives 不够可靠；
- 延迟不是完整端到端。

### Q50. 下一步最值得做什么？

最值得做的是训练一个 answer-conditioned evidence scorer：

- 输入问题和选项；
- 输出每个时间片的重要性；
- 使用验证过的证据帧监督；
- 同时优化准确率和检索成本。

---

## Interview closing statement

我不会把项目描述成“复杂采样显著提升了性能”，因为 Frozen Set 没有支持这个结论。这个项目真正证明的是，我能够发现训练捷径、审计数据质量、设计受控反事实实验，并根据 Frozen 结果做出更稳健的工程决策。
