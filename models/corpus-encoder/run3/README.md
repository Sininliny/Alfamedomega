# 结构编码器权重 · 第三轮（按听感加权）

`model.pt`：`amo.corpus-encoder/1`。它由 `training/amo/corpus/train.py` 训练，可以用 `training/amo/corpus/evaluate.py` 加载。输入一段音频的特征和演奏条件，输出结构族、每个节点的算子和全部参数值。它不是插件加载的 `amo.runtime.gru/1` 模型。

- 训练数据：corpus-v3，每族 2048 个构造，共 524,288 条，32 个结构族，全部由我们自己的原生 DSP 渲染，不含任何第三方插件或录音。
- 训练：Kaggle T4，第 15 epoch 最佳，第 25 epoch 早停。参数 loss 按 `sensitivity.json` 加权。
- test 集：重渲染距离中位数 0.180（第二轮 0.184），结构族准确率 0.866。完整报告见 `evaluation-test.json`。
- SHA-256：`e23a7bc2347875fb94f26f269108a9e42ddad415a62d588cf098fe5afd83850e`

加载时需要 `torch.load(..., weights_only=False)`，因为包里带有 numpy 统计量。只加载你信任来源的 `.pt` 文件。
