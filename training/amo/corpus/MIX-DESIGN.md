# 混音片段 → 多乐器 preset：训练方法（设计稿 v1，待确认）

状态：方法设计，尚未写代码。现有 `corpus-encoder` run 3 只解决“一个音符、一个乐器、一个结构”的 one-shot。本文把它扩展为：输入一段 **非 one-shot、可能是多乐器混音** 的音频，输出片段里每个乐器的 preset（以及它们演奏的音符和总线效果）。

## 1. 问题的数学表述

一个“场景”（scene）是

$$
S = \big(B,\ \{(I_k, F_k, g_k, \phi_k, N_k)\}_{k=1}^{K}\big)
$$

- $I_k$：第 $k$ 个乐器的结构与参数（就是现在 corpus encoder 的预测目标：结构族、节点算子、parameters/routing/modulation/assets）。
- $F_k$：该乐器的插入效果链（0–3 个 FX）。
- $g_k,\ \phi_k$：电平与声像。
- $N_k$：该乐器演奏的音符事件（音高、起止、力度）。
- $B$：总线效果链（混响、延迟、EQ、饱和、宽度……）。

混音由原生 DSP 渲染：

$$
x(S) = B\Big(\sum_{k=1}^{K} \mathrm{pan}_{\phi_k}\big(g_k \cdot F_k(R(I_k, N_k))\big)\Big)
$$

用户得到的 **preset** 是 $P_k = (I_k, F_k)$：AMO 是元模块化的，乐器和它的插入效果本来就在同一张图里。$B$ 单独导出成一个 FX preset，$N_k$ 导出成每个乐器一条 MIDI（试听和验证都需要它）。

距离使用已有的 `resynth.distance`（多分辨率 log-STFT L1），记作 $D(x, \hat x)$。

## 2. 为什么必须限制组合方式（你说的“升调 echo”问题）

混音到场景的映射不是单射。例如一个自带升调 echo 的合成器：

- $S_1$：1 个乐器，$F_1$ = 带移调的反馈延迟；
- $S_2$：3 个乐器，第 2、3 个是同一音色但延后若干毫秒、移高若干半音、更小声。

$x(S_1) \approx x(S_2)$，只看频谱无法区分。所以需要一个先验来选“最自然的解释”。我们用最小描述长度（MDL）形式：

$$
\mathcal{J}(S) = D\big(x, x(S)\big) + \lambda_I K + \lambda_N \sum_k |N_k| + \lambda_F \sum_k |F_k| + \lambda_B |B|
$$

取 $\lambda_I \gg \lambda_N > \lambda_F$：**多一个乐器很贵，多一些音符次之，多一个效果最便宜**。于是同一音符流经过“时移/移调/滤波/增益”得到的副本，会被优先解释成该乐器的 FX，而不是新乐器。

**“一个乐器”的定义**（训练数据按它打标签）：两个声部属于同一乐器，当且仅当其中一个能由另一个的同一音符流经过我们 FX 词表里的变换（延迟、移调、滤波、增益、声像、调制）得到。不同音色恰好演奏平行旋律（例如八度叠奏但音色不同）算两个乐器。

### 自动数量模式

$$
\hat S = \arg\min_{S} \mathcal{J}(S)
$$

网络直接预测候选和数量分布，最终用 $\mathcal J$ 在少量候选里重打分（见 §5）。

### 手动数量模式（用户规定 $K$）

$$
\hat S_K = \arg\min_{S:\ |S| = K} \Big[D\big(x, x(S)\big) + \lambda_N \textstyle\sum_k |N_k| + \lambda_F \sum_k |F_k|\Big]
$$

即在恰好 $K$ 个乐器的约束下，选融合后频谱最接近输入的组合。乐器项 $\lambda_I K$ 是常数，不再起作用。

- $K$ 小于自然数量：必须合并或舍弃。能合并的（音符流相同的叠层）合成一个 layered preset；音符流不同的只能舍弃最弱的或由一个乐器近似，频谱误差会上升，报告里如实给出。
- $K$ 大于自然数量：沿“层级阶梯”拆开（见 §3.3）：叠层拆成各层，再往下才把 echo 拆成独立乐器。用户要的数量超过最细层级时，多出的槽位拟合残差，并标注“残差槽，能量很低”。

升调 echo 合成器在 $K=1$ 下一定得到“1 个合成器 + 移调延迟”，这正是你要的限制效果。

## 3. 训练数据：算法生成的多乐器场景

新模块 `training/amo/corpus/mixscene.py`（CPU，复用 `prepare.py` 的构造采样和原生渲染）。

### 3.1 单个乐器

- 构造 $I_k$：从现有 32 个结构族里采样（LHS，同 corpus-v3），保证和 one-shot 模型同一套标签空间。
- 角色（决定音区和音符生成器）：bass、和弦/pad、lead、arp、打击（噪声/冲激/模态族）、texture。
- 音符 $N_k$：随机速度 70–160 BPM、调式、和弦进行、节奏型；打击用固定音高的 hit。片段长 4–8 s。这一步就把“非 one-shot”纳入了：同一乐器演奏多个音符、重叠、连奏、松键尾音。
- 插入 FX $F_k$：0–2 个，来自现有 fx 族，加上新的：反馈延迟（`time.ring_delay`）、**移调延迟**（`pitchtime.phase_vocoder_locked` 放在延迟反馈环里）、chorus（调制分数延迟）、小混响。

### 3.2 混音与总线

- 自然数量 $K \in [1, 6]$，权重偏向 2–4。
- 电平：每个乐器相对响度在 $[-18, 0]$ dB 内随机，声像随机。
- 总线 $B$：混响（`time.fdn_hadamard8` 或 `time.partitioned_convolution`）70%、延迟 30%、EQ/滤波、饱和、立体声宽度。压缩器目前仓库里没有原语，先不做（见 §8）。

### 3.3 层级标签（“阶梯”）

生成器按层级构建场景，因此每个场景天生带多种粒度的正确答案：

| 层级 | 内容 | 乐器数 |
|---|---|---|
| L0 | 叠层合并成 layered preset，echo/双声部作为 FX | $K_0$（最少） |
| L1 | 叠层拆成各自的乐器，echo 仍是 FX | $K_1$ |
| L2 | echo/移调副本也拆成独立乐器（延后、移调的音符流） | $K_2$（最多） |

L0 就是 MDL 意义下的自然解释，自动模式的目标。手动模式给 $K$ 时，取阶梯里乐器数为 $K$ 的层级做标签；阶梯里没有恰好 $K$ 的，离线枚举“舍弃最弱 / 合并同音符流”的方案，用原生渲染算 $D$，取最小者做标签。

### 3.4 必须刻意加入的难例（≥ 20% 场景）

- 移调 echo、普通 echo、乒乓延迟（标签：1 个乐器）。
- unison/八度叠层同一音色（1 个）。
- 不同音色平行或八度叠奏（2 个）。
- 呼应式（call-response）的两个相似音色（2 个）。
- 打击与有音高乐器重叠的低频冲突。

没有这些难例，模型学不到 §2 的先验。

### 3.5 规模估计（粗算，生成前再测）

起步 100k 场景，平均 3 个乐器、6 s，约 30 万个乐器片段渲染、50 万秒音频。按 corpus-v3 的速度（4 核约 25 分钟生成 13 万条 1.2–2.4 s），约 3–5 小时 CPU，Kaggle CPU kernel 可以分片跑。特征按下面的格式存 fp16，每场景约 120 KB，共约 12 GB。

## 4. 模型

### 4.1 输入

现在的 32 个时间窗 × 32 个频点太粗，看不清多个音符和多个乐器。新的输入：

- log 频率谱图，24 kHz，hop ≈ 21 ms（8 s 约 375 帧），Mid 128 个 log 频点（30 Hz–12 kHz）+ Side 32 个频点；
- 不再输入“已知演奏条件”（真实混音里没有）。音高、时序由网络自己从谱图里读。

### 4.2 结构

- 编码器：卷积前端 + Transformer（时间轴上的注意力）。
- **数量 token**：$c \in \{\text{auto}, 1, \dots, K_{\max}\}$，嵌入后拼进编码器。训练时 30% 用 auto（目标 L0），其余从阶梯里随机选一个 $K$（目标是对应层级）。
- 解码器：$K_{\max} = 8$ 个可学习的槽（DETR / slot attention），交叉注意编码器。每个槽输出：
  - 存在概率 $e_k$；
  - 结构族 / 节点算子 / 参数值：复用 corpus encoder 的输出头和 `TargetStats` 归一化，按族条件化；可从 run 3 权重初始化这些头；
  - 插入 FX 链：最多 3 个 FX，每个是 FX 类型 + 参数；
  - 电平、声像；
  - 音符：帧 × 128 音高的 onset/持续/力度（Onsets-and-Frames 式）；
  - 辅助输出：该乐器的时频 mask $m_k(t,f)$（源分离），用于匹配和稳定训练。
- 全局 token 输出总线 $B$ 和数量分布 $p(K)$。

手动模式：取 $e_k$ 最高的恰好 $K$ 个槽。自动模式：$e_k > 0.5$ 的槽，之后由 §5 重打分。

### 4.3 损失

槽的顺序没有意义，先用 Hungarian 匹配：

$$
\sigma^\star = \arg\min_{\sigma} \sum_{k} C\big(\text{slot}_{\sigma(k)}, \text{true}_k\big)
$$

$$
C = \alpha\,\mathrm{CE}_{\text{family}} + \beta\,\lVert W(\hat v - v)\rVert_1 + \gamma\,\lVert \log \hat Y_k - \log Y_k \rVert_1 + \delta\,(1 - \mathrm{F1}_{\text{notes}})
$$

其中 $W$ 是 run 3 的感知敏感度权重（`sensitivity.json`），$Y_k$ 是该乐器的干声 stem 谱图（生成时保留）。然后

$$
\mathcal L = \sum_{\text{matched}} \big[\mathcal L_{\text{family}} + \mathcal L_{\text{ops}} + \mathcal L_{\text{values}}^{W} + \mathcal L_{\text{FX}} + \mathcal L_{\text{notes}} + \mathcal L_{\text{mask}}\big] + \mathcal L_{\text{exist}} + \mathcal L_{B} + [c=\text{auto}]\,\mathrm{CE}(p(K), K_0)
$$

未匹配的槽只吃存在损失（目标 0）。另外约束 $\sum_k m_k \approx 1$。原生 DSP 不可微，所以混音重建距离 $D$ 不进梯度，只用于评估和推理时的重打分，这和现有 resynth 评估一致。

### 4.4 训练阶段（Kaggle T4）

1. **S1** 单乐器、非 one-shot 片段、无总线。先验收：在 one-shot 测试集上不差于 run 3（重渲染中位数 0.180）。
2. **S2** 1–3 个乐器 + 插入 FX。
3. **S3** 1–6 个乐器 + 总线 + 难例 + 数量 token 阶梯。

每阶段都可 `--resume`，沿用现有 train-a/train-b 交替 kernel。

## 5. 推理（AMO Centre “Wav to Preset”）

1. 输入 wav，选择“乐器数量：自动 / N”。
2. 切成 ≤ 8 s 的窗（有重叠）。v1 只做单窗；长曲目跨窗把槽按 preset 相似度连接，放到 v2。
3. 网络给出候选：每槽前 2 个结构族 × 阶梯相邻的 $K$（自动模式）或固定 $K$（手动），共十几个候选场景。
4. 每个候选用原生 DSP 渲染，算 $D$；自动模式按 $\mathcal J$、手动模式按带约束的目标选出最优。
5. 导出一个文件夹：每个乐器一个 preset、一个 MIDI、总线 FX preset、`report.json`（每个乐器的 stem 距离、总距离、是否残差槽），并支持单独试听每个乐器。

## 6. 评估指标

- 混音重渲染距离 $D$（主指标）；
- 匹配后每个乐器的 stem 距离；
- 自动模式数量准确率（对 L0）；
- **echo 测试集**：FX 型 echo 场景被解释成 1 个乐器的比例；
- 音符 F1；
- one-shot 回归：$K=1$ 时不差于 run 3。

## 7. 和现有部分的关系

- run 3 继续服务 one-shot 路径，新模型在 one-shot 上追平之后再替换。
- 新模型输出的每个 $P_k$ 和 corpus encoder 用同一个标签空间，所以 AMO Centre 写 preset 的代码可以共用。
- 改动集中在 `training/amo/corpus/`：新增 `mixscene.py`（场景生成）、`mixtrain.py`（训练）、`mixevaluate.py`（评估），`prepare.py`/`resynth.py` 只做小的复用性改动。

## 8. 已知限制与待定

- 只认识我们自己 32 个结构族能表达的声音。真实录音（人声、原声乐器、第三方插件）只能被近似，报告里的距离会说明近似程度。
- 仓库里没有压缩器/包络跟随类原语，总线压缩暂不建模；真实混音里的压缩会表现为残差。
- 待 Sinin 决定：
  1. 是否同时导出每个乐器的 MIDI（建议是）；
  2. $K_{\max}$：v1 用 6 还是 8（建议训练 8 个槽、数据最多 6 个乐器）；
  3. 总线效果单独导出成 FX preset，还是可选“折叠进每个乐器”（建议单独导出，另给折叠选项）；
  4. $\lambda_I, \lambda_N, \lambda_F$ 的初值：先在 echo 测试集上调，使 FX 型 echo 被解释为 1 个乐器的比例 ≥ 90%，同时不误合并不同音色的叠奏。
