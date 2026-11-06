# 已完成：大规模、多构造 CPU 数据预处理

本批数据已实际生成、提取特征和全量核验，没有启动模型训练。原先的 `round-next-20261014/dataset-v1` 保留用于比较；新的主数据目录是`training/data/round-expanded-20261014/corpus-v2/`。

| 指标 | 上一批 | 本批 |
|---|---:|---:|
| 独立原始构造 C | 96 | **16,384** |
| DSP 结构族 | 1 | **32** |
| 原生运算覆盖 | ADSR＋加法振荡 | **23 种运算** |
| datapoint | 9,216 | **131,072** |
| 音频时长 | 约 3.58 小时 | **约 47.50 小时** |
| 声道/采样率 | 单声道 / 16 kHz | **立体声 / 16、24 kHz** |
| FX 输入/输出配对 | 无 | **32,768 条** |

独立原始构造约增加 170.7 倍，样本数约增加 14.2 倍。每个 C 都是分层采样的独立原生配置；它和受边界约束的 I 各覆盖四种条件。没有把录音增益复制计作新增独立构造。

## 验证得到的实际多样性

- 131,072 个 WAV 内容指纹均不相同。
- 特征按 1e-3 精度量化后，有 **130,800** 个不同指纹。其余 272 条属于这个特征精度下的重复响应，不夸大它们的区分度。
- 对每个结构族统一演奏和激励，并删除绝对音量/持续时间后，**16,384 个原始配置在各自结构族内均有不同的相对声学响应**。
- 这是构造与声学多样性诊断，不是统计 effective sample size 或感知独立性的证明。

结构覆盖包括稀疏/稠密/非整数加法、减法滤波、波表、FM/AM/环形调制、时变延迟、phaser、allpass/comb、非线性与反馈、噪声激励、模态/共振器/调谐波导，以及具有真实条件输入的 FX。路由、时钟、状态布局、运算身份、assets 和真正的 padding/presence 都作为构造监督保留。详见 [设计说明](DESIGN.md)。

## 分区

| 默认分区 | 原始配置 | datapoint |
|---|---:|---:|
| train | 12,288 | **98,304** |
| validation | 2,048 | **16,384** |
| test | 2,048 | **16,384** |

所有演奏和实例变体跟随完整 C 分区。全量核验没有发现原始组或相同音频跨分区泄漏。

另有 `unseen-topology-protocol.json`：完全留出四类结构组合，相关原生操作在其余组合中已有出现。它和默认协议独立选择，不能普通训练之后再把已见过的结构宣称为未见结构。

## 产物和读取

主目录约 **23 GiB**，分为 **256 个分片**，每个分片 64 个原始配置/512 条样本。

- `report.json`：实际数量、运算范围、时长和限制。
- `verification.json`：全量 131,072 条检查；196,608 次内容指纹核验；1,536 次原生重放与特征重算。
- `diversity-audit.json`：每个结构族的统一条件响应与邻近距离。
- `catalog.sqlite`：原始组、样本、内容指纹、分区和分片索引。
- `operation-vocabulary.json` / `coordinate-semantics.json`：运算 token 和参数语义/边界。
- `shards/*/index.jsonl`：音频路径、C/I、输入、演奏、质量、内容指纹和实例扰动。
- `shards/*/prepared.npz`：音频特征、条件输入、完整双头数值/图/资产目标和 padding masks。
- `*-legacy-manifest.json`：22 个已知结构的参考 WAV learner 子集，合计 48,662 条 train/validation 样本。噪声/FX 或基频条件不适用的样本不会被强塞入该参考谐波分析器。
- `generation-source.py`：本次实际渲染脚本的内容快照；`metadata-refinement.json` 记录采样调制频率的导出边界修正，原始原生文档与扰动文档在 96 组比较中保持相同。

全量数据通过 [reader.py](reader.py) 使用，已验证 CPU Torch 批次、尾批，以及两种评估协议。它分开返回 observed inputs、construction targets 和 metadata，避免把真实结构 ID 混入未知 WAV encoder 的输入。

```python
from pathlib import Path
from training.amo.corpus.reader import batches

root = Path("/workspace/Alfamedomega/training/data/round-expanded-20261014/corpus-v2")
for batch in batches(root, split="train", batch_size=64,
                     protocol="originals", as_torch=True):
    audio_features = batch["inputs"]["X_audio"]  # [batch, 2244]，CPU
    performance = batch["inputs"]["conditions"]
    targets = batch["targets"]  # canonical_* / instance_*；包含真实 masks、图与 assets
    # 下一轮模型在此使用监督目标；本次预处理没有执行 optimizer。
```

每次只解压和加载一个有限分片。批次可能少于 `batch_size`，不要把尾批视为失败。完整 C/I JSON 保留 compact pack 未展开的初始状态和可听输出混音等内容。

现有 WAV-only learner 可以直接读取上述同构 reference manifests；它本身没有未知拓扑预测或成对 FX 条件输入接口。全量数据已准备为结构监督和成对输入任务，不宣称旧模型已自动具备这些能力。也不把合成课程宣称为真实历史音频、真实用户反馈或任意构造的唯一逆解。

## 复核与重做

```bash
cd /workspace/Alfamedomega
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
python -m training.amo.corpus.prepare validate training/data/round-expanded-20261014/corpus-v2 --verification-output /tmp/corpus-verification.json
python -m training.amo.corpus.diversity_audit training/data/round-expanded-20261014/corpus-v2 --workers 4
```

重做须选择新版本目录：

```bash
python -m training.amo.corpus.prepare build NEW_DIRECTORY --canonical-per-type 512 --workers 4 [--no-wav]
```

原生渲染使用 `build/data-release/src/tensor/libalfa_tensor_c.so` 的 Release 库。库和源码版本指纹已保存；只在 CPU 上运行。生成时峰值不超过 0.406357；重放最大逐样本误差 5.96e-8，符合 PCM24 量化精度。音频、状态时钟、路由/图、资产、mask 和特征均已核验，未启动 optimizer。原始生成版本与审计结果见 [reports/report.json](reports/report.json) 和 [reports/verification.json](reports/verification.json)。

## 代码与历史数据版本

此目录保存可维护的生成、读取与审计程序；音频、标签、特征分片和索引保存在忽略的 `training/data/` 下，完整数据单独传输，不进入 Git 历史。`reports/` 保存本批历史审计证据。

本批原始数据在提交 `342642d` 上生成，仍保留旧 `aom.*` schema 和准确的生成脚本快照，避免改写已有标签身份和内容指纹。当前程序使用 `training.amo`，新数据写入 `amo.*`；仓库兼容层支持读取旧 schema。重新生成是新批次，不能据此声称与旧批次逐字节一致。原生库需要从当前源码重建；可以用 `AMO_TENSOR_LIBRARY` 指定其绝对路径。

```bash
cmake --preset core -B build/data-release -DCMAKE_BUILD_TYPE=Release -DALFA_BUILD_TESTS=OFF
cmake --build build/data-release --target alfa_tensor_c --parallel 4
python -m training.amo.corpus.prepare probe training/data/corpus-probe
```

`generation-source.py` 是历史证据，请运行上面的模块入口，不要直接执行快照文件。对旧数据的严格重放应使用报告中记录的原始源码及原生库；新版运行时的兼容性另行验证。

当前 main (`b43c6cb`) 的额外兼容性抽查完成 1,536 次重放，其中 5 次略超原始 PCM24 严格阈值，最大逐样本误差为 `4.58e-7`，涉及加法振荡及其级联。新版原生运算已更改计算实现；不放宽历史验证阈值来隐藏差异。详见 [版本兼容报告](reports/runtime-compatibility.json)。原始生成版本的全量验证仍然保留。

## 在 Windows 本机恢复完整数据

下载所有 `corpus-v2.tar.zst.part-*` 分卷、`download-manifest.json` 和 [restore.py](restore.py) 到同一个临时下载目录。在该目录打开 PowerShell，使用 Python 3.12 或更新版本：

```powershell
py -3 -m pip install zstandard==0.25.0
py -3 .\restore.py --source . --project 'F:\source\code\Alfamedomega'
```

恢复程序先核对每卷大小及 SHA-256，再流式解压，无须拼接出额外的大压缩包。它把数据放到 `training\data\round-expanded-20261014\corpus-v2`，要求解压数据量加 1 GiB 的空余空间，拒绝覆盖已有 corpus。下载目录与目标目录可以位于不同磁盘。校验清单只验证分卷内容与完整性，不代替原生运行时版本验证。

## 2026-10-22 审查修订（corpus-v3）

- **修复**：`prepare.py` 在 import 时设置 `CUDA_VISIBLE_DEVICES=""`，而 `reader.py` 会 import 它，导致任何用 reader 的 GPU 训练静默退回 CPU。现在只在命令行入口设置。
- **修复（数据偏差）**：v2 中物理/模态 5 个结构族的时长单独取 1.6–2.4 秒，其余 0.9–1.45 秒；时长同时在 `X_audio[0]` 和 `conditions[1]` 里，模型可以只凭时长认出这些结构族。现在所有族统一取 1.2–2.4 秒。因此重新生成的数据是新版本 `corpus-v3`，与 v2 不逐字节一致。
- **新增 `--no-wav`**：WAV 只在内存中量化并计算指纹，不写盘。全量约 0.8 GB（含 WAV 约 30 GB）。`validate` 在此模式下重新渲染并要求逐字节相同的 PCM24 指纹；旧 WAV learner 的 legacy manifest 需要 WAV 文件，因此跳过。
- `--workers` 上限改为本机 CPU 数；无 git 的代码包（Kaggle）用 `AMO_SOURCE_COMMIT` 记录来源提交。
- 全量生成很快：4 核约 25 分钟（4096 条 47 秒）。

## 训练结构编码器

[train.py](train.py) 从 `X_audio` 与已知演奏条件学习：结构族（32 类，只作预测目标）、8 个节点槽的原生运算、所有 padding 坐标的 presence，以及 canonical / instance 两头的 parameters、routing、modulation、assets 数值。扁平参数下标在不同结构族含义不同，所以数值按（结构族, 坐标）标准化（正值且跨度 ≥ 8 倍时用 log），数值头以结构族为条件：训练时用真实族，推理时用预测族。族内恒定的坐标直接由统计量解码，不计损失。

```bash
python -m training.amo.corpus.train CORPUS_DIR runs/corpus-encoder --epochs 120
python -m training.amo.corpus.train CORPUS_DIR runs/corpus-encoder --epochs 120 --resume auto
```

- 每个 epoch 原子写入 `last.pt`，验证集改善时写 `best.pt`；`--checkpoint-minutes`（默认 20）在 epoch 内也存 `last.pt`；`--keep-every N` 另存 `epoch-NNNN.pt`。
- `--resume auto` 从 `OUTPUT/last.pt` 或 `--resume-search` 目录下最新的 `last.pt` 精确续训（权重、AdamW、GradScaler、所有 RNG、epoch 内位置）。语料、协议或模型尺寸不同则拒绝续训。
- `--time-budget-hours` 在时限前保存并以退出码 2 结束（Kaggle 用 11）。
- 监控：每 `--log-every` 步打印 loss 各分量、学习率、梯度范数、samples/s、本 epoch 与总 ETA、GPU 显存；每个 epoch 打印验证 loss、结构族准确率、运算槽/整图准确率、presence 准确率、各张量 MAE(z) 以及用预测族解码的端到端 MAE。全部写入 `train.log` 与 `metrics.jsonl`。
- 结束时导出 `model.pt`（`amo.corpus-encoder/1`：最佳权重、标准化统计、运算词表）。

在 Kaggle 上生成与训练见 [kaggle/README.md](kaggle/README.md)。
