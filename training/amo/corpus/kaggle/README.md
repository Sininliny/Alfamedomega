# 在 Kaggle 上生成语料并训练结构编码器

不需要上传数据或代码数据集：每个 kernel 内嵌所需源码（约 670 KiB），语料直接在 Kaggle 上生成。

| Kernel | 机器 | 做什么 | 输出 |
|---|---|---|---|
| `amo-corpus-generate` | CPU，开网络（CMake 依赖） | 编译 `libalfa_tensor_c`，`prepare build --no-wav`（512/族，131,072 条），`validate` | `corpus.tar`（约 0.8 GB）、`report.json`、`verification.json` |
| `amo-corpus-sensitivity` | CPU，开网络 | 编译 `libalfa_tensor_c`，`sensitivity.py`：每族抽 64 个训练实例，把每个可学习坐标移动 ±0.25σ 后用原生 runtime 重渲染，量出它对频谱距离的影响 | `sensitivity.npz`（按听感的 loss 权重）、`sensitivity.json`（每族权重排序） |
| `amo-corpus-train-a` / `-b` | GPU T4，不开网络 | 解出语料，`train.py --resume auto --time-budget-hours 11` | `run/`：`last.pt`、`best.pt`、`model.pt`、`metrics.jsonl`、`train.log` |

两个训练 kernel 轮流运行：每次推送都挂载另一个的最新输出，`--resume auto` 选全局步数最大的 `last.pt` 继续。一次 12 小时会话训不完时，推送另一个即可接着训。

按听感加权（`make_kernels.py --perceptual`）：参数 loss 按每个（族，坐标）对声音的实际影响加权，听不出来的坐标少花容量，音高、分音、衰减这类敏感坐标多花。权重在每族内部归一到均值 1，所以各族之间的比重不变。训练 kernel 会挂载 `amo-corpus-sensitivity` 的输出并传 `--perceptual-weights`；验证 loss 用同样的权重，`evaluate.py` 的指标不加权，可以直接和不加权的模型比。

## 步骤

```bash
# 1. 生成 kernel 文件夹（在仓库根目录）
python training/amo/corpus/kaggle/make_kernels.py --user KAGGLE_USERNAME --out build/kaggle -- --epochs 120

# 2. 生成语料（约 30–40 分钟）
python training/amo/corpus/kaggle/kaggle_push.py push build/kaggle/generate
python training/amo/corpus/kaggle/kaggle_push.py status amo-corpus-generate

# 2b. 按听感加权时（make_kernels.py 加 --perceptual）：先量敏感度（约 10–20 分钟）
python training/amo/corpus/kaggle/kaggle_push.py push build/kaggle/sensitivity

# 3. 训练；每次未训完就再运行一次 train（自动换 a/b 并续训）
python training/amo/corpus/kaggle/kaggle_push.py train build/kaggle
python training/amo/corpus/kaggle/kaggle_push.py status amo-corpus-train-a

# 4. 取回日志与模型
python training/amo/corpus/kaggle/kaggle_push.py output amo-corpus-train-a build/kaggle/out-a
```

`--user` 是 Kaggle 主页地址里的用户名（kaggle.com/<用户名>），不是登录邮箱；`status`/`output` 也请写全 `用户名/kernel`。凭据只从环境读取：`KAGGLE_USERNAME` + `KAGGLE_KEY`（或 `KAGGLE_API_TOKEN`），否则读 `~/.kaggle/kaggle.json`。装了官方 CLI 时也可以直接 `kaggle kernels push -p build/kaggle/generate`；训练 kernel 第二次起需要把另一个训练 kernel 加进 `kernel-metadata.json` 的 `kernel_sources`。

额外训练参数写在 `--` 之后，例如 `-- --epochs 200 --width 512`。改动模型尺寸后旧 checkpoint 不能续训，训练会明确报错。
