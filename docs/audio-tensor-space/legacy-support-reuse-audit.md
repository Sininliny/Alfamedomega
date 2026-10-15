# 旧支持系统复用审计

本审计覆盖旧推理、训练、工具、AMO Centre 的数据与任务服务、偏好数据以及旧预设。DSP 数学内核和插件界面另有专项审计。这里的“可复用”是源码检查结论，不等于已经接入新插件；本次只新增本文件，没有启动训练或修改实现。

分类含义：**直接复用**表示独立于旧音色表示的工具或机制可保留；**适配**表示保留机制但必须更换数据契约或补齐边界；**参考重写**表示只能提取算法或工程经验；**不带入**表示旧默认值、表示和产品方向不能成为新系统约束。一个文件可能需要拆分后分别处理。

## 推理与模型加载

| 路径 | 判断 | 可保留内容与必须改变的部分 |
| --- | --- | --- |
| `src/inference/src/TripleBuffer.h` | 直接复用机制 | SPSC 最新完整值发布、三缓冲所有权交换。新载荷须在准备阶段分配，复制不能在音频线程分配；不能把多生产者访问当作 SPSC。 |
| `src/inference/src/Sha256.cpp` | 直接复用 | 文件校验算法不依赖音色表示；放在加载线程。 |
| `src/inference/include/alfa/inference/ModelLibrary.h`、`src/inference/src/ModelLibrary.cpp` | 适配 | 清单版本、文件长度/哈希检查、模型版本固定和平台路径有价值。更换 schema、ontology、模型种类与独立预算字段；不继承 `frameRateHz=250` 或 tier 顺序自动寻找“最近模型”的声音语义。相对路径检查还需结合实际解析路径/符号链接做根目录限制。 |
| `src/inference/include/alfa/inference/InferenceHost.h`、`src/inference/src/InferenceHost.cpp` | 参考重写 | 保留准备/发布/退休引擎分离、加载失败保留当前引擎、离线等待和显式 reset 的思路。旧实现固定 16 个 voice、8 个特征、32 个旋钮、method 和单一控制帧；不能直接成为张量执行宿主。 |
| `src/inference/src/FramePipeline.h`、`src/inference/src/FramePipeline.cpp` | 参考重写 | ONNX 张量名称/形状验证、递归状态清零可参考。patch encoder + method experts 的调度、固定 VoiceFrame 展开和 hybrid A/B 合并不带入。ONNX 会分配内存的路径仍需隔离音频线程。 |
| `src/inference/include/alfa/inference/FrameClock.h` | 参考重写 | 块内边界偏移、跨块计数与 reset 测试可参考。旧 48 kHz/192 sample hop 不能作为所有 primitive 或模型的共同更新频率。 |
| `src/inference/src/Npy.h`、`src/inference/src/Npy.cpp` | 适配 | 可作为旧资产导入器，不是通用张量契约。只接受 little-endian float32、C order、二维数组；加载前需验证 header 大小、维度乘积溢出、文件长度与资产预算，当前直接按 header 分配不可直接接收不受限新资产。 |
| `src/inference/src/LogMel.h`、`src/inference/src/LogMel.cpp` | 参考重写 | FFT/mel 数值与跨语言对齐可参考；固定 48 kHz mono、64 mel 的表示是旧模型接口。不能据此声称消除任意音高/录音条件，也不能仅用全局谱统计丢弃 articulation 和时序。 |
| `src/inference/include/alfa/inference/Tier.h`、`ControlFrameLayout.h`、`src/inference/MODEL_CONTRACT.md` | 不带入新契约 | 固定 method9、knob32、voice16、250 Hz、线程/缓冲/推理时限等只属于旧格式。保留文件用于旧实现与旧测试；新 tier 必须是同一张量 ontology 的容量预算。 |
| `src/inference/tests/AllocationGuard.h`、`AllocationGuard.cpp`、`RealtimeSafetyTests.cpp`、`RtsanTests.cpp` | 直接复用工具，适配测试 | 分配捕捉、实时线程验证工具可迁移；替换被测宿主与数据。Windows/glibc 覆盖不同，不能把只捕捉 `operator new` 视为所有库分配均已覆盖。 |
| `src/inference/tests/ModelLibraryTests.cpp`、`FrameClockTests.cpp`、`PipelineTests.cpp`、`HostTests.cpp` | 参考重写 | 保留损坏文件、失败加载、边界/reset、跨线程完整帧的测试场景；固定旧布局的断言继续测试旧实现，不改名冒充新张量测试。 |

## AMO Centre 服务与存储

| 路径 | 判断 | 可保留内容与必须改变的部分 |
| --- | --- | --- |
| `apps/control-center/src/alfacc/settings.py` | 直接复用工具，适配设置 | argv 分词、显式路径覆盖、解释器发现、atomic JSON 与 Windows 短暂共享冲突重试可保留。新训练默认入口必须显式区分旧脚本；并发写同一文件时应改用唯一临时文件/协调，现有固定 `.tmp` 不是多写者事务。 |
| `apps/control-center/src/alfacc/jobs.py` | 直接复用任务框架 | 命令 argv、独立进程、日志、取消、超时、状态持久化/中断识别不依赖 method。心跳/日志节流是管理任务时间尺度，与音频控制更新率无关。 |
| `apps/control-center/src/alfacc/context.py` | 适配 | 保存用户设置、发现仓库与服务组合可用；当前构造会创建目录并初始化旧 ModelStore/PreferenceStore。不要让音频实例构造或纯 AMO 命令被迫初始化旧训练服务。新 AMO CLI 已独立路由。 |
| `apps/control-center/src/alfacc/models.py` | 适配 | 哈希、导入 staging、版本清单、bundle 校验、先取消列出再删除、后台任务可保留。旧 REQUIRED_GRAPHS、tier 名单、schema1、frame250 是旧 ONNX 包契约。替换期间 final→old→staging 的序列与清单提交不是完整事务，第二次 rename/清单失败应可回滚；新包还须限制压缩解包大小并严格校验 Windows 文件名与解析后的根路径。 |
| `apps/control-center/src/alfacc/tiers.py` | 不带入 | 4M/16M/64M/256M 模型参数量、安装体积和 expert 图列表不能替代新运行容量预算。名称可以经产品决定继续使用，数字不能自动继承。 |
| `apps/control-center/src/alfacc/training.py` | 参考重写 | 后台运行、不在 GUI 进程导入 Torch、环境检查、导出步骤、Colab notebook 生成可参考。TrainSpec 的旧阶段/tier/default steps 和脚本导出图名称不能自动接入新训练。当前环境检查只查询 CUDA 是否存在，并未估算可用 RAM/VRAM。 |
| `apps/control-center/src/alfacc/preference.py` | 适配 | 用户选择的库路径、增量文件索引、手动标签、局部删除、分类授权和独立 learner 进程可保留。文件/文件夹名只是弱标签；不能变成对音色构造的真实监督。`learner_supports` 源码字符串探测应改为版本化能力接口。 |
| `apps/control-center/src/alfacc/assimilate/discovery.py` | 直接复用扫描机制 | 插件与预设文件发现、显式额外目录、格式支持查询可用于采集。发现某种格式不代表当前 render-host 实际支持它。 |
| `apps/control-center/src/alfacc/assimilate/harvest.py` | 适配 | 分进程扫描/渲染、采集记录、分阶段探测与日志可保留。音高探测阈值、采样网格与 rendered WAV 不等于已知 AMO canonical target；外部插件样本需要单独记录来源和不可观测构造。 |
| `apps/control-center/src/alfacc/assimilate/plans.py` | 适配 | 渲染清单、note/velocity/duration 组合、稳定 probe 标识可保留。`estimate()` 估算的是音频秒数/渲染数量，既不是实际处理 ETA，也不是 RAM/VRAM 安全预算。预置密度不能成为用户未确认的新系统优先级。 |
| `apps/control-center/src/alfacc/assimilate/analyze_worker.py` | 参考重写 | PCM 解码、包络/音高测量可参考；mono 化、固定帧长、静音和 pitch 阈值是采集策略，不能覆盖空间信息与完整 articulation 的新训练要求。 |

## 偏好与既有预设

| 路径 | 判断 | 可保留内容与必须改变的部分 |
| --- | --- | --- |
| `preference/alfapref/consent.py`、`tokens.py`、`library.py` | 直接复用工具/适配存储 | 本地授权分类、文本分词、库记录可用。token 不构成音色 ontology；保留用户标签和来源，版本化存储。 |
| `preference/alfapref/model.py`、`profile.py`、`textprior.py` | 参考重写 | 标准化/PCA、局部偏好评分、成对学习/后验探索、保存模型可参考。当前 features 包含 method onehot、固定 context，必须换成版本化张量/测量/上下文特征，并独立预算。 |
| `preference/alfapref/events.py` | 适配 | 行为事件来源、明确喜欢/拒绝和成对记录可保留。旧 `{method,pair,knobs[32]}` 需要 tensor document ID/版本、锁与可比较配置；固定 dwell/reward/衰减数值不自动作为新偏好策略。 |
| `preference/alfapref/descriptor.py` | 参考重写 | `logmel-stats-v1` 可继续读取旧索引；均值/标准差及前 10 秒截取不能作为新 WAV 双监督的全部输入。保留 descriptor ID，避免新旧向量同名混用。 |
| `preference/alfapref/generate.py`、`renderers.py`、`simulate.py` | 不带入旧 prior；适配渲染接口 | method prevalence/difficulty、32-knob 高斯扰动、hybrid 排除表是旧方向。可保留 renderer 插件化/可重复实验形式；改用有声明约束与预算的张量配置候选，而不是换名继续生成旧 synth preset。 |
| `src/plugin/include/alfa/plugin/PreferenceLog.h`、`src/plugin/src/PreferenceLog.cpp` | 适配 | 默认关闭、类别授权、音频线程仅计数/message-thread 写日志的分工可复用。`patchToVar/capturePatch` 的旧方法/旋钮表示需要替换；不把旧统计自动解释为新的“音乐意图”标签。 |
| `src/plugin/include/alfa/plugin/PresetLibrary.h`、`src/plugin/src/PresetLibrary.cpp` | 参考重写 | 名称/分类、锁定参数不被覆盖、声音与界面布局分开是可保留产品行为。旧工厂声音明确是按 method 制作的 placeholder，APVTS 参数值不包含完整张量状态结构；不能直接宣称已迁移声音。 |
| `programs/*.amo.json`、`programs/factory/*.amo.json`、`src/tensor_plugin/src/PresetStore.h/.cpp` | 新体系已有实现 | 完整版本化 tensor JSON、实际 native 编译验证、独立 UI 状态、atomic 保存与显式覆盖应作为新存储边界。工厂配置是可执行示例，不是数字音频历史已被学习的证据。旧参数预设迁移需明确映射、验证声音并报告不可映射部分。 |

## 工具与训练留档

仓库没有根目录 `tools/`；实际工具在组件内部。

| 路径 | 判断 | 可保留内容与必须改变的部分 |
| --- | --- | --- |
| `src/inference/tools/export_dummy_models.py` | 参考测试方法 | 确定性模型探针和可生成测试夹具有价值。输出完全依赖旧 ONNX 图/250 Hz/VoiceFrame 契约；保留用于旧回归，不视为新模型。 |
| `src/dsp/tools/DumpLayout.cpp`、`alfa_dsp_py/alfa_dsp_c.cpp`、`alfa_dsp_py/alfa_dsp.py` | 参考重写 | 跨语言 ABI 校验、native 渲染与 parity 检查方法可复用；导出布局和调用数据是旧 ControlFrame。新工具应调用 tensor native API 并验证状态、时钟和编译预算。 |
| `src/dsp/tools/Calibrate.cpp` | 参考测量方法 | 真实 DSP 成本测量思路可复用。新的成本需包含输入形状、资产、状态、更新时钟与交互；不能把旧 family/slot 测量表直接当作 tensor tier 容量。 |
| `src/dsp/tools/gen_tier_limits.py`、`docs/tiers/tier_budget.py` | 不带入旧分配；参考生成流程 | 单一配置生成 C++ 常量/检查 stale 的流程可参考。旧 fixed ceilings、family share 与“优先降低某 family”属于旧策略，不能替代 learned importance + 受约束跨 tier 评估。 |
| `apps/control-center/render-host/src/Main.cpp`、`render-host/test-synth/TestSynth.cpp` | 适配/复用测试夹具 | 实际插件加载、MIDI 渲染、状态/预设采集和隔离运行是有效基础；以新 VST3 二进制和完整声音文档加入验证，不要求旧方法选择器。 |
| `training/alfamedomega_train.py`、`training/alfamedomega_colab.ipynb`、`training/LEGACY_METHOD.md` | 留档，参考重写 | 随机 performance、native parity、多尺度音频损失、checkpoint、导出/蒸馏工程流程可参考。method teachers/experts、knob32、frame250、固定位宽、手写 prevalence/difficulty、hybrid 合并和旧 tier 配额不带入。旧 teacher 的监督不满足原始 canonical 主目标 + 扰动 instance 次目标。 |
| `training/amo/contracts.py`、`learner.py`、`pipeline.py`、`runtime_model.py`、`local_learning.py`、`tiers.py` | 新体系，保持独立 | 已有共享 schema/padding mask、双监督、runtime/offline/preference 分类、可编译 JSON 与受约束候选接口；目前是可执行参考实现与受控验证，不是任意 WAV 或历史设备的大规模已训练模型。本轮不继续训练。 |
| `training/amo/hardware.py` | 新体系已有容量探测 | Windows 可用物理 RAM、Linux MemAvailable、显式 CUDA device 的可用 VRAM 和显式 reserve 是正确边界；其他平台目前明确失败并要求实测 allowance，不伪造容量。CPU 核数不能直接推断可达延迟；内存快照也不能保证共享宿主之后不发生压力。 |

## 接入次序与结论

先保留与声音表示无关的加载校验、任务隔离、授权、库索引和测试工具；随后以新 tensor schema 适配模型包、事件和采集元数据。旧推理与生成接口需要参考重写，不能因为它们已经能运行就把 method、固定槽位或 250 Hz 带回新执行模型。

每次迁移必须区分“源码机制复用”“数据格式兼容”“声音行为保持”。特别是旧 APVTS preset、外部插件 rendered WAV 和历史 preference profile，均不能无损等同于 canonical AMO representation。未确定的 tier 数字、优先级、归一化策略与声音身份约束继续由明确契约/用户方向决定。本次没有自动采用旧数值，也没有删除旧组件或测试。
