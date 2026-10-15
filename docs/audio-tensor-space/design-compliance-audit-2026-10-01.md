# 最新设计落实检查，2026-10-01

> 此文记录优先修复前的审计结果。随后按用户指定完成了 DSP、FX 混合、自由组合、
> 状态迁移与 Tier 的本轮实现；当前能力及边界以 [完成契约](product-completion.md)
> 和 [最新验证](validation.md#priority-completion-validation--2026-10-01) 为准。

## 结论和依据

当前工程已经落实独立递归张量执行、显式时钟、instrument/FX/hybrid、分层 X 与矩阵、原始参数/锁、片段组合、primitive 创建、WAV 资产导入，以及浏览与加载分离。它尚未达到对话要求的完整成品；“有接口”“有参考算法”“已有自动测试”不能替代完整功能和真实宿主验收。

审计基线为本地 HEAD `c02c332`。审计开始时没有已跟踪文件修改。设计依据：

- 同工程聊天《重构音频张量编译器》的可读取记录，尤其最新一轮要求：继续迁移到成品级；支持“元合成器/元效果器”的自由组合；查看组件不改变声音；旧 UI 只参考绘制风格和帮助理解的局部可视化，不能反向规定声音架构。
- `mission.md`、`runtime-contract.md`、`production-composition.md`、`training/METHOD.md`、精度策略及迁移/产品审计。
- 同一对话明确训练暂停；本次不把未训练的最终模型列为当前 VST 交付必须立即解决的缺陷。

较旧的 method/32-knob/DAG 文档仅作为历史实现参考，不能覆盖上述要求。此次检查不修改产品实现，不启动训练，也不自行制定 tier 数值或新的声音方向。

## 需要优先处理的实现问题

### 1. 状态继承没有完整检查结构兼容性（P1，源码证据）

位置：`src/tensor/src/Runtime.cpp` 的 `RuntimeT::adoptStateFrom`；`src/tensor_plugin/src/TensorExecution.cpp` 的 `Execution::adopt`；`TensorProcessor.cpp` 的块边界发布。

当前 DSP 状态继承检查采样率、实例/状态/输出总尺寸、各位置的 primitive ID 和切片大小，然后按物理位置复制状态。它没有检查实例身份、实例时钟、参数切片关系或资产/绑定的结构兼容性。编译器也没有把 JSON 的 `instance_id` 保留成 runtime 身份。

例如，两份相同 primitive 且相同状态宽度的实例在 JSON 中交换位置后，仍会按位置继承旧状态，而非按实例身份迁移；改变某实例的时钟也不会因这一变化自动拒绝继承。不同操作 ID 或尺寸变化会拒绝，不能据此推断所有结构变化都已正确处理。

这与 runtime 文档“兼容编辑保留状态、结构变化明确重置”的宽泛描述不完全一致。应明确哪些编辑可以保留状态，为实例身份及结构兼容性建立规则，覆盖同类型实例重排和时钟修改等回归。此项为源码推导，尚未增加专项行为复现测试。

### 2. 大状态编辑仍在音频回调中整块复制（P1，性能风险待测）

位置：`TensorProcessor.cpp::processBlock` → `Execution::adopt` → `RuntimeT::adoptStateFrom`。

稀疏 journal 已解决正常 tick 时复制整个长延迟状态的问题；但兼容参数编辑发布时，音频线程仍复制完整 `state_`，并清理/复制其他缓冲。Panic 的 `reset()` 也恢复完整初始状态。其工作量随状态容量增长。

这不是已证明的每次编辑必然爆音，但现有长环形状态 benchmark 测的是 tick，并没有证明巨大状态配置在低 block size 下反复编辑、撤销重做和 Panic 能满足 deadline。需增加真实回调峰值耗时测试，再选择满足实时要求的状态交接/复位方案。

### 3. 宿主全局 dry/wet 和 bypass 没有延迟对齐（P1，源码确认的产品边界）

位置：`TensorProcessor.cpp::processBlock` 的 wet/dry 合成；`production-composition.md` 的 Authored native host latency。

当前 `host_latency_samples` 只声明 DAW 补偿；全局 mix 直接混合当前 dry sample 与当前 wet output，bypass 也直接输出当前 dry。对于具有算法延迟的 STFT 或线性相位过采样声音，部分混合时会叠加不对齐的信号；完全 bypass 时还需验证插件声明延迟与实际直通输出之间的关系。

文档已经明确承认不自动插入 dry/wet 对齐缓冲，因此此项不是“违反现有底层契约”，而是成品 FX 工作流仍未完成的边界。作者可以在 tensor 内做对齐混合并保持全局 mix=1，但这不能使全局 mix 本身获得对齐。需要明确全局控件的可用语义，并验证冲激、部分混合及旁路切换；不应把艺术性 delay 自动算成补偿延迟。

## 尚未完整落地的设计能力

| 项目 | 当前实际行为 | 未满足的部分与边界 |
| --- | --- | --- |
| 四档 tier 的复杂度预算与转换 | `Precision.cpp` 选择 Lite/Standard=f32、Pro/Studio=f64；`Processor::setTier` 改类型及输入转换预算。其他预算来自声音 JSON。 | 尚不是完整的不同容量/分辨率档位转换。`training/amo/tiers.py` 只对调用者给出的候选和评估器做受约束选择，没有接入插件 tier 切换。具体档位数字未获指定，不能擅自补成旧版配额。 |
| 任意片段自由组合 | 已支持插入、克隆、移除、音频/参数递归绑定、offset 重映射与显式 Apply。 | `TensorComposition::lower` 拒绝含 `runtimeModel` 的片段；不同 `performance` 或 `latentSound` 策略要求预先显式协调，面板没有完整的协调流程。编辑 flattened 参数后 recipe 会过期，必须将当前声音重新作为 base，原有分片编辑工作流不能直接连续保留。明确拒绝比静默丢数据正确，但自由组合仍有边界。 |
| 成品 DSP 质量和成本 | 注册表共 68 个操作；有 production ring、streaming STFT、版本化 1x/2x/4x nonlinear。 | 分区卷积/优化 FFT 接入、随播放比率变化的抗混叠重采样、较完整的物理模型/phase-vocoder 等尚未完成。直接 FIR 的成本随 IR 长度增长；参考操作的存在不等于旧版完整品质、延迟或性能 parity。 |
| 功能解释与局部可视化 UX | 有输出波形/FFT、仪表、实例列表、参数行、矩阵系数编辑和 JSON inspection；选择实例只改变查看对象。 | 组合和关系编辑仍主要依赖扁平数字坐标及 JSON。未形成完整的局部输入/输出/状态关系、控制矩阵影响、时钟更新行为的可视化解释。不能把“需要局部可视化”解释为恢复旧 method pipeline；也没有要求强制采用某种节点图 UI。 |
| 偏好学习进入运行产品 | `training/amo/local_learning.py` 有离线参考流程；host 有独立 preference 输入类别。 | 新 Processor 的 `preferences_` 初始化为零，源码未见加载/更新偏好向量的接入。现有类别入口不是端到端本地偏好适配。按训练暂停决定，此项记录为后续集成，不立即开启训练。 |
| 成品运行验收 | 有核心、直接 Processor、独立 JUCE VST3 host 与先前真实设备 Standalone 测试。 | 仍缺新 Processor 完整回调 heap-allocation 捕捉、多实例/长时/大资产/重复编译的 deadline 压力和主要 DAW 工程保存关闭重开验收；Linux/macOS 也没有本轮结果。 |
| 大工程资产和预设规模 | 完整 JSON 和原 fragment 存在 recipe；校验、原子保存和过期保护已有。 | 原始 fragment 与 flattened 数据会增加序列化体积，没有资产去重/压缩及巨大工程的资源验收。 |

## 训练暂停期间保留的长期目标

没有覆盖通用数字音频历史的预训练构造模型，也没有任意 WAV 的重建质量证明。`training/amo/pipeline.py::export_prediction` 依赖结构模板，预测不能添加模板之外的 primitive 实现；`analysis.py` 是受控声音的参考分析，不支持未经约定的任意录音归一化策略。

这意味着运行时、离线学习和偏好学习的类别已分开，但“从任意声音学习其可编辑构造与关系”的最终能力仍未达成。双监督 canonical/perturbed-instance 契约已实现，不应误判为尚未采用双监督。训练仍暂停，不能把训练完成作为当前审计的隐含行动。

## 文档不一致

- `README.md` 仍写 Fifty-two operations，最新注册表与 runtime/production 文档是 68。
- `docs/ui/ui-design.md` 仍以旧 method、32 knobs、按档位限制 UI 为主体。README 已声明旧文档不能覆盖新方向，但该文档的 Draft 状态及根 README 的 UI 链接仍容易使人误把它当作当前产品需求。
- `validation.md` 保留多个阶段的计数和旧入口路径，应以最后的 2026-10-01 集成结果为准。阶段性历史不能累加或作为当前功能缺失的单独证据。
- `runtime-contract.md` 中“结构变化 reset”等概括应收紧到真实兼容性检查；shared prefix、整数 sample clocks 等初始边界已公开，不能误称为无限制 shared writes 或任意异步时钟。

## 本轮验证

本轮先运行现有 Release 二进制：

- tensor：89 cases / 602424 assertions，全部通过。
- native host：55 cases / 218745 assertions，全部通过。
- 独立 JUCE host 加载已有 VST3：1 case / 5127 assertions，全部通过。

随后重新构建当前源码的 `alfa_tensor_tests` 和 `alfa_tensor_host_tests`，构建成功，重建后的两套测试再次全部通过，分别为 89 cases / 602424 assertions 和 55 cases / 218745 assertions。首次沙箱构建遭 MSBuild FileTracker 权限错误，获自动审批后正常重建成功，不作为产品源码失败。

VST3 检查针对已有产品二进制，本轮未重新构建 VST3。以上测试证明既有覆盖仍通过，不覆盖上文所有缺口；本轮未进行实际 DAW 或听感验收、未启动训练。

## 建议处理顺序

1. 明确并验证状态身份/结构迁移、巨型状态编辑时的回调 deadline，以及全局 mix/bypass 的延迟语义。
2. 补足高成本 DSP 的 production 实现，完善片段编辑/策略协调与局部关系可视化。
3. 完成真实宿主、多实例和长时验收；统一当前设计文档。
4. 在已约定方向与预算下完善 tier 容量策略和转换集成；用户恢复训练后再推进构造模型与偏好学习。
