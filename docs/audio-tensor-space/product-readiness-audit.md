# 可用产品就绪审计

范围：当前 `Alfamedomega` 张量插件壳、状态与自动化、预设库、组件编辑器和组合/primitive 弹窗。训练在本轮暂停。本文件区分源码检查、测试源码与实际运行证据；最新中央运行结果记录在 validation.md，不把构建成功等同于完整产品验收。

## 可验证的工程边界

`TensorProcessor.cpp` 把 JSON 解析、确定性编译、参数编辑、预设磁盘访问与模型准备放在音频回调之外。准备后的 `Program` 用原子 pending 指针在块边界发布；旧程序通过退休队列由消息线程释放。`processBlock` 不取得 `configurationMutex_`。此设计的线程边界值得保留，但源码检查不证明全部第三方库调用无分配。

`prepareToPlay` 为合并 MIDI 预留 65536 字节；回调拒绝超过 60000 字节的宿主 MIDI，并限制键盘队列。`TensorExecution::tick/updateControls` 使用已准备的类型统一张量、边界缓冲和矩阵空间。f32/f64 的原子精度报告、CPU runtime 与 neural compiler 状态是不同的状态对象。

当前 APVTS 的稳定宿主参数是 `x0..x7`、输入/输出增益、dry/wet、bypass、输入模式。更深层控制维度和原始参数通过完整配置编辑，不自动获得大量 DAW automation ID。八个 X 是宿主边界，不是张量 ontology 的维度上限。`getBypassParameter()` 返回实际 APVTS bypass 对象。

`isBusesLayoutSupported` 支持 mono/stereo 输出及同宽或关闭的输入；其他布局拒绝。核心多通道数学操作可存在于配置中，但插件实际公开的音频总线仍是 mono/stereo。不能仅因为核心有空间矩阵就声称已提供任意 DAW 多通道布局。

预设库保存完整 JSON、声音与 UI 设置分开、UUID 文件名、校验后 temporary-file replacement；显式覆盖时检查库身份、磁盘内容指纹和符号链接。坏文件在 refresh 时报告，同时保留有效条目。不同宿主采样率仍须由实际 Processor 再验证；库预验证不是最终执行保证。

## 发现的具体问题及交给主线程的修复

### 宿主状态恢复不是完整事务

检查时 `setStateInformation` 先调用 `loadJSON(configuration)`，之后才读取 `mode`、`x_controls` 和其他字段。将 configuration 换成另一有效声音，但令 `mode` 为字符串或 `x_controls` 为对象，会在新声音已经发布后抛出类型错误并被空 catch 吞掉，留下混合状态。它也未检查 `state_schema`，未来 `aom.host/999` 会按当前格式加载。

建议：先解析、验证完整 wrapper 和可选旧格式，准备所有候选数据，再提交声音/参数/UI；未支持版本拒绝。新 `ProductSafetyTests.cpp` 覆盖 malformed mode、X controls、未来 schema、错误 XML 根和 APVTS 越域值的事务拒绝。主线程已实现 wrapper 预验证，接受 absent/host1/host2 并拒绝未来 schema；最终运行证据应以最新中央回归记录为准，本审计只更新源码状态。

### 非有限音频被清零后，峰值仍可能为 Infinity

检查时回调先将 dry 输入写入 inputPeak、将混合后的输出写入 outputPeak，再检查最终 sample 是否有限。bypass 或 mix=0 下的 `+inf` 输入最终被清零，但峰值已变成 Infinity。音频有限性和表头有限性是两条不同要求。

建议：在读入 native audio 边界处理非有限值，或在更新峰值前使用已清理值；失败计数保持可见。主线程已实现在 input/output 峰值更新前清理非有限值，尚待中央回归确认。新测试给左右声道 Infinity/NaN，再验证音频与 meters 有限，以及正常 dry audio 恢复。

### 带 runtimeModel 的普通参数编辑会重置正在演奏的状态

`TensorExecution::adopt` 要求新旧执行对象均不带 `model_`。Processor 只有 adopt 成功才保留 performance；因此即便 tensor shapes、clock、primitive、模型权重均未改变，带 runtimeModel 配置的一次普通参数编辑也会进入清空 MIDI performance 的路径。已有不带模型的 phase/held-note 连续性测试不能证明带模型版本具有同样行为。

建议：为兼容的同精度、同模型布局显式转移 DSP 和 neural recurrent state；模型结构/权重变化则使用明确的新模型加载语义。保留参数锁的权威。本轮训练暂停不意味着应忽略已可导入 runtimeModel 的产品行为；当前源码已实现同精度、同模型身份且兼容布局的 DSP/GRU state adoption；模型身份或结构不兼容时明确拒绝 adoption。最终连续性测试结果应以中央回归为准。

## 本轮新增产品回归

`src/tensor_plugin/tests/ProductSafetyTests.cpp` 新增：

- 完整宿主 wrapper 的 malformed/future schema 事务拒绝。
- Infinity/NaN 输入后的有限音频、有限仪表和 dry 恢复。
- mono/stereo/disabled input 接受与不匹配、多声道输出拒绝。
- 第一次音频块前连续发布 40 个已准备配置，实际执行最后一个配置。
- 带真实 runtimeModel 的两个相同宿主接收同一 held MIDI；其中一个编辑 base 参数，在后续 compiler events 后比较实际音频与 held-note 状态，以检查 DSP/GRU 递归状态转移。
- 通过实际 Editor 的 Tensor space → Compose fragments → Create primitive 打开嵌套 JUCE DialogWindow，销毁 Editor 后检查两层 SafePointer 失效，再重新打开/关闭组合弹窗。

这些测试已列入当前宿主测试构建；本次文档更新没有启动构建，也不新增通过数量。最终结果以最新中央运行记录为准。弹窗测试检查组件生命周期和消息循环，不能代替 OS 缩放、焦点、无障碍、实际 DAW 窗口管理与肉眼布局检查。

## 现有证据与仍需验收的内容

当前主界面的配置名称和 inspection 是只读呈现。独立 `SoundLibraryPanel` 可搜索、筛选、选择并查看完整配置，只有显式 Load 才调用声音发布；浏览或检查不更换正在播放的声音。其 preview 也是只读。直接控件调整、导入与加载是独立操作，不能用“浏览预设”概括成隐式加载。

当前注册表为 68 个数学操作：64 个 Universe 操作加 4 个基础操作。新增 production ring、流式 STFT 和显式 1x/2x/4x nonlinear 算法都有独立版本、Scalar state、资产与预算契约。稀疏 journal 按修改项分配；FFT 临时工作空间使用独立 `scratch_elements`，不冒充共享持久状态。数量增加不表示完整通用音频历史、所有旧 DSP parity 或所有配置的实时性能已经完成。详见 [production 契约](production-composition.md)。

已有测试涵盖真实 MIDI 多声部/踏板/释放/panic、原始参数编辑 phase 连续性、锁、自动化 X fanout、FX 增益/mix/bypass、状态保存/恢复、主题恢复、f32/f64 tier 切换、双精度 JSON、预设存储、组合 offset/feedback、可编辑 recipe 和 primitive/WAV 导入。每项最终状态应以中央测试结果为准，不能仅因测试源码存在而标为通过。

目前没有新的完整宿主回调 heap-allocation 捕捉证明；旧 inference 的 AllocationGuard 也没有自动覆盖这个新 Processor。仍需带实际 VST3 的有界 MIDI/重编译压力、多个实例、长时播放和实际宿主保存/关闭/重新打开工程验证。JSON/DSP 预算验证不是实测 CPU deadline 保证。

Creator 展示参数域与 repeated layout 描述，检查列出的必要参数数量；部分参考 primitive 对越域值采用保持/静音而不是统一报错。仅通过 `prepareExecution` 不能声称所有数学操作的所有参数组合都已验证可发声。生产 journal 操作有明确拒绝路径，参考操作和生产操作的错误语义仍需产品化一致。

WAV import 使用显式 source channel、不重采样、不混音、不改参数，并在解码前检查剩余 tensor capacity。JUCE reader 的 float32 解码边界在界面和 `asset_sources` 中公开；f64 存储不会恢复导入时已经丢失的源精度。用户手动编辑资产后，来源记录应视为来源历史，不能当作当前数组的完整校验保证。

组合 recipe 对整个 flattened sound 有指纹；直接 X/raw 编辑后不会静默重放旧 recipe，而需要明确将当前声音作为新 base。原始 fragment 保存在 recipe 中，因此序列化规模随组成内容增长；这还不是针对巨大工程的资产去重或压缩格式。

以上问题按已经存在的行为审计，不引入新的音色方向、tier 数值、语义宏映射或训练默认。这些源码修复和中央回归仍需实际宿主/长时资源验证；测试通过也不等于任意配置的 CPU deadline 保证。
