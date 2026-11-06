Alfamedomega 套件（插件 + 独立版 + AMO Centre）
================================================

安装（Windows 10/11，64 位）
1. 把整个压缩包解压到任意文件夹（不要直接在压缩包里双击）。
2. 双击 Install.bat。第一次安装会下载 Python、PyTorch 等，约 400 MB，需要几分钟。
   中途 Windows 会询问管理员权限：这是把 VST3 插件复制到
   C:\Program Files\Common Files\VST3。点"是"。
3. 看到 "Done." 就装好了。解压出来的文件夹之后可以删掉。

装好之后
- 桌面上有两个图标：
  Alfamedomega   独立版合成器（不用宿主就能弹，可在设置里选声卡和 MIDI 键盘）
  AMO Centre     控制中心
- 在宿主（FL Studio、Ableton、Reaper 等）里重新扫描插件，就能找到 Alfamedomega（VST3）。

把采样变成音色：WAV to Preset
1. 打开 AMO Centre，左边第一项就是 "WAV to Preset"，模型已经内置好了。
2. 把 .wav 采样拖进虚线框（也可以点框选择文件）。单音的 one-shot 效果最好：
   一个音符、开头干净、没有混响和别的乐器。
3. 音高默认自动识别；识别不准时，在 "Sample pitch" 里手动选采样的音高。
4. 转换完成后，打开插件（或独立版）的音色库，点刷新按钮，新音色在 "WAV import" 分类里。
   音色文件保存在 %APPDATA%\Alfamedomega\TensorPresets。

须知
- 模型做的是"用我们自己的合成结构，找最接近这个采样的那一个"，不是采样回放。
  合成器能做的声音越接近采样，结果越像；人声、整段音乐、复杂真实乐器只会得到大致相近的音色。
- 转换在 CPU 上跑，每个采样十几秒到一分钟，期间只用两个线程，电脑不会太吵。
- 卸载：开始菜单 → Alfamedomega → Uninstall Alfamedomega。你做的音色会保留。

遇到问题请把窗口截图发给 Sinin。
