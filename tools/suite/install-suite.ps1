# Alfamedomega suite installer: plugin (VST3), standalone app and AMO Centre with the built-in
# WAV to Preset model. Installs for the current user into %LOCALAPPDATA%\Programs\Alfamedomega;
# only the VST3 copy into C:\Program Files\Common Files\VST3 asks for administrator rights.
# Test switches (not used by Install.bat): -Dest elsewhere, -SkipVst, -NoShortcuts; set
# ALFA_CC_HOME to keep AMO Centre settings out of the real profile.
param([string]$Dest = '', [switch]$SkipVst, [switch]$NoShortcuts)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$src = $PSScriptRoot
$dest = if ($Dest) { $Dest } else { Join-Path $env:LOCALAPPDATA 'Programs\Alfamedomega' }
$runtime = Join-Path $dest 'runtime'
$uv = Join-Path $runtime 'uv\uv.exe'
$venv = Join-Path $runtime 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
$app = Join-Path $dest 'app'
$vstRoot = Join-Path $env:CommonProgramFiles 'VST3'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:UV_CACHE_DIR = Join-Path $runtime 'cache'

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Check($code, $what) { if ($code -ne 0) { throw "$what failed (exit code $code)" } }
# Windows PowerShell 5.1 turns native stderr (uv's progress lines) into terminating errors
# under 'Stop' when output is redirected; run native tools with 'Continue' and use Check.
function Native { $ErrorActionPreference = 'Continue'; $exe, $rest = $args; & $exe @rest }
function Mirror($from, $to) {
    & robocopy $from $to /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying $from failed (robocopy $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
}

Step "Copying Alfamedomega to $dest"
New-Item -ItemType Directory -Force -Path $dest, $runtime | Out-Null
Mirror (Join-Path $src 'app') $app
Mirror (Join-Path $src 'plugin') (Join-Path $dest 'plugin')
Copy-Item -Force (Join-Path $src 'Alfamedomega.exe') $dest
Copy-Item -Force (Join-Path $src 'uninstall-suite.ps1'), (Join-Path $src 'README.txt') $dest

if (-not (Test-Path $uv)) {
    Step 'Downloading uv (Python installer, about 15 MB)'
    $zip = Join-Path $runtime 'uv.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.5.11/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    Expand-Archive -Force -Path $zip -DestinationPath (Join-Path $runtime 'uv')
    Remove-Item $zip
}
# A venv from an older setup may sit on a Python already on the machine (e.g. the Microsoft
# Store one, which silently redirects writes under %APPDATA%); rebuild it on the private Python.
$cfg = Join-Path $venv 'pyvenv.cfg'
if ((Test-Path $cfg) -and -not (Select-String -Quiet -SimpleMatch -Path $cfg -Pattern $env:UV_PYTHON_INSTALL_DIR)) {
    Remove-Item -Recurse -Force $venv
}
if (-not (Test-Path $python)) {
    Step 'Installing a private Python 3.12'
    Native $uv venv --python-preference only-managed --python 3.12 $venv
    Check $LASTEXITCODE 'Creating the Python environment'
}
Step 'Installing AMO Centre, PySide6, numpy and PyTorch (CPU, about 350 MB download; a few minutes)'
Native $uv pip install --python $python --reinstall-package alfamedomega-control-center 'numpy>=1.26' 'torch>=2.2' 'PySide6-Essentials>=6.6' (Join-Path $app 'control-center')
Check $LASTEXITCODE 'Installing the Python packages'

Step 'Pointing AMO Centre at the built-in model'
$env:PYTHONPATH = $app
Native $python -c "import sys; from alfacc.settings import default_home, load_settings, save_settings; h = default_home(); s = load_settings(h); s.repo_dir = sys.argv[1]; s.training_python = sys.argv[2]; s.extra.pop('wav_preset_model', None); save_settings(h, s); print('settings', h)" $app $python
Check $LASTEXITCODE 'Writing AMO Centre settings'
Native $python -c "import torch, numpy, PySide6; from training.amo.corpus import wav_to_preset; from training.amo.native import _library_path; print('torch', torch.__version__, '| native', _library_path())"
Check $LASTEXITCODE 'The installation check'
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $env:UV_CACHE_DIR

if ($SkipVst) { Write-Host 'Skipping the VST3 copy (-SkipVst)' } else {
Step "Installing the VST3 plugin into $vstRoot (Windows asks for administrator rights)"
$plugin = Join-Path $dest 'plugin\Alfamedomega.vst3'
$command = "New-Item -ItemType Directory -Force -Path '$vstRoot' | Out-Null; " +
           "if (Test-Path '$vstRoot\Alfamedomega.vst3') { Remove-Item -Recurse -Force '$vstRoot\Alfamedomega.vst3' }; " +
           "Copy-Item -Recurse '$plugin' '$vstRoot'"
$vstOk = $false
try {
    Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', $command
    $vstOk = Test-Path (Join-Path $vstRoot 'Alfamedomega.vst3')
} catch { }
if (-not $vstOk) {
    Write-Host "The VST3 was not copied. In your DAW, add this folder to its plugin search paths instead:" -ForegroundColor Yellow
    Write-Host "  $(Join-Path $dest 'plugin')" -ForegroundColor Yellow
}
}

if (-not $NoShortcuts) {
Step 'Creating shortcuts'
$shell = New-Object -ComObject WScript.Shell
$menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Alfamedomega'
New-Item -ItemType Directory -Force -Path $menu | Out-Null
function Shortcut($path, $target, $arguments, $workdir) {
    $s = $shell.CreateShortcut($path)
    $s.TargetPath = $target
    if ($arguments) { $s.Arguments = $arguments }
    $s.WorkingDirectory = $workdir
    $s.Save()
}
$centre = Join-Path $venv 'Scripts\alfa-cc-gui.exe'
$desktop = [Environment]::GetFolderPath('Desktop')
foreach ($folder in $menu, $desktop) {
    Shortcut (Join-Path $folder 'Alfamedomega.lnk') (Join-Path $dest 'Alfamedomega.exe') '' $dest
    Shortcut (Join-Path $folder 'AMO Centre.lnk') $centre '' $app
}
Shortcut (Join-Path $menu 'Uninstall Alfamedomega.lnk') 'powershell.exe' "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $dest 'uninstall-suite.ps1')`"" $dest
}

Write-Host ""
Write-Host 'Done. Desktop shortcuts: Alfamedomega (standalone) and AMO Centre.' -ForegroundColor Green
Write-Host 'In AMO Centre open "WAV to Preset" and drop one-shot samples; then press Refresh in the plugin''s sound library.' -ForegroundColor Green
