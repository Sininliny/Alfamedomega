# AMO listening tool: one-time setup. Everything goes into .\runtime next to this script;
# nothing is installed system-wide. Delete the whole folder to uninstall.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
$runtime = Join-Path $root 'runtime'
$uvDir = Join-Path $runtime 'uv'
$uv = Join-Path $uvDir 'uv.exe'
$venv = Join-Path $runtime 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:UV_CACHE_DIR = Join-Path $runtime 'cache'

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Check($code, $what) { if ($code -ne 0) { throw "$what failed (exit code $code)" } }

New-Item -ItemType Directory -Force -Path $runtime | Out-Null

if (-not (Test-Path $uv)) {
    Step 'Downloading uv (Python installer, about 15 MB)'
    $zip = Join-Path $runtime 'uv.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.5.11/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    Expand-Archive -Force -Path $zip -DestinationPath $uvDir
    Remove-Item $zip
}

if (-not (Test-Path $python)) {
    Step 'Installing a private Python 3.12'
    & $uv venv --python 3.12 $venv
    Check $LASTEXITCODE 'Creating the Python environment'
}

Step 'Installing numpy and PyTorch (CPU, about 250 MB download; this takes a few minutes)'
& $uv pip install --python $python 'numpy>=1.26' 'torch>=2.2'
Check $LASTEXITCODE 'Installing numpy and torch'

Step 'Checking the installation'
$env:PYTHONPATH = Join-Path $root 'app'
$env:AMO_TENSOR_LIBRARY = Join-Path $root 'bin\alfa_tensor_c.dll'
& $python -c "import torch, numpy; from training.amo.corpus import resynth_wav; from training.amo.native import NativeRuntime; print('torch', torch.__version__, 'numpy', numpy.__version__)"
Check $LASTEXITCODE 'The installation check'

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $env:UV_CACHE_DIR
Write-Host ""
Write-Host 'Done. Double-click 2-demo.bat to try it.' -ForegroundColor Green
