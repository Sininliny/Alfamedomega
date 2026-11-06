<#
.SYNOPSIS
Build the AMO listening tool zip (Windows): structure-encoder weights + our DSP + one-click setup.

.DESCRIPTION
    scripts\package-listen-tool.ps1              build alfa_tensor_c with 2 jobs, then write release\AMO-Listen-<commit>.zip
    scripts\package-listen-tool.ps1 -Jobs 4      more build jobs (louder fan)

Only the native tensor library is compiled (no JUCE, no plugin). The zip holds:
    1-install.bat / 2-demo.bat / 3-resynth-my-wav.bat / install.ps1 / README.txt   (tools\listen-tool)
    app\training\amo\        the Python sources the tool needs
    model\model.pt           models\corpus-encoder\run3\model.pt
    bin\alfa_tensor_c.dll    plus the VC++ runtime DLLs it needs (app-local deployment)
The recipient's setup downloads its own Python and CPU PyTorch into the unpacked folder.
#>
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-listen-tool',
    [int]$Jobs = 2,
    [string]$Model = 'models\corpus-encoder\run3\model.pt'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo $BuildDir

function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    # Windows PowerShell 5.1 turns native stderr (e.g. CMake warnings) into terminating
    # errors under 'Stop' when output is redirected; rely on the exit code instead.
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

# Build at below-normal priority so the machine stays quiet and responsive.
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }

if (-not (Test-Path (Join-Path $build 'CMakeCache.txt'))) {
    Invoke-Native cmake @('-S', $repo, '-B', $build, '-A', 'x64', '-DCMAKE_BUILD_TYPE=Release',
        '-DALFA_BUILD_PLUGIN=OFF', '-DALFA_BUILD_TESTS=OFF')
}
Invoke-Native cmake @('--build', $build, '--config', 'Release', '--parallel', "$Jobs", '--target', 'alfa_tensor_c')
$dll = Join-Path $build 'src\tensor\Release\alfa_tensor_c.dll'
if (-not (Test-Path $dll)) { throw "The build did not produce $dll" }
$modelPath = Join-Path $repo $Model
if (-not (Test-Path $modelPath)) { throw "Model not found: $modelPath" }

$commit = (& git -C $repo rev-parse --short HEAD).Trim()
$stage = Join-Path $build 'AMO-Listen'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path (Join-Path $stage 'bin'), (Join-Path $stage 'model'), (Join-Path $stage 'app\training') | Out-Null

Copy-Item (Join-Path $repo 'tools\listen-tool\*') $stage
Copy-Item $modelPath (Join-Path $stage 'model\model.pt')
Copy-Item $dll (Join-Path $stage 'bin')
foreach ($name in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    $system = Join-Path $env:SystemRoot "System32\$name"
    if (Test-Path $system) { Copy-Item $system (Join-Path $stage 'bin') }
}
$source = Join-Path $repo 'training\amo'
$target = Join-Path $stage 'app\training\amo'
# robocopy exit codes below 8 mean success.
& robocopy $source $target /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ reports kaggle testdata /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with $LASTEXITCODE" }
$global:LASTEXITCODE = 0
Set-Content -Path (Join-Path $stage 'BUILD-INFO.txt') -Value "commit $commit`nbuilt $(Get-Date -Format s)" -Encoding UTF8

$releaseDir = Join-Path $repo 'release'
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$zip = Join-Path $releaseDir "AMO-Listen-$commit.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "wrote $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"
