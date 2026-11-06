<#
.SYNOPSIS
Build the Alfamedomega suite zip (Windows): VST3 + standalone + AMO Centre with WAV to Preset.

.DESCRIPTION
    scripts\package-suite.ps1               build with 2 jobs at below-normal priority, write release\Alfamedomega-Suite-<commit>.zip
    scripts\package-suite.ps1 -Jobs 4       more build jobs (louder fan)
    scripts\package-suite.ps1 -Test         also build and run alfa_tensor_host_tests first

The zip holds Install.bat (tools\suite), the plugin and standalone, and app\ with:
    training\amo\                          the Python sources WAV to Preset runs
    models\corpus-encoder\run3\model.pt    the built-in weights (AMO Centre's default model path)
    bin\alfa_tensor_c.dll                  our DSP runtime plus the VC++ runtime DLLs (app-local)
    control-center\                        the AMO Centre source, installed by Install.bat
The recipient's Install.bat downloads its own Python, PySide6 and CPU PyTorch.
#>
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-suite',
    [int]$Jobs = 2,
    [string]$Model = 'models\corpus-encoder\run3\model.pt',
    [switch]$Test
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo $BuildDir

function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    # Windows PowerShell 5.1 turns native stderr (CMake warnings) into terminating errors; use the exit code.
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}
function Mirror($from, $to, [string[]]$extra = @()) {
    & robocopy $from $to /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache build dist *.egg-info @extra /XF *.pyc | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $from failed with $LASTEXITCODE" }
    $global:LASTEXITCODE = 0
}

try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }

if (-not (Test-Path (Join-Path $build 'CMakeCache.txt'))) {
    $tests = if ($Test) { 'ON' } else { 'OFF' }
    Invoke-Native cmake @('-S', $repo, '-B', $build, '-A', 'x64', '-DCMAKE_BUILD_TYPE=Release',
        '-DALFA_BUILD_PLUGIN=ON', "-DALFA_BUILD_TESTS=$tests")
}
$targets = @('Alfamedomega_VST3', 'Alfamedomega_Standalone', 'alfa_tensor_c')
if ($Test) { $targets += 'alfa_tensor_host_tests' }
Invoke-Native cmake (@('--build', $build, '--config', 'Release', '--parallel', "$Jobs", '--target') + $targets)
if ($Test) {
    $hostTests = Get-ChildItem -Recurse -Path $build -Filter 'alfa_tensor_host_tests.exe' | Select-Object -First 1
    if (-not $hostTests) { throw 'alfa_tensor_host_tests.exe not found' }
    Invoke-Native $hostTests.FullName @()
}

$artefacts = Join-Path $build 'src\plugin\Alfamedomega_artefacts\Release'
$vst3 = Join-Path $artefacts 'VST3\Alfamedomega.vst3'
$standalone = Join-Path $artefacts 'Standalone\Alfamedomega.exe'
$dll = Join-Path $build 'src\tensor\Release\alfa_tensor_c.dll'
$modelPath = Join-Path $repo $Model
foreach ($path in $vst3, $standalone, $dll, $modelPath) {
    if (-not (Test-Path $path)) { throw "Missing $path" }
}

$commit = (& git -C $repo rev-parse --short HEAD).Trim()
$stage = Join-Path $build 'Alfamedomega-Suite'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$app = Join-Path $stage 'app'
New-Item -ItemType Directory -Force -Path (Join-Path $stage 'plugin'), (Join-Path $app 'bin'),
    (Join-Path $app 'training'), (Split-Path -Parent (Join-Path $app $Model)) | Out-Null

Copy-Item (Join-Path $repo 'tools\suite\*') $stage
Copy-Item -Recurse $vst3 (Join-Path $stage 'plugin')
Copy-Item $standalone $stage
Get-ChildItem (Split-Path -Parent $standalone) -Filter *.dll | Copy-Item -Destination $stage
Copy-Item $modelPath (Join-Path $app $Model)
Get-ChildItem (Split-Path -Parent $modelPath) -Filter *.md | Copy-Item -Destination (Split-Path -Parent (Join-Path $app $Model))
Copy-Item $dll (Join-Path $app 'bin')
foreach ($name in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    $system = Join-Path $env:SystemRoot "System32\$name"
    if (Test-Path $system) { Copy-Item $system (Join-Path $app 'bin') }
}
Mirror (Join-Path $repo 'training\amo') (Join-Path $app 'training\amo') @('reports', 'kaggle', 'testdata')
Mirror (Join-Path $repo 'apps\control-center') (Join-Path $app 'control-center') @('tests')
Set-Content -Path (Join-Path $stage 'BUILD-INFO.txt') -Value "commit $commit`nbuilt $(Get-Date -Format s)" -Encoding UTF8

$releaseDir = Join-Path $repo 'release'
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$zip = Join-Path $releaseDir "Alfamedomega-Suite-$commit.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "wrote $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"
