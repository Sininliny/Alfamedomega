<#
.SYNOPSIS
Build Release from the current checkout and refresh the release folder (Windows).

.DESCRIPTION
    scripts\build-release.ps1              build, then replace release\ with the new output
    scripts\build-release.ps1 -BuildDir b  use another CMake build folder (default: build)

release\ ends up holding:
    Alfamedomega.vst3\      the VST3 bundle; copy it to %CommonProgramFiles%\VST3 to use it in a DAW
    Alfamedomega.exe        the standalone app
    AOM Centre\             alfa-cc-gui.exe (the window) and alfa-cc.exe (the command line)
    BUILD-INFO.txt          the commit and time this output was built from

The AOM Centre launchers start the control center from .venv in this checkout, so they always
run the current source, and they stop working if the checkout or .venv is moved.

The previous release\ is removed only after the build succeeds, so a failed build leaves it as
it was. Close the standalone app and the AOM Centre first if they were started from release\.
#>
[CmdletBinding()]
param(
    [string]$BuildDir = 'build'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo $BuildDir
$release = Join-Path $repo 'release'
$staging = Join-Path $build 'release-staging'
$venv = Join-Path $repo '.venv'

function Invoke-Native {
    # Windows PowerShell does not stop on a failing program, so check the exit code here.
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

# 1. Plugin and standalone app. The first configure downloads JUCE and ONNX Runtime.
#    CMAKE_BUILD_TYPE=Release is required by the ONNX Runtime dependency (docs/local-testing.md).
if (-not (Test-Path (Join-Path $build 'CMakeCache.txt'))) {
    Invoke-Native cmake @('-S', $repo, '-B', $build, '-A', 'x64',
        '-DCMAKE_BUILD_TYPE=Release', '-DALFA_BUILD_PLUGIN=ON')
}
Invoke-Native cmake @('--build', $build, '--config', 'Release', '--parallel',
    '--target', 'Alfamedomega_VST3', 'Alfamedomega_Standalone')

$artefacts = Join-Path $build 'src\plugin\Alfamedomega_artefacts\Release'
$vst3 = Join-Path $artefacts 'VST3\Alfamedomega.vst3'
$standalone = Join-Path $artefacts 'Standalone\Alfamedomega.exe'
foreach ($path in @($vst3, $standalone)) {
    if (-not (Test-Path $path)) { throw "The build did not produce $path" }
}

# 2. AOM Centre. An editable install follows the source, so it is only installed when missing.
$scripts = Join-Path $venv 'Scripts'
$centre = @('alfa-cc-gui.exe', 'alfa-cc.exe') | ForEach-Object { Join-Path $scripts $_ }
if (@($centre | Where-Object { -not (Test-Path $_) }).Count -gt 0) {
    if (-not (Test-Path (Join-Path $scripts 'python.exe'))) {
        Invoke-Native python @('-m', 'venv', $venv)
    }
    Invoke-Native (Join-Path $scripts 'python.exe') @('-m', 'pip', 'install', '-e',
        "$(Join-Path $repo 'apps\control-center')[gui]")
}

# 3. Stage everything, then swap it in for the previous release folder.
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $staging 'AOM Centre') | Out-Null
Copy-Item $vst3 (Join-Path $staging 'Alfamedomega.vst3') -Recurse
Copy-Item $standalone $staging
Copy-Item $centre (Join-Path $staging 'AOM Centre')

$commit = (& git -C $repo rev-parse --short HEAD)
$branch = (& git -C $repo rev-parse --abbrev-ref HEAD)
$dirty = if (& git -C $repo status --porcelain) { ' (with uncommitted changes)' } else { '' }
@(
    "Commit: $commit$dirty"
    "Branch: $branch"
    "Built:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
) | Set-Content (Join-Path $staging 'BUILD-INFO.txt') -Encoding ascii

if (Test-Path $release) {
    try {
        Remove-Item $release -Recurse -Force
    } catch {
        throw "Could not replace $release. Close anything running from it and try again. $_"
    }
}
Move-Item $staging $release

Write-Host ''
Write-Host "Release refreshed from $commit$dirty"
Write-Host "  VST3:        $(Join-Path $release 'Alfamedomega.vst3')"
Write-Host "  Standalone:  $(Join-Path $release 'Alfamedomega.exe')"
Write-Host "  AOM Centre:  $(Join-Path $release 'AOM Centre\alfa-cc-gui.exe')"
