# Removes the Alfamedomega suite. Presets you made stay in %APPDATA%\Alfamedomega\TensorPresets.
$ErrorActionPreference = 'Continue'
$dest = Join-Path $env:LOCALAPPDATA 'Programs\Alfamedomega'
$vst = Join-Path $env:CommonProgramFiles 'VST3\Alfamedomega.vst3'
if (Test-Path $vst) {
    try { Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', "Remove-Item -Recurse -Force '$vst'" } catch { }
}
Remove-Item -Recurse -Force (Join-Path ([Environment]::GetFolderPath('Programs')) 'Alfamedomega') -ErrorAction SilentlyContinue
$desktop = [Environment]::GetFolderPath('Desktop')
Remove-Item -Force (Join-Path $desktop 'Alfamedomega.lnk'), (Join-Path $desktop 'AMO Centre.lnk') -ErrorAction SilentlyContinue
Set-Location $env:TEMP
Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue
Write-Host 'Alfamedomega removed. Your presets are still in %APPDATA%\Alfamedomega\TensorPresets.'
Read-Host 'Press Enter to close'
