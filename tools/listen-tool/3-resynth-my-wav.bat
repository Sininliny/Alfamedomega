@echo off
cd /d "%~dp0"
if not exist "runtime\venv\Scripts\python.exe" (
  echo Please double-click 1-install.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag one or more WAV files onto this file to use it.
  pause
  exit /b 1
)
set "AMO_TENSOR_LIBRARY=%~dp0bin\alfa_tensor_c.dll"
set "PYTHONPATH=%~dp0app"
set "OMP_NUM_THREADS=2"
start "" /belownormal /wait /b "runtime\venv\Scripts\python.exe" -m training.amo.corpus.resynth_wav --threads 2 wav "model\model.pt" "results\my-wav" %*
start "" explorer "%~dp0results\my-wav"
pause
