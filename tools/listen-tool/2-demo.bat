@echo off
cd /d "%~dp0"
if not exist "runtime\venv\Scripts\python.exe" (
  echo Please double-click 1-install.bat first.
  pause
  exit /b 1
)
set "AMO_TENSOR_LIBRARY=%~dp0bin\alfa_tensor_c.dll"
set "PYTHONPATH=%~dp0app"
set "OMP_NUM_THREADS=2"
echo Making 8 random sounds and letting the model guess how each one is built...
start "" /belownormal /wait /b "runtime\venv\Scripts\python.exe" -m training.amo.corpus.resynth_wav --threads 2 demo "model\model.pt" "results\demo" --count 8
start "" explorer "%~dp0results\demo"
pause
