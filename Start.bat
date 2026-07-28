@echo off
title APP INSTALLER
cd /d "%~dp0"

:: 1. Checking if the Python folder exists
if not exist "python_embed\python.exe" (
    echo [INFO] First run detected. Install python...
    echo.
    powershell.exe -ExecutionPolicy Bypass -NoProfile -File "build_portable.ps1"
)

:: 2. Pre-compile the code
"python_embed\python.exe" -m compileall -q "app"

:: 3. Run application with CMD minimized in the taskbar
start /min "" cmd /c "python_embed\python.exe app\detector_app.py --data data.yaml > antrenare_log.txt 2>&1"