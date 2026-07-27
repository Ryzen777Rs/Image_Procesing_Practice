@echo off
title Lansator Aplicatie
cd /d "%~dp0"

:: 1. Verificam daca folderul cu Python exista
if not exist "python_embed\python.exe" (
    echo [INFO] Prima rulare detectata! Se instaleaza mediul portabil...
    echo.
    powershell.exe -ExecutionPolicy Bypass -NoProfile -File "build_portable.ps1"
)

:: 2. Pre-compilam codul din folderul app (creeaza fisiere .pyc optimizate)
"python_embed\python.exe" -m compileall -q "app"

:: 3. Pornim aplicatia
 "python_embed\python.exe" "app\detector_app.py"