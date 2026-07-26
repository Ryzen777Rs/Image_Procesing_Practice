@echo off
title Lansator Aplicatie
cd /d "%~dp0"

:: 1. Verificam daca folderul cu Python exista deja
if not exist "python_embed\python.exe" (
    echo [INFO] Prima rulare detectata! Se instaleaza mediul portabil...
    echo.
    powershell.exe -ExecutionPolicy Bypass -NoProfile -File "build_portable.ps1"
)

:: 2. Pornim aplicatia direct si inchidem consola neagra
start "" "python_embed\pythonw.exe" "app\detector_app.py"