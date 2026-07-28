if (Test-Path "python_embed") { Remove-Item "python_embed" -Recurse -Force }
if (Test-Path "python") { Remove-Item "python" -Recurse -Force }
if (Test-Path "uv_tool") { Remove-Item "uv_tool" -Recurse -Force }

Write-Host "[1/4] Download Python 3.12 Standalone..." -ForegroundColor Green
$url = "https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.12.7+20241016-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
Invoke-WebRequest -Uri $url -OutFile "python_standalone.tar.gz"

Write-Host "[2/4] Dezarhivez pachetul Python..." -ForegroundColor Green
tar -xzf "python_standalone.tar.gz"
Rename-Item -Path "python" -NewName "python_embed"
Remove-Item "python_standalone.tar.gz" -Force

Write-Host "[3/4] Download UV..." -ForegroundColor Green
$uv_url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
Invoke-WebRequest -Uri $uv_url -OutFile "uv.zip"
Expand-Archive -Path "uv.zip" -DestinationPath "uv_tool" -Force
Remove-Item "uv.zip" -Force

Write-Host "[4/4] Installing dependencies from pyproject.toml using UV..." -ForegroundColor Green
.\uv_tool\uv.exe pip install -r pyproject.toml --target "python_embed\Lib\site-packages" --python "python_embed\python.exe"
Remove-Item "uv_tool" -Recurse -Force