# Установка зависимостей из requirements.txt
# Запуск: .\install_deps.ps1  или  pwsh -File install_deps.ps1
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
python -m pip install -r requirements.txt
