$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

$env:OLLAMA_MODELS = "D:\.ollama\models"
$env:LLAMA_ARG_REPACK = "0"

Write-Host "[1/3] Checking Ollama service..." -ForegroundColor Cyan
try {
    $res = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "Ollama is already running." -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama server in background..." -ForegroundColor Yellow
    Start-Process -FilePath "D:\Ollama\program\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

Write-Host "[2/3] Checking port 8501..." -ForegroundColor Cyan
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}

Write-Host "[3/3] Starting Local AI Assistant..." -ForegroundColor Green
Write-Host "Browser will open automatically at http://localhost:8501" -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" server.py
