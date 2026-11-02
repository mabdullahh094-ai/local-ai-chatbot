@echo off
setlocal
cd /d "%~dp0"

set "OLLAMA_MODELS=D:\.ollama\models"
set "LLAMA_ARG_REPACK=0"

title Local AI Assistant and Realtime Voice Agent

echo [1/3] Checking Ollama service...
curl.exe -s --max-time 2 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama server in background...
    start "" /b "D:\Ollama\program\ollama.exe" serve
    timeout /t 2 /nobreak >nul
) else (
    echo Ollama is already running.
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Virtual environment nahi mili. Pehle setup check karein.
    pause
    exit /b 1
)

echo [2/3] Checking port 8501...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

echo [3/3] Starting Local AI Assistant and Voice Agent...
echo The app will open in your browser automatically at http://localhost:8501
echo Press CTRL+C to stop the server anytime.
echo.
".venv\Scripts\python.exe" server.py

if errorlevel 1 (
    echo.
    echo [ERROR] Chatbot server stop ho gaya ya koi error aaya.
    pause
)
