@echo off
REM Windows batch file to start the bot with auto-reload

echo 🎯 Starting Phemex Bot with Auto-Reload
echo ==========================================

REM Check if we should run in API mode
if "%1"=="api" (
    echo 🌐 Starting in API mode...
    set MODE=api
) else (
    echo 📊 Starting in Scanner mode...
    set MODE=scanner
)

REM Change to bot directory
cd /d "%~dp0"

REM Run with auto-reload
python run_with_reload.py

pause
