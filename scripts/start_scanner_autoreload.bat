@echo off
REM Batch script to start the scanner with auto-reload (no execution policy issues)

cd /d "%~dp0"

echo ============================================================
echo Starting Phemex Scanner with INSTANT AUTO-RELOAD...
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo Python detected successfully

REM Set environment variables for testing
set LIVE_TRADE=false
set USE_TUI=true
set PHEMEX_API_KEY=test
set PHEMEX_API_SECRET=test

echo.
echo INSTANT AUTO-RELOAD ENABLED!
echo The bot will restart automatically when you:
echo    - Edit ANY .py file
echo    - Modify .env files
echo    - Change .mdc or .json config files
echo    - Even just add a space and save!
echo.
echo Restart delay: 0.5 seconds (super fast!)
echo Press Ctrl+C to stop the auto-reloader
echo ============================================================

REM Start the auto-reload system
python auto_reload.py --mode scanner

if errorlevel 1 (
    echo Failed to start auto-reload scanner
    pause
    exit /b 1
)
