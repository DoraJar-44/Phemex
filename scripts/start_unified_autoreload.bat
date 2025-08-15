@echo off
REM Batch script to start the UNIFIED bot with auto-reload (no execution policy issues)

cd /d "%~dp0"

echo ============================================================
echo Starting UNIFIED Phemex Trading Bot with INSTANT AUTO-RELOAD...
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo Python detected successfully

REM Set environment variables for testing (Bitget as per workspace rules)
set LIVE_TRADE=false
set USE_TUI=true
set BITGET_API_KEY=test
set BITGET_API_SECRET=test
set BITGET_API_PASSWORD=test

echo.
echo Exchange: Bitget Futures (as per workspace rules)
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

REM Start the auto-reload system for unified bot
python auto_reload.py --mode unified

if errorlevel 1 (
    echo Failed to start auto-reload unified bot
    pause
    exit /b 1
)
