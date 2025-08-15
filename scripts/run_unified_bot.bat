@echo off
echo 🎯 Starting Unified Phemex Trading Bot
echo =======================================

cd /d "C:\Users\user\Desktop\NEW-PHEMEX-main"

if "%BITGET_API_KEY%"=="" (
    echo ❌ Missing BITGET_API_KEY environment variable!
    echo Please set your Bitget API credentials before running the bot.
    pause
    exit /b 1
)

echo 🚀 Starting unified bot with TUI, scanning, scoring, and trading...
echo 💰 Exchange: Bitget Futures (Swaps)
if "%LIVE_TRADE%"=="false" (
    echo 🎛️ Mode: PAPER TRADING
) else (
    echo 🎛️ Mode: LIVE TRADING
)
echo 📊 Press 'q' in TUI to quit
echo =======================================

python unified_trading_bot.py

if errorlevel 1 (
    echo ❌ Bot exited with error
    pause
)
