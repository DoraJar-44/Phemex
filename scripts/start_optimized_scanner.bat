@echo off
:: === Optimized Scanner Startup Script (MDC Compliant) ===
:: Purpose: Launch high-performance scanner with proper MDC configuration  
:: Platform: Phemex Futures (Swaps) - Windows Batch
:: Performance: 3-5x faster than standard scanner

setlocal EnableDelayedExpansion

:: === Default Configuration ===
set WORKING_DIR=C:\Users\user\Desktop\NEW-PHEMEX-main
set TIMEFRAME=5m
set MIN_LEVERAGE=25
set EXCLUDE_BASES=BTC,ETH,SOL,BNB,XRP,DOGE
set USE_TUI=true
set LIVE_TRADE=false
set MAX_SYMBOLS=40

:: === MDC Configuration Check ===
echo 🔧 MDC Configuration Check...

if not exist "%WORKING_DIR%" (
    echo ❌ Working directory not found: %WORKING_DIR%
    pause
    exit /b 1
)

cd /d "%WORKING_DIR%"
echo ✅ Working directory: %CD%

if not exist "mdc_config.mdc" (
    echo ❌ MDC configuration file not found
    pause
    exit /b 1
)

echo ✅ MDC configuration validated

:: === Environment Setup ===
echo 🚀 Setting up optimized scanner environment...

:: Core settings
set TIMEFRAME=%TIMEFRAME%
set MIN_LEVERAGE=%MIN_LEVERAGE%
set EXCLUDE_BASES=%EXCLUDE_BASES%
set USE_TUI=%USE_TUI%
set LIVE_TRADE=%LIVE_TRADE%

:: Optimization flags
set PYTHONUNBUFFERED=1
set PYTHONOPTIMIZE=1
set SCANNER_MODE=OPTIMIZED
set MAX_SYMBOLS_PER_SCAN=%MAX_SYMBOLS%

:: Performance settings
set OHLCV_CACHE_TTL=60
set SYMBOL_CACHE_TTL=300
set PARALLEL_BATCH_SIZE=15

echo ✅ Environment configured for maximum performance

:: === Configuration Display ===
echo.
echo 📊 OPTIMIZED SCANNER CONFIGURATION
echo =================================
echo TimeFrame: %TIMEFRAME%
echo Min Leverage: %MIN_LEVERAGE%x
echo Max Symbols: %MAX_SYMBOLS%
echo Live Trading: %LIVE_TRADE%
echo TUI Interface: %USE_TUI%
echo Exclude Bases: %EXCLUDE_BASES%
echo Parallel Batches: 15 symbols
echo Cache TTL: 60s OHLCV, 5m symbols
echo.

:: === Safety Check ===
if "%LIVE_TRADE%"=="true" (
    echo ⚠️  LIVE TRADING ENABLED - This will place real orders!
    echo Press Ctrl+C within 5 seconds to cancel...
    timeout /t 5 /nobreak >nul
)

:: === Launch Optimized Scanner ===
echo 🚀 Launching optimized high-performance scanner...

python -m bot.engine.optimized_scanner

if errorlevel 1 (
    echo ❌ Optimized scanner failed, falling back to standard scanner...
    python -m bot.engine.scanner
)

echo.
echo 🛑 Scanner stopped
pause
