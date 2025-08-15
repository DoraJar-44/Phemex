# PowerShell script to run the unified trading bot

Write-Host "🎯 Starting Unified Phemex Trading Bot" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Set the working directory to the script's directory
Set-Location $PSScriptRoot

# Check if required environment variables are set
if (-not $env:BITGET_API_KEY) {
    Write-Host "❌ Missing BITGET_API_KEY environment variable!" -ForegroundColor Red
    Write-Host "Please set your Bitget API credentials before running the bot." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🚀 Starting unified bot with TUI, scanning, scoring, and trading..." -ForegroundColor Green
Write-Host "💰 Exchange: Bitget Futures (Swaps)" -ForegroundColor Yellow
Write-Host "🎛️ Mode: $(if ($env:LIVE_TRADE -eq 'false') { 'PAPER TRADING' } else { 'LIVE TRADING' })" -ForegroundColor $(if ($env:LIVE_TRADE -eq 'false') { 'Yellow' } else { 'Red' })
Write-Host "📊 Press 'q' in TUI to quit" -ForegroundColor Yellow
Write-Host "=======================================" -ForegroundColor Cyan

try {
    # Run the unified bot
    cd C:\Users\user\Desktop\NEW-PHEMEX-main
    python unified_trading_bot.py
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
