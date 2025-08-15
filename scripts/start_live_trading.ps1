# Start Live Trading with Phemex - All Pairs Scanning
Write-Host "🚀 Starting Phemex Live Trading Bot - All Pairs Mode" -ForegroundColor Green

# Set environment variables
$env:PHEMEX_API_KEY = "47a52259-6ee5-4096-9f26-fb206fefa4ea"
$env:PHEMEX_API_SECRET = "8u4nIrfP8C1z-7ioxzd_3k4S4iPE2Y5XiXv8ShfNTr4yODA4NTEyZi05YjBjLTRlYmItYmRiMy1lNDZiMTBhNzc0NTk"
$env:LIVE_TRADE = "true"
$env:SYMBOLS = ""
$env:EXCLUDE_BASES = ""
$env:MIN_LEVERAGE = "1"
$env:QUICK_RUN = "false"

Write-Host "✅ Environment configured for live trading with all pairs" -ForegroundColor Green
Write-Host "Exchange: Phemex" -ForegroundColor Cyan
Write-Host "Mode: Live Trading" -ForegroundColor Cyan
Write-Host "Pairs: ALL AVAILABLE" -ForegroundColor Cyan
Write-Host ""

# Start the bot
Write-Host "Starting bot..." -ForegroundColor Yellow
python unified_trading_bot.py
