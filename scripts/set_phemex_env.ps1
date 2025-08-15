# Phemex API Credentials Setup Script
Write-Host "🔧 Setting up Phemex API credentials for live trading..." -ForegroundColor Yellow

# Set your Phemex API credentials here
$env:PHEMEX_API_KEY = "47a52259-6ee5-4096-9f26-fb206fefa4ea"
$env:PHEMEX_API_SECRET = "8u4nIrfP8C1z-7ioxzd_3k4S4iPE2Y5XiXv8ShfNTr4yODA4NTEyZi05YjBjLTRlYmItYmRiMy1lNDZiMTBhNzc0NTk"

# Live trading configuration
$env:LIVE_TRADE = "true"
$env:SYMBOLS = ""  # Empty = scan all pairs
$env:EXCLUDE_BASES = ""  # No exclusions
$env:MIN_LEVERAGE = "1"
$env:QUICK_RUN = "false"
$env:SCORE_MIN = "85"
$env:RISK_PER_TRADE_PCT = "0.5"
$env:MAX_POSITIONS = "5"

Write-Host "✅ Environment variables set for live trading with all pairs!" -ForegroundColor Green
Write-Host ""
Write-Host "Configuration:" -ForegroundColor White
Write-Host "  Exchange: Phemex" -ForegroundColor Gray
Write-Host "  LIVE_TRADE: $env:LIVE_TRADE" -ForegroundColor Gray
Write-Host "  SYMBOLS: $(if($env:SYMBOLS) {$env:SYMBOLS} else {'ALL PAIRS'})" -ForegroundColor Gray
Write-Host "  RISK_PER_TRADE: $env:RISK_PER_TRADE_PCT%" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: .\set_phemex_env.ps1"
Write-Host "2. Test: python setup_live_trading.py"
Write-Host "3. Start: python unified_trading_bot.py"
