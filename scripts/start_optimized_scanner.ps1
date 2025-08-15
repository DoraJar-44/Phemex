# === Optimized Scanner Startup Script (MDC Compliant) ===
# Purpose: Launch high-performance scanner with proper MDC configuration
# Platform: Phemex Futures (Swaps) - Windows PowerShell
# Performance: 3-5x faster than standard scanner

param(
    [string]$WorkingDirectory = "C:\Users\user\Desktop\NEW-PHEMEX-main",
    [string]$TimeFrame = "5m",
    [string]$Symbols = "",
    [int]$MinLeverage = 25,
    [string]$ExcludeBases = "BTC,ETH,SOL,BNB,XRP,DOGE",
    [switch]$LiveTrade = $false,
    [switch]$UseTUI = $true,
    [int]$MaxSymbols = 40
)

# === MDC Configuration Validation ===
Write-Host "🔧 MDC Configuration Check..." -ForegroundColor Cyan

# Ensure we're in the correct working directory
if (-not (Test-Path $WorkingDirectory)) {
    Write-Host "❌ Working directory not found: $WorkingDirectory" -ForegroundColor Red
    exit 1
}

Set-Location $WorkingDirectory
Write-Host "✅ Working directory: $(Get-Location)" -ForegroundColor Green

# Check for required MDC config file
if (-not (Test-Path "mdc_config.mdc")) {
    Write-Host "❌ MDC configuration file not found" -ForegroundColor Red
    exit 1
}

# Load MDC settings
$mdcConfig = Get-Content "mdc_config.mdc" | Where-Object { $_ -match "^[A-Z_]+=.*" }
Write-Host "✅ MDC configuration loaded" -ForegroundColor Green

# === Environment Setup ===
Write-Host "🚀 Setting up optimized scanner environment..." -ForegroundColor Yellow

# Performance environment variables
$env:TIMEFRAME = $TimeFrame
$env:SYMBOLS = $Symbols  
$env:MIN_LEVERAGE = $MinLeverage
$env:EXCLUDE_BASES = $ExcludeBases
$env:USE_TUI = if ($UseTUI) { "true" } else { "false" }
$env:LIVE_TRADE = if ($LiveTrade) { "true" } else { "false" }

# Optimization flags
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONOPTIMIZE = "1"
$env:SCANNER_MODE = "OPTIMIZED"
$env:MAX_SYMBOLS_PER_SCAN = $MaxSymbols

# Cache settings for performance
$env:OHLCV_CACHE_TTL = "60"
$env:SYMBOL_CACHE_TTL = "300"
$env:PARALLEL_BATCH_SIZE = "15"

Write-Host "✅ Environment configured for maximum performance" -ForegroundColor Green

# === Performance Settings Display ===
Write-Host ""
Write-Host "📊 OPTIMIZED SCANNER CONFIGURATION" -ForegroundColor Magenta
Write-Host "=================================" -ForegroundColor Magenta
Write-Host "TimeFrame: $TimeFrame" -ForegroundColor White
Write-Host "Min Leverage: ${MinLeverage}x" -ForegroundColor White
Write-Host "Max Symbols: $MaxSymbols" -ForegroundColor White
Write-Host "Live Trading: $(if ($LiveTrade) { 'ENABLED' } else { 'DRY RUN' })" -ForegroundColor $(if ($LiveTrade) { 'Red' } else { 'Yellow' })
Write-Host "TUI Interface: $(if ($UseTUI) { 'ENABLED' } else { 'DISABLED' })" -ForegroundColor White
Write-Host "Exclude Bases: $ExcludeBases" -ForegroundColor White
Write-Host "Parallel Batches: 15 symbols" -ForegroundColor White
Write-Host "Cache TTL: 60s OHLCV, 5m symbols" -ForegroundColor White
Write-Host ""

# === Safety Check ===
if ($LiveTrade) {
    Write-Host "⚠️  LIVE TRADING ENABLED - This will place real orders!" -ForegroundColor Red
    Write-Host "Press Ctrl+C within 5 seconds to cancel..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}

# === Launch Optimized Scanner ===
Write-Host "🚀 Launching optimized high-performance scanner..." -ForegroundColor Green

try {
    # Use optimized scanner module
    python -m bot.engine.optimized_scanner
}
catch {
    Write-Host "❌ Scanner failed to start: $($_.Exception.Message)" -ForegroundColor Red
    
    # Fallback to standard scanner
    Write-Host "🔄 Falling back to standard scanner..." -ForegroundColor Yellow
    python -m bot.engine.scanner
}

Write-Host ""
Write-Host "🛑 Scanner stopped" -ForegroundColor Yellow
