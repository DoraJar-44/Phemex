# PowerShell script to start the UNIFIED bot with auto-reload for instant development

# Change to script directory
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "Starting UNIFIED Phemex Trading Bot with INSTANT AUTO-RELOAD..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found! Please install Python 3.8+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set environment variables for testing (Bitget as per workspace rules)
$env:LIVE_TRADE = "false"
$env:USE_TUI = "true"
$env:BITGET_API_KEY = "test"
$env:BITGET_API_SECRET = "test"
$env:BITGET_API_PASSWORD = "test"

Write-Host "Exchange: Bitget Futures (as per workspace rules)" -ForegroundColor Blue
Write-Host "INSTANT AUTO-RELOAD ENABLED!" -ForegroundColor Magenta
Write-Host "The bot will restart automatically when you:" -ForegroundColor Green
Write-Host "   - Edit ANY .py file" -ForegroundColor White
Write-Host "   - Modify .env files" -ForegroundColor White  
Write-Host "   - Change .mdc or .json config files" -ForegroundColor White
Write-Host "   - Even just add a space and save!" -ForegroundColor White
Write-Host "" 
Write-Host "Restart delay: 0.5 seconds (super fast!)" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the auto-reloader" -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Yellow

try {
    # Use the enhanced auto-reload system for unified bot
    python auto_reload.py --mode unified
} catch {
    Write-Host "Failed to start auto-reload unified bot: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}