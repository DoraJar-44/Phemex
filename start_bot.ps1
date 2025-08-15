# PowerShell script to start the bot with auto-reload

param(
    [string]$Mode = "scanner"
)

Write-Host "🎯 Starting Phemex Bot with Auto-Reload" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Set the working directory to the script's directory
Set-Location $PSScriptRoot

# Set environment variable
$env:MODE = $Mode.ToLower()

if ($env:MODE -eq "api") {
    Write-Host "🌐 Starting in API mode..." -ForegroundColor Green
} else {
    Write-Host "📊 Starting in Scanner mode..." -ForegroundColor Green
}

Write-Host "🔄 Auto-restart enabled for code changes" -ForegroundColor Yellow
Write-Host "⏹️ Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

try {
    # Run with auto-reload
    python run_with_reload.py
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
