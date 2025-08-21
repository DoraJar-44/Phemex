#!/bin/bash
# Professional Bounce Trading Bot Startup Script
# MDC Compliant - Always includes cd command as required
# Platform: Bitget Futures (Swaps)

echo "🚀 Starting Professional Bounce Trading Bot"
echo "📊 Smart Money Concepts + Predictive Ranges Strategy"

# MDC Compliance: Always cd to workspace first
cd /workspace

# Check environment variables
if [ -z "$BITGET_API_KEY" ] && [ -z "$PHEMEX_API_KEY" ]; then
    echo "❌ Error: No API credentials found"
    echo "Please set BITGET_API_KEY and BITGET_SECRET for Bitget"
    echo "Or set PHEMEX_API_KEY and PHEMEX_SECRET as fallback"
    exit 1
fi

# Set default values following MDC config
export SYMBOL=${SYMBOL:-"BTC/USDT:USDT"}
export TIMEFRAME=${TIMEFRAME:-"4h"}
export LEVERAGE=${LEVERAGE:-25}  # User's preferred leverage - DO NOT CHANGE
export MODE="professional_bounce"

echo "⚙️  Configuration:"
echo "   Symbol: $SYMBOL"
echo "   Timeframe: $TIMEFRAME"  
echo "   Leverage: ${LEVERAGE}x"
echo "   Platform: Bitget Futures (Swaps)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the professional bounce bot
echo "🔥 Launching Professional Bounce Strategy..."
python3 professional_bounce_live.py

echo "👋 Professional Bounce Bot stopped"