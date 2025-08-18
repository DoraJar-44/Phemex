#!/usr/bin/env python3
"""
NEXT STEPS: Ready for Live Trading Activation
"""

import os
import asyncio
from datetime import datetime

def print_banner():
    print("🚀" * 20)
    print("🔥 AGGRESSIVE 50X LIVE TRADING SYSTEM")
    print("🚀" * 20)
    print("⚡ STATUS: READY TO ACTIVATE")
    print("⚡ WAITING FOR: ACCOUNT FUNDING")
    print("🚀" * 20)

def show_funding_requirements():
    print("\n💰 FUNDING REQUIREMENTS:")
    print("=" * 50)
    print("❌ Current Balance: $0.00 USDT")
    print("✅ Minimum Required: $10 USDT")
    print("🎯 Recommended Amounts:")
    print("   • Conservative: $50 USDT")
    print("   • Standard: $100 USDT") 
    print("   • Aggressive: $500+ USDT")

def show_system_specs():
    print("\n⚙️ AGGRESSIVE SYSTEM SPECIFICATIONS:")
    print("=" * 50)
    print("🔥 Max Leverage: 50x")
    print("🎯 Score Threshold: 65 (LOWERED for more trades)")
    print("💰 Risk per Trade: 1.0% (INCREASED for bigger positions)")
    print("📊 Max Positions: 8 (INCREASED for more opportunities)")
    print("⚡ Scan Frequency: Every 15 seconds (FASTER)")
    print("🎲 Min Trade Size: $5 USDT")
    print("🛡️ Auto Stop-Loss: 2% (liquidation protection)")

def show_trading_pairs():
    print("\n🎯 50X LEVERAGE PAIRS READY:")
    print("=" * 50)
    pairs = [
        "BTC/USDT:USDT (100x)", "ETH/USDT:USDT (100x)", 
        "SOL/USDT:USDT (50x)", "ADA/USDT:USDT (50x)",
        "DOT/USDT:USDT (50x)", "LINK/USDT:USDT (50x)",
        "ATOM/USDT:USDT (50x)", "AVAX/USDT:USDT (50x)",
        "UNI/USDT:USDT (50x)", "DOGE/USDT:USDT (50x)"
    ]
    
    for i, pair in enumerate(pairs, 1):
        print(f"{i:2d}. {pair}")
    print("    ...and 282 more pairs")

def show_activation_steps():
    print("\n🚀 ACTIVATION STEPS:")
    print("=" * 50)
    print("1️⃣ DEPOSIT FUNDS:")
    print("   • Login to Phemex → Assets → Deposit")
    print("   • Deposit USDT (minimum $10, recommended $50+)")
    print("   • Ensure funds are in 'Futures' wallet")
    
    print("\n2️⃣ ACTIVATE LIVE TRADING:")
    print("   cd /workspace && python3 aggressive_live_trading.py")
    
    print("\n3️⃣ MONITOR SYSTEM:")
    print("   • View live logs: tail -f /workspace/aggressive_trading.log")
    print("   • Check positions: python3 live_trading_status.py")
    print("   • Stop system: pkill -f aggressive_live_trading.py")

def show_demo_mode():
    print("\n🎮 DEMO MODE AVAILABLE:")
    print("=" * 50)
    print("Want to see how it works before funding?")
    print("Run: cd /workspace && LIVE_TRADE=false python3 aggressive_live_trading.py")
    print("This will show paper trades without using real money")

def show_risk_warning():
    print("\n⚠️  RISK DISCLAIMER:")
    print("=" * 50)
    print("🔥 50x leverage trading is EXTREMELY HIGH RISK")
    print("💸 You can lose your entire deposit quickly")
    print("🎯 Only trade with money you can afford to lose")
    print("🛡️ System has stop-losses but crypto moves fast")
    print("📚 Consider starting with smaller amounts first")

def show_current_market():
    print(f"\n📊 CURRENT TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔍 System is ready to scan these opportunities:")
    print("   • Momentum breakouts")
    print("   • Volume surges") 
    print("   • Volatility spikes")
    print("   • Technical pattern confirmations")

async def main():
    print_banner()
    show_funding_requirements()
    show_system_specs()
    show_trading_pairs()
    show_activation_steps()
    show_demo_mode()
    show_current_market()
    show_risk_warning()
    
    print("\n" + "🔥" * 60)
    print("🎯 SYSTEM STATUS: READY FOR ACTIVATION")
    print("⚡ NEXT ACTION: DEPOSIT FUNDS TO PHEMEX ACCOUNT")
    print("🚀 THEN RUN: python3 aggressive_live_trading.py")
    print("🔥" * 60)

if __name__ == "__main__":
    asyncio.run(main())