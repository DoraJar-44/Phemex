#!/usr/bin/env python3
"""
Live Trading Status Dashboard
"""

import asyncio
import json
import os
from dotenv import load_dotenv
import ccxt.async_support as ccxt

load_dotenv()

async def check_live_status():
    print("🚀 LIVE TRADING SYSTEM STATUS DASHBOARD")
    print("=" * 60)
    
    # Check if system is running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'live_trading_50x.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
            print(f"✅ Trading System: RUNNING (PID: {pid})")
        else:
            print(f"❌ Trading System: STOPPED")
    except:
        print(f"⚠️ Trading System: UNKNOWN")
    
    # Check account status
    try:
        phemex = ccxt.phemex({
            'apiKey': os.getenv("PHEMEX_API_KEY", ""),
            'secret': os.getenv("PHEMEX_API_SECRET", ""),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        balance = await phemex.fetch_balance()
        positions = await phemex.fetch_positions()
        
        total_usdt = balance.get('USDT', {}).get('total', 0)
        free_usdt = balance.get('USDT', {}).get('free', 0)
        used_usdt = balance.get('USDT', {}).get('used', 0)
        
        active_positions = [p for p in positions if float(p.get('size', 0)) != 0]
        
        print(f"💰 Account Balance: ${total_usdt:.2f} USDT")
        print(f"💰 Available: ${free_usdt:.2f} USDT")
        print(f"💰 Used: ${used_usdt:.2f} USDT")
        print(f"📊 Active Positions: {len(active_positions)}")
        
        total_unrealized = 0
        for pos in active_positions:
            symbol = pos['symbol']
            size = float(pos['size'])
            side = pos['side']
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            percentage = float(pos.get('percentage', 0))
            
            print(f"📍 {symbol}: {side} {abs(size):.4f} PnL: ${unrealized_pnl:.2f} ({percentage:+.2f}%)")
            total_unrealized += unrealized_pnl
        
        if active_positions:
            print(f"💼 Total Unrealized P&L: ${total_unrealized:.2f}")
        
        await phemex.close()
        
    except Exception as e:
        print(f"❌ Account check failed: {e}")
    
    # Configuration status
    print(f"\n⚙️ CONFIGURATION:")
    print(f"   Live Trading: {'✅ ENABLED' if os.getenv('LIVE_TRADE', '').lower() in ('1', 'true', 'yes') else '❌ DISABLED'}")
    print(f"   Max Leverage: {os.getenv('LEVERAGE_MAX', '5')}x")
    print(f"   Risk per Trade: {os.getenv('RISK_PER_TRADE_PCT', '0.5')}%")
    print(f"   Max Positions: {os.getenv('MAX_POSITIONS', '5')}")
    print(f"   Score Threshold: {os.getenv('SCORE_MIN', '85')}")
    
    # Check logs
    try:
        with open('/workspace/live_trading.log', 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-10:]
            
        print(f"\n📋 RECENT ACTIVITY:")
        for line in recent_lines:
            if 'Score:' in line or 'OPPORTUNITY:' in line or 'ERROR' in line:
                print(f"   {line.strip()}")
                
    except:
        print(f"⚠️ Could not read log file")
    
    print(f"\n🎯 QUICK ACTIONS:")
    print(f"   • View logs: tail -f /workspace/live_trading.log")
    print(f"   • Stop system: pkill -f live_trading_50x.py")
    print(f"   • Lower score threshold: export SCORE_MIN=70")
    print(f"   • Check positions: python3 -c \"import asyncio; import ccxt.async_support as ccxt; asyncio.run(check_positions())\"")

if __name__ == "__main__":
    asyncio.run(check_live_status())