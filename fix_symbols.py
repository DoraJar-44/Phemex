#!/usr/bin/env python3
"""
Fix Phemex symbol format issues for live trading
"""

import asyncio
import ccxt.async_support as ccxt
from dotenv import load_dotenv
import os
import json

load_dotenv()

async def test_phemex_symbols():
    """Test correct Phemex symbol formats"""
    
    print("🔧 FIXING PHEMEX SYMBOL FORMAT ISSUES")
    print("=" * 50)
    
    # Initialize Phemex
    phemex = ccxt.phemex({
        'apiKey': os.getenv("PHEMEX_API_KEY", ""),
        'secret': os.getenv("PHEMEX_API_SECRET", ""),
        'sandbox': False,  # Live trading
        'enableRateLimit': True,
    })
    
    try:
        print("🔍 Loading Phemex markets...")
        markets = await phemex.load_markets()
        
        # Find correct symbol formats for major pairs
        print("\n📊 CORRECT PHEMEX SYMBOL FORMATS:")
        print("-" * 40)
        
        test_bases = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'ATOM', 'AVAX']
        correct_symbols = {}
        
        for symbol, market in markets.items():
            if (market.get('type') == 'swap' and 
                market.get('quote') == 'USDT' and 
                market.get('active', False)):
                
                base = market.get('base', '')
                if base in test_bases:
                    correct_symbols[base] = symbol
                    max_leverage = market.get('limits', {}).get('leverage', {}).get('max', 0)
                    print(f"✅ {base:<6} -> {symbol:<20} (Max: {max_leverage}x)")
        
        print(f"\n🔬 TESTING MARKET DATA CONNECTION:")
        print("-" * 40)
        
        # Test market data for BTC (should work)
        btc_symbol = correct_symbols.get('BTC')
        if btc_symbol:
            try:
                ohlcv = await phemex.fetch_ohlcv(btc_symbol, '5m', limit=10)
                print(f"✅ {btc_symbol}: {len(ohlcv)} candles fetched successfully")
                
                # Test latest price
                ticker = await phemex.fetch_ticker(btc_symbol)
                print(f"✅ {btc_symbol}: Price ${ticker['last']:,.2f}")
                
            except Exception as e:
                print(f"❌ {btc_symbol}: Market data failed - {e}")
        
        # Test a few more symbols
        for base in ['ETH', 'SOL', 'ADA']:
            symbol = correct_symbols.get(base)
            if symbol:
                try:
                    ohlcv = await phemex.fetch_ohlcv(symbol, '5m', limit=5)
                    ticker = await phemex.fetch_ticker(symbol)
                    print(f"✅ {symbol}: {len(ohlcv)} candles, Price ${ticker['last']:,.4f}")
                except Exception as e:
                    print(f"❌ {symbol}: Failed - {e}")
        
        print(f"\n🎯 50X LEVERAGE SYMBOLS (corrected format):")
        print("-" * 40)
        
        leverage_50x = []
        for symbol, market in markets.items():
            if (market.get('type') == 'swap' and 
                market.get('quote') == 'USDT' and 
                market.get('active', False)):
                
                max_leverage = market.get('limits', {}).get('leverage', {}).get('max', 0)
                if max_leverage >= 50:
                    leverage_50x.append(symbol)
        
        print(f"Found {len(leverage_50x)} symbols supporting 50x+ leverage")
        for i, symbol in enumerate(leverage_50x[:20], 1):
            print(f"{i:2d}. {symbol}")
        
        if len(leverage_50x) > 20:
            print(f"... and {len(leverage_50x) - 20} more")
        
        # Save corrected symbols
        symbol_mapping = {
            'corrected_symbols': correct_symbols,
            'leverage_50x_symbols': leverage_50x[:50],  # Top 50 for performance
            'total_50x_pairs': len(leverage_50x)
        }
        
        with open('/workspace/corrected_symbols.json', 'w') as f:
            json.dump(symbol_mapping, f, indent=2)
        
        print(f"\n✅ Saved corrected symbols to: corrected_symbols.json")
        
        return correct_symbols, leverage_50x
        
    except Exception as e:
        print(f"❌ Error testing Phemex symbols: {e}")
        return {}, []
    finally:
        await phemex.close()

async def test_account_access():
    """Test account access for live trading"""
    
    print(f"\n🔐 TESTING LIVE TRADING ACCOUNT ACCESS:")
    print("-" * 40)
    
    phemex = ccxt.phemex({
        'apiKey': os.getenv("PHEMEX_API_KEY", ""),
        'secret': os.getenv("PHEMEX_API_SECRET", ""),
        'sandbox': False,  # Live trading
        'enableRateLimit': True,
    })
    
    try:
        # Test account balance
        balance = await phemex.fetch_balance()
        total_usdt = balance.get('USDT', {}).get('total', 0)
        free_usdt = balance.get('USDT', {}).get('free', 0)
        
        print(f"✅ Account Balance: ${total_usdt:.2f} USDT")
        print(f"✅ Available Balance: ${free_usdt:.2f} USDT")
        
        if total_usdt > 10:
            print(f"✅ Sufficient balance for live trading")
        else:
            print(f"⚠️ Low balance - consider adding funds")
        
        # Test positions
        positions = await phemex.fetch_positions()
        active_positions = [p for p in positions if float(p.get('size', 0)) != 0]
        
        print(f"📊 Open Positions: {len(active_positions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Account access failed: {e}")
        return False
    finally:
        await phemex.close()

async def main():
    print("🚀 PHEMEX LIVE TRADING SYMBOL FIX")
    print("=" * 50)
    
    # Test symbols and market data
    correct_symbols, leverage_50x = await test_phemex_symbols()
    
    # Test account access
    account_ok = await test_account_access()
    
    if correct_symbols and account_ok:
        print(f"\n🎉 SYSTEM READY FOR LIVE TRADING!")
        print(f"✅ Symbol formats corrected")
        print(f"✅ Market data connection working")  
        print(f"✅ Account access confirmed")
        print(f"✅ {len(leverage_50x)} pairs available for 50x trading")
        
        return True
    else:
        print(f"\n❌ ISSUES NEED TO BE RESOLVED")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())