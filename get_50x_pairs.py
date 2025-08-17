#!/usr/bin/env python3
"""
Script to identify all Phemex pairs that support 50x leverage
"""

import asyncio
import ccxt.async_support as ccxt
from dotenv import load_dotenv
import os
import json
from typing import List, Dict

load_dotenv(override=True)

async def get_50x_leverage_pairs():
    """Fetch all pairs that support 50x leverage from Phemex"""
    
    # Initialize Phemex exchange
    phemex = ccxt.phemex({
        'apiKey': os.getenv("PHEMEX_API_KEY", ""),
        'secret': os.getenv("PHEMEX_API_SECRET", ""),
        'sandbox': os.getenv("PHEMEX_TESTNET", "false").lower() in ("1", "true", "yes"),
        'enableRateLimit': True,
    })
    
    try:
        print("🔍 Loading markets from Phemex...")
        markets = await phemex.load_markets()
        
        pairs_50x = []
        pairs_100x = []
        all_leverage_info = {}
        
        print(f"📊 Found {len(markets)} total markets")
        print("\n🔎 Analyzing leverage limits...")
        
        for symbol, market in markets.items():
            try:
                # Focus on USDT perpetual swaps only
                if (market.get('type') == 'swap' and 
                    market.get('quote') == 'USDT' and 
                    market.get('active', False) and
                    'USDT' in symbol and
                    ':USDT' in symbol):  # Perpetual futures format
                    
                    # Get leverage information
                    leverage_limits = market.get('limits', {}).get('leverage', {})
                    max_leverage = leverage_limits.get('max', 0)
                    min_leverage = leverage_limits.get('min', 1)
                    
                    # Store leverage info
                    all_leverage_info[symbol] = {
                        'max_leverage': max_leverage,
                        'min_leverage': min_leverage,
                        'base': market.get('base'),
                        'quote': market.get('quote'),
                        'type': market.get('type'),
                        'active': market.get('active')
                    }
                    
                    # Categorize by leverage
                    if max_leverage >= 100:
                        pairs_100x.append(symbol)
                        print(f"💯 {symbol:<20} - Max: {max_leverage}x")
                    elif max_leverage >= 50:
                        pairs_50x.append(symbol)
                        print(f"🔥 {symbol:<20} - Max: {max_leverage}x")
                    elif max_leverage >= 25:
                        print(f"⚡ {symbol:<20} - Max: {max_leverage}x")
                    else:
                        print(f"🔻 {symbol:<20} - Max: {max_leverage}x")
                        
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue
        
        # Sort pairs alphabetically
        pairs_50x.sort()
        pairs_100x.sort()
        
        print(f"\n📈 PAIRS SUPPORTING 50x+ LEVERAGE ({len(pairs_50x)} pairs):")
        print("=" * 60)
        for i, pair in enumerate(pairs_50x, 1):
            leverage_info = all_leverage_info.get(pair, {})
            max_lev = leverage_info.get('max_leverage', 0)
            print(f"{i:2d}. {pair:<25} (Max: {max_lev}x)")
        
        print(f"\n🚀 PAIRS SUPPORTING 100x+ LEVERAGE ({len(pairs_100x)} pairs):")
        print("=" * 60)
        for i, pair in enumerate(pairs_100x, 1):
            leverage_info = all_leverage_info.get(pair, {})
            max_lev = leverage_info.get('max_leverage', 0)
            print(f"{i:2d}. {pair:<25} (Max: {max_lev}x)")
        
        # Save to files
        with open('/workspace/data/phemex_50x_pairs.json', 'w') as f:
            json.dump({
                'pairs_50x': pairs_50x,
                'pairs_100x': pairs_100x,
                'all_leverage_info': all_leverage_info,
                'total_50x': len(pairs_50x),
                'total_100x': len(pairs_100x)
            }, f, indent=2)
        
        # Save simple text files
        with open('/workspace/data/phemex_watchlist_50x.txt', 'w') as f:
            for pair in pairs_50x:
                f.write(f"{pair}\n")
        
        with open('/workspace/data/phemex_watchlist_100x.txt', 'w') as f:
            for pair in pairs_100x:
                f.write(f"{pair}\n")
        
        print(f"\n✅ Results saved to:")
        print(f"   📁 data/phemex_50x_pairs.json")
        print(f"   📁 data/phemex_watchlist_50x.txt")
        print(f"   📁 data/phemex_watchlist_100x.txt")
        
        return pairs_50x, pairs_100x, all_leverage_info
        
    except Exception as e:
        print(f"❌ Error fetching leverage information: {e}")
        return [], [], {}
    finally:
        await phemex.close()

async def main():
    print("🔥 PHEMEX 50X LEVERAGE PAIR DISCOVERY")
    print("=" * 50)
    
    pairs_50x, pairs_100x, all_info = await get_50x_leverage_pairs()
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Total 50x+ pairs: {len(pairs_50x)}")
    print(f"   • Total 100x+ pairs: {len(pairs_100x)}")
    print(f"   • Combined total: {len(pairs_50x) + len(pairs_100x)}")
    
    if pairs_50x or pairs_100x:
        print(f"\n🎯 RECOMMENDED FOR 50X TRADING:")
        # Show top 20 most liquid/popular pairs
        recommended = (pairs_100x + pairs_50x)[:20]
        for i, pair in enumerate(recommended, 1):
            max_lev = all_info.get(pair, {}).get('max_leverage', 0)
            print(f"  {i:2d}. {pair:<25} (Max: {max_lev}x)")

if __name__ == "__main__":
    asyncio.run(main())