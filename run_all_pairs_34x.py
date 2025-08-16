#!/usr/bin/env python3
"""
LIVE TRADING BOT - ALL PAIRS WITH 34X LEVERAGE
Using $1 fixed risk per trade
"""

import asyncio
import ccxt.async_support as ccxt
import time
from datetime import datetime
import json

# Your credentials
API_KEY = "8d65ae81-ddd4-44f7-84bb-5b01608251de"
API_SECRET = "_NKwZcNx8JMrpJD7NORH8abxVOA1Jw6G-JM3jl2-18phOWY4NTc4NS00YzkyLTQzZWQtYTk0MS1hZDEwNTU3MzUyOWQ"

# Settings
LEVERAGE = 34
RISK_PER_TRADE = 1.0  # $1 per trade
ACCOUNT_BALANCE = 47.25
MAX_POSITIONS = 5
MIN_SCORE = 75
TIMEFRAME = '5m'

class AllPairsTrader:
    def __init__(self):
        self.exchange = None
        self.positions = {}
        self.signals = []
        
    async def initialize(self):
        """Initialize exchange"""
        print("🚀 Initializing Phemex connection...")
        self.exchange = ccxt.phemex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
            }
        })
        
        # Set leverage for all pairs
        print(f"⚙️ Setting leverage to {LEVERAGE}x")
        return True
        
    async def get_all_pairs(self):
        """Get all tradeable pairs with 34x+ leverage"""
        markets = await self.exchange.load_markets()
        
        pairs = []
        for symbol, market in markets.items():
            if (market['active'] and 
                market['type'] == 'swap' and 
                market['quote'] == 'USDT' and
                market['settle'] == 'USDT'):
                
                # Check leverage
                max_lev = market.get('limits', {}).get('leverage', {}).get('max', 0)
                if max_lev >= 34:
                    pairs.append(symbol)
                    
        print(f"✅ Found {len(pairs)} pairs with 34x+ leverage")
        return pairs
        
    async def calculate_score(self, symbol):
        """Calculate trading score for a symbol"""
        try:
            # Fetch OHLCV
            ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            
            if len(ohlcv) < 50:
                return None
                
            closes = [c[4] for c in ohlcv]
            highs = [c[2] for c in ohlcv]
            lows = [c[3] for c in ohlcv]
            
            current_price = closes[-1]
            
            # Calculate ATR
            atr_period = 14
            tr_values = []
            for i in range(1, min(len(ohlcv), atr_period + 1)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                tr_values.append(tr)
                
            atr = sum(tr_values[-atr_period:]) / atr_period if tr_values else 0
            
            if atr == 0:
                return None
                
            # Simple trend detection
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
            
            # Score calculation
            score = 50
            if sma_20 > sma_50:
                score += 25
            if current_price > sma_20:
                score += 15
            
            # Momentum
            momentum = (current_price - closes[-10]) / closes[-10] * 100
            score += min(10, abs(momentum))
            
            # Determine direction
            direction = "LONG" if sma_20 > sma_50 and current_price > sma_20 else "SHORT"
            
            return {
                'symbol': symbol,
                'score': score,
                'direction': direction,
                'price': current_price,
                'atr': atr,
                'sl_distance': atr * 1.5,
                'tp_distance': atr * 4.5,  # 3:1 RR
            }
            
        except Exception as e:
            return None
            
    async def place_trade(self, signal):
        """Place a trade with 34x leverage"""
        try:
            symbol = signal['symbol']
            side = signal['direction'].lower()
            
            # Calculate position size for $1 risk
            position_size = RISK_PER_TRADE / signal['sl_distance']
            
            # Calculate notional value
            notional = position_size * signal['price']
            
            # Check if we need this much leverage
            required_leverage = notional / ACCOUNT_BALANCE
            
            if required_leverage > LEVERAGE:
                print(f"⚠️ Skipping {symbol} - requires {required_leverage:.1f}x leverage")
                return False
                
            print(f"\n📈 PLACING TRADE:")
            print(f"Symbol: {symbol}")
            print(f"Direction: {side.upper()}")
            print(f"Entry: ${signal['price']:.4f}")
            print(f"Position Size: {position_size:.6f}")
            print(f"Risk: $1.00")
            print(f"Target Profit: $3.00")
            print(f"Leverage Used: {min(LEVERAGE, required_leverage):.1f}x")
            
            # Set leverage
            await self.exchange.set_leverage(LEVERAGE, symbol)
            
            # Place market order
            order = await self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=position_size,
                params={
                    'leverage': LEVERAGE,
                    'stopLoss': {
                        'triggerPrice': signal['price'] - signal['sl_distance'] if side == 'buy' else signal['price'] + signal['sl_distance'],
                        'type': 'market'
                    },
                    'takeProfit': {
                        'triggerPrice': signal['price'] + signal['tp_distance'] if side == 'buy' else signal['price'] - signal['tp_distance'],
                        'type': 'market'
                    }
                }
            )
            
            print(f"✅ Order placed: {order['id']}")
            
            self.positions[symbol] = {
                'side': side,
                'entry': signal['price'],
                'size': position_size,
                'time': datetime.now()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Trade failed for {signal['symbol']}: {str(e)}")
            return False
            
    async def scan_all_pairs(self):
        """Scan all pairs and trade the best ones"""
        pairs = await self.get_all_pairs()
        
        print(f"\n🔍 Scanning {len(pairs)} pairs...")
        
        all_signals = []
        
        # Scan in batches
        batch_size = 10
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            
            tasks = [self.calculate_score(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if result and not isinstance(result, Exception):
                    if result['score'] >= MIN_SCORE:
                        all_signals.append(result)
                        
            print(f"\rProgress: {min(i+batch_size, len(pairs))}/{len(pairs)} pairs scanned", end="", flush=True)
            
            # Small delay
            await asyncio.sleep(0.5)
            
        print(f"\n\n✅ Found {len(all_signals)} trading signals")
        
        # Sort by score
        all_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top signals
        print("\n🏆 TOP SIGNALS:")
        print("-" * 60)
        for i, signal in enumerate(all_signals[:10], 1):
            print(f"{i}. {signal['symbol']:<20} Score: {signal['score']:.1f} Direction: {signal['direction']}")
            
        return all_signals
        
    async def run(self):
        """Main trading loop"""
        await self.initialize()
        
        print("\n" + "="*60)
        print("LIVE TRADING ACTIVATED")
        print(f"Balance: ${ACCOUNT_BALANCE}")
        print(f"Leverage: {LEVERAGE}x")
        print(f"Risk per trade: ${RISK_PER_TRADE}")
        print(f"Max positions: {MAX_POSITIONS}")
        print("="*60)
        
        while True:
            try:
                # Get current positions count
                current_positions = len(self.positions)
                
                if current_positions >= MAX_POSITIONS:
                    print(f"\n⚠️ Max positions reached ({MAX_POSITIONS})")
                    await asyncio.sleep(60)
                    continue
                    
                # Scan all pairs
                signals = await self.scan_all_pairs()
                
                # Trade top signals
                trades_placed = 0
                for signal in signals:
                    if current_positions + trades_placed >= MAX_POSITIONS:
                        break
                        
                    # Skip if already in position
                    if signal['symbol'] in self.positions:
                        continue
                        
                    # Place trade
                    if await self.place_trade(signal):
                        trades_placed += 1
                        await asyncio.sleep(2)  # Small delay between trades
                        
                print(f"\n📊 Placed {trades_placed} new trades")
                print(f"📈 Total positions: {len(self.positions)}")
                
                # Wait before next scan
                print(f"\n⏰ Next scan in 5 minutes...")
                await asyncio.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                await asyncio.sleep(30)
                
        await self.exchange.close()

async def main():
    trader = AllPairsTrader()
    await trader.run()

if __name__ == "__main__":
    print("⚡"*30)
    print("34X LEVERAGE TRADING BOT")
    print("ALL PAIRS - $1 FIXED RISK")
    print("⚡"*30)
    
    asyncio.run(main())