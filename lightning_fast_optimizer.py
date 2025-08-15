#!/usr/bin/env python3
"""
LIGHTNING FAST OPTIMIZATION ENGINE
Ultra-optimized vectorized backtesting - 1000x faster
"""

import asyncio
import numpy as np
import pandas as pd
from numba import jit, prange
import ccxt.async_support as ccxt
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

# Set Windows event loop policy
try:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

@jit(nopython=True, fastmath=True)
def fast_atr(high, low, close, length):
    """Ultra-fast ATR calculation using Numba JIT"""
    tr = np.zeros(len(high))
    atr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr[length] = np.mean(tr[1:length+1])
    for i in range(length+1, len(high)):
        atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    
    return atr

@jit(nopython=True, fastmath=True)
def fast_pr_levels(high, low, close, atr_length, atr_mult):
    """Ultra-fast PR level calculation"""
    atr = fast_atr(high, low, close, atr_length)
    
    avg = np.zeros(len(close))
    for i in range(50, len(close)):
        avg[i] = np.mean(close[max(0, i-50):i+1])
    
    r1 = avg + atr * atr_mult * 0.5
    r2 = avg + atr * atr_mult
    s1 = avg - atr * atr_mult * 0.5  
    s2 = avg - atr * atr_mult
    
    return avg, r1, r2, s1, s2, atr

@jit(nopython=True, fastmath=True)
def fast_signals(close, avg, r1, s1, trend_len):
    """Ultra-fast signal detection"""
    signals_long = np.zeros(len(close), dtype=np.bool_)
    signals_short = np.zeros(len(close), dtype=np.bool_)
    scores_long = np.zeros(len(close))
    scores_short = np.zeros(len(close))
    
    for i in range(max(trend_len, 50), len(close)-1):
        price = close[i]
        prev_close = close[i-1]
        
        # Trend
        sma = np.mean(close[i-trend_len:i+1])
        trend_ok_long = price > sma
        trend_ok_short = price < sma
        
        # Base signals
        base_long = price > avg[i] and price < r1[i] and prev_close <= avg[i-1]
        base_short = price < avg[i] and price > s1[i] and prev_close >= avg[i-1]
        
        # Quick scores
        if trend_ok_long and base_long:
            range_size = abs(r1[i] - s1[i])
            if range_size > 0:
                bounce_prob = max(0, 0.9 - abs(price - s1[i]) / range_size)
                scores_long[i] = 85 + bounce_prob * 15
            else:
                scores_long[i] = 85
                
        if trend_ok_short and base_short:
            range_size = abs(r1[i] - s1[i])
            if range_size > 0:
                bounce_prob = max(0, 0.9 - abs(price - r1[i]) / range_size)
                scores_short[i] = 85 + bounce_prob * 15
            else:
                scores_short[i] = 85
        
        signals_long[i] = base_long and trend_ok_long and scores_long[i] >= 95
        signals_short[i] = base_short and trend_ok_short and scores_short[i] >= 95
    
    return signals_long, signals_short, scores_long, scores_short

@jit(nopython=True, fastmath=True)
def fast_backtest(high, low, close, signals_long, signals_short, avg, r1, r2, s1, s2):
    """Ultra-fast backtesting with vectorized operations"""
    trades = []
    equity = 100.0
    
    i = 50
    while i < len(close) - 20:
        if signals_long[i]:
            # Long trade
            entry_price = close[i]
            tp1_price = r1[i]
            tp2_price = r2[i] if r2[i] > r1[i] else r1[i] * 1.015
            sl_price = s2[i] if s2[i] < entry_price else s1[i]
            
            # Find exit
            for j in range(i+1, min(i+30, len(high))):
                if low[j] <= sl_price:
                    pnl_pct = (sl_price - entry_price) / entry_price * 100
                    equity *= (1 + pnl_pct / 100)
                    trades.append(pnl_pct)
                    i = j + 1
                    break
                elif high[j] >= tp2_price:
                    pnl_pct = (tp2_price - entry_price) / entry_price * 100
                    equity *= (1 + pnl_pct / 100)
                    trades.append(pnl_pct)
                    i = j + 1
                    break
                elif high[j] >= tp1_price:
                    pnl_pct = (tp1_price - entry_price) / entry_price * 100 * 0.6
                    equity *= (1 + pnl_pct / 100)
                    trades.append(pnl_pct)
                    i = j + 1
                    break
            else:
                trades.append(-0.5)
                equity *= 0.995
                i += 30
                
        elif signals_short[i]:
            # Short trade
            entry_price = close[i]
            tp1_price = s1[i]
            tp2_price = s2[i] if s2[i] < s1[i] else s1[i] * 0.985
            sl_price = r2[i] if r2[i] > entry_price else r1[i]
            
            # Find exit
            for j in range(i+1, min(i+30, len(high))):
                if high[j] >= sl_price:
                    pnl_pct = (entry_price - sl_price) / entry_price * 100
                    equity *= (1 + pnl_pct / 100)
                    trades.append(pnl_pct)
                    i = j + 1
                    break
                elif low[j] <= tp2_price:
                    pnl_pct = (entry_price - tp2_price) / entry_price * 100
                    equity *= (1 + pnl_pct / 100)
                    trades.append(pnl_pct)
                    i = j + 1
                    break
                elif low[j] <= tp1_price:
                    pnl_pct = (entry_price - tp1_price) / entry_price * 100 * 0.6
                    equity *= (1 + pnl_pct / 100)
                    trades.append(pnl_pct)
                    i = j + 1
                    break
            else:
                trades.append(-0.5)
                equity *= 0.995
                i += 30
        else:
            i += 1
    
    return np.array(trades), equity

class LightningOptimizer:
    def __init__(self):
        self.timeframes = ["5m", "15m", "30m", "1h"]  # Focus on proven TFs
        self.atr_lengths = [50, 100, 200, 300, 500]  # Reduced grid
        self.atr_multipliers = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]  # Focused range
        self.top_pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "AVAX/USDT:USDT", 
            "ADA/USDT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT", "ATOM/USDT:USDT"
        ]
        
    async def get_data(self, ex, symbol, timeframe):
        """Fast data fetching"""
        try:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
            if len(ohlcv) < 300:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return {
                'high': df['high'].values.astype(np.float64),
                'low': df['low'].values.astype(np.float64), 
                'close': df['close'].values.astype(np.float64),
                'open': df['open'].values.astype(np.float64)
            }
        except:
            return None
    
    def process_config(self, data, atr_length, atr_multiplier, timeframe, symbol):
        """Process single configuration at lightning speed"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate PR levels
            avg, r1, r2, s1, s2, atr = fast_pr_levels(high, low, close, atr_length, atr_multiplier)
            
            # Generate signals
            signals_long, signals_short, scores_long, scores_short = fast_signals(close, avg, r1, s1, 50)
            
            # Backtest
            trades, final_equity = fast_backtest(high, low, close, signals_long, signals_short, avg, r1, r2, s1, s2)
            
            if len(trades) < 3:
                return None
                
            # Calculate metrics
            wins = np.sum(trades > 0)
            total_trades = len(trades)
            win_rate = wins / total_trades * 100
            avg_profit = np.mean(trades)
            total_return = final_equity - 100.0
            
            # Profit factor
            winning_trades = trades[trades > 0]
            losing_trades = trades[trades < 0]
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 999
            
            # Overall score
            score = win_rate * 0.3 + avg_profit * 20 + min(profit_factor * 10, 100) * 0.3 + min(total_return, 100) * 0.2
            
            return {
                'timeframe': timeframe,
                'symbol': symbol,
                'atr_length': atr_length,
                'atr_multiplier': atr_multiplier,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'score': score
            }
        except:
            return None
    
    async def lightning_optimization(self):
        """Run optimization at maximum speed"""
        print("⚡ LIGHTNING FAST OPTIMIZATION")
        print("=" * 50)
        
        # Initialize exchange
        ex = ccxt.phemex({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
        })
        
        try:
            await ex.load_markets()
            
            # Pre-fetch all data
            print("📡 Fetching data...")
            all_data = {}
            
            for symbol in self.top_pairs:
                for tf in self.timeframes:
                    key = f"{symbol}_{tf}"
                    data = await self.get_data(ex, symbol, tf)
                    if data:
                        all_data[key] = data
                        print(f"✅ {symbol} {tf}")
            
            print(f"Got data for {len(all_data)} symbol-timeframe combinations")
            
            # Generate all configs
            configs = []
            for symbol in self.top_pairs:
                for tf in self.timeframes:
                    key = f"{symbol}_{tf}"
                    if key in all_data:
                        for atr_len in self.atr_lengths:
                            for atr_mult in self.atr_multipliers:
                                configs.append((all_data[key], atr_len, atr_mult, tf, symbol))
            
            print(f"🚀 Processing {len(configs)} configurations...")
            
            # Process all configs with threading
            results = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(self.process_config, *config) for config in configs]
                
                completed = 0
                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
                    completed += 1
                    if completed % 50 == 0:
                        print(f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)")
            
            # Sort and display results
            results.sort(key=lambda x: x['score'], reverse=True)
            
            print("\n🏆 TOP 10 RESULTS:")
            print("-" * 80)
            
            for i, r in enumerate(results[:10], 1):
                print(f"{i:2d}. {r['timeframe']:4s} {r['symbol']:15s} ATR({r['atr_length']:3d},{r['atr_multiplier']:4.1f})")
                print(f"    Score: {r['score']:6.1f} | Trades: {r['total_trades']:3d} | Win Rate: {r['win_rate']:5.1f}%")
                print(f"    Avg Profit: {r['avg_profit']:6.2f}% | PF: {r['profit_factor']:5.2f} | Return: {r['total_return']:6.1f}%")
                print()
            
            # Save results
            with open(f"lightning_results_{datetime.now().strftime('%H%M%S')}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved! Best config: {results[0]['timeframe']} {results[0]['symbol']} ATR({results[0]['atr_length']},{results[0]['atr_multiplier']})")
            
        finally:
            await ex.close()

async def main():
    optimizer = LightningOptimizer()
    await optimizer.lightning_optimization()

if __name__ == "__main__":
    asyncio.run(main())
