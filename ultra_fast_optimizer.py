#!/usr/bin/env python3
"""
ULTRA FAST OPTIMIZER - Pure Python with maximum speed optimizations
Using proven performance techniques: vectorization, built-ins, list comprehensions
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import statistics
import itertools

# Windows event loop policy
try:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import ccxt.async_support as ccxt

class UltraFastOptimizer:
    def __init__(self):
        # Focused parameter grid for speed
        self.timeframes = ["1m", "3m", "5m", "15m", "30m", "1h"]
        self.atr_lengths = [50, 100, 200, 300, 500]
        self.atr_multipliers = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "AVAX/USDT:USDT",
            "ADA/USDT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT", "ATOM/USDT:USDT"
        ]
        
    async def get_data_fast(self, ex, symbol, timeframe):
        """Ultra-fast data fetching with minimal overhead"""
        try:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
            if len(ohlcv) < 200:
                return None
            
            # Direct list creation - fastest method
            return {
                'h': [float(x[2]) for x in ohlcv],  # high
                'l': [float(x[3]) for x in ohlcv],  # low  
                'c': [float(x[4]) for x in ohlcv],  # close
                'o': [float(x[1]) for x in ohlcv]   # open
            }
        except:
            return None
    
    @staticmethod
    def fast_atr(high, low, close, length):
        """Optimized ATR using built-in functions"""
        if len(high) < length + 5:
            return [0] * len(high)
            
        # Calculate True Range using list comprehensions (faster than loops)
        tr = [max(high[i] - low[i], 
                 abs(high[i] - close[i-1]), 
                 abs(low[i] - close[i-1])) 
              for i in range(1, len(high))]
        
        # Initialize ATR list
        atr = [0] * len(high)
        if len(tr) >= length:
            atr[length] = sum(tr[:length]) / length
            
            # Fast ATR calculation using built-in sum
            for i in range(length + 1, len(high)):
                atr[i] = (atr[i-1] * (length-1) + tr[i-1]) / length
                
        return atr
    
    @staticmethod
    def fast_pr_signals(data, atr_length, atr_multiplier):
        """Ultra-fast PR calculation and signal generation"""
        h, l, c = data['h'], data['l'], data['c']
        
        if len(c) < 100:
            return []
            
        # Fast ATR calculation
        atr = UltraFastOptimizer.fast_atr(h, l, c, atr_length)
        
        # Fast SMA using built-in sum and slicing
        sma_len = 50
        sma = [sum(c[max(0, i-sma_len):i+1]) / min(sma_len, i+1) 
               for i in range(len(c))]
        
        # PR levels using list comprehensions
        avg = sma  # Use SMA as average
        r1 = [avg[i] + atr[i] * atr_multiplier * 0.5 for i in range(len(avg))]
        r2 = [avg[i] + atr[i] * atr_multiplier for i in range(len(avg))]
        s1 = [avg[i] - atr[i] * atr_multiplier * 0.5 for i in range(len(avg))]
        s2 = [avg[i] - atr[i] * atr_multiplier for i in range(len(avg))]
        
        # Fast signal detection
        signals = []
        start_idx = max(atr_length, sma_len) + 10
        
        for i in range(start_idx, len(c) - 5):
            price = c[i]
            prev_price = c[i-1]
            
            # Trend check (optimized)
            trend_long = price > sma[i]
            trend_short = price < sma[i]
            
            # Entry conditions (optimized logic)
            long_entry = (trend_long and price > avg[i] and price < r1[i] and 
                         prev_price <= avg[i-1])
            short_entry = (trend_short and price < avg[i] and price > s1[i] and 
                          prev_price >= avg[i-1])
            
            # Quick scoring (simplified for speed)
            if long_entry or short_entry:
                range_size = abs(r1[i] - s1[i])
                if range_size > 0:
                    if long_entry:
                        bounce_prob = max(0, 0.9 - abs(price - s1[i]) / range_size)
                        score = 85 + bounce_prob * 15
                    else:
                        bounce_prob = max(0, 0.9 - abs(price - r1[i]) / range_size)
                        score = 85 + bounce_prob * 15
                    
                    if score >= 95:  # Only high-quality signals
                        signals.append({
                            'i': i, 'side': 'long' if long_entry else 'short',
                            'price': price, 'tp1': r1[i] if long_entry else s1[i],
                            'tp2': r2[i] if long_entry else s2[i],
                            'sl': s2[i] if long_entry else r2[i], 'score': score
                        })
        
        return signals
    
    @staticmethod
    def fast_backtest(signals, h, l, c):
        """Ultra-fast backtesting using optimized loops"""
        if not signals:
            return {'trades': 0, 'pnl': 0, 'win_rate': 0, 'total_return': 0}
            
        trades = []
        equity = 100.0
        
        for sig in signals:
            entry_idx = sig['i']
            side = sig['side']
            entry_price = sig['price']
            tp1, tp2, sl = sig['tp1'], sig['tp2'], sig['sl']
            
            # Fast exit detection (limited lookforward for speed)
            exit_found = False
            for j in range(entry_idx + 1, min(entry_idx + 25, len(h))):
                bar_high, bar_low = h[j], l[j]
                
                if side == 'long':
                    if bar_low <= sl:
                        pnl = (sl - entry_price) / entry_price * 100
                        trades.append(pnl)
                        equity *= (1 + pnl / 100)
                        exit_found = True
                        break
                    elif bar_high >= tp2:
                        pnl = (tp2 - entry_price) / entry_price * 100
                        trades.append(pnl)
                        equity *= (1 + pnl / 100)
                        exit_found = True
                        break
                    elif bar_high >= tp1:
                        pnl = (tp1 - entry_price) / entry_price * 100 * 0.6
                        trades.append(pnl)
                        equity *= (1 + pnl / 100)
                        exit_found = True
                        break
                else:  # short
                    if bar_high >= sl:
                        pnl = (entry_price - sl) / entry_price * 100
                        trades.append(pnl)
                        equity *= (1 + pnl / 100)
                        exit_found = True
                        break
                    elif bar_low <= tp2:
                        pnl = (entry_price - tp2) / entry_price * 100
                        trades.append(pnl)
                        equity *= (1 + pnl / 100)
                        exit_found = True
                        break
                    elif bar_low <= tp1:
                        pnl = (entry_price - tp1) / entry_price * 100 * 0.6
                        trades.append(pnl)
                        equity *= (1 + pnl / 100)
                        exit_found = True
                        break
            
            if not exit_found:
                trades.append(-0.5)  # Small timeout loss
                equity *= 0.995
        
        if not trades:
            return {'trades': 0, 'pnl': 0, 'win_rate': 0, 'total_return': 0}
        
        # Fast metrics calculation using built-ins
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades) * 100
        avg_pnl = sum(trades) / len(trades)
        total_return = equity - 100.0
        
        return {
            'trades': len(trades), 'pnl': avg_pnl, 'win_rate': win_rate, 
            'total_return': total_return
        }
    
    def process_single_config(self, args):
        """Process single configuration - optimized for multiprocessing"""
        data, atr_length, atr_multiplier, timeframe, symbol = args
        
        try:
            # Generate signals
            signals = self.fast_pr_signals(data, atr_length, atr_multiplier)
            
            # Backtest
            results = self.fast_backtest(signals, data['h'], data['l'], data['c'])
            
            if results['trades'] < 3:
                return None
            
            # Calculate composite score
            score = (results['win_rate'] * 0.4 + 
                    min(results['pnl'] * 20, 50) + 
                    min(results['total_return'], 100) * 0.3)
            
            return {
                'timeframe': timeframe, 'symbol': symbol,
                'atr_length': atr_length, 'atr_multiplier': atr_multiplier,
                'trades': results['trades'], 'win_rate': results['win_rate'],
                'avg_pnl': results['pnl'], 'total_return': results['total_return'],
                'score': score
            }
        except:
            return None
    
    async def run_optimization(self):
        """Run ultra-fast optimization"""
        print("🚀 ULTRA FAST OPTIMIZATION ENGINE")
        print("=" * 50)
        start_time = time.time()
        
        # Initialize exchange
        ex = ccxt.phemex({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
        })
        
        try:
            await ex.load_markets()
            
            # Phase 1: Lightning data collection
            print("📡 Fetching data at lightning speed...")
            all_data = {}
            
            fetch_tasks = []
            for symbol in self.pairs:
                for tf in self.timeframes:
                    fetch_tasks.append(self.get_data_fast(ex, symbol, tf))
            
            # Concurrent data fetching
            data_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Map results back
            idx = 0
            for symbol in self.pairs:
                for tf in self.timeframes:
                    result = data_results[idx]
                    if result and not isinstance(result, Exception):
                        all_data[f"{symbol}_{tf}"] = result
                        print(f"✅ {symbol} {tf}")
                    idx += 1
            
            print(f"📊 Got data for {len(all_data)} combinations")
            
            # Phase 2: Ultra-fast processing
            print("⚡ Processing configurations...")
            
            # Generate all configs for parallel processing
            configs = []
            for symbol in self.pairs:
                for tf in self.timeframes:
                    key = f"{symbol}_{tf}"
                    if key in all_data:
                        for atr_len in self.atr_lengths:
                            for atr_mult in self.atr_multipliers:
                                configs.append((all_data[key], atr_len, atr_mult, tf, symbol))
            
            print(f"🔥 Processing {len(configs)} configurations with parallel processing...")
            
            # Use ThreadPoolExecutor for CPU-bound tasks (faster than ProcessPool for this workload)
            results = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all tasks
                futures = [executor.submit(self.process_single_config, config) for config in configs]
                
                # Collect results with progress
                completed = 0
                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
                    completed += 1
                    
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(configs) - completed) / rate
                        print(f"⚡ {completed}/{len(configs)} ({completed/len(configs)*100:.0f}%) "
                              f"Rate: {rate:.0f}/s ETA: {eta:.0f}s")
            
            # Phase 3: Results analysis
            elapsed = time.time() - start_time
            print(f"\n🏁 COMPLETED in {elapsed:.1f}s ({len(configs)/elapsed:.0f} configs/sec)")
            print(f"📊 Valid results: {len(results)}")
            
            if not results:
                print("❌ No valid results found!")
                return
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Display top results
            print("\n🏆 TOP 15 CONFIGURATIONS:")
            print("-" * 90)
            print("Rank TF   Symbol          ATR(Len,Mult) Score  Trades WinRate  AvgPnL TotalRet")
            print("-" * 90)
            
            for i, r in enumerate(results[:15], 1):
                print(f"{i:2d}.  {r['timeframe']:4s} {r['symbol']:12s} "
                      f"ATR({r['atr_length']:3d},{r['atr_multiplier']:4.1f}) "
                      f"{r['score']:6.1f} {r['trades']:6d} {r['win_rate']:6.1f}% "
                      f"{r['avg_pnl']:7.2f}% {r['total_return']:7.1f}%")
            
            # Category analysis
            print(f"\n📊 BEST BY TIMEFRAME:")
            tf_best = {}
            for r in results:
                if r['timeframe'] not in tf_best:
                    tf_best[r['timeframe']] = r
            
            for tf in ["1m", "3m", "5m", "15m", "30m", "1h"]:
                if tf in tf_best:
                    r = tf_best[tf]
                    print(f"  {tf:4s}: {r['symbol']:12s} ATR({r['atr_length']:3d},{r['atr_multiplier']:4.1f}) "
                          f"Score:{r['score']:5.1f} WR:{r['win_rate']:5.1f}%")
            
            print(f"\n🥇 WINNER: {results[0]['timeframe']} {results[0]['symbol']} "
                  f"ATR({results[0]['atr_length']},{results[0]['atr_multiplier']}) "
                  f"Score: {results[0]['score']:.1f}")
            
            # Save results
            filename = f"ultra_fast_results_{datetime.now().strftime('%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {filename}")
            
        finally:
            await ex.close()

async def main():
    optimizer = UltraFastOptimizer()
    await optimizer.run_optimization()

if __name__ == "__main__":
    asyncio.run(main())
