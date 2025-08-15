#!/usr/bin/env python3
"""
ULTRA FAST OPTIMIZER - PERFORMANCE OPTIMIZED VERSION
With NumPy vectorization and proper parallel processing
"""

import asyncio
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import statistics
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
import warnings

# Suppress NumPy warnings for cleaner output
warnings.filterwarnings('ignore')

# Windows event loop policy
try:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import ccxt.async_support as ccxt

class UltraFastOptimizer:
    """Optimized backtesting engine with NumPy vectorization"""
    
    def __init__(self):
        self.score_threshold = 95
        self.max_workers = mp.cpu_count()  # Use all CPU cores
        self.results = []
        
    def compute_predictive_ranges_vectorized(self, 
                                            ohlcv: Dict[str, np.ndarray],
                                            atr_length: int = 100,
                                            atr_multiplier: float = 3.0) -> Dict[str, np.ndarray]:
        """Vectorized computation of predictive ranges using NumPy"""
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            
            if len(close) < atr_length + 1:
                return {}
            
            # Vectorized True Range calculation
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr = np.maximum.reduce([
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            ])
            
            # Vectorized ATR calculation using convolution
            atr = np.convolve(tr, np.ones(atr_length)/atr_length, mode='valid')
            
            # Pad ATR to match original length
            atr_padded = np.zeros_like(close)
            atr_padded[atr_length-1:] = atr
            
            # Calculate SMA for average price
            sma_len = 20
            sma = np.convolve(close, np.ones(sma_len)/sma_len, mode='same')
            
            # Vectorized range calculations
            r1 = sma + atr_padded * atr_multiplier * 0.5
            r2 = sma + atr_padded * atr_multiplier
            s1 = sma - atr_padded * atr_multiplier * 0.5
            s2 = sma - atr_padded * atr_multiplier
            
            return {
                'r1': r1,
                'r2': r2,
                's1': s1,
                's2': s2,
                'atr': atr_padded,
                'sma': sma
            }
            
        except Exception as e:
            print(f"Error in vectorized PR calculation: {e}")
            return {}
    
    def backtest_strategy_vectorized(self,
                                    ohlcv: Dict[str, np.ndarray],
                                    ranges: Dict[str, np.ndarray],
                                    entry_type: str = "close_confirmed") -> Dict:
        """Vectorized backtesting using NumPy operations"""
        try:
            if not ranges or 'r1' not in ranges:
                return {'trades': [], 'metrics': {}}
            
            close = ohlcv['close']
            high = ohlcv['high']
            low = ohlcv['low']
            
            r1 = ranges['r1']
            r2 = ranges['r2']
            s1 = ranges['s1']
            s2 = ranges['s2']
            
            # Vectorized signal generation
            if entry_type == "close_confirmed":
                long_signals = (close < s1) & np.roll(close < s1, 1)
                short_signals = (close > r1) & np.roll(close > r1, 1)
            else:  # wick_entry
                long_signals = low < s1
                short_signals = high > r1
            
            # Find signal indices
            long_indices = np.where(long_signals)[0]
            short_indices = np.where(short_signals)[0]
            
            trades = []
            
            # Process long trades (vectorized where possible)
            for entry_idx in long_indices[long_indices < len(close) - 5]:
                entry_price = close[entry_idx]
                
                # Vectorized exit search
                future_highs = high[entry_idx+1:min(entry_idx+26, len(high))]
                future_lows = low[entry_idx+1:min(entry_idx+26, len(low))]
                
                # Find TP and SL hits
                tp1_hit = np.where(future_highs >= r1[entry_idx])[0]
                tp2_hit = np.where(future_highs >= r2[entry_idx])[0]
                sl_hit = np.where(future_lows <= s2[entry_idx])[0]
                
                if len(tp1_hit) > 0:
                    exit_idx = tp1_hit[0] + entry_idx + 1
                    exit_price = r1[entry_idx]
                    exit_type = "TP1"
                elif len(sl_hit) > 0:
                    exit_idx = sl_hit[0] + entry_idx + 1
                    exit_price = s2[entry_idx]
                    exit_type = "SL"
                else:
                    continue
                
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'type': 'LONG',
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'exit_type': exit_type,
                    'pnl_pct': float(pnl_pct),
                    'bars_held': exit_idx - entry_idx
                })
            
            # Process short trades (similar vectorized approach)
            for entry_idx in short_indices[short_indices < len(close) - 5]:
                entry_price = close[entry_idx]
                
                future_highs = high[entry_idx+1:min(entry_idx+26, len(high))]
                future_lows = low[entry_idx+1:min(entry_idx+26, len(low))]
                
                tp1_hit = np.where(future_lows <= s1[entry_idx])[0]
                tp2_hit = np.where(future_lows <= s2[entry_idx])[0]
                sl_hit = np.where(future_highs >= r2[entry_idx])[0]
                
                if len(tp1_hit) > 0:
                    exit_idx = tp1_hit[0] + entry_idx + 1
                    exit_price = s1[entry_idx]
                    exit_type = "TP1"
                elif len(sl_hit) > 0:
                    exit_idx = sl_hit[0] + entry_idx + 1
                    exit_price = r2[entry_idx]
                    exit_type = "SL"
                else:
                    continue
                
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                trades.append({
                    'type': 'SHORT',
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'exit_type': exit_type,
                    'pnl_pct': float(pnl_pct),
                    'bars_held': exit_idx - entry_idx
                })
            
            # Calculate metrics using NumPy
            if trades:
                pnls = np.array([t['pnl_pct'] for t in trades])
                wins = pnls > 0
                
                metrics = {
                    'total_trades': len(trades),
                    'wins': int(np.sum(wins)),
                    'losses': int(np.sum(~wins)),
                    'win_rate': float(np.mean(wins) * 100),
                    'avg_profit_pct': float(np.mean(pnls)),
                    'total_return_pct': float(np.sum(pnls)),
                    'max_drawdown_pct': float(np.min(np.minimum.accumulate(pnls))) if len(pnls) > 0 else 0,
                    'profit_factor': float(np.sum(pnls[wins]) / -np.sum(pnls[~wins])) if np.any(~wins) else float('inf'),
                    'avg_bars_held': float(np.mean([t['bars_held'] for t in trades]))
                }
            else:
                metrics = {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'avg_profit_pct': 0,
                    'total_return_pct': 0,
                    'max_drawdown_pct': 0,
                    'profit_factor': 0,
                    'avg_bars_held': 0
                }
            
            return {'trades': trades, 'metrics': metrics}
            
        except Exception as e:
            print(f"Error in vectorized backtest: {e}")
            return {'trades': [], 'metrics': {}}
    
    def process_single_config(self, config_data: Tuple) -> Optional[Dict]:
        """Process a single configuration (for parallel execution)"""
        timeframe, symbol, ohlcv_dict, atr_length, atr_multiplier, entry_type = config_data
        
        try:
            # Convert to NumPy arrays for vectorization
            ohlcv = {
                'open': np.array(ohlcv_dict['open']),
                'high': np.array(ohlcv_dict['high']),
                'low': np.array(ohlcv_dict['low']),
                'close': np.array(ohlcv_dict['close']),
                'volume': np.array(ohlcv_dict['volume'])
            }
            
            # Compute ranges with vectorization
            ranges = self.compute_predictive_ranges_vectorized(
                ohlcv, atr_length, atr_multiplier
            )
            
            if not ranges:
                return None
            
            # Run backtest with vectorization
            result = self.backtest_strategy_vectorized(
                ohlcv, ranges, entry_type
            )
            
            metrics = result['metrics']
            
            # Filter by performance
            if (metrics['win_rate'] >= 60 and 
                metrics['profit_factor'] >= 1.5 and
                metrics['total_trades'] >= 10):
                
                return {
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'atr_length': atr_length,
                    'atr_multiplier': atr_multiplier,
                    'entry_type': entry_type,
                    'metrics': metrics,
                    'score': self.calculate_score(metrics)
                }
            
            return None
            
        except Exception as e:
            print(f"Error processing config: {e}")
            return None
    
    def calculate_score(self, metrics: Dict) -> float:
        """Calculate overall score for ranking"""
        if metrics['total_trades'] == 0:
            return 0
        
        score = (
            metrics['win_rate'] * 0.3 +
            min(metrics['profit_factor'] * 20, 100) * 0.3 +
            min(metrics['avg_profit_pct'] * 10, 100) * 0.2 +
            (100 - abs(metrics['max_drawdown_pct']) * 2) * 0.2
        )
        
        return min(100, max(0, score))
    
    async def fetch_ohlcv_data(self, exchange, symbol: str, timeframe: str) -> Dict:
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
            
            return {
                'open': [x[1] for x in ohlcv],
                'high': [x[2] for x in ohlcv],
                'low': [x[3] for x in ohlcv],
                'close': [x[4] for x in ohlcv],
                'volume': [x[5] for x in ohlcv]
            }
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return {}
    
    async def run_optimization(self):
        """Run optimization across all parameters using ProcessPoolExecutor"""
        print("=" * 60)
        print("ULTRA FAST OPTIMIZER - VECTORIZED VERSION")
        print("=" * 60)
        
        # Initialize exchange
        exchange = ccxt.phemex({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        
        # Parameters to test
        timeframes = ['5m', '15m', '1h', '4h']
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        atr_lengths = [50, 100, 200, 500]
        atr_multipliers = [1.0, 2.0, 3.0, 5.0]
        entry_types = ['close_confirmed', 'wick_entry']
        
        print(f"Testing {len(timeframes)} timeframes x {len(symbols)} symbols")
        print(f"ATR parameters: {len(atr_lengths)} lengths x {len(atr_multipliers)} multipliers")
        print(f"Entry types: {entry_types}")
        print(f"Total combinations: {len(timeframes) * len(symbols) * len(atr_lengths) * len(atr_multipliers) * len(entry_types)}")
        print("=" * 60)
        
        # Fetch all data first
        print("Fetching market data...")
        data_cache = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                data = await self.fetch_ohlcv_data(exchange, symbol, timeframe)
                if data:
                    data_cache[key] = data
                    print(f"✓ Fetched {symbol} {timeframe}: {len(data['close'])} candles")
        
        await exchange.close()
        
        # Prepare all configurations
        configs = []
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                if key not in data_cache:
                    continue
                    
                ohlcv_data = data_cache[key]
                
                for atr_length in atr_lengths:
                    for atr_multiplier in atr_multipliers:
                        for entry_type in entry_types:
                            configs.append((
                                timeframe, symbol, ohlcv_data,
                                atr_length, atr_multiplier, entry_type
                            ))
        
        print(f"\n🚀 Processing {len(configs)} configurations with {self.max_workers} CPU cores...")
        start_time = time.time()
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.process_single_config, config) for config in configs]
            
            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    print(f"Progress: {completed}/{len(configs)} ({rate:.1f} configs/sec)")
                
                result = future.result()
                if result:
                    results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Display results
        print("\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETE in {elapsed_time:.2f} seconds")
        print(f"Processing rate: {len(configs)/elapsed_time:.1f} configs/second")
        print(f"Found {len(results)} profitable configurations")
        print("=" * 60)
        
        if results:
            print("\n🏆 TOP 10 CONFIGURATIONS:")
            print("-" * 60)
            
            for i, result in enumerate(results[:10], 1):
                metrics = result['metrics']
                print(f"\n#{i}. {result['symbol']} - {result['timeframe']}")
                print(f"   ATR: {result['atr_length']} x {result['atr_multiplier']}")
                print(f"   Entry: {result['entry_type']}")
                print(f"   Score: {result['score']:.1f}")
                print(f"   Win Rate: {metrics['win_rate']:.1f}%")
                print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"   Avg Profit: {metrics['avg_profit_pct']:.2f}%")
                print(f"   Total Return: {metrics['total_return_pct']:.1f}%")
                print(f"   Trades: {metrics['total_trades']}")
        
        # Save results
        if results:
            filename = f"ultra_fast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results[:20], f, indent=2)
            print(f"\n💾 Top 20 results saved to {filename}")
        
        return results

async def main():
    """Main entry point"""
    optimizer = UltraFastOptimizer()
    await optimizer.run_optimization()

if __name__ == "__main__":
    # For Windows compatibility
    if __name__ == '__main__':
        mp.freeze_support()
    
    asyncio.run(main())