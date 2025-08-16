#!/usr/bin/env python3
"""
ULTRA-FAST PINE SCRIPT OPTIMIZER
Tests essential configurations only for quick results
"""

import asyncio
import ccxt.async_support as ccxt
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "8d65ae81-ddd4-44f7-84bb-5b01608251de"
API_SECRET = "_NKwZcNx8JMrpJD7NORH8abxVOA1Jw6G-JM3jl2-18phOWY4NTc4NS00YzkyLTQzZWQtYTk0MS1hZDEwNTU3MzUyOWQ"

class FastPineStrategy:
    """Simplified Pine Script strategy for fast testing"""
    
    def __init__(self, config: Dict):
        self.atr_length = config['atr_length']
        self.atr_mult = config['atr_mult']
        self.score_min = config['score_min']
        self.trend_len = config['trend_len']
        
    def calculate_signals(self, closes: np.ndarray) -> Dict:
        """Fast signal calculation"""
        if len(closes) < max(self.atr_length, self.trend_len):
            return {'signals': 0, 'score': 0}
        
        # Simple ATR approximation
        volatility = np.std(closes[-self.atr_length:])
        atr = volatility * 2.0
        
        # Current price and trend
        current = closes[-1]
        trend_sma = np.mean(closes[-self.trend_len:])
        
        # Predictive ranges
        avg = current
        hold_atr = atr * self.atr_mult * 0.5
        r1 = avg + hold_atr
        s1 = avg - hold_atr
        
        # Signal scoring
        score = 0
        signals = 0
        
        # Range position (30 points)
        if current > avg and current < r1:
            proximity = 1.0 - abs(current - s1) / (r1 - s1)
            score += 30 * proximity
            signals += 1
        
        # Trend alignment (20 points)
        if current > trend_sma:
            trend_strength = (current - trend_sma) / trend_sma
            score += 20 * min(trend_strength * 100, 1.0)
        
        # Momentum (10 points)
        if len(closes) >= 6:
            momentum = (current - closes[-6]) / closes[-6]
            if momentum > 0:
                score += 10 * min(abs(momentum) * 50, 1.0)
        
        # Volatility bonus (15 points)
        vol_ratio = volatility / current
        if vol_ratio > 0.01:
            score += 15 * min(vol_ratio * 50, 1.0)
        
        return {
            'signals': signals if score >= self.score_min else 0,
            'score': score,
            'r1': r1,
            's1': s1,
            'avg': avg
        }

async def test_configuration(exchange, symbol: str, timeframe: str, config: Dict) -> Dict:
    """Test a single configuration"""
    try:
        # Fetch minimal data for speed (using sync within async)
        import ccxt
        sync_exchange = ccxt.phemex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': False,
            'options': {'defaultType': 'swap'}
        })
        sync_exchange.load_markets()
        ohlcv = sync_exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        if not ohlcv or len(ohlcv) < 100:
            return {'score': -1}
        
        closes = np.array([x[4] for x in ohlcv])
        
        # Create strategy
        strategy = FastPineStrategy(config)
        
        # Simulate trading
        total_signals = 0
        winning_signals = 0
        total_pnl = 0
        
        for i in range(100, len(closes)):
            window = closes[:i+1]
            result = strategy.calculate_signals(window)
            
            if result['signals'] > 0:
                total_signals += 1
                
                # Simulate trade outcome
                future_price = closes[min(i+5, len(closes)-1)]
                pnl = (future_price - closes[i]) / closes[i] * 100
                
                if pnl > 0:
                    winning_signals += 1
                total_pnl += pnl
        
        if total_signals == 0:
            return {'score': 0}
        
        win_rate = winning_signals / total_signals * 100
        avg_pnl = total_pnl / total_signals
        
        # Composite score
        score = win_rate * 0.5 + avg_pnl * 10 + min(total_signals, 20)
        
        return {
            'score': score,
            'signals': total_signals,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'symbol': symbol,
            'timeframe': timeframe,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Error testing {symbol} {timeframe}: {e}")
        return {'score': -1}

async def optimize_fast():
    """Fast optimization of Pine Script strategy"""
    
    # Initialize exchange
    exchange = ccxt.phemex({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': False,  # Faster for testing
        'options': {'defaultType': 'swap'}
    })
    
    await exchange.load_markets()
    
    # Best perpetual swap symbols
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    timeframes = ['1m', '5m', '15m']
    
    # Essential configurations only
    configs = [
        {'atr_length': 50, 'atr_mult': 4.0, 'score_min': 30, 'trend_len': 20},
        {'atr_length': 100, 'atr_mult': 6.0, 'score_min': 40, 'trend_len': 50},
        {'atr_length': 200, 'atr_mult': 6.0, 'score_min': 30, 'trend_len': 50},
        {'atr_length': 200, 'atr_mult': 8.0, 'score_min': 50, 'trend_len': 100},
        {'atr_length': 100, 'atr_mult': 4.0, 'score_min': 40, 'trend_len': 50},
        {'atr_length': 150, 'atr_mult': 5.0, 'score_min': 35, 'trend_len': 75},
    ]
    
    results = []
    tasks = []
    
    logger.info(f"Testing {len(symbols)} symbols × {len(timeframes)} timeframes × {len(configs)} configs")
    
    # Create all tasks
    for symbol in symbols:
        for timeframe in timeframes:
            for config in configs:
                task = test_configuration(exchange, symbol, timeframe, config)
                tasks.append(task)
    
    # Run all tasks in parallel
    logger.info(f"Running {len(tasks)} tests in parallel...")
    start_time = time.time()
    
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in task_results:
        if isinstance(result, dict) and result.get('score', -1) > 0:
            results.append(result)
    
    elapsed = time.time() - start_time
    logger.info(f"Completed {len(tasks)} tests in {elapsed:.1f} seconds")
    
    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'pinescript_fast_results_{timestamp}.json', 'w') as f:
        json.dump(results[:20], f, indent=2, default=str)
    
    await exchange.close()
    
    return results

def display_results(results: List[Dict]):
    """Display optimization results"""
    
    print("\n" + "="*80)
    print("PINE SCRIPT STRATEGY - OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\n🏆 TOP 5 CONFIGURATIONS:")
    print("-"*80)
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n#{i} CONFIGURATION")
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        print(f"Timeframe: {result.get('timeframe', 'N/A')}")
        print(f"Score: {result.get('score', 0):.2f}")
        print(f"Win Rate: {result.get('win_rate', 0):.1f}%")
        print(f"Avg P&L: {result.get('avg_pnl', 0):.2f}%")
        print(f"Total Signals: {result.get('signals', 0)}")
        
        config = result.get('config', {})
        print(f"\nParameters:")
        print(f"  ATR Length: {config.get('atr_length')}")
        print(f"  ATR Multiplier: {config.get('atr_mult')}")
        print(f"  Score Minimum: {config.get('score_min')}")
        print(f"  Trend Length: {config.get('trend_len')}")
    
    print("\n" + "="*80)
    print("BEST OVERALL SETTINGS:")
    print("-"*80)
    
    if results:
        best = results[0]
        config = best.get('config', {})
        
        print(f"""
✅ OPTIMAL CONFIGURATION:
  • ATR Length: {config.get('atr_length')}
  • ATR Multiplier: {config.get('atr_mult')}  
  • Score Minimum: {config.get('score_min')}
  • Trend Length: {config.get('trend_len')}
  
📊 PERFORMANCE:
  • Win Rate: {best.get('win_rate', 0):.1f}%
  • Average P&L: {best.get('avg_pnl', 0):.2f}%
  • Signal Frequency: {best.get('signals', 0)} per 100 candles
  
🎯 BEST FOR:
  • Symbol: {best.get('symbol', 'N/A')}
  • Timeframe: {best.get('timeframe', 'N/A')}
  
💰 TRADING SETUP:
  • Use $1 risk per trade
  • Set leverage to 34x
  • Stop Loss at S2 level
  • Take Profit at R1 level
""")
    
    print("="*80)

async def main():
    """Main entry point"""
    
    print("🚀 Starting Pine Script Fast Optimizer...")
    
    try:
        results = await optimize_fast()
        display_results(results)
        
        print("\n✅ Optimization complete!")
        print(f"📁 Results saved to pinescript_fast_results_*.json")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())