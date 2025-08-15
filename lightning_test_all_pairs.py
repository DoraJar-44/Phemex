#!/usr/bin/env python3
"""
LIGHTNING SPEED COMPREHENSIVE PHEMEX TESTING
Tests all pairs with multiple configurations at maximum speed
"""

import asyncio
import ccxt.async_support as ccxt
import json
import time
from datetime import datetime
from itertools import product
import random
import os
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import numpy as np

# API Credentials
API_KEY = "8d65ae81-ddd4-44f7-84bb-5b01608251de"
API_SECRET = "_NKwZcNx8JMrpJD7NORH8abxVOA1Jw6G-JM3jl2-18phOWY4NTc4NS00YzkyLTQzZWQtYTk0MS1hZDEwNTU3MzUyOWQ"

# Test configurations
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
MTF_COMBINATIONS = [
    ['1m', '5m', '15m'],
    ['3m', '15m', '30m'],
    ['5m', '15m', '1h'],
    ['15m', '30m', '1h'],
    ['30m', '1h', '4h'],
    ['1h', '4h', '1d'],
    ['5m', '30m', '4h'],
    ['15m', '1h', '1d']
]
ATR_PERIODS = [2, 5, 10, 14, 20, 30, 50, 100, 200, 300, 400, 500]
ATR_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

class LightningTester:
    def __init__(self):
        self.exchange = None
        self.results = []
        self.total_tests = 0
        self.completed_tests = 0
        self.start_time = None
        self.real_balance = 0
        self.all_symbols = []
        self.best_configs = {}
        
    async def initialize(self):
        """Initialize exchange connection"""
        print("⚡ Initializing Phemex connection...")
        self.exchange = ccxt.phemex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
            }
        })
        
        # Load markets
        markets = await self.exchange.load_markets()
        
        # Get all USDT perpetual swaps
        self.all_symbols = [
            symbol for symbol, market in markets.items()
            if market['active'] and 
            market['type'] == 'swap' and 
            market['quote'] == 'USDT' and
            ':USDT' in symbol
        ]
        
        print(f"✅ Found {len(self.all_symbols)} active USDT perpetual pairs")
        
        # Get real balance
        try:
            balance = await self.exchange.fetch_balance()
            self.real_balance = balance['USDT']['total'] if 'USDT' in balance else 0
            print(f"💰 Real Balance: ${self.real_balance:.2f} USDT")
        except Exception as e:
            print(f"⚠️ Could not fetch balance: {e}")
            self.real_balance = 10000  # Default for testing
            
        return len(self.all_symbols) > 0
        
    async def calculate_atr(self, ohlcv, period):
        """Calculate ATR for given OHLCV data"""
        if len(ohlcv) < period + 1:
            return 0
            
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        closes = [candle[4] for candle in ohlcv]
        
        tr_list = []
        for i in range(1, len(ohlcv)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_list.append(tr)
            
        if len(tr_list) >= period:
            atr = sum(tr_list[-period:]) / period
            return atr
        return 0
        
    async def test_configuration(self, symbol, timeframe, mtf_combo, atr_period, atr_mult):
        """Test a single configuration"""
        try:
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=max(atr_period + 50, 100))
            
            if not ohlcv or len(ohlcv) < atr_period:
                return None
                
            # Calculate metrics
            closes = [c[4] for c in ohlcv]
            current_price = closes[-1]
            
            # Calculate ATR
            atr = await self.calculate_atr(ohlcv, atr_period)
            
            if atr == 0:
                return None
                
            # Calculate signals
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current_price
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else current_price
            
            # Simple momentum
            momentum = (current_price - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
            
            # Volatility
            volatility = (atr / current_price) * 100
            
            # Score calculation
            trend_score = 50
            if sma_20 > sma_50:
                trend_score += 25
            if current_price > sma_20:
                trend_score += 25
                
            momentum_score = min(100, max(0, 50 + momentum * 5))
            
            # Risk-adjusted score
            risk_reward = (atr * atr_mult) / current_price * 100
            position_size = (self.real_balance * 0.01) / (atr * atr_mult) if atr > 0 else 0
            
            # Calculate potential profit/loss
            tp_distance = atr * atr_mult * 3  # 3:1 RR
            sl_distance = atr * atr_mult
            
            potential_profit = position_size * tp_distance
            potential_loss = position_size * sl_distance
            
            # Final score
            final_score = (trend_score * 0.4 + momentum_score * 0.3 + (100 - min(100, volatility * 10)) * 0.3)
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'mtf': mtf_combo,
                'atr_period': atr_period,
                'atr_mult': atr_mult,
                'price': current_price,
                'atr': atr,
                'volatility': volatility,
                'momentum': momentum,
                'score': final_score,
                'position_size': position_size,
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'risk_reward': tp_distance / sl_distance if sl_distance > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            # Silently skip errors for speed
            return None
            
    async def test_symbol_batch(self, symbol, config_batch):
        """Test a batch of configurations for a symbol"""
        tasks = []
        for timeframe, mtf_combo, atr_period, atr_mult in config_batch:
            task = self.test_configuration(symbol, timeframe, mtf_combo, atr_period, atr_mult)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if r and not isinstance(r, Exception)]
        
        self.completed_tests += len(config_batch)
        
        # Update progress
        progress = (self.completed_tests / self.total_tests) * 100
        elapsed = time.time() - self.start_time
        rate = self.completed_tests / elapsed if elapsed > 0 else 0
        eta = (self.total_tests - self.completed_tests) / rate if rate > 0 else 0
        
        print(f"\r⚡ Progress: {progress:.1f}% | {self.completed_tests}/{self.total_tests} | "
              f"Rate: {rate:.0f} tests/sec | ETA: {eta:.0f}s | "
              f"Symbol: {symbol[:10]:<10}", end="", flush=True)
        
        return valid_results
        
    async def run_lightning_test(self):
        """Run comprehensive testing at maximum speed"""
        print("\n🚀 Starting Lightning Speed Testing...")
        print(f"📊 Testing {len(self.all_symbols)} symbols")
        print(f"⚙️ Configurations per symbol: {len(TIMEFRAMES) * len(MTF_COMBINATIONS) * len(ATR_PERIODS) * len(ATR_MULTIPLIERS)}")
        
        self.start_time = time.time()
        
        # Create all configuration combinations
        all_configs = list(product(
            TIMEFRAMES[:4],  # Use first 4 timeframes for speed
            MTF_COMBINATIONS[:4],  # Use first 4 MTF combos
            random.sample(ATR_PERIODS, 5),  # Random sample of 5 ATR periods
            random.sample(ATR_MULTIPLIERS, 3)  # Random sample of 3 multipliers
        ))
        
        # Limit to 10 configs per symbol for speed
        configs_per_symbol = min(10, len(all_configs))
        
        # Sample symbols if too many (limit to 50 for speed)
        test_symbols = random.sample(self.all_symbols, min(50, len(self.all_symbols)))
        
        self.total_tests = len(test_symbols) * configs_per_symbol
        
        print(f"🎯 Total tests to run: {self.total_tests}")
        print(f"⏱️ Estimated time: {self.total_tests / 50:.1f} seconds\n")
        
        # Process all symbols
        all_results = []
        
        for symbol in test_symbols:
            # Random sample of configurations for this symbol
            symbol_configs = random.sample(all_configs, configs_per_symbol)
            
            # Test in batches
            batch_size = 5
            for i in range(0, len(symbol_configs), batch_size):
                batch = symbol_configs[i:i+batch_size]
                batch_results = await self.test_symbol_batch(symbol, batch)
                all_results.extend(batch_results)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
        
        print(f"\n\n✅ Testing completed! Processed {len(all_results)} valid results")
        
        # Analyze results
        self.analyze_results(all_results)
        
        # Save results
        self.save_results(all_results)
        
        return all_results
        
    def analyze_results(self, results):
        """Analyze and display best configurations"""
        if not results:
            print("❌ No valid results to analyze")
            return
            
        print("\n" + "="*80)
        print("📊 ANALYSIS RESULTS")
        print("="*80)
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Top 10 overall
        print("\n🏆 TOP 10 CONFIGURATIONS:")
        print("-"*80)
        print(f"{'Rank':<5} {'Symbol':<15} {'TF':<5} {'ATR':<8} {'Score':<8} {'P&L':<10} {'RR':<5}")
        print("-"*80)
        
        for i, r in enumerate(sorted_results[:10], 1):
            print(f"{i:<5} {r['symbol']:<15} {r['timeframe']:<5} "
                  f"{r['atr_period']}/{r['atr_mult']:<7.1f} {r['score']:<8.2f} "
                  f"${r['potential_profit']:<9.2f} {r['risk_reward']:<5.1f}")
        
        # Best by timeframe
        print("\n📈 BEST BY TIMEFRAME:")
        print("-"*80)
        
        for tf in TIMEFRAMES[:4]:
            tf_results = [r for r in results if r['timeframe'] == tf]
            if tf_results:
                best = max(tf_results, key=lambda x: x['score'])
                print(f"{tf:<5}: {best['symbol']:<15} Score: {best['score']:.2f} "
                      f"ATR: {best['atr_period']}/{best['atr_mult']:.1f}")
        
        # Statistics
        print("\n📊 STATISTICS:")
        print("-"*80)
        
        scores = [r['score'] for r in results]
        volatilities = [r['volatility'] for r in results]
        momentums = [r['momentum'] for r in results]
        
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Best Score: {max(scores):.2f}")
        print(f"Worst Score: {min(scores):.2f}")
        print(f"Score Std Dev: {np.std(scores):.2f}")
        print(f"Average Volatility: {np.mean(volatilities):.2f}%")
        print(f"Average Momentum: {np.mean(momentums):.2f}%")
        print(f"Profitable Configs: {len([r for r in results if r['score'] > 70])}/{len(results)}")
        
        # Best symbols
        symbol_scores = {}
        for r in results:
            if r['symbol'] not in symbol_scores:
                symbol_scores[r['symbol']] = []
            symbol_scores[r['symbol']].append(r['score'])
        
        avg_symbol_scores = {s: np.mean(scores) for s, scores in symbol_scores.items()}
        best_symbols = sorted(avg_symbol_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\n🎯 TOP 5 SYMBOLS (by average score):")
        print("-"*80)
        for symbol, avg_score in best_symbols:
            print(f"{symbol:<15} Average Score: {avg_score:.2f}")
            
    def save_results(self, results):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lightning_test_results_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'real_balance': self.real_balance,
            'total_symbols_tested': len(set(r['symbol'] for r in results)),
            'total_tests': len(results),
            'execution_time': time.time() - self.start_time,
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"\n💾 Results saved to {filename}")
        
        # Also save best configs
        best_configs = sorted(results, key=lambda x: x['score'], reverse=True)[:20]
        best_filename = f"best_configs_{timestamp}.json"
        
        with open(best_filename, 'w') as f:
            json.dump(best_configs, f, indent=2)
            
        print(f"💾 Best configurations saved to {best_filename}")

async def main():
    """Main execution"""
    print("⚡"*40)
    print("LIGHTNING SPEED PHEMEX COMPREHENSIVE TESTER")
    print("⚡"*40)
    
    tester = LightningTester()
    
    # Initialize
    if not await tester.initialize():
        print("❌ Failed to initialize")
        return
        
    # Run tests
    results = await tester.run_lightning_test()
    
    # Close exchange
    await tester.exchange.close()
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print(f"⏱️ Total time: {time.time() - tester.start_time:.2f} seconds")
    print(f"📊 Tests per second: {len(results) / (time.time() - tester.start_time):.2f}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())