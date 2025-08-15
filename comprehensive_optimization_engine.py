#!/usr/bin/env python3
"""
COMPREHENSIVE TRADING BOT OPTIMIZATION ENGINE
Lightning-fast testing across all timeframes, pairs, and ATR parameters
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import itertools
import statistics
from datetime import datetime

# Windows event loop policy
try:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import ccxt.async_support as ccxt
from bot.strategy.pr import compute_predictive_ranges
from bot.strategy.score import ScoreInputs, compute_total_score
from bot.config import settings

@dataclass
class OptimizationConfig:
    timeframe: str
    symbol: str
    atr_length: int
    atr_multiplier: float
    entry_type: str  # "close_confirmed" or "wick_entry"

@dataclass
class PerformanceMetrics:
    config: OptimizationConfig
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_profit_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float
    signals_per_day: float
    avg_hold_bars: float
    total_return_pct: float
    risk_score: float  # Lower is better
    overall_score: float  # Higher is better

class ComprehensiveOptimizer:
    def __init__(self):
        self.score_threshold = 95
        self.max_workers = 8  # Parallel processing
        self.results: List[PerformanceMetrics] = []
        
        # Parameter grids
        self.atr_lengths = [20, 50, 100, 150, 200, 300, 500, 750, 1000]
        self.atr_multipliers = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
        self.entry_types = ["close_confirmed", "wick_entry"]
        
    async def discover_supported_timeframes(self, ex: ccxt.Exchange) -> List[str]:
        """Discover all Phemex supported timeframes"""
        print("🔍 Discovering supported timeframes...")
        
        test_symbol = "BTC/USDT:USDT"
        candidate_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
        supported = []
        
        for tf in candidate_timeframes:
            try:
                data = await ex.fetch_ohlcv(test_symbol, timeframe=tf, limit=5)
                if data and len(data) >= 3:
                    supported.append(tf)
                    print(f"  ✅ {tf}")
                else:
                    print(f"  ❌ {tf} - Insufficient data")
            except Exception as e:
                print(f"  ❌ {tf} - Error: {str(e)[:40]}")
        
        print(f"Found {len(supported)} supported timeframes: {supported}")
        return supported
    
    async def discover_leverage_pairs(self, ex: ccxt.Exchange, min_leverage: float = 20.0) -> List[str]:
        """Find all USDT pairs with minimum leverage support"""
        print(f"🔍 Discovering pairs with {min_leverage}x+ leverage...")
        
        markets = await ex.load_markets()
        qualified_pairs = []
        
        for symbol, market in markets.items():
            try:
                if (market.get('type') == 'swap' and 
                    market.get('quote') == 'USDT' and 
                    market.get('active') and
                    'USDT' in symbol):
                    
                    # Check leverage
                    leverage_info = market.get('limits', {}).get('leverage', {})
                    max_leverage = leverage_info.get('max', 0)
                    
                    if max_leverage >= min_leverage:
                        qualified_pairs.append(symbol)
                        
            except Exception:
                continue
        
        # Prioritize major pairs first
        major_pairs = []
        other_pairs = []
        
        priority_bases = ['BTC', 'ETH', 'SOL', 'AVAX', 'ADA', 'DOGE', 'DOT', 'MATIC', 
                         'LTC', 'ATOM', 'LINK', 'UNI', 'XRP', 'ARB', 'OP', 'APT', 
                         'SUI', 'FTM', 'NEAR', 'ICP', 'AAVE', 'MKR', 'COMP']
        
        for pair in qualified_pairs:
            base = pair.split('/')[0]
            if base in priority_bases:
                major_pairs.append(pair)
            else:
                other_pairs.append(pair)
        
        # Return major pairs first, then others (limit to 50 total for speed)
        final_pairs = (major_pairs + other_pairs)[:50]
        print(f"Found {len(final_pairs)} qualified pairs (showing first 20):")
        for i, pair in enumerate(final_pairs[:20], 1):
            print(f"  {i:2d}. {pair}")
        
        return final_pairs
    
    async def fetch_ohlcv_data(self, ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 500) -> Optional[Dict[str, List[float]]]:
        """Fetch OHLCV data with error handling"""
        try:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if len(ohlcv) < 100:  # Need sufficient history
                return None
                
            return {
                "open": [float(x[1]) for x in ohlcv],
                "high": [float(x[2]) for x in ohlcv],
                "low": [float(x[3]) for x in ohlcv],
                "close": [float(x[4]) for x in ohlcv],
                "timestamp": [int(x[0]) for x in ohlcv]
            }
        except Exception:
            return None
    
    def detect_wick_entry(self, i: int, side: str, ohlcv: Dict[str, List[float]], 
                         avg: float, r1: float, s1: float) -> bool:
        """Detect liquidation hunt wick entries"""
        if i < 2:
            return False
            
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        open_ = ohlcv["open"]
        
        current_open = open_[i]
        current_high = high[i]
        current_low = low[i]
        current_close = close[i]
        
        body_size = abs(current_close - current_open)
        
        if side == "long":
            # Look for hammer-style wick (long lower wick, small body)
            lower_wick = min(current_open, current_close) - current_low
            upper_wick = current_high - max(current_open, current_close)
            
            # Wick entry conditions for long:
            # 1. Long lower wick (at least 2x body size)
            # 2. Price drops near/below support then recovers
            # 3. Close above wick midpoint
            wick_ratio = lower_wick / max(body_size, current_close * 0.001)
            price_near_support = current_low <= s1 * 1.02  # Within 2% of support
            recovery = current_close > (current_low + lower_wick * 0.5)
            
            return wick_ratio >= 2.0 and price_near_support and recovery
            
        else:  # short
            # Look for inverted hammer (long upper wick, small body)
            upper_wick = current_high - max(current_open, current_close)
            lower_wick = min(current_open, current_close) - current_low
            
            # Wick entry conditions for short:
            # 1. Long upper wick (at least 2x body size)
            # 2. Price spikes near/above resistance then falls
            # 3. Close below wick midpoint
            wick_ratio = upper_wick / max(body_size, current_close * 0.001)
            price_near_resistance = current_high >= r1 * 0.98  # Within 2% of resistance
            rejection = current_close < (current_high - upper_wick * 0.5)
            
            return wick_ratio >= 2.0 and price_near_resistance and rejection
    
    def calculate_performance(self, config: OptimizationConfig, ohlcv: Dict[str, List[float]]) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics for a configuration"""
        try:
            high = ohlcv["high"]
            low = ohlcv["low"]
            close = ohlcv["close"]
            open_ = ohlcv["open"]
            
            trades = []
            equity_curve = []
            current_equity = 100.0  # Start with $100
            
            # Minimum bars for analysis
            min_bars = max(config.atr_length + 50, 100)
            
            for i in range(min_bars, len(close) - 5):
                price = close[i]
                prev_close = close[i-1]
                
                # Compute PR levels with config parameters
                try:
                    avg, r1, r2, s1, s2 = compute_predictive_ranges(
                        high[:i+1], low[:i+1], close[:i+1], 
                        config.atr_length, config.atr_multiplier
                    )
                except Exception:
                    continue
                
                if avg == 0:
                    continue
                
                # Trend analysis
                trend_len = min(50, i)
                sma = sum(close[i-trend_len:i+1]) / trend_len
                trend_ok_long = price > sma
                trend_ok_short = price < sma
                
                # Entry conditions based on type
                if config.entry_type == "close_confirmed":
                    # Standard close-confirmed entries (5min+ timeframes)
                    tf_minutes = self._timeframe_to_minutes(config.timeframe)
                    if tf_minutes < 5:
                        continue  # Skip sub-5min for close confirmed
                        
                    base_long = price > avg and price < r1 and prev_close <= avg
                    base_short = price < avg and price > s1 and prev_close >= avg
                    
                elif config.entry_type == "wick_entry":
                    # Wick-based liquidation hunt entries
                    base_long = self.detect_wick_entry(i, "long", ohlcv, avg, r1, s1)
                    base_short = self.detect_wick_entry(i, "short", ohlcv, avg, r1, s1)
                
                # Calculate scores (simplified for speed)
                range_size = abs(r1 - s1)
                if range_size > 0:
                    long_bounce_prob = max(0, 0.9 - (abs(price - s1) / range_size))
                    short_bounce_prob = max(0, 0.9 - (abs(price - r1) / range_size))
                else:
                    long_bounce_prob = 0.5
                    short_bounce_prob = 0.5
                
                # Quick score calculation (optimized)
                long_score = self._quick_score(price, avg, r1, s1, long_bounce_prob, trend_ok_long)
                short_score = self._quick_score(price, avg, s1, r1, short_bounce_prob, trend_ok_short)
                
                # Entry signals
                long_entry = base_long and trend_ok_long and long_score >= self.score_threshold
                short_entry = base_short and trend_ok_short and short_score >= self.score_threshold
                
                # Execute trades
                if long_entry:
                    trade_result = self._simulate_trade(i, "long", price, r1, r2, s1, s2, high, low, close)
                    if trade_result:
                        trades.append(trade_result)
                        current_equity *= (1 + trade_result["return_pct"] / 100)
                        
                elif short_entry:
                    trade_result = self._simulate_trade(i, "short", price, r1, r2, s1, s2, high, low, close)
                    if trade_result:
                        trades.append(trade_result)
                        current_equity *= (1 + trade_result["return_pct"] / 100)
                
                equity_curve.append(current_equity)
            
            # Calculate metrics
            if len(trades) < 3:  # Need minimum trades for valid analysis
                return None
            
            returns = [t["return_pct"] for t in trades]
            wins = sum(1 for r in returns if r > 0)
            losses = len(returns) - wins
            
            avg_profit = statistics.mean(returns)
            win_rate = wins / len(returns) * 100
            
            # Risk metrics
            losing_trades = [r for r in returns if r < 0]
            avg_loss = statistics.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum([r for r in returns if r > 0]) / sum(losing_trades)) if losing_trades else 999
            
            # Drawdown calculation
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            # Sharpe ratio (simplified)
            returns_std = statistics.stdev(returns) if len(returns) > 1 else 1
            sharpe_ratio = avg_profit / returns_std if returns_std > 0 else 0
            
            # Signals per day
            tf_minutes = self._timeframe_to_minutes(config.timeframe)
            total_time_days = len(close) * tf_minutes / (24 * 60)
            signals_per_day = len(trades) / max(total_time_days, 1)
            
            # Overall score calculation
            overall_score = self._calculate_overall_score(win_rate, avg_profit, profit_factor, max_drawdown, sharpe_ratio)
            
            return PerformanceMetrics(
                config=config,
                total_trades=len(trades),
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                avg_profit_pct=avg_profit,
                max_drawdown_pct=max_drawdown,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                signals_per_day=signals_per_day,
                avg_hold_bars=statistics.mean([t["hold_bars"] for t in trades]),
                total_return_pct=(current_equity - 100.0),
                risk_score=max_drawdown / max(avg_profit, 0.1),  # Risk-adjusted
                overall_score=overall_score
            )
            
        except Exception as e:
            return None
    
    def _timeframe_to_minutes(self, tf: str) -> int:
        """Convert timeframe to minutes"""
        multipliers = {"m": 1, "h": 60, "d": 1440}
        for unit, mult in multipliers.items():
            if tf.endswith(unit):
                return int(tf[:-1]) * mult
        return 5  # Default
    
    def _quick_score(self, price: float, avg: float, target: float, stop: float, bounce_prob: float, trend_ok: bool) -> int:
        """Quick score calculation optimized for speed"""
        if not trend_ok:
            return 50
            
        # Distance to target vs stop
        target_dist = abs(price - target)
        stop_dist = abs(price - stop)
        
        if target_dist < stop_dist:  # Closer to target = higher score
            base_score = 85
        else:
            base_score = 65
            
        # Add bounce probability
        bounce_bonus = int(bounce_prob * 20)
        
        return min(100, base_score + bounce_bonus)
    
    def _simulate_trade(self, entry_idx: int, side: str, entry_price: float, 
                       r1: float, r2: float, s1: float, s2: float,
                       high: List[float], low: List[float], close: List[float]) -> Optional[Dict[str, any]]:
        """Simulate trade execution with bracket orders"""
        
        if side == "long":
            tp1_price = r1
            tp2_price = r2 if r2 > r1 else r1 * 1.015
            sl_price = s2 if s2 < entry_price else s1
        else:
            tp1_price = s1
            tp2_price = s2 if s2 < s1 else s1 * 0.985
            sl_price = r2 if r2 > entry_price else r1
        
        # Look for exit within next 50 bars
        for j in range(entry_idx + 1, min(entry_idx + 50, len(high))):
            bar_high = high[j]
            bar_low = low[j]
            
            if side == "long":
                if bar_low <= sl_price:
                    return_pct = (sl_price - entry_price) / entry_price * 100
                    return {"return_pct": return_pct, "exit_type": "SL", "hold_bars": j - entry_idx}
                elif bar_high >= tp2_price:
                    return_pct = (tp2_price - entry_price) / entry_price * 100
                    return {"return_pct": return_pct, "exit_type": "TP2", "hold_bars": j - entry_idx}
                elif bar_high >= tp1_price:
                    return_pct = (tp1_price - entry_price) / entry_price * 100 * 0.6  # Partial exit
                    return {"return_pct": return_pct, "exit_type": "TP1", "hold_bars": j - entry_idx}
            else:  # short
                if bar_high >= sl_price:
                    return_pct = (entry_price - sl_price) / entry_price * 100
                    return {"return_pct": return_pct, "exit_type": "SL", "hold_bars": j - entry_idx}
                elif bar_low <= tp2_price:
                    return_pct = (entry_price - tp2_price) / entry_price * 100
                    return {"return_pct": return_pct, "exit_type": "TP2", "hold_bars": j - entry_idx}
                elif bar_low <= tp1_price:
                    return_pct = (entry_price - tp1_price) / entry_price * 100 * 0.6  # Partial exit
                    return {"return_pct": return_pct, "exit_type": "TP1", "hold_bars": j - entry_idx}
        
        # Timeout - small loss
        return {"return_pct": -0.5, "exit_type": "TIMEOUT", "hold_bars": 50}
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if len(equity_curve) < 2:
            return 0.0
            
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_overall_score(self, win_rate: float, avg_profit: float, profit_factor: float, 
                               max_drawdown: float, sharpe_ratio: float) -> float:
        """Calculate composite score for ranking configurations"""
        
        # Normalize metrics (0-100 scale)
        win_rate_score = min(win_rate, 100)
        profit_score = max(min(avg_profit * 10, 100), -100)  # Scale avg profit
        pf_score = min(profit_factor * 10, 100)  # Scale profit factor
        dd_score = max(100 - max_drawdown * 2, 0)  # Lower drawdown = higher score
        sharpe_score = max(min(sharpe_ratio * 20, 100), -100)  # Scale Sharpe
        
        # Weighted combination
        overall = (win_rate_score * 0.25 + profit_score * 0.30 + pf_score * 0.20 + 
                  dd_score * 0.15 + sharpe_score * 0.10)
        
        return max(overall, 0)

    async def run_comprehensive_optimization(self):
        """Run the complete optimization suite"""
        print("🚀 STARTING COMPREHENSIVE OPTIMIZATION")
        print("=" * 80)
        
        # Initialize exchange
        ex = ccxt.phemex({
            'apiKey': settings.phemex_api_key,
            'secret': settings.phemex_api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
        })
        
        try:
            # Discovery phase
            timeframes = await self.discover_supported_timeframes(ex)
            pairs = await self.discover_leverage_pairs(ex, 20.0)
            
            print(f"\n📊 OPTIMIZATION SCOPE:")
            print(f"   Timeframes: {len(timeframes)} ({timeframes})")
            print(f"   Pairs: {len(pairs)}")
            print(f"   ATR Lengths: {len(self.atr_lengths)} ({self.atr_lengths})")
            print(f"   ATR Multipliers: {len(self.atr_multipliers)} ({self.atr_multipliers})")
            print(f"   Entry Types: {len(self.entry_types)} ({self.entry_types})")
            
            total_combinations = (len(timeframes) * len(pairs) * len(self.atr_lengths) * 
                                len(self.atr_multipliers) * len(self.entry_types))
            print(f"   TOTAL COMBINATIONS: {total_combinations:,}")
            
            # Start optimization
            start_time = time.time()
            print(f"\n⚡ Starting lightning-speed optimization at {datetime.now().strftime('%H:%M:%S')}")
            
            # Create all configurations
            configs = []
            for tf, symbol, atr_len, atr_mult, entry_type in itertools.product(
                timeframes, pairs[:20], self.atr_lengths, self.atr_multipliers, self.entry_types):
                configs.append(OptimizationConfig(tf, symbol, atr_len, atr_mult, entry_type))
            
            print(f"Created {len(configs):,} configurations to test...")
            
            # Parallel processing
            results = []
            processed = 0
            
            # Process in batches for memory management
            batch_size = 100
            for i in range(0, len(configs), batch_size):
                batch = configs[i:i+batch_size]
                batch_results = await self._process_batch(ex, batch)
                results.extend([r for r in batch_results if r is not None])
                
                processed += len(batch)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(configs) - processed) / rate if rate > 0 else 0
                
                print(f"Progress: {processed:,}/{len(configs):,} ({processed/len(configs)*100:.1f}%) "
                      f"Rate: {rate:.1f}/s ETA: {eta/60:.1f}min")
            
            self.results = results
            
            # Analysis and ranking
            print(f"\n📊 OPTIMIZATION COMPLETE!")
            print(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")
            print(f"   Valid results: {len(results):,}")
            print(f"   Processing rate: {len(configs)/(time.time() - start_time):.1f} configs/sec")
            
            await self._analyze_and_rank_results()
            
        finally:
            await ex.close()
    
    async def _process_batch(self, ex: ccxt.Exchange, configs: List[OptimizationConfig]) -> List[Optional[PerformanceMetrics]]:
        """Process a batch of configurations in parallel"""
        
        # Group by symbol-timeframe for efficient data fetching
        data_cache = {}
        
        # Pre-fetch all required data
        for config in configs:
            cache_key = f"{config.symbol}_{config.timeframe}"
            if cache_key not in data_cache:
                data_cache[cache_key] = await self.fetch_ohlcv_data(ex, config.symbol, config.timeframe)
        
        # Process configurations
        results = []
        for config in configs:
            cache_key = f"{config.symbol}_{config.timeframe}"
            ohlcv_data = data_cache.get(cache_key)
            
            if ohlcv_data:
                result = self.calculate_performance(config, ohlcv_data)
                results.append(result)
            else:
                results.append(None)
        
        return results
    
    async def _analyze_and_rank_results(self):
        """Analyze and rank all results with detailed explanations"""
        if not self.results:
            print("❌ No valid results to analyze!")
            return
        
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE RESULTS ANALYSIS")
        print("="*80)
        
        # Sort by overall score
        sorted_results = sorted(self.results, key=lambda x: x.overall_score, reverse=True)
        
        # Top 10 configurations
        print("\n🥇 TOP 10 CONFIGURATIONS:")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:10], 1):
            config = result.config
            print(f"\n{i:2d}. {config.timeframe:4s} {config.symbol:15s} "
                  f"ATR({config.atr_length},{config.atr_multiplier:4.1f}) {config.entry_type}")
            print(f"    Overall Score: {result.overall_score:6.1f}/100")
            print(f"    Win Rate: {result.win_rate:5.1f}% | Avg Profit: {result.avg_profit_pct:6.2f}%")
            print(f"    Profit Factor: {result.profit_factor:5.2f} | Max DD: {result.max_drawdown_pct:5.1f}%")
            print(f"    Trades: {result.total_trades:3d} | Signals/Day: {result.signals_per_day:4.1f}")
            print(f"    Total Return: {result.total_return_pct:6.1f}%")
        
        # Best by category analysis
        await self._analyze_by_categories(sorted_results)
        
        # Save results
        self._save_results(sorted_results)
    
    async def _analyze_by_categories(self, sorted_results: List[PerformanceMetrics]):
        """Analyze best configurations by different categories"""
        
        print("\n📊 CATEGORY ANALYSIS:")
        print("-" * 50)
        
        # Best by timeframe
        tf_best = {}
        for result in sorted_results:
            tf = result.config.timeframe
            if tf not in tf_best:
                tf_best[tf] = result
        
        print("\n🕐 BEST BY TIMEFRAME:")
        for tf in sorted(tf_best.keys(), key=lambda x: self._timeframe_to_minutes(x)):
            result = tf_best[tf]
            print(f"  {tf:4s}: Score {result.overall_score:5.1f} | "
                  f"Win Rate {result.win_rate:5.1f}% | "
                  f"Avg Profit {result.avg_profit_pct:5.2f}%")
        
        # Best by entry type
        entry_best = {}
        for result in sorted_results:
            entry_type = result.config.entry_type
            if entry_type not in entry_best:
                entry_best[entry_type] = result
        
        print("\n🎯 BEST BY ENTRY TYPE:")
        for entry_type, result in entry_best.items():
            print(f"  {entry_type:15s}: Score {result.overall_score:5.1f} | "
                  f"Win Rate {result.win_rate:5.1f}% | "
                  f"Signals/Day {result.signals_per_day:4.1f}")
        
        # Best ATR parameters
        print("\n📈 OPTIMAL ATR PARAMETERS:")
        atr_analysis = {}
        for result in sorted_results[:50]:  # Top 50 results
            atr_key = f"{result.config.atr_length}_{result.config.atr_multiplier}"
            if atr_key not in atr_analysis:
                atr_analysis[atr_key] = []
            atr_analysis[atr_key].append(result.overall_score)
        
        atr_scores = {k: statistics.mean(v) for k, v in atr_analysis.items()}
        top_atr = sorted(atr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for atr_params, avg_score in top_atr:
            length, mult = atr_params.split('_')
            print(f"  ATR({length}, {mult}): Avg Score {avg_score:5.1f}")
    
    def _save_results(self, sorted_results: List[PerformanceMetrics]):
        """Save comprehensive results to JSON"""
        
        results_data = []
        for result in sorted_results:
            data = asdict(result)
            # Convert config to dict
            data['config'] = asdict(result.config)
            results_data.append(data)
        
        filename = f"comprehensive_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Also save top 20 as CSV for easy viewing
        import csv
        csv_filename = f"top_20_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Rank', 'Timeframe', 'Symbol', 'ATR_Length', 'ATR_Multiplier', 'Entry_Type',
                'Overall_Score', 'Win_Rate', 'Avg_Profit_Pct', 'Profit_Factor', 'Max_Drawdown_Pct',
                'Total_Trades', 'Signals_Per_Day', 'Total_Return_Pct'
            ])
            
            for i, result in enumerate(sorted_results[:20], 1):
                writer.writerow([
                    i, result.config.timeframe, result.config.symbol,
                    result.config.atr_length, result.config.atr_multiplier, result.config.entry_type,
                    f"{result.overall_score:.1f}", f"{result.win_rate:.1f}%", f"{result.avg_profit_pct:.2f}%",
                    f"{result.profit_factor:.2f}", f"{result.max_drawdown_pct:.1f}%",
                    result.total_trades, f"{result.signals_per_day:.1f}", f"{result.total_return_pct:.1f}%"
                ])
        
        print(f"💾 Top 20 results saved to: {csv_filename}")

async def main():
    optimizer = ComprehensiveOptimizer()
    await optimizer.run_comprehensive_optimization()

if __name__ == "__main__":
    asyncio.run(main())
