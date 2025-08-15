#!/usr/bin/env python3
"""
COMPREHENSIVE TIMEFRAME OPTIMIZATION TEST
Tests 1m, 2m, 3m, 5m, 10m, 15m, 30m timeframes using real OHLCV data
Analyzes 20 top liquid USDT pairs for best entry performance
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import statistics

# Set Windows event loop policy
try:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import ccxt.async_support as ccxt

# Import our bot modules
from bot.strategy.pr import compute_predictive_ranges
from bot.strategy.score import ScoreInputs, compute_total_score
from bot.config import settings

@dataclass
class TimeframeResults:
    timeframe: str
    total_trades: int
    wins: int
    losses: int
    avg_pnl_pct: float
    win_rate: float
    avg_hold_bars: float
    signals_per_hour: float
    quality_score: float  # Combined metric

class TimeframeOptimizer:
    def __init__(self):
        self.timeframes = ["1m", "2m", "3m", "5m", "10m", "15m", "30m"]
        self.top_pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "AVAX/USDT:USDT", "ADA/USDT:USDT",
            "DOGE/USDT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT", "LTC/USDT:USDT", "ATOM/USDT:USDT",
            "LINK/USDT:USDT", "UNI/USDT:USDT", "XRP/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
            "APT/USDT:USDT", "SUI/USDT:USDT", "FTM/USDT:USDT", "NEAR/USDT:USDT", "ICP/USDT:USDT"
        ]
        self.score_threshold = 95
        
    async def fetch_ohlcv(self, ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 300) -> Optional[Dict[str, List[float]]]:
        """Fetch OHLCV data with error handling"""
        try:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if len(ohlcv) < 200:
                return None
                
            return {
                "open": [float(x[1]) for x in ohlcv],
                "high": [float(x[2]) for x in ohlcv],
                "low": [float(x[3]) for x in ohlcv],
                "close": [float(x[4]) for x in ohlcv]
            }
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def calculate_signals_and_trades(self, ohlcv: Dict[str, List[float]], timeframe: str) -> Dict[str, any]:
        """Calculate signals and simulate trades for given OHLCV data"""
        try:
            high = ohlcv["high"]
            low = ohlcv["low"]
            close = ohlcv["close"]
            open_ = ohlcv["open"]
            
            trades = []
            signals = 0
            
            # Start analysis from bar 50 onwards (enough history)
            for i in range(50, len(close) - 2):
                price = close[i]
                prev_close = close[i-1]
                
                # Compute PR levels
                avg, r1, r2, s1, s2 = compute_predictive_ranges(
                    high[:i+1], low[:i+1], close[:i+1], 
                    min(settings.pr_atr_len, i), settings.pr_atr_mult
                )
                
                if avg == 0:
                    continue
                    
                # Trend analysis
                sma = sum(close[max(0, i-settings.trend_len):i+1]) / min(settings.trend_len, i+1)
                trend_ok_long = price > sma
                trend_ok_short = price < sma
                
                # Entry conditions (simplified)
                base_long = price > avg and price < r1 and prev_close <= avg
                base_short = price < avg and price > s1 and prev_close >= avg
                
                # Calculate scores
                range_size = abs(r1 - s1)
                if range_size > 0:
                    long_bounce_prob = max(0, 0.9 - (abs(price - s1) / range_size))
                    short_bounce_prob = max(0, 0.9 - (abs(price - r1) / range_size))
                else:
                    long_bounce_prob = 0.5
                    short_bounce_prob = 0.5
                
                trend_strength = abs(price - sma) / sma if sma > 0 else 0
                trend_conf = min(0.8, trend_strength * 10)
                
                # Score inputs
                si_long = ScoreInputs(
                    avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
                    close=price, open=open_[i],
                    bounce_prob=long_bounce_prob,
                    bias_up_conf=trend_conf if trend_ok_long else 0.2,
                    bias_dn_conf=0.0
                )
                
                si_short = ScoreInputs(
                    avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
                    close=price, open=open_[i],
                    bounce_prob=short_bounce_prob,
                    bias_up_conf=0.0,
                    bias_dn_conf=trend_conf if trend_ok_short else 0.2
                )
                
                long_score = compute_total_score(si_long, "long", self.score_threshold)
                short_score = compute_total_score(si_short, "short", self.score_threshold)
                
                # Check for signals
                if long_score >= self.score_threshold and base_long and trend_ok_long:
                    signals += 1
                    # Simulate trade
                    trades.append(self._simulate_trade(i, "long", price, r1, r2, s1, s2, high, low, close))
                    
                elif short_score >= self.score_threshold and base_short and trend_ok_short:
                    signals += 1
                    # Simulate trade
                    trades.append(self._simulate_trade(i, "short", price, r1, r2, s1, s2, high, low, close))
            
            # Calculate metrics
            if not trades:
                return {
                    "trades": 0, "wins": 0, "losses": 0, "avg_pnl": 0.0, 
                    "win_rate": 0.0, "avg_hold": 0.0, "signals": signals
                }
            
            wins = sum(1 for t in trades if t["pnl"] > 0)
            losses = len(trades) - wins
            avg_pnl = statistics.mean([t["pnl"] for t in trades])
            win_rate = wins / len(trades) * 100
            avg_hold = statistics.mean([t["hold_bars"] for t in trades])
            
            return {
                "trades": len(trades), "wins": wins, "losses": losses, 
                "avg_pnl": avg_pnl, "win_rate": win_rate, "avg_hold": avg_hold, 
                "signals": signals
            }
            
        except Exception as e:
            print(f"Error in signal calculation: {e}")
            return {"trades": 0, "wins": 0, "losses": 0, "avg_pnl": 0.0, "win_rate": 0.0, "avg_hold": 0.0, "signals": 0}
    
    def _simulate_trade(self, entry_idx: int, side: str, entry_price: float, r1: float, r2: float, s1: float, s2: float, 
                       high: List[float], low: List[float], close: List[float]) -> Dict[str, any]:
        """Simulate a single trade with bracket orders"""
        
        if side == "long":
            tp1_price = r1
            tp2_price = r2 if r2 > r1 else r1 * 1.01
            sl_price = s2 if s2 < entry_price else s1
        else:
            tp1_price = s1
            tp2_price = s2 if s2 < s1 else s1 * 0.99
            sl_price = r2 if r2 > entry_price else r1
        
        # Look for exit within next 100 bars
        for j in range(entry_idx + 1, min(entry_idx + 100, len(high))):
            bar_high = high[j]
            bar_low = low[j]
            
            if side == "long":
                if bar_low <= sl_price:
                    # Hit SL
                    pnl_pct = (sl_price - entry_price) / entry_price * 100
                    return {"pnl": pnl_pct, "exit_type": "SL", "hold_bars": j - entry_idx}
                elif bar_high >= tp2_price:
                    # Hit TP2 - full target
                    pnl_pct = (tp2_price - entry_price) / entry_price * 100
                    return {"pnl": pnl_pct, "exit_type": "TP2", "hold_bars": j - entry_idx}
                elif bar_high >= tp1_price:
                    # Hit TP1 - partial target
                    pnl_pct = (tp1_price - entry_price) / entry_price * 100 * 0.5  # 50% position
                    return {"pnl": pnl_pct, "exit_type": "TP1", "hold_bars": j - entry_idx}
            else:  # short
                if bar_high >= sl_price:
                    # Hit SL
                    pnl_pct = (entry_price - sl_price) / entry_price * 100
                    return {"pnl": pnl_pct, "exit_type": "SL", "hold_bars": j - entry_idx}
                elif bar_low <= tp2_price:
                    # Hit TP2 - full target
                    pnl_pct = (entry_price - tp2_price) / entry_price * 100
                    return {"pnl": pnl_pct, "exit_type": "TP2", "hold_bars": j - entry_idx}
                elif bar_low <= tp1_price:
                    # Hit TP1 - partial target
                    pnl_pct = (entry_price - tp1_price) / entry_price * 100 * 0.5  # 50% position
                    return {"pnl": pnl_pct, "exit_type": "TP1", "hold_bars": j - entry_idx}
        
        # No exit found - assume small loss
        return {"pnl": -0.5, "exit_type": "TIMEOUT", "hold_bars": 100}
    
    def _calculate_signals_per_hour(self, signals: int, bars: int, timeframe: str) -> float:
        """Calculate signals per hour based on timeframe"""
        tf_minutes = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30}
        minutes_per_bar = tf_minutes.get(timeframe, 5)
        total_hours = (bars * minutes_per_bar) / 60
        return signals / max(total_hours, 1)
    
    def _calculate_quality_score(self, win_rate: float, avg_pnl: float, signals_per_hour: float, avg_hold: float) -> float:
        """Calculate composite quality score for timeframe"""
        # Normalize metrics and combine
        win_rate_norm = min(win_rate / 70, 1.0)  # Target 70% win rate
        pnl_norm = max(min(avg_pnl / 2.0, 1.0), -1.0)  # Target 2% avg profit
        frequency_norm = min(signals_per_hour / 2.0, 1.0)  # Target 2 signals/hour
        hold_norm = max(1.0 - (avg_hold / 50), 0.0)  # Prefer shorter holds
        
        # Weighted combination
        quality = (win_rate_norm * 0.3 + pnl_norm * 0.4 + frequency_norm * 0.2 + hold_norm * 0.1) * 100
        return max(quality, 0)
    
    async def test_timeframe(self, ex: ccxt.Exchange, timeframe: str) -> TimeframeResults:
        """Test a single timeframe across all pairs"""
        print(f"\n📊 Testing {timeframe} timeframe...")
        
        all_trades = []
        all_signals = []
        total_bars = 0
        
        tested_pairs = 0
        for symbol in self.top_pairs:
            try:
                ohlcv = await self.fetch_ohlcv(ex, symbol, timeframe)
                if not ohlcv:
                    continue
                
                results = self.calculate_signals_and_trades(ohlcv, timeframe)
                if results["trades"] > 0:
                    all_trades.extend([{"pnl": results["avg_pnl"]}] * results["trades"])
                    
                all_signals.append(results["signals"])
                total_bars += len(ohlcv["close"])
                tested_pairs += 1
                
                print(f"  {symbol:15} Trades:{results['trades']:3d} Win:{results['win_rate']:5.1f}% PnL:{results['avg_pnl']:6.2f}%")
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"  {symbol:15} ERROR: {e}")
                continue
        
        # Calculate aggregate metrics
        if not all_trades:
            return TimeframeResults(
                timeframe=timeframe, total_trades=0, wins=0, losses=0,
                avg_pnl_pct=0.0, win_rate=0.0, avg_hold_bars=0.0,
                signals_per_hour=0.0, quality_score=0.0
            )
        
        total_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t["pnl"] > 0)
        losses = total_trades - wins
        avg_pnl = statistics.mean([t["pnl"] for t in all_trades])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        total_signals = sum(all_signals)
        signals_per_hour = self._calculate_signals_per_hour(total_signals, total_bars, timeframe)
        avg_hold_bars = 25  # Estimated average
        
        quality_score = self._calculate_quality_score(win_rate, avg_pnl, signals_per_hour, avg_hold_bars)
        
        return TimeframeResults(
            timeframe=timeframe,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            avg_pnl_pct=avg_pnl,
            win_rate=win_rate,
            avg_hold_bars=avg_hold_bars,
            signals_per_hour=signals_per_hour,
            quality_score=quality_score
        )
    
    async def run_optimization(self):
        """Run complete timeframe optimization"""
        print("🔍 TIMEFRAME OPTIMIZATION TEST")
        print("=" * 60)
        print(f"Testing timeframes: {', '.join(self.timeframes)}")
        print(f"Score threshold: {self.score_threshold}+")
        print(f"Testing {len(self.top_pairs)} major USDT pairs")
        print("=" * 60)
        
        # Initialize exchange
        ex = ccxt.phemex({
            'apiKey': settings.phemex_api_key,
            'secret': settings.phemex_api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
        })
        
        try:
            await ex.load_markets()
            
            results = []
            for timeframe in self.timeframes:
                result = await self.test_timeframe(ex, timeframe)
                results.append(result)
                
                print(f"\n📈 {timeframe} SUMMARY:")
                print(f"   Trades: {result.total_trades}")
                print(f"   Win Rate: {result.win_rate:.1f}%")
                print(f"   Avg PnL: {result.avg_pnl_pct:.2f}%")
                print(f"   Signals/Hour: {result.signals_per_hour:.2f}")
                print(f"   Quality Score: {result.quality_score:.1f}/100")
            
            # Final analysis
            print("\n" + "=" * 60)
            print("🏆 FINAL TIMEFRAME RANKING")
            print("=" * 60)
            
            # Sort by quality score
            results.sort(key=lambda x: x.quality_score, reverse=True)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.timeframe:4s} Quality:{result.quality_score:5.1f} "
                      f"WinRate:{result.win_rate:5.1f}% PnL:{result.avg_pnl_pct:6.2f}% "
                      f"Signals/h:{result.signals_per_hour:4.1f}")
            
            best_timeframe = results[0]
            print(f"\n🥇 BEST TIMEFRAME: {best_timeframe.timeframe}")
            print(f"   Quality Score: {best_timeframe.quality_score:.1f}/100")
            print(f"   Win Rate: {best_timeframe.win_rate:.1f}%")
            print(f"   Average PnL: {best_timeframe.avg_pnl_pct:.2f}%")
            print(f"   Signals per Hour: {best_timeframe.signals_per_hour:.2f}")
            
            return results
            
        finally:
            await ex.close()

async def main():
    optimizer = TimeframeOptimizer()
    results = await optimizer.run_optimization()
    
    # Save results
    results_data = []
    for r in results:
        results_data.append({
            "timeframe": r.timeframe,
            "total_trades": r.total_trades,
            "wins": r.wins,
            "losses": r.losses,
            "avg_pnl_pct": r.avg_pnl_pct,
            "win_rate": r.win_rate,
            "avg_hold_bars": r.avg_hold_bars,
            "signals_per_hour": r.signals_per_hour,
            "quality_score": r.quality_score
        })
    
    with open("timeframe_optimization_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to timeframe_optimization_results.json")

if __name__ == "__main__":
    asyncio.run(main())
