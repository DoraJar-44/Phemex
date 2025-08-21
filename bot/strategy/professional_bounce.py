#!/usr/bin/env python3
"""
PROFESSIONAL BOUNCE STRATEGY - SMART MONEY EDITION
Combines Predictive Ranges with Smart Money Concepts for institutional-grade trading
Author: Advanced Trading Systems
Platform: Bitget Futures (Swaps)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from .pr import compute_predictive_ranges, compute_atr
from .score import ScoreInputs, compute_total_score


@dataclass
class OrderBlock:
    """Represents an institutional order block (demand/supply zone)"""
    price_high: float
    price_low: float
    volume: float
    direction: str  # "bullish" or "bearish"
    strength: float  # 0-100
    timestamp: int


@dataclass
class LiquidityZone:
    """Represents areas where retail stops cluster (equal highs/lows)"""
    price: float
    zone_type: str  # "equal_highs" or "equal_lows"
    touches: int
    last_touch: int
    strength: float  # 0-100


@dataclass
class MarketStructure:
    """Tracks overall market structure and trend"""
    trend_direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-100
    last_swing_high: float
    last_swing_low: float
    structure_broken: bool


@dataclass
class VolumeProfile:
    """Volume analysis for institutional activity detection"""
    current_volume: float
    avg_volume: float
    volume_ratio: float  # current/average
    is_spike: bool
    institutional_activity: bool


@dataclass
class ConfluenceFactors:
    """Six confluence factors for professional bounce signals"""
    ma_support: bool = False  # Price bouncing off moving averages
    rsi_oversold: bool = False  # RSI oversold with momentum shift
    volume_spike: bool = False  # Institutional volume activity
    bullish_pattern: bool = False  # Hammer, engulfing, doji patterns
    support_level: bool = False  # Price at historical support
    market_structure: bool = False  # Favorable trend context
    
    def count_factors(self) -> int:
        """Count how many confluence factors are present"""
        return sum([
            self.ma_support,
            self.rsi_oversold, 
            self.volume_spike,
            self.bullish_pattern,
            self.support_level,
            self.market_structure
        ])
    
    def get_score(self) -> float:
        """Calculate confluence score (0-100)"""
        count = self.count_factors()
        return (count / 6.0) * 100.0


class ProfessionalBounceStrategy:
    """Professional bounce strategy with smart money concepts"""
    
    def __init__(self, 
                 atr_length: int = 50,
                 atr_multiplier: float = 5.0,
                 ma_periods: List[int] = [21, 50, 200],
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 volume_period: int = 20,
                 volume_spike_threshold: float = 1.5,
                 min_confluence_factors: int = 3,
                 order_block_lookback: int = 10,
                 liquidity_lookback: int = 20):
        
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.ma_periods = ma_periods
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.volume_period = volume_period
        self.volume_spike_threshold = volume_spike_threshold
        self.min_confluence_factors = min_confluence_factors
        self.order_block_lookback = order_block_lookback
        self.liquidity_lookback = liquidity_lookback
    
    def detect_order_blocks(self, 
                           high: List[float], 
                           low: List[float], 
                           close: List[float], 
                           volume: List[float],
                           lookback: int = None) -> List[OrderBlock]:
        """Detect institutional order blocks (demand/supply zones)"""
        if lookback is None:
            lookback = self.order_block_lookback
            
        order_blocks = []
        
        for i in range(lookback, len(close) - 3):
            # Look for strong moves that create imbalances
            price_change = abs(close[i] - close[i-1]) / close[i-1]
            volume_ratio = volume[i] / np.mean(volume[max(0, i-20):i]) if i >= 20 else 1.0
            
            # Criteria for order block
            if price_change > 0.02 and volume_ratio > 1.5:  # 2% move with 1.5x volume
                # Determine if bullish or bearish order block
                if close[i] > close[i-1]:  # Bullish move
                    # Order block is the consolidation before the move
                    block_high = max(high[max(0, i-5):i])
                    block_low = min(low[max(0, i-5):i])
                    direction = "bullish"
                else:  # Bearish move
                    block_high = max(high[max(0, i-5):i])
                    block_low = min(low[max(0, i-5):i])
                    direction = "bearish"
                
                # Calculate order block strength based on volume and price action
                strength = min(100.0, (volume_ratio * 30) + (price_change * 1000))
                
                order_blocks.append(OrderBlock(
                    price_high=block_high,
                    price_low=block_low,
                    volume=volume[i],
                    direction=direction,
                    strength=strength,
                    timestamp=i
                ))
        
        return order_blocks
    
    def detect_liquidity_zones(self, 
                              high: List[float], 
                              low: List[float],
                              lookback: int = None) -> List[LiquidityZone]:
        """Detect liquidity zones (equal highs/lows where stops cluster)"""
        if lookback is None:
            lookback = self.liquidity_lookback
            
        liquidity_zones = []
        tolerance = 0.001  # 0.1% tolerance for "equal" levels
        
        # Find equal highs
        for i in range(lookback, len(high) - 1):
            current_high = high[i]
            touches = 1
            
            # Look for similar highs within tolerance
            for j in range(max(0, i-lookback), i):
                if abs(high[j] - current_high) / current_high <= tolerance:
                    touches += 1
            
            if touches >= 3:  # At least 3 touches to be significant
                strength = min(100.0, touches * 20)
                liquidity_zones.append(LiquidityZone(
                    price=current_high,
                    zone_type="equal_highs",
                    touches=touches,
                    last_touch=i,
                    strength=strength
                ))
        
        # Find equal lows
        for i in range(lookback, len(low) - 1):
            current_low = low[i]
            touches = 1
            
            # Look for similar lows within tolerance
            for j in range(max(0, i-lookback), i):
                if abs(low[j] - current_low) / current_low <= tolerance:
                    touches += 1
            
            if touches >= 3:  # At least 3 touches to be significant
                strength = min(100.0, touches * 20)
                liquidity_zones.append(LiquidityZone(
                    price=current_low,
                    zone_type="equal_lows", 
                    touches=touches,
                    last_touch=i,
                    strength=strength
                ))
        
        return liquidity_zones
    
    def analyze_market_structure(self, 
                                high: List[float], 
                                low: List[float], 
                                close: List[float],
                                lookback: int = 50) -> MarketStructure:
        """Analyze market structure for trend direction and strength"""
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(high) - 5):
            # Swing high: higher than 5 bars before and after
            if all(high[i] > high[j] for j in range(i-5, i)) and all(high[i] > high[j] for j in range(i+1, i+6)):
                swing_highs.append((i, high[i]))
            
            # Swing low: lower than 5 bars before and after
            if all(low[i] < low[j] for j in range(i-5, i)) and all(low[i] < low[j] for j in range(i+1, i+6)):
                swing_lows.append((i, low[i]))
        
        # Determine trend based on recent swing points
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            
            # Check for higher highs and higher lows (bullish structure)
            higher_highs = all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, len(recent_highs)))
            higher_lows = all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, len(recent_lows)))
            
            # Check for lower highs and lower lows (bearish structure)
            lower_highs = all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, len(recent_highs)))
            lower_lows = all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, len(recent_lows)))
            
            if higher_highs and higher_lows:
                trend_direction = "bullish"
                strength = 80.0
            elif lower_highs and lower_lows:
                trend_direction = "bearish"
                strength = 80.0
            else:
                trend_direction = "neutral"
                strength = 40.0
                
            last_swing_high = swing_highs[-1][1] if swing_highs else close[-1]
            last_swing_low = swing_lows[-1][1] if swing_lows else close[-1]
            
            # Check if structure is broken (price breaks recent swing points)
            current_price = close[-1]
            structure_broken = (trend_direction == "bullish" and current_price < last_swing_low) or \
                             (trend_direction == "bearish" and current_price > last_swing_high)
        else:
            trend_direction = "neutral"
            strength = 30.0
            last_swing_high = max(high[-lookback:])
            last_swing_low = min(low[-lookback:])
            structure_broken = False
        
        return MarketStructure(
            trend_direction=trend_direction,
            strength=strength,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            structure_broken=structure_broken
        )
    
    def analyze_volume_profile(self, 
                              volume: List[float], 
                              period: int = None) -> VolumeProfile:
        """Analyze volume for institutional activity"""
        if period is None:
            period = self.volume_period
            
        current_volume = volume[-1]
        avg_volume = np.mean(volume[-period:]) if len(volume) >= period else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike detection
        is_spike = volume_ratio >= self.volume_spike_threshold
        
        # Institutional activity (high volume + sustained activity)
        recent_high_volume_bars = sum(1 for v in volume[-5:] if v > avg_volume * 1.2)
        institutional_activity = is_spike and recent_high_volume_bars >= 3
        
        return VolumeProfile(
            current_volume=current_volume,
            avg_volume=avg_volume,
            volume_ratio=volume_ratio,
            is_spike=is_spike,
            institutional_activity=institutional_activity
        )
    
    def calculate_rsi(self, close: List[float], period: int = None) -> List[float]:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
            
        if len(close) < period + 1:
            return [50.0] * len(close)
        
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Wilder's smoothing
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rs_values = []
        for i in range(period, len(deltas)):
            if i == period:
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
            else:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
            
            rsi = 100 - (100 / (1 + rs))
            rs_values.append(rsi)
        
        # Pad with initial RSI value
        return [50.0] * (period + 1) + rs_values
    
    def calculate_moving_averages(self, close: List[float]) -> Dict[str, List[float]]:
        """Calculate multiple EMAs for confluence analysis"""
        mas = {}
        
        for period in self.ma_periods:
            if len(close) < period:
                mas[f"ema_{period}"] = close.copy()
                continue
                
            ema = [close[0]]  # Start with first price
            multiplier = 2.0 / (period + 1)
            
            for i in range(1, len(close)):
                ema_value = (close[i] * multiplier) + (ema[-1] * (1 - multiplier))
                ema.append(ema_value)
            
            mas[f"ema_{period}"] = ema
        
        return mas
    
    def detect_reversal_patterns(self, 
                                open_prices: List[float],
                                high: List[float], 
                                low: List[float], 
                                close: List[float]) -> bool:
        """Detect bullish reversal patterns (hammer, engulfing, doji)"""
        if len(close) < 3:
            return False
        
        # Current and previous candle
        o0, h0, l0, c0 = open_prices[-1], high[-1], low[-1], close[-1]
        o1, h1, l1, c1 = open_prices[-2], high[-2], low[-2], close[-2]
        
        body_0 = abs(c0 - o0)
        body_1 = abs(c1 - o1)
        range_0 = h0 - l0
        range_1 = h1 - l1
        
        # Hammer pattern
        lower_shadow = min(o0, c0) - l0
        upper_shadow = h0 - max(o0, c0)
        hammer = (lower_shadow > 2 * body_0) and (upper_shadow < body_0 * 0.5) and (c0 > o0)
        
        # Bullish engulfing
        engulfing = (c1 < o1) and (c0 > o0) and (c0 > o1) and (o0 < c1) and (body_0 > body_1)
        
        # Doji (indecision that often precedes reversal)
        doji = body_0 < (range_0 * 0.1) and range_0 > 0
        
        return hammer or engulfing or doji
    
    def check_support_level_interaction(self, 
                                      close: List[float], 
                                      price: float,
                                      order_blocks: List[OrderBlock]) -> bool:
        """Check if price is interacting with significant support levels"""
        tolerance = 0.002  # 0.2% tolerance
        
        # Check interaction with order blocks
        for block in order_blocks:
            if block.direction == "bullish":  # Demand zone
                if (block.price_low <= price <= block.price_high * (1 + tolerance)) and \
                   block.strength > 60:
                    return True
        
                     # Check historical support levels
             recent_lows = []
             if len(close) > 100:  # Add safety check
                 for i in range(max(0, len(close) - 100), len(close)):
                     if i >= 2 and i < len(close) - 2:
                         if len(low) > i+2:  # Ensure we have enough data
                             if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                                 recent_lows.append(low[i])
        
        # Check if current price is near any significant low
        for support_level in recent_lows:
            if abs(price - support_level) / price <= tolerance:
                return True
        
        return False
    
    def calculate_confluence_factors(self,
                                   open_prices: List[float],
                                   high: List[float],
                                   low: List[float], 
                                   close: List[float],
                                   volume: List[float],
                                   price: float,
                                   avg: float,
                                   r1: float,
                                   s1: float) -> ConfluenceFactors:
        """Calculate all six confluence factors for professional analysis"""
        
        # 1. MA Support - Check if price is bouncing off moving averages
        mas = self.calculate_moving_averages(close)
        ma_support = False
        for ma_name, ma_values in mas.items():
            if len(ma_values) > 0:
                ma_price = ma_values[-1]
                # Price bouncing off MA (within 0.5% tolerance)
                if abs(price - ma_price) / price <= 0.005 and price > ma_price * 0.999:
                    ma_support = True
                    break
        
        # 2. RSI Oversold - RSI below threshold with momentum shift
        rsi_values = self.calculate_rsi(close)
        rsi_oversold = False
        if len(rsi_values) >= 2:
            current_rsi = rsi_values[-1]
            prev_rsi = rsi_values[-2]
            rsi_oversold = (current_rsi < self.rsi_oversold) and (current_rsi > prev_rsi)
        
        # 3. Volume Spike - Institutional volume activity
        volume_profile = self.analyze_volume_profile(volume)
        volume_spike = volume_profile.is_spike or volume_profile.institutional_activity
        
        # 4. Bullish Reversal Pattern - Candlestick patterns
        bullish_pattern = self.detect_reversal_patterns(open_prices, high, low, close)
        
        # 5. Support Level - Price at historical support or order blocks
        order_blocks = self.detect_order_blocks(high, low, close, volume)
        support_level = self.check_support_level_interaction(close, price, order_blocks)
        
        # 6. Market Structure - Favorable trend context
        market_structure = self.analyze_market_structure(high, low, close)
        favorable_structure = (market_structure.trend_direction == "bullish" or 
                             market_structure.trend_direction == "neutral") and \
                             not market_structure.structure_broken
        
        return ConfluenceFactors(
            ma_support=ma_support,
            rsi_oversold=rsi_oversold,
            volume_spike=volume_spike,
            bullish_pattern=bullish_pattern,
            support_level=support_level,
            market_structure=favorable_structure
        )
    
    def generate_professional_signals(self,
                                    open_prices: List[float],
                                    high: List[float],
                                    low: List[float],
                                    close: List[float],
                                    volume: List[float]) -> List[Dict[str, Any]]:
        """Generate professional bounce signals with full analysis"""
        
        if len(close) < max(self.atr_length, max(self.ma_periods)) + 10:
            return []
        
        signals = []
        
        # Calculate base predictive ranges
        pr_avg, pr_r1, pr_r2, pr_s1, pr_s2 = compute_predictive_ranges(
            high, low, close, self.atr_length, self.atr_multiplier
        )
        
        # Analyze current market conditions
        current_price = close[-1]
        
        # Generate long signal analysis
        if pr_s1 < current_price < pr_avg:  # In bounce zone
            confluence = self.calculate_confluence_factors(
                open_prices, high, low, close, volume, current_price, pr_avg, pr_r1, pr_s1
            )
            
            factor_count = confluence.count_factors()
            confluence_score = confluence.get_score()
            
            if factor_count >= self.min_confluence_factors:
                # Calculate enhanced score combining PR score with confluence
                pr_score_inputs = ScoreInputs(
                    avg=pr_avg, r1=pr_r1, r2=pr_r2, s1=pr_s1, s2=pr_s2,
                    close=current_price, open=open_prices[-1]
                )
                base_score = compute_total_score(pr_score_inputs, "long")
                
                # Enhance score with confluence factors
                enhanced_score = base_score + (confluence_score * 0.2)  # 20% bonus from confluence
                
                signals.append({
                    "side": "long",
                    "price": current_price,
                    "score": enhanced_score,
                    "confluence_factors": factor_count,
                    "confluence_score": confluence_score,
                    "pr_levels": {
                        "avg": pr_avg,
                        "r1": pr_r1,
                        "r2": pr_r2,
                        "s1": pr_s1,
                        "s2": pr_s2
                    },
                    "confluence_details": {
                        "ma_support": confluence.ma_support,
                        "rsi_oversold": confluence.rsi_oversold,
                        "volume_spike": confluence.volume_spike,
                        "bullish_pattern": confluence.bullish_pattern,
                        "support_level": confluence.support_level,
                        "market_structure": confluence.market_structure
                    },
                    "entry_price": current_price,
                    "tp1": pr_r1,
                    "tp2": pr_r2,
                    "sl": pr_s2,
                    "risk_reward": abs(pr_r1 - current_price) / abs(current_price - pr_s2)
                })
        
        # Generate short signal analysis (for completeness)
        if pr_avg < current_price < pr_r1:  # In short bounce zone
            # Similar logic for short signals
            confluence = self.calculate_confluence_factors(
                open_prices, high, low, close, volume, current_price, pr_avg, pr_r1, pr_s1
            )
            
            factor_count = confluence.count_factors()
            if factor_count >= self.min_confluence_factors:
                # For shorts, we need bearish confluence factors
                # This is simplified - you'd want specific bearish confluence logic
                pass  # Skip shorts for now since focus is on bounce (long) strategy
        
        return signals


def run_professional_backtest(strategy: ProfessionalBounceStrategy,
                            ohlcv_data: Dict[str, List[float]],
                            leverage: float = 25.0) -> Dict[str, Any]:
    """Run comprehensive backtest with the professional bounce strategy"""
    
    open_prices = ohlcv_data["open"]
    high = ohlcv_data["high"]
    low = ohlcv_data["low"]
    close = ohlcv_data["close"]
    volume = ohlcv_data["volume"]
    
    trades = []
    account_balance = 10000.0  # Starting balance
    peak_balance = account_balance
    max_drawdown = 0.0
    
    start_idx = max(strategy.atr_length, max(strategy.ma_periods)) + 20
    
    for i in range(start_idx, len(close) - 1):
        # Get data up to current point
        current_data = {
            "open": open_prices[:i+1],
            "high": high[:i+1], 
            "low": low[:i+1],
            "close": close[:i+1],
            "volume": volume[:i+1]
        }
        
        signals = strategy.generate_professional_signals(**current_data)
        
        for signal in signals:
            if signal["score"] >= 95 and signal["confluence_factors"] >= strategy.min_confluence_factors:
                # Execute trade
                entry_price = signal["entry_price"]
                tp1_price = signal["tp1"]
                tp2_price = signal["tp2"]
                sl_price = signal["sl"]
                
                # Position sizing (1% risk)
                risk_amount = account_balance * 0.01
                stop_distance = abs(entry_price - sl_price)
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                
                if position_size > 0:
                    # Check exit conditions in subsequent bars
                    for j in range(i + 1, min(i + 50, len(close))):  # Max 50 bars hold
                        exit_price = None
                        exit_reason = ""
                        
                        # Check if hit TP1 (50% position)
                        if high[j] >= tp1_price:
                            exit_price = tp1_price
                            exit_reason = "TP1"
                            break
                        # Check if hit SL
                        elif low[j] <= sl_price:
                            exit_price = sl_price
                            exit_reason = "SL"
                            break
                    
                    if exit_price:
                        # Calculate profit
                        profit_pct = ((exit_price - entry_price) / entry_price) * leverage
                        profit_amount = account_balance * (profit_pct / 100)
                        account_balance += profit_amount
                        
                        # Track drawdown
                        if account_balance > peak_balance:
                            peak_balance = account_balance
                        drawdown = ((peak_balance - account_balance) / peak_balance) * 100
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                        
                        trades.append({
                            "entry_bar": i,
                            "exit_bar": j,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "profit_pct": profit_pct,
                            "profit_amount": profit_amount,
                            "confluence_factors": signal["confluence_factors"],
                            "confluence_score": signal["confluence_score"],
                            "signal_score": signal["score"]
                        })
    
    # Calculate performance metrics
    wins = [t for t in trades if t["profit_pct"] > 0]
    losses = [t for t in trades if t["profit_pct"] <= 0]
    
    total_return = ((account_balance - 10000) / 10000) * 100
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0
    avg_profit = np.mean([t["profit_pct"] for t in trades]) if trades else 0
    profit_factor = (sum(t["profit_amount"] for t in wins) / 
                    abs(sum(t["profit_amount"] for t in losses))) if losses else float('inf')
    
    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_profit_pct": avg_profit,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": profit_factor,
        "final_balance": account_balance,
        "trades": trades[-10:] if trades else []  # Last 10 trades for review
    }


# Optimized parameter sets based on professional analysis
PROFESSIONAL_CONFIGS = [
    # High-frequency professional setups
    {"timeframe": "5m", "atr_length": 50, "atr_multiplier": 5.0, "min_confluence": 3},
    {"timeframe": "15m", "atr_length": 50, "atr_multiplier": 5.0, "min_confluence": 3},
    {"timeframe": "1h", "atr_length": 50, "atr_multiplier": 5.0, "min_confluence": 4},
    {"timeframe": "4h", "atr_length": 50, "atr_multiplier": 5.0, "min_confluence": 4},
    
    # Conservative professional setups
    {"timeframe": "1h", "atr_length": 100, "atr_multiplier": 6.0, "min_confluence": 5},
    {"timeframe": "4h", "atr_length": 100, "atr_multiplier": 6.0, "min_confluence": 5},
    
    # Ultra-selective professional setups
    {"timeframe": "4h", "atr_length": 200, "atr_multiplier": 8.0, "min_confluence": 6},
]