#!/usr/bin/env python3
"""
PINE SCRIPT STRATEGY OPTIMIZER
Converts Pine Script v6 strategy to Python and tests all configurations
"""

import asyncio
import ccxt.async_support as ccxt
import numpy as np
from itertools import product
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "8d65ae81-ddd4-44f7-84bb-5b01608251de"
API_SECRET = "_NKwZcNx8JMrpJD7NORH8abxVOA1Jw6G-JM3jl2-18phOWY4NTc4NS00YzkyLTQzZWQtYTk0MS1hZDEwNTU3MzUyOWQ"

class PineScriptStrategy:
    """Pine Script strategy implementation in Python"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.atr_length = config.get('atr_length', 200)
        self.atr_multiplier = config.get('atr_multiplier', 6.0)
        self.enable_scoring = config.get('enable_scoring', True)
        self.score_min = config.get('score_min', 30)
        self.trend_len = config.get('trend_len', 50)
        self.use_rsi_filter = config.get('use_rsi_filter', False)
        self.rsi_len = config.get('rsi_len', 14)
        self.min_body_atr = config.get('min_body_atr', 0.20)
        self.risk_per_trade = config.get('risk_per_trade', 0.5)
        self.max_leverage = config.get('max_leverage', 5.0)
        
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Average True Range"""
        if len(highs) < self.atr_length:
            return 0
            
        tr_list = []
        for i in range(1, min(len(highs), self.atr_length + 1)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
            
        return np.mean(tr_list[-self.atr_length:]) if tr_list else 0
    
    def calculate_predictive_ranges(self, ohlcv: np.ndarray) -> Dict:
        """Calculate predictive range levels like Pine Script"""
        closes = ohlcv[:, 4]
        highs = ohlcv[:, 2]
        lows = ohlcv[:, 3]
        
        atr = self.calculate_atr(highs, lows, closes)
        atr_mult = atr * self.atr_multiplier
        
        # Initialize average
        avg = closes[-1]
        
        # Calculate hold ATR (simplified version)
        hold_atr = atr_mult * 0.5
        hold_atr = np.clip(hold_atr, atr * 0.25, atr_mult)
        
        # Calculate levels
        pr_avg = avg
        pr_r1 = avg + hold_atr
        pr_r2 = avg + hold_atr * 2.0
        pr_s1 = avg - hold_atr
        pr_s2 = avg - hold_atr * 2.0
        
        return {
            'avg': pr_avg,
            'r1': pr_r1,
            'r2': pr_r2,
            's1': pr_s1,
            's2': pr_s2,
            'atr': atr,
            'hold_atr': hold_atr
        }
    
    def calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI"""
        if len(closes) < self.rsi_len + 1:
            return 50
            
        deltas = np.diff(closes[-self.rsi_len-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_score(self, side: str, close: float, open_price: float, 
                       levels: Dict, ohlcv: np.ndarray) -> float:
        """Calculate enhanced scoring like Pine Script"""
        score = 0.0
        closes = ohlcv[:, 4]
        
        # 1. Range Position Score (30 points)
        range_width = levels['r1'] - levels['s1']
        if side == "long":
            if range_width > 0 and close > levels['avg']:
                dist_from_s1 = abs(close - levels['s1'])
                proximity_s1 = np.clip(1.0 - (dist_from_s1 / range_width), 0.0, 1.0)
                score += 30.0 * proximity_s1
        else:
            if range_width > 0 and close < levels['avg']:
                dist_from_r1 = abs(close - levels['r1'])
                proximity_r1 = np.clip(1.0 - (dist_from_r1 / range_width), 0.0, 1.0)
                score += 30.0 * proximity_r1
        
        # 2. Trend Alignment Score (20 points)
        if len(closes) >= self.trend_len:
            trend_sma = np.mean(closes[-self.trend_len:])
            if side == "long" and close > trend_sma:
                trend_strength = (close - trend_sma) / trend_sma
                score += 20.0 * np.clip(trend_strength * 100, 0.0, 1.0)
            elif side == "short" and close < trend_sma:
                trend_strength = (trend_sma - close) / trend_sma
                score += 20.0 * np.clip(trend_strength * 100, 0.0, 1.0)
            else:
                score += 10.0
        
        # 3. RSI Confirmation Score (15 points)
        rsi = self.calculate_rsi(closes)
        if self.use_rsi_filter:
            if side == "long":
                if rsi < 30:
                    score += 15.0
                elif rsi < 50:
                    score += 15.0 * (1.0 - (rsi - 30) / 20)
                elif rsi < 70:
                    score += 7.5
            else:
                if rsi > 70:
                    score += 15.0
                elif rsi > 50:
                    score += 15.0 * ((rsi - 50) / 20)
                elif rsi > 30:
                    score += 7.5
        else:
            score += 7.5
        
        # 4. Body Size Filter (10 points)
        body_size = abs(close - open_price)
        body_atr_ok = body_size >= levels['atr'] * self.min_body_atr
        if body_atr_ok:
            score += 10.0
        
        # 5. Range Strength Bonus (15 points)
        range_strength = range_width / max(levels['avg'], 1e-10)
        if range_strength > 0.01:
            score += 15.0 * np.clip(range_strength * 50, 0.0, 1.0)
        else:
            score += 5.0
        
        # 6. Momentum Alignment (10 points)
        if len(closes) >= 6:
            momentum = (close - closes[-6]) / closes[-6] * 100
            if side == "long" and momentum > 0:
                score += 10.0 * np.clip(momentum / 2.0, 0.0, 1.0)
            elif side == "short" and momentum < 0:
                score += 10.0 * np.clip(abs(momentum) / 2.0, 0.0, 1.0)
        
        return max(score, 10.0)
    
    def generate_signals(self, ohlcv: np.ndarray) -> Dict:
        """Generate trading signals based on Pine Script logic"""
        if len(ohlcv) < max(self.atr_length, self.trend_len, self.rsi_len):
            return {'long': False, 'short': False, 'long_score': 0, 'short_score': 0}
        
        closes = ohlcv[:, 4]
        opens = ohlcv[:, 1]
        highs = ohlcv[:, 2]
        lows = ohlcv[:, 3]
        
        close = closes[-1]
        open_price = opens[-1]
        prev_close = closes[-2] if len(closes) > 1 else close
        
        # Calculate levels
        levels = self.calculate_predictive_ranges(ohlcv)
        
        # Base signals
        long_signal = close > levels['avg'] and close < levels['r1'] and prev_close <= levels['avg']
        short_signal = close < levels['avg'] and close > levels['s1'] and prev_close >= levels['avg']
        
        # Breakout signals
        breakout_long = close > levels['r1']
        breakout_short = close < levels['s1']
        
        # Trend filter
        trend_sma = np.mean(closes[-self.trend_len:]) if len(closes) >= self.trend_len else close
        trend_ok_long = close > trend_sma
        trend_ok_short = close < trend_sma
        
        # RSI filter
        rsi = self.calculate_rsi(closes)
        rsi_ok_long = not self.use_rsi_filter or (rsi > 50 and rsi > self.calculate_rsi(closes[:-1]))
        rsi_ok_short = not self.use_rsi_filter or (rsi < 50 and rsi < self.calculate_rsi(closes[:-1]))
        
        # Body size filter
        body_size = abs(close - open_price)
        body_atr_ok = body_size >= levels['atr'] * self.min_body_atr
        
        # Calculate scores
        long_score = self.calculate_score("long", close, open_price, levels, ohlcv) if self.enable_scoring else 50.0
        short_score = self.calculate_score("short", close, open_price, levels, ohlcv) if self.enable_scoring else 50.0
        
        # Final entry conditions
        long_entry = (long_signal and trend_ok_long and rsi_ok_long and body_atr_ok and 
                     (not self.enable_scoring or long_score >= self.score_min))
        short_entry = (short_signal and trend_ok_short and rsi_ok_short and body_atr_ok and 
                      (not self.enable_scoring or short_score >= self.score_min))
        
        strong_long_entry = long_entry and long_score >= 70
        strong_short_entry = short_entry and short_score >= 70
        
        return {
            'long': long_entry,
            'short': short_entry,
            'strong_long': strong_long_entry,
            'strong_short': strong_short_entry,
            'breakout_long': breakout_long and not long_entry,
            'breakout_short': breakout_short and not short_entry,
            'long_score': long_score,
            'short_score': short_score,
            'levels': levels,
            'rsi': rsi,
            'trend_sma': trend_sma
        }

class StrategyOptimizer:
    """Optimize Pine Script strategy across all configurations"""
    
    def __init__(self):
        self.exchange = None
        self.results = []
        self.best_config = None
        self.best_score = -float('inf')
        
    async def initialize(self):
        """Initialize exchange connection"""
        self.exchange = ccxt.phemex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        
        await self.exchange.load_markets()
        logger.info("Exchange initialized")
        
    async def fetch_data(self, symbol: str, timeframe: str = '5m', limit: int = 500) -> np.ndarray:
        """Fetch OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return np.array(ohlcv) if ohlcv else None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def backtest_strategy(self, strategy: PineScriptStrategy, ohlcv: np.ndarray) -> Dict:
        """Backtest strategy on historical data"""
        if ohlcv is None or len(ohlcv) < 200:
            return {'score': -1, 'trades': 0}
        
        trades = []
        position = None
        
        for i in range(200, len(ohlcv)):
            window = ohlcv[:i+1]
            signals = strategy.generate_signals(window)
            
            close = window[-1, 4]
            
            # Entry logic
            if position is None:
                if signals['strong_long'] or signals['long']:
                    position = {
                        'type': 'long',
                        'entry': close,
                        'sl': signals['levels']['s2'],
                        'tp': signals['levels']['r1'],
                        'score': signals['long_score'],
                        'strong': signals['strong_long']
                    }
                elif signals['strong_short'] or signals['short']:
                    position = {
                        'type': 'short',
                        'entry': close,
                        'sl': signals['levels']['r2'],
                        'tp': signals['levels']['s1'],
                        'score': signals['short_score'],
                        'strong': signals['strong_short']
                    }
            
            # Exit logic
            elif position:
                if position['type'] == 'long':
                    if close <= position['sl'] or close >= position['tp']:
                        pnl = (close - position['entry']) / position['entry'] * 100
                        trades.append({
                            'type': 'long',
                            'pnl': pnl,
                            'win': close >= position['tp'],
                            'score': position['score'],
                            'strong': position['strong']
                        })
                        position = None
                elif position['type'] == 'short':
                    if close >= position['sl'] or close <= position['tp']:
                        pnl = (position['entry'] - close) / position['entry'] * 100
                        trades.append({
                            'type': 'short',
                            'pnl': pnl,
                            'win': close <= position['tp'],
                            'score': position['score'],
                            'strong': position['strong']
                        })
                        position = None
        
        # Calculate metrics
        if not trades:
            return {'score': 0, 'trades': 0}
        
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['win'])
        win_rate = wins / total_trades * 100
        
        avg_pnl = np.mean([t['pnl'] for t in trades])
        total_pnl = sum(t['pnl'] for t in trades)
        
        strong_trades = [t for t in trades if t['strong']]
        strong_win_rate = (sum(1 for t in strong_trades if t['win']) / len(strong_trades) * 100) if strong_trades else 0
        
        # Composite score
        score = (win_rate * 0.3 + 
                avg_pnl * 10 + 
                total_pnl * 0.1 + 
                strong_win_rate * 0.2 + 
                min(total_trades, 50) * 0.5)
        
        return {
            'score': score,
            'trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'strong_trades': len(strong_trades),
            'strong_win_rate': strong_win_rate
        }
    
    async def optimize(self, symbols: List[str], timeframes: List[str]):
        """Optimize strategy across all configurations"""
        
        # Define parameter ranges
        param_ranges = {
            'atr_length': [50, 100, 200, 300],
            'atr_multiplier': [2.0, 4.0, 6.0, 8.0, 10.0],
            'score_min': [20, 30, 40, 50, 60, 70],
            'trend_len': [20, 50, 100, 200],
            'rsi_len': [7, 14, 21, 28],
            'min_body_atr': [0.1, 0.2, 0.3, 0.5],
            'use_rsi_filter': [True, False],
            'enable_scoring': [True, False]
        }
        
        # Generate all combinations
        param_combinations = list(product(
            param_ranges['atr_length'],
            param_ranges['atr_multiplier'],
            param_ranges['score_min'],
            param_ranges['trend_len'],
            param_ranges['rsi_len'],
            param_ranges['min_body_atr'],
            param_ranges['use_rsi_filter'],
            param_ranges['enable_scoring']
        ))
        
        total_tests = len(param_combinations) * len(symbols) * len(timeframes)
        logger.info(f"Starting optimization: {total_tests} total tests")
        
        test_count = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Fetch data once for this symbol/timeframe
                ohlcv = await self.fetch_data(symbol, timeframe)
                
                if ohlcv is None:
                    continue
                
                for params in param_combinations:
                    config = {
                        'atr_length': params[0],
                        'atr_multiplier': params[1],
                        'score_min': params[2],
                        'trend_len': params[3],
                        'rsi_len': params[4],
                        'min_body_atr': params[5],
                        'use_rsi_filter': params[6],
                        'enable_scoring': params[7],
                        'risk_per_trade': 1.0,  # Fixed $1 risk
                        'max_leverage': 34.0
                    }
                    
                    strategy = PineScriptStrategy(config)
                    result = self.backtest_strategy(strategy, ohlcv)
                    
                    result.update({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'config': config
                    })
                    
                    self.results.append(result)
                    
                    # Track best configuration
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        self.best_config = result
                    
                    test_count += 1
                    
                    # Progress update
                    if test_count % 100 == 0:
                        progress = test_count / total_tests * 100
                        logger.info(f"Progress: {progress:.1f}% | Best score: {self.best_score:.2f}")
                
                # Small delay between symbols
                await asyncio.sleep(0.1)
        
        logger.info(f"Optimization complete: {test_count} tests performed")
        
    def get_top_configurations(self, n: int = 10) -> List[Dict]:
        """Get top N configurations"""
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]
    
    def save_results(self):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        with open(f'pinescript_optimization_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save top configurations
        top_configs = self.get_top_configurations(20)
        with open(f'pinescript_top_configs_{timestamp}.json', 'w') as f:
            json.dump(top_configs, f, indent=2, default=str)
        
        logger.info(f"Results saved to pinescript_optimization_{timestamp}.json")
        
        return top_configs

async def main():
    """Main optimization routine"""
    logger.info("="*60)
    logger.info("PINE SCRIPT STRATEGY OPTIMIZER")
    logger.info("="*60)
    
    optimizer = StrategyOptimizer()
    await optimizer.initialize()
    
    # Test on top performing symbols from previous analysis
    symbols = [
        'ALPINE/USDT:USDT',
        'ILV/USDT:USDT',
        'LA/USDT:USDT',
        'VINE/USDT:USDT',
        'BTC/USDT:USDT',
        'ETH/USDT:USDT'
    ]
    
    timeframes = ['5m', '15m', '30m', '1h']
    
    # Run optimization
    await optimizer.optimize(symbols, timeframes)
    
    # Get results
    top_configs = optimizer.save_results()
    
    # Display best configurations
    logger.info("\n" + "="*60)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("="*60)
    
    for i, config in enumerate(top_configs[:5], 1):
        logger.info(f"\n#{i} Configuration:")
        logger.info(f"Symbol: {config['symbol']} | Timeframe: {config['timeframe']}")
        logger.info(f"Score: {config['score']:.2f}")
        logger.info(f"Win Rate: {config.get('win_rate', 0):.1f}%")
        logger.info(f"Avg P&L: {config.get('avg_pnl', 0):.2f}%")
        logger.info(f"Total Trades: {config.get('trades', 0)}")
        logger.info(f"Strong Win Rate: {config.get('strong_win_rate', 0):.1f}%")
        logger.info("Parameters:")
        for key, value in config['config'].items():
            logger.info(f"  {key}: {value}")
    
    # Best overall configuration
    if optimizer.best_config:
        logger.info("\n" + "="*60)
        logger.info("BEST OVERALL CONFIGURATION")
        logger.info("="*60)
        logger.info(f"Symbol: {optimizer.best_config['symbol']}")
        logger.info(f"Timeframe: {optimizer.best_config['timeframe']}")
        logger.info(f"Score: {optimizer.best_config['score']:.2f}")
        logger.info(f"Win Rate: {optimizer.best_config.get('win_rate', 0):.1f}%")
        logger.info(f"Total P&L: {optimizer.best_config.get('total_pnl', 0):.2f}%")
        
    await optimizer.exchange.close()

if __name__ == "__main__":
    asyncio.run(main())