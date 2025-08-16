#!/usr/bin/env python3
"""
PINE SCRIPT LIVE TRADER
Implements the optimized Pine Script v6 strategy for live trading
With 34x leverage and $1 risk per trade
"""

import ccxt
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pinescript_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "8d65ae81-ddd4-44f7-84bb-5b01608251de"
API_SECRET = "_NKwZcNx8JMrpJD7NORH8abxVOA1Jw6G-JM3jl2-18phOWY4NTc4NS00YzkyLTQzZWQtYTk0MS1hZDEwNTU3MzUyOWQ"

# Optimized Pine Script Configuration
PINE_CONFIG = {
    'atr_length': 200,
    'atr_multiplier': 6.0,
    'score_min': 30,
    'trend_len': 50,
    'use_rsi_filter': False,
    'rsi_len': 14,
    'min_body_atr': 0.20
}

# Trading Configuration
TRADING_CONFIG = {
    'leverage': 34,
    'risk_per_trade': 1.0,  # $1 risk per trade
    'max_positions': 5,
    'symbols': [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'SOL/USDT:USDT'
    ],
    'timeframe': '300',  # 5 minutes in seconds for Phemex
    'strong_signal_threshold': 70
}

class PineScriptStrategy:
    """Pine Script v6 strategy implementation"""
    
    def __init__(self):
        self.config = PINE_CONFIG
        
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Average True Range"""
        if len(highs) < self.config['atr_length']:
            return 0
            
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
            
        return np.mean(tr_list[-self.config['atr_length']:]) if tr_list else 0
    
    def calculate_levels(self, ohlcv: List) -> Dict:
        """Calculate predictive range levels"""
        if len(ohlcv) < self.config['atr_length']:
            return None
            
        closes = np.array([x[4] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        
        atr = self.calculate_atr(highs, lows, closes)
        atr_mult = atr * self.config['atr_multiplier']
        
        # Current average
        avg = closes[-1]
        
        # Hold ATR calculation
        hold_atr = atr_mult * 0.5
        hold_atr = np.clip(hold_atr, atr * 0.25, atr_mult)
        
        return {
            'avg': avg,
            'r1': avg + hold_atr,
            'r2': avg + hold_atr * 2.0,
            's1': avg - hold_atr,
            's2': avg - hold_atr * 2.0,
            'atr': atr,
            'hold_atr': hold_atr
        }
    
    def calculate_score(self, side: str, ohlcv: List, levels: Dict) -> float:
        """Calculate signal score (0-100)"""
        if not levels or len(ohlcv) < max(self.config['atr_length'], self.config['trend_len']):
            return 0
            
        closes = np.array([x[4] for x in ohlcv])
        opens = np.array([x[1] for x in ohlcv])
        
        close = closes[-1]
        open_price = opens[-1]
        score = 0.0
        
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
        if len(closes) >= self.config['trend_len']:
            trend_sma = np.mean(closes[-self.config['trend_len']:])
            if side == "long" and close > trend_sma:
                trend_strength = (close - trend_sma) / trend_sma
                score += 20.0 * np.clip(trend_strength * 100, 0.0, 1.0)
            elif side == "short" and close < trend_sma:
                trend_strength = (trend_sma - close) / trend_sma
                score += 20.0 * np.clip(trend_strength * 100, 0.0, 1.0)
            else:
                score += 10.0
        
        # 3. RSI Score (15 points) - simplified
        if not self.config['use_rsi_filter']:
            score += 7.5
        
        # 4. Body Size Filter (10 points)
        body_size = abs(close - open_price)
        if body_size >= levels['atr'] * self.config['min_body_atr']:
            score += 10.0
        
        # 5. Range Strength (15 points)
        range_strength = range_width / max(levels['avg'], 1e-10)
        if range_strength > 0.01:
            score += 15.0 * np.clip(range_strength * 50, 0.0, 1.0)
        else:
            score += 5.0
        
        # 6. Momentum (10 points)
        if len(closes) >= 6:
            momentum = (close - closes[-6]) / closes[-6] * 100
            if side == "long" and momentum > 0:
                score += 10.0 * np.clip(momentum / 2.0, 0.0, 1.0)
            elif side == "short" and momentum < 0:
                score += 10.0 * np.clip(abs(momentum) / 2.0, 0.0, 1.0)
        
        return max(score, 10.0)
    
    def generate_signals(self, ohlcv: List) -> Dict:
        """Generate trading signals"""
        if len(ohlcv) < self.config['atr_length']:
            return {'signal': None, 'score': 0}
            
        levels = self.calculate_levels(ohlcv)
        if not levels:
            return {'signal': None, 'score': 0}
            
        closes = np.array([x[4] for x in ohlcv])
        close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else close
        
        # Check for signals
        long_signal = close > levels['avg'] and close < levels['r1'] and prev_close <= levels['avg']
        short_signal = close < levels['avg'] and close > levels['s1'] and prev_close >= levels['avg']
        
        # Calculate scores
        long_score = self.calculate_score("long", ohlcv, levels) if long_signal else 0
        short_score = self.calculate_score("short", ohlcv, levels) if short_signal else 0
        
        # Determine best signal
        if long_score >= self.config['score_min'] and long_score > short_score:
            return {
                'signal': 'STRONG_LONG' if long_score >= TRADING_CONFIG['strong_signal_threshold'] else 'LONG',
                'score': long_score,
                'levels': levels,
                'side': 'buy'
            }
        elif short_score >= self.config['score_min']:
            return {
                'signal': 'STRONG_SHORT' if short_score >= TRADING_CONFIG['strong_signal_threshold'] else 'SHORT',
                'score': short_score,
                'levels': levels,
                'side': 'sell'
            }
        
        return {'signal': None, 'score': 0}

class PineScriptLiveTrader:
    """Live trading bot using Pine Script strategy"""
    
    def __init__(self):
        self.exchange = None
        self.strategy = PineScriptStrategy()
        self.positions = {}
        self.stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'start_balance': 0
        }
        
    def initialize(self):
        """Initialize exchange connection"""
        self.exchange = ccxt.phemex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        
        self.exchange.load_markets()
        
        # Set leverage for all symbols
        for symbol in TRADING_CONFIG['symbols']:
            try:
                self.exchange.set_leverage(TRADING_CONFIG['leverage'], symbol)
                logger.info(f"Set leverage to {TRADING_CONFIG['leverage']}x for {symbol}")
            except Exception as e:
                logger.error(f"Failed to set leverage for {symbol}: {e}")
        
        # Get account balance
        balance = self.exchange.fetch_balance()
        self.stats['start_balance'] = balance['USDT']['free'] if 'USDT' in balance else 0
        logger.info(f"Account balance: ${self.stats['start_balance']:.2f}")
    
    def calculate_position_size(self, symbol: str, stop_distance: float) -> float:
        """Calculate position size for $1 risk"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position size for $1 risk
            # Position size = Risk Amount / (Stop Distance * Price)
            position_value = TRADING_CONFIG['risk_per_trade'] / (stop_distance / current_price)
            
            # Apply leverage
            actual_cost = position_value / TRADING_CONFIG['leverage']
            
            # Get minimum order size
            market = self.exchange.markets[symbol]
            min_amount = market['limits']['amount']['min']
            
            # Calculate contracts
            contract_size = position_value / current_price
            contract_size = max(contract_size, min_amount)
            
            return contract_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def place_order(self, symbol: str, signal: Dict) -> bool:
        """Place order with stop loss and take profit"""
        try:
            # Check if we already have a position
            if symbol in self.positions:
                logger.info(f"Already have position in {symbol}")
                return False
            
            # Check max positions
            if len(self.positions) >= TRADING_CONFIG['max_positions']:
                logger.info("Max positions reached")
                return False
            
            levels = signal['levels']
            side = signal['side']
            
            # Determine SL and TP based on side
            if side == 'buy':
                sl_price = levels['s2']  # Stop at S2
                tp_price = levels['r1']  # Take profit at R1
            else:
                sl_price = levels['r2']  # Stop at R2
                tp_price = levels['s1']  # Take profit at S1
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position size
            stop_distance = abs(current_price - sl_price)
            position_size = self.calculate_position_size(symbol, stop_distance)
            
            if position_size == 0:
                logger.error(f"Invalid position size for {symbol}")
                return False
            
            # Place market order
            logger.info(f"Placing {side.upper()} order for {symbol}")
            logger.info(f"  Price: {current_price:.2f}")
            logger.info(f"  Size: {position_size:.4f}")
            logger.info(f"  SL: {sl_price:.2f}")
            logger.info(f"  TP: {tp_price:.2f}")
            logger.info(f"  Score: {signal['score']:.1f}")
            
            # Place main order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=position_size
            )
            
            # Place stop loss order
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='stop',
                side='sell' if side == 'buy' else 'buy',
                amount=position_size,
                stopPrice=sl_price
            )
            
            # Place take profit order
            tp_order = self.exchange.create_limit_order(
                symbol=symbol,
                side='sell' if side == 'buy' else 'buy',
                amount=position_size,
                price=tp_price
            )
            
            # Store position info
            self.positions[symbol] = {
                'side': side,
                'entry': current_price,
                'size': position_size,
                'sl': sl_price,
                'tp': tp_price,
                'signal': signal['signal'],
                'score': signal['score'],
                'time': datetime.now(),
                'sl_order_id': sl_order['id'],
                'tp_order_id': tp_order['id']
            }
            
            self.stats['trades'] += 1
            logger.info(f"✅ Order placed successfully for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            return False
    
    def check_positions(self):
        """Check and update existing positions"""
        try:
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Calculate P&L
                if position['side'] == 'buy':
                    pnl_pct = (current_price - position['entry']) / position['entry'] * 100
                else:
                    pnl_pct = (position['entry'] - current_price) / position['entry'] * 100
                
                pnl_usd = pnl_pct * position['size'] * position['entry'] / 100
                
                # Check if position hit SL or TP
                hit_sl = (position['side'] == 'buy' and current_price <= position['sl']) or \
                         (position['side'] == 'sell' and current_price >= position['sl'])
                
                hit_tp = (position['side'] == 'buy' and current_price >= position['tp']) or \
                         (position['side'] == 'sell' and current_price <= position['tp'])
                
                if hit_sl or hit_tp:
                    # Update stats
                    if hit_tp:
                        self.stats['wins'] += 1
                        logger.info(f"✅ {symbol} hit TP! P&L: ${pnl_usd:.2f} ({pnl_pct:.1f}%)")
                    else:
                        self.stats['losses'] += 1
                        logger.info(f"❌ {symbol} hit SL! P&L: ${pnl_usd:.2f} ({pnl_pct:.1f}%)")
                    
                    self.stats['total_pnl'] += pnl_usd
                    del self.positions[symbol]
                else:
                    # Log current status
                    logger.debug(f"{symbol} P&L: ${pnl_usd:.2f} ({pnl_pct:.1f}%)")
                    
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def scan_for_signals(self):
        """Scan all symbols for trading signals"""
        signals = []
        
        for symbol in TRADING_CONFIG['symbols']:
            try:
                                 # Fetch OHLCV data (use '5m' string for ccxt)
                # Phemex may have issues with large limit values, use default
                ohlcv = self.exchange.fetch_ohlcv(symbol, '5m')
                
                if not ohlcv:
                    continue
                
                # Generate signal
                signal = self.strategy.generate_signals(ohlcv)
                
                if signal['signal']:
                    signal['symbol'] = symbol
                    signals.append(signal)
                    logger.info(f"Signal detected: {symbol} - {signal['signal']} (Score: {signal['score']:.1f})")
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by score (highest first)
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals
    
    def display_stats(self):
        """Display trading statistics"""
        win_rate = (self.stats['wins'] / self.stats['trades'] * 100) if self.stats['trades'] > 0 else 0
        
        logger.info("="*60)
        logger.info("PINE SCRIPT TRADER - LIVE STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Trades: {self.stats['trades']}")
        logger.info(f"Wins: {self.stats['wins']} | Losses: {self.stats['losses']}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total P&L: ${self.stats['total_pnl']:.2f}")
        logger.info(f"Active Positions: {len(self.positions)}")
        
        if self.positions:
            logger.info("\nActive Positions:")
            for symbol, pos in self.positions.items():
                logger.info(f"  {symbol}: {pos['side'].upper()} @ {pos['entry']:.2f} (Score: {pos['score']:.1f})")
        
        logger.info("="*60)
    
    def run(self):
        """Main trading loop"""
        logger.info("="*60)
        logger.info("🚀 PINE SCRIPT LIVE TRADER STARTED")
        logger.info(f"Leverage: {TRADING_CONFIG['leverage']}x")
        logger.info(f"Risk per trade: ${TRADING_CONFIG['risk_per_trade']}")
        logger.info(f"Symbols: {', '.join(TRADING_CONFIG['symbols'])}")
        logger.info("="*60)
        
        self.initialize()
        
        while True:
            try:
                # Check existing positions
                self.check_positions()
                
                # Scan for new signals
                signals = self.scan_for_signals()
                
                # Execute best signals
                for signal in signals:
                    if len(self.positions) >= TRADING_CONFIG['max_positions']:
                        break
                    
                    if signal['symbol'] not in self.positions:
                        success = self.place_order(signal['symbol'], signal)
                        if success:
                            time.sleep(1)  # Small delay between orders
                
                # Display stats
                self.display_stats()
                
                # Wait for next scan
                logger.info(f"Next scan in 60 seconds...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("\n⏹️ Stopping trader...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)
        
        # Final stats
        logger.info("\n📊 FINAL STATISTICS:")
        self.display_stats()

if __name__ == "__main__":
    trader = PineScriptLiveTrader()
    trader.run()