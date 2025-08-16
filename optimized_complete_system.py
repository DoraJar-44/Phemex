#!/usr/bin/env python3
"""
OPTIMIZED COMPLETE TRADING SYSTEM
Systematically debugged and optimized for maximum performance
"""

import asyncio
import ccxt.async_support as ccxt
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import signal
import sys

# ==================== CONFIGURATION ====================
API_KEY = "8d65ae81-ddd4-44f7-84bb-5b01608251de"
API_SECRET = "_NKwZcNx8JMrpJD7NORH8abxVOA1Jw6G-JM3jl2-18phOWY4NTc4NS00YzkyLTQzZWQtYTk0MS1hZDEwNTU3MzUyOWQ"

@dataclass
class OptimizedConfig:
    """Optimized configuration with all parameters"""
    # Account settings
    balance: float = 47.25
    leverage: int = 34
    risk_per_trade: float = 1.0
    
    # Position management
    max_positions: int = 5
    min_score: int = 75
    
    # Timeframes (optimized order)
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '30m'])
    
    # ATR settings (optimized)
    atr_period: int = 14
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 4.5  # 3:1 RR
    
    # Performance tracking
    max_daily_loss_pct: float = 10.0
    win_rate_target: float = 35.0
    
    # Execution
    scan_interval: int = 300  # 5 minutes
    order_timeout: int = 10
    max_retries: int = 3
    
    # Optimization flags
    use_cache: bool = True
    parallel_processing: bool = True
    adaptive_scoring: bool = True

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATA STRUCTURES ====================
@dataclass
class MarketData:
    """Optimized market data structure"""
    symbol: str
    ohlcv: np.ndarray
    atr: float
    sma_fast: float
    sma_slow: float
    momentum: float
    volatility: float
    timestamp: datetime

@dataclass
class Signal:
    """Trading signal with all parameters"""
    symbol: str
    direction: str  # 'long' or 'short'
    score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage_required: float
    risk_amount: float
    potential_profit: float
    timestamp: datetime

@dataclass
class Position:
    """Active position tracking"""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    open_time: datetime
    order_id: str

# ==================== OPTIMIZED COMPONENTS ====================

class DataCache:
    """Caching layer for market data"""
    def __init__(self, ttl: int = 60):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl)
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.now())
    
    def clear_expired(self):
        now = datetime.now()
        expired = [k for k, (_, t) in self.cache.items() if now - t >= self.ttl]
        for key in expired:
            del self.cache[key]

class OptimizedScanner:
    """Optimized market scanner with parallel processing"""
    
    def __init__(self, exchange, config: OptimizedConfig):
        self.exchange = exchange
        self.config = config
        self.cache = DataCache()
        self.markets = {}
        
    async def load_markets(self) -> Dict:
        """Load and filter markets efficiently"""
        if not self.markets:
            self.markets = await self.exchange.load_markets()
            
        # Filter for 34x+ leverage USDT perpetuals
        filtered = {
            symbol: market for symbol, market in self.markets.items()
            if (market.get('active') and 
                market.get('type') == 'swap' and
                market.get('quote') == 'USDT' and
                market.get('settle') == 'USDT' and
                market.get('info', {}).get('maxLeverage', 0) >= self.config.leverage)
        }
        
        logger.info(f"Filtered {len(filtered)} markets with {self.config.leverage}x+ leverage")
        return filtered
    
    async def fetch_ohlcv_batch(self, symbols: List[str], timeframe: str) -> Dict[str, np.ndarray]:
        """Fetch OHLCV data in parallel batches"""
        results = {}
        batch_size = 20  # Optimal batch size
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = []
            
            for symbol in batch:
                # Check cache first
                cache_key = f"{symbol}_{timeframe}"
                cached = self.cache.get(cache_key)
                if cached is not None:
                    results[symbol] = cached
                else:
                    tasks.append(self._fetch_single_ohlcv(symbol, timeframe))
            
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for symbol, result in zip(batch, batch_results):
                    if not isinstance(result, Exception) and result is not None:
                        results[symbol] = result
                        self.cache.set(f"{symbol}_{timeframe}", result)
            
            # Rate limit management
            await asyncio.sleep(0.1)
        
        return results
    
    async def _fetch_single_ohlcv(self, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Fetch single OHLCV with error handling"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            if ohlcv and len(ohlcv) >= 50:
                return np.array(ohlcv)
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {str(e)}")
        return None

class OptimizedAnalyzer:
    """Optimized technical analysis engine"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        
    def calculate_indicators_vectorized(self, ohlcv: np.ndarray) -> Dict:
        """Vectorized indicator calculations for speed"""
        closes = ohlcv[:, 4]
        highs = ohlcv[:, 2]
        lows = ohlcv[:, 3]
        
        # ATR calculation (vectorized)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
        atr = np.mean(tr[-self.config.atr_period:]) if len(tr) >= self.config.atr_period else 0
        
        # SMA calculations (vectorized)
        sma_fast = np.mean(closes[-20:])
        sma_slow = np.mean(closes[-50:]) if len(closes) >= 50 else sma_fast
        
        # Momentum (vectorized)
        momentum = ((closes[-1] - closes[-10]) / closes[-10] * 100) if len(closes) >= 10 else 0
        
        # Volatility
        volatility = (atr / closes[-1] * 100) if atr > 0 and closes[-1] > 0 else 0
        
        # RSI (vectorized)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'price': closes[-1],
            'atr': atr,
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'momentum': momentum,
            'volatility': volatility,
            'rsi': rsi,
            'volume': np.mean(ohlcv[-20:, 5])
        }
    
    def calculate_adaptive_score(self, indicators: Dict) -> Tuple[float, str]:
        """Adaptive scoring based on market conditions"""
        score = 50  # Base score
        
        # Trend component (40% weight)
        if indicators['sma_fast'] > indicators['sma_slow']:
            score += 20
            direction = 'long'
        else:
            score += 10
            direction = 'short'
            
        if (direction == 'long' and indicators['price'] > indicators['sma_fast']) or \
           (direction == 'short' and indicators['price'] < indicators['sma_fast']):
            score += 20
        
        # Momentum component (30% weight)
        momentum_score = min(15, abs(indicators['momentum']) * 3)
        if (direction == 'long' and indicators['momentum'] > 0) or \
           (direction == 'short' and indicators['momentum'] < 0):
            score += momentum_score
        
        # RSI component (20% weight)
        if direction == 'long' and 30 < indicators['rsi'] < 70:
            score += 10
        elif direction == 'short' and 30 < indicators['rsi'] < 70:
            score += 10
        elif (direction == 'long' and indicators['rsi'] < 30) or \
             (direction == 'short' and indicators['rsi'] > 70):
            score += 20
        
        # Volatility penalty (10% weight)
        if indicators['volatility'] < 2:
            score += 10
        elif indicators['volatility'] > 5:
            score -= 10
        
        return score, direction

class OptimizedRiskManager:
    """Optimized risk management system"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.daily_pnl = 0
        self.daily_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def calculate_position_size(self, price: float, stop_distance: float) -> float:
        """Calculate optimal position size"""
        if stop_distance <= 0:
            return 0
        
        # Fixed $1 risk
        position_size = self.config.risk_per_trade / stop_distance
        
        # Check leverage requirement
        notional = position_size * price
        required_leverage = notional / self.config.balance
        
        if required_leverage > self.config.leverage:
            # Adjust position size to fit leverage limit
            max_notional = self.config.balance * self.config.leverage
            position_size = max_notional / price
            
        return position_size
    
    def validate_trade(self, signal: Signal, active_positions: List[Position]) -> bool:
        """Validate trade against risk rules"""
        # Check position limit
        if len(active_positions) >= self.config.max_positions:
            logger.info(f"Max positions reached: {len(active_positions)}/{self.config.max_positions}")
            return False
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / self.config.balance * 100)
        if self.daily_pnl < 0 and daily_loss_pct >= self.config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2f}%")
            return False
        
        # Check if already in position
        if any(pos.symbol == signal.symbol for pos in active_positions):
            logger.debug(f"Already in position for {signal.symbol}")
            return False
        
        # Check leverage requirement
        if signal.leverage_required > self.config.leverage:
            logger.debug(f"Leverage too high for {signal.symbol}: {signal.leverage_required:.1f}x")
            return False
        
        return True
    
    def update_metrics(self, pnl: float, is_win: bool):
        """Update performance metrics"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        win_rate = (self.winning_trades / self.daily_trades * 100) if self.daily_trades > 0 else 0
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f} | Win Rate: {win_rate:.1f}% | Trades: {self.daily_trades}")

class OptimizedExecutor:
    """Optimized order execution system"""
    
    def __init__(self, exchange, config: OptimizedConfig):
        self.exchange = exchange
        self.config = config
        self.pending_orders = {}
        
    async def execute_signal(self, signal: Signal) -> Optional[str]:
        """Execute trading signal with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                # Set leverage
                await self.exchange.set_leverage(self.config.leverage, signal.symbol)
                
                # Prepare order parameters
                params = {
                    'stopLoss': {
                        'triggerPrice': signal.stop_loss,
                        'type': 'Market'
                    },
                    'takeProfit': {
                        'triggerPrice': signal.take_profit,
                        'type': 'Market'
                    }
                }
                
                # Place market order
                order = await self.exchange.create_market_order(
                    symbol=signal.symbol,
                    side='buy' if signal.direction == 'long' else 'sell',
                    amount=signal.position_size,
                    params=params
                )
                
                logger.info(f"✅ Order executed: {signal.symbol} {signal.direction.upper()} @ ${signal.entry_price:.4f}")
                return order['id']
                
            except Exception as e:
                logger.error(f"Execution attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        return None

class OptimizedMonitor:
    """Optimized position monitoring system"""
    
    def __init__(self, exchange, config: OptimizedConfig):
        self.exchange = exchange
        self.config = config
        self.positions: Dict[str, Position] = {}
        
    async def update_positions(self) -> List[Position]:
        """Update all position data"""
        try:
            positions = await self.exchange.fetch_positions()
            
            updated = []
            for pos in positions:
                if pos['contracts'] > 0:
                    position = Position(
                        symbol=pos['symbol'],
                        side=pos['side'],
                        entry_price=pos['markPrice'],
                        current_price=pos['markPrice'],
                        size=pos['contracts'],
                        stop_loss=0,  # Would need to fetch from orders
                        take_profit=0,  # Would need to fetch from orders
                        pnl=pos['unrealizedPnl'],
                        pnl_percent=pos['percentage'],
                        open_time=datetime.now(),
                        order_id=pos['id']
                    )
                    updated.append(position)
                    self.positions[pos['symbol']] = position
                    
            return updated
            
        except Exception as e:
            logger.error(f"Failed to update positions: {str(e)}")
            return list(self.positions.values())

# ==================== MAIN TRADING SYSTEM ====================

class OptimizedTradingSystem:
    """Complete optimized trading system"""
    
    def __init__(self):
        self.config = OptimizedConfig()
        self.exchange = None
        self.scanner = None
        self.analyzer = None
        self.risk_manager = None
        self.executor = None
        self.monitor = None
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Optimized Trading System...")
        
        # Initialize exchange
        self.exchange = ccxt.phemex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True
            }
        })
        
        # Initialize components
        self.scanner = OptimizedScanner(self.exchange, self.config)
        self.analyzer = OptimizedAnalyzer(self.config)
        self.risk_manager = OptimizedRiskManager(self.config)
        self.executor = OptimizedExecutor(self.exchange, self.config)
        self.monitor = OptimizedMonitor(self.exchange, self.config)
        
        # Load markets
        await self.scanner.load_markets()
        
        logger.info("✅ System initialized successfully")
        return True
    
    async def scan_and_analyze(self) -> List[Signal]:
        """Scan markets and generate signals"""
        markets = await self.scanner.load_markets()
        symbols = list(markets.keys())
        
        all_signals = []
        
        # Process each timeframe
        for timeframe in self.config.timeframes:
            logger.info(f"Scanning {len(symbols)} symbols on {timeframe}...")
            
            # Fetch OHLCV data in batches
            ohlcv_data = await self.scanner.fetch_ohlcv_batch(symbols, timeframe)
            
            # Analyze each symbol
            for symbol, ohlcv in ohlcv_data.items():
                if ohlcv is None:
                    continue
                    
                # Calculate indicators
                indicators = self.analyzer.calculate_indicators_vectorized(ohlcv)
                
                # Calculate score
                score, direction = self.analyzer.calculate_adaptive_score(indicators)
                
                if score >= self.config.min_score:
                    # Create signal
                    signal = Signal(
                        symbol=symbol,
                        direction=direction,
                        score=score,
                        entry_price=indicators['price'],
                        stop_loss=indicators['price'] - indicators['atr'] * self.config.atr_multiplier_sl 
                                  if direction == 'long' else 
                                  indicators['price'] + indicators['atr'] * self.config.atr_multiplier_sl,
                        take_profit=indicators['price'] + indicators['atr'] * self.config.atr_multiplier_tp 
                                   if direction == 'long' else 
                                   indicators['price'] - indicators['atr'] * self.config.atr_multiplier_tp,
                        position_size=self.risk_manager.calculate_position_size(
                            indicators['price'], 
                            indicators['atr'] * self.config.atr_multiplier_sl
                        ),
                        leverage_required=(self.config.risk_per_trade / 
                                         (indicators['atr'] * self.config.atr_multiplier_sl) * 
                                         indicators['price'] / self.config.balance),
                        risk_amount=self.config.risk_per_trade,
                        potential_profit=self.config.risk_per_trade * 3,
                        timestamp=datetime.now()
                    )
                    
                    all_signals.append(signal)
        
        # Sort by score
        all_signals.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Generated {len(all_signals)} signals with score >= {self.config.min_score}")
        return all_signals
    
    async def execute_signals(self, signals: List[Signal]):
        """Execute trading signals"""
        # Get current positions
        positions = await self.monitor.update_positions()
        
        executed = 0
        for signal in signals:
            # Validate trade
            if not self.risk_manager.validate_trade(signal, positions):
                continue
            
            # Execute trade
            order_id = await self.executor.execute_signal(signal)
            
            if order_id:
                executed += 1
                positions.append(Position(
                    symbol=signal.symbol,
                    side=signal.direction,
                    entry_price=signal.entry_price,
                    current_price=signal.entry_price,
                    size=signal.position_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl=0,
                    pnl_percent=0,
                    open_time=datetime.now(),
                    order_id=order_id
                ))
                
                # Check position limit
                if len(positions) >= self.config.max_positions:
                    break
        
        logger.info(f"Executed {executed} trades")
        return executed
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        
        logger.info("="*60)
        logger.info("OPTIMIZED TRADING SYSTEM STARTED")
        logger.info(f"Balance: ${self.config.balance}")
        logger.info(f"Leverage: {self.config.leverage}x")
        logger.info(f"Risk per trade: ${self.config.risk_per_trade}")
        logger.info("="*60)
        
        while self.running:
            try:
                # Scan and analyze markets
                signals = await self.scan_and_analyze()
                
                # Display top signals
                if signals:
                    logger.info("\n🏆 TOP SIGNALS:")
                    for i, signal in enumerate(signals[:5], 1):
                        logger.info(f"{i}. {signal.symbol} {signal.direction.upper()} "
                                  f"Score: {signal.score:.1f} Leverage: {signal.leverage_required:.1f}x")
                
                # Execute signals
                await self.execute_signals(signals)
                
                # Update positions
                positions = await self.monitor.update_positions()
                if positions:
                    total_pnl = sum(pos.pnl for pos in positions)
                    logger.info(f"Active positions: {len(positions)} | Total P&L: ${total_pnl:.2f}")
                
                # Wait for next scan
                logger.info(f"Next scan in {self.config.scan_interval} seconds...")
                await asyncio.sleep(self.config.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                self.running = False
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(30)
        
        # Cleanup
        await self.exchange.close()
        logger.info("System shutdown complete")

# ==================== ENTRY POINT ====================

async def main():
    """Main entry point"""
    system = OptimizedTradingSystem()
    
    if await system.initialize():
        await system.run()

if __name__ == "__main__":
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the system
    asyncio.run(main())