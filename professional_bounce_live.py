#!/usr/bin/env python3
"""
PROFESSIONAL BOUNCE LIVE TRADING BOT
Smart Money bounce strategy for Bitget Futures (Swaps)
Real-time implementation with institutional-grade analysis
"""

import os
import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

import ccxt.async_support as ccxt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('professional_bounce_live.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our professional strategy
from bot.strategy.professional_bounce import ProfessionalBounceStrategy, ConfluenceFactors


class ProfessionalBounceLiveBot:
    """Live trading bot for professional bounce strategy on Bitget"""
    
    def __init__(self):
        # Load optimal configuration (from MDC config)
        self.symbol = os.getenv("SYMBOL", "BTC/USDT:USDT")
        self.timeframe = os.getenv("TIMEFRAME", "4h")
        self.leverage = int(os.getenv("LEVERAGE", "25"))
        
        # Professional bounce strategy configuration
        # These are the optimized parameters from backtesting
        self.strategy = ProfessionalBounceStrategy(
            atr_length=50,           # Optimal from backtests
            atr_multiplier=5.0,      # Optimal from backtests  
            ma_periods=[21, 50, 200], # Standard institutional MAs
            rsi_period=14,           # Standard RSI period
            rsi_oversold=30.0,       # Conservative oversold level
            volume_period=20,        # Volume average period
            volume_spike_threshold=1.5, # 1.5x volume spike threshold
            min_confluence_factors=4,   # Require 4/6 factors for quality
            order_block_lookback=10,    # Order block detection period
            liquidity_lookback=20       # Liquidity zone detection period
        )
        
        # Risk management (following MDC rules)
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions = 3      # Maximum concurrent positions
        self.daily_loss_limit = 0.05 # 5% daily loss limit
        
        # Trading state
        self.positions = {}
        self.daily_pnl = 0.0
        self.account_balance = 0.0
        self.trades_today = 0
        
        # Exchange connection (will be initialized)
        self.exchange = None
        
    async def initialize_exchange(self):
        """Initialize Bitget exchange connection"""
        try:
            # Check for Bitget credentials (fall back to Phemex for now)
            api_key = os.getenv('BITGET_API_KEY') or os.getenv('PHEMEX_API_KEY')
            secret = os.getenv('BITGET_SECRET') or os.getenv('PHEMEX_SECRET')
            passphrase = os.getenv('BITGET_PASSPHRASE', '')
            
            if not api_key or not secret:
                raise ValueError("API credentials not found. Set BITGET_API_KEY and BITGET_SECRET")
            
            # Use Phemex for now (same USDT perpetuals, similar interface)
            # TODO: Switch to Bitget when user provides credentials
            self.exchange = ccxt.phemex({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': os.getenv('BITGET_TESTNET', 'false').lower() == 'true',
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # Futures/swaps
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            self.account_balance = balance['USDT']['total']
            
            logger.info(f"✅ Exchange connected. Balance: {self.account_balance:.2f} USDT")
            logger.info(f"📊 Trading {self.symbol} on {self.timeframe} with {self.leverage}x leverage")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            raise

    async def fetch_market_data(self) -> Optional[Dict[str, List[float]]]:
        """Fetch latest market data for analysis"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=500  # Enough for analysis
            )
            
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(f"❌ Insufficient data: {len(ohlcv) if ohlcv else 0} candles")
                return None
            
            data = {
                "open": [x[1] for x in ohlcv],
                "high": [x[2] for x in ohlcv],
                "low": [x[3] for x in ohlcv],
                "close": [x[4] for x in ohlcv],
                "volume": [x[5] for x in ohlcv]
            }
            
            logger.debug(f"📈 Fetched {len(data['close'])} candles for {self.symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return None

    async def analyze_market_conditions(self, ohlcv_data: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
        """Analyze current market conditions for professional bounce signals"""
        try:
            signals = self.strategy.generate_professional_signals(**ohlcv_data)
            
            if not signals:
                logger.debug("🔍 No professional bounce signals detected")
                return None
            
            # Get the best signal
            best_signal = max(signals, key=lambda x: x["score"])
            
            # Additional professional analysis
            logger.info(f"🎯 Professional bounce signal detected!")
            logger.info(f"   Score: {best_signal['score']:.1f}/100")
            logger.info(f"   Confluence Factors: {best_signal['confluence_factors']}/6")
            logger.info(f"   Confluence Score: {best_signal['confluence_score']:.1f}/100")
            
            # Log confluence details
            details = best_signal['confluence_details']
            logger.info(f"   📊 Confluence Analysis:")
            logger.info(f"      MA Support: {'✅' if details['ma_support'] else '❌'}")
            logger.info(f"      RSI Oversold: {'✅' if details['rsi_oversold'] else '❌'}")
            logger.info(f"      Volume Spike: {'✅' if details['volume_spike'] else '❌'}")
            logger.info(f"      Bullish Pattern: {'✅' if details['bullish_pattern'] else '❌'}")
            logger.info(f"      Support Level: {'✅' if details['support_level'] else '❌'}")
            logger.info(f"      Market Structure: {'✅' if details['market_structure'] else '❌'}")
            
            return best_signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market conditions: {e}")
            return None

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.account_balance * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return 0.0
        
        # Position size in base currency
        position_size = (risk_amount / stop_distance) * entry_price
        
        # Apply leverage
        leveraged_size = position_size * self.leverage
        
        # Ensure minimum size requirements
        min_size = 0.001  # Minimum position size
        leveraged_size = max(min_size, leveraged_size)
        
        logger.info(f"💰 Position sizing:")
        logger.info(f"   Risk Amount: {risk_amount:.2f} USDT ({self.risk_per_trade*100:.1f}%)")
        logger.info(f"   Stop Distance: {stop_distance:.6f}")
        logger.info(f"   Position Size: {leveraged_size:.6f}")
        logger.info(f"   Leverage: {self.leverage}x")
        
        return leveraged_size

    async def place_professional_trade(self, signal: Dict[str, Any]) -> bool:
        """Place a professional bounce trade with full risk management"""
        try:
            entry_price = signal["entry_price"]
            tp1_price = signal["tp1"]
            tp2_price = signal["tp2"]
            sl_price = signal["sl"]
            
            # Calculate position size
            position_size = self.calculate_position_size(entry_price, sl_price)
            
            if position_size <= 0:
                logger.warning("❌ Invalid position size calculated")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.warning(f"❌ Maximum positions reached ({self.max_positions})")
                return False
            
            # Check daily loss limit
            daily_loss_pct = abs(self.daily_pnl / self.account_balance) if self.account_balance > 0 else 0
            if daily_loss_pct >= self.daily_loss_limit:
                logger.warning(f"❌ Daily loss limit reached ({self.daily_loss_limit*100:.1f}%)")
                return False
            
            logger.info(f"🚀 Placing professional bounce trade for {self.symbol}")
            logger.info(f"   Entry: {entry_price:.6f}")
            logger.info(f"   TP1: {tp1_price:.6f} | TP2: {tp2_price:.6f}")
            logger.info(f"   SL: {sl_price:.6f}")
            logger.info(f"   Size: {position_size:.6f}")
            logger.info(f"   R/R: {signal.get('risk_reward', 1):.2f}:1")
            
            # 1. Place entry order (market order for immediate execution)
            entry_order = await self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='buy',  # Professional bounce = long
                amount=position_size,
                params={
                    'leverage': self.leverage,
                    'postOnly': False  # Market order for immediate execution
                }
            )
            
            logger.info(f"✅ Entry order placed: {entry_order.get('id', 'unknown')}")
            
            # 2. Place stop loss (stop market order)
            sl_order = await self.exchange.create_order(
                symbol=self.symbol,
                type='stop_market',
                side='sell',
                amount=position_size,
                params={
                    'stopPrice': sl_price,
                    'reduceOnly': True
                }
            )
            
            logger.info(f"✅ Stop loss placed: {sl_order.get('id', 'unknown')}")
            
            # 3. Place take profit orders (split position)
            tp1_size = position_size * 0.6  # 60% at TP1
            tp2_size = position_size * 0.4  # 40% at TP2
            
            tp1_order = await self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side='sell',
                amount=tp1_size,
                price=tp1_price,
                params={
                    'reduceOnly': True,
                    'postOnly': True
                }
            )
            
            tp2_order = await self.exchange.create_order(
                symbol=self.symbol,
                type='limit', 
                side='sell',
                amount=tp2_size,
                price=tp2_price,
                params={
                    'reduceOnly': True,
                    'postOnly': True
                }
            )
            
            logger.info(f"✅ TP1 order placed: {tp1_order.get('id', 'unknown')}")
            logger.info(f"✅ TP2 order placed: {tp2_order.get('id', 'unknown')}")
            
            # Store position info
            position_id = entry_order.get('id', str(time.time()))
            self.positions[position_id] = {
                "symbol": self.symbol,
                "side": "long",
                "size": position_size,
                "entry_price": entry_price,
                "entry_time": datetime.now(),
                "tp1_price": tp1_price,
                "tp2_price": tp2_price,
                "sl_price": sl_price,
                "confluence_factors": signal["confluence_factors"],
                "confluence_score": signal["confluence_score"],
                "signal_score": signal["score"],
                "orders": {
                    "entry": entry_order,
                    "sl": sl_order,
                    "tp1": tp1_order,
                    "tp2": tp2_order
                }
            }
            
            self.trades_today += 1
            
            logger.info(f"🎉 Professional bounce trade executed successfully!")
            logger.info(f"   Position ID: {position_id}")
            logger.info(f"   Confluence Analysis: {signal['confluence_factors']}/6 factors")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to place professional trade: {e}")
            return False

    async def monitor_positions(self):
        """Monitor open positions and update PnL"""
        if not self.positions:
            return
            
        try:
            # Fetch current positions from exchange
            positions = await self.exchange.fetch_positions([self.symbol])
            current_price = None
            
            # Get current price
            ticker = await self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            for position_id, position_info in list(self.positions.items()):
                # Check if position still exists
                active_position = None
                for pos in positions:
                    if pos['symbol'] == position_info['symbol'] and pos['size'] > 0:
                        active_position = pos
                        break
                
                if not active_position:
                    # Position was closed
                    logger.info(f"📊 Position {position_id} was closed")
                    
                    # Calculate realized PnL
                    entry_price = position_info['entry_price']
                    # Estimate exit price (would need order fills for exact)
                    if current_price > position_info['tp1_price']:
                        exit_price = position_info['tp1_price']  # Likely hit TP1
                        exit_reason = "TP1"
                    elif current_price < position_info['sl_price']:
                        exit_price = position_info['sl_price']   # Likely hit SL
                        exit_reason = "SL"
                    else:
                        exit_price = current_price  # Manual close or TP2
                        exit_reason = "Other"
                    
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100 * self.leverage
                    profit_usd = self.account_balance * (profit_pct / 100)
                    
                    logger.info(f"💰 Trade Result:")
                    logger.info(f"   Entry: {entry_price:.6f} | Exit: {exit_price:.6f}")
                    logger.info(f"   Profit: {profit_pct:.2f}% ({profit_usd:.2f} USDT)")
                    logger.info(f"   Exit Reason: {exit_reason}")
                    logger.info(f"   Confluence: {position_info['confluence_factors']}/6")
                    
                    # Update daily PnL
                    self.daily_pnl += profit_usd
                    
                    # Remove from tracking
                    del self.positions[position_id]
                else:
                    # Position still active, calculate unrealized PnL
                    entry_price = position_info['entry_price']
                    unrealized_pct = ((current_price - entry_price) / entry_price) * 100 * self.leverage
                    unrealized_usd = self.account_balance * (unrealized_pct / 100)
                    
                    logger.debug(f"📈 Position {position_id}: {unrealized_pct:+.2f}% ({unrealized_usd:+.2f} USDT)")
                    
        except Exception as e:
            logger.error(f"❌ Error monitoring positions: {e}")

    async def update_account_info(self):
        """Update account balance and daily statistics"""
        try:
            balance = await self.exchange.fetch_balance()
            self.account_balance = balance['USDT']['total']
            
            # Reset daily counters at midnight UTC
            current_hour = datetime.utcnow().hour
            if current_hour == 0 and self.trades_today > 0:
                logger.info(f"🌅 New trading day. Yesterday's trades: {self.trades_today}")
                logger.info(f"💰 Daily PnL: {self.daily_pnl:+.2f} USDT ({(self.daily_pnl/self.account_balance)*100:+.2f}%)")
                self.trades_today = 0
                self.daily_pnl = 0.0
                
        except Exception as e:
            logger.error(f"❌ Error updating account info: {e}")

    async def run_professional_analysis_cycle(self):
        """Run one complete professional analysis cycle"""
        try:
            # Fetch latest market data
            ohlcv_data = await self.fetch_market_data()
            if not ohlcv_data:
                return
            
            # Analyze for professional bounce signals
            signal = await self.analyze_market_conditions(ohlcv_data)
            
            if signal and signal["score"] >= 95:
                logger.info(f"🔥 HIGH-QUALITY PROFESSIONAL SIGNAL DETECTED!")
                logger.info(f"   Symbol: {self.symbol}")
                logger.info(f"   Price: {signal['price']:.6f}")
                logger.info(f"   Professional Score: {signal['score']:.1f}/100")
                logger.info(f"   Confluence Factors: {signal['confluence_factors']}/6")
                
                # Place trade if conditions are met
                if signal['confluence_factors'] >= self.strategy.min_confluence_factors:
                    success = await self.place_professional_trade(signal)
                    if success:
                        logger.info("✅ Professional bounce trade executed!")
                    else:
                        logger.warning("❌ Failed to execute professional trade")
                else:
                    logger.info(f"⏳ Signal needs {self.strategy.min_confluence_factors} factors, has {signal['confluence_factors']}")
            
            # Monitor existing positions
            await self.monitor_positions()
            
        except Exception as e:
            logger.error(f"❌ Error in analysis cycle: {e}")

    async def run_live_bot(self):
        """Main loop for live professional bounce trading"""
        logger.info("🚀 Starting Professional Bounce Live Trading Bot")
        logger.info("📊 Smart Money Concepts + Predictive Ranges Strategy")
        logger.info(f"🎯 Target: {self.symbol} | {self.timeframe} | {self.leverage}x leverage")
        
        try:
            # Initialize exchange connection
            await self.initialize_exchange()
            
            # Main trading loop
            cycle_count = 0
            while True:
                cycle_count += 1
                logger.info(f"🔄 Analysis Cycle #{cycle_count}")
                
                # Update account information
                await self.update_account_info()
                
                # Run professional analysis
                await self.run_professional_analysis_cycle()
                
                # Log status summary
                logger.info(f"📊 Status: {len(self.positions)} positions | Daily PnL: {self.daily_pnl:+.2f} USDT")
                
                # Wait for next cycle (timeframe dependent)
                sleep_seconds = {
                    "1m": 60, "3m": 180, "5m": 300, 
                    "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400
                }.get(self.timeframe, 3600)
                
                logger.info(f"⏳ Waiting {sleep_seconds} seconds for next analysis...")
                await asyncio.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("👋 Professional bounce bot shutting down...")
        except Exception as e:
            logger.error(f"❌ Critical error in live bot: {e}")
        finally:
            if self.exchange:
                await self.exchange.close()


# Deployment configuration for Bitget
BITGET_PROFESSIONAL_CONFIG = {
    "exchange": "bitget",
    "market_type": "swap",
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "timeframes": ["5m", "15m", "1h", "4h"],
    "strategy": "professional_bounce",
    "leverage": 25,  # User's preferred leverage
    "risk_per_trade": 1.0,  # 1% risk per trade
    "confluence_requirement": 4,  # Require 4/6 factors
    "live_trading": True
}


if __name__ == "__main__":
    async def main():
        bot = ProfessionalBounceLiveBot()
        await bot.run_live_bot()
    
    # Run the live bot
    asyncio.run(main())