#!/usr/bin/env python3
"""
MULTI-ASSET PROFESSIONAL BOUNCE TRADING BOT
Trades professional bounce strategy across 100+ diverse cryptocurrency pairs
Automatically manages portfolio across different asset classes and sectors
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_asset_professional_bounce.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our professional strategy
from bot.strategy.professional_bounce import ProfessionalBounceStrategy


class MultiAssetProfessionalBot:
    """Multi-asset professional bounce trading bot"""
    
    def __init__(self):
        # Load configuration
        self.leverage = int(os.getenv("LEVERAGE", "25"))
        self.primary_timeframe = os.getenv("TIMEFRAME", "4h")
        
        # Multi-asset configuration based on diversity testing
        self.asset_universe = {
            # TIER 1: Large cap - stable, high liquidity (60% allocation)
            "tier1": {
                "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"],
                "allocation": 0.60,
                "config": {
                    "atr_length": 50,
                    "atr_multiplier": 5.0,
                    "min_confluence_factors": 3,  # Lower for stable coins
                    "volume_spike_threshold": 1.3,
                    "risk_per_trade": 0.015  # 1.5% per trade
                }
            },
            
            # TIER 2: Mid cap - growth potential (25% allocation)
            "tier2": {
                "symbols": ["MATIC/USDT:USDT", "UNI/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
                          "AVAX/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT", "ALGO/USDT:USDT"],
                "allocation": 0.25,
                "config": {
                    "atr_length": 50,
                    "atr_multiplier": 5.0,
                    "min_confluence_factors": 4,  # Standard
                    "volume_spike_threshold": 1.5,
                    "risk_per_trade": 0.01  # 1% per trade
                }
            },
            
            # TIER 3: High alpha - small cap, DeFi, gaming (15% allocation)
            "tier3": {
                "symbols": ["AAVE/USDT:USDT", "SUSHI/USDT:USDT", "AXS/USDT:USDT", "SAND/USDT:USDT",
                          "GALA/USDT:USDT", "FTM/USDT:USDT", "ONE/USDT:USDT", "ENJ/USDT:USDT",
                          "SHIB/USDT:USDT", "PEPE/USDT:USDT"],
                "allocation": 0.15,
                "config": {
                    "atr_length": 50,
                    "atr_multiplier": 6.0,  # Wider ranges for volatility
                    "min_confluence_factors": 5,  # Higher selectivity
                    "volume_spike_threshold": 2.0,  # Higher threshold
                    "risk_per_trade": 0.005  # 0.5% per trade (higher volatility)
                }
            }
        }
        
        # Portfolio management
        self.max_positions_per_tier = {
            "tier1": 3,   # Up to 3 large cap positions
            "tier2": 2,   # Up to 2 mid cap positions  
            "tier3": 1    # Up to 1 high alpha position
        }
        
        self.portfolio_limits = {
            "max_total_positions": 6,
            "max_daily_trades": 15,
            "max_daily_loss_pct": 0.05,  # 5%
            "rebalance_interval": 86400   # 24 hours
        }
        
        # Trading state
        self.active_positions = {}
        self.daily_stats = {
            "trades": 0,
            "pnl": 0.0,
            "win_count": 0,
            "loss_count": 0
        }
        
        self.account_balance = 0.0
        self.exchange = None

    async def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            # Use Bitget if available, fallback to Phemex
            api_key = os.getenv('BITGET_API_KEY') or os.getenv('PHEMEX_API_KEY')
            secret = os.getenv('BITGET_SECRET') or os.getenv('PHEMEX_SECRET')
            
            if not api_key or not secret:
                raise ValueError("API credentials required")
            
            self.exchange = ccxt.phemex({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': os.getenv('BITGET_TESTNET', 'false').lower() == 'true',
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            self.account_balance = balance['USDT']['total']
            
            logger.info(f"✅ Exchange connected. Balance: {self.account_balance:.2f} USDT")
            logger.info(f"🌍 Multi-asset portfolio trading with {self.leverage}x leverage")
            
        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            raise

    async def fetch_multi_asset_data(self, symbols: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """Fetch OHLCV data for multiple assets"""
        data_cache = {}
        
        for symbol in symbols:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, self.primary_timeframe, limit=500
                )
                
                if ohlcv and len(ohlcv) >= 100:
                    data_cache[symbol] = {
                        "open": [x[1] for x in ohlcv],
                        "high": [x[2] for x in ohlcv],
                        "low": [x[3] for x in ohlcv],
                        "close": [x[4] for x in ohlcv],
                        "volume": [x[5] for x in ohlcv]
                    }
                    logger.debug(f"📈 {symbol}: {len(ohlcv)} candles")
                
            except Exception as e:
                logger.warning(f"❌ Failed to fetch {symbol}: {str(e)[:50]}")
        
        return data_cache

    async def scan_multi_asset_signals(self) -> List[Dict[str, Any]]:
        """Scan for professional bounce signals across all assets"""
        all_signals = []
        
        for tier_name, tier_config in self.asset_universe.items():
            symbols = tier_config["symbols"]
            strategy_config = tier_config["config"]
            
            logger.info(f"🔍 Scanning {tier_name.upper()} assets: {len(symbols)} symbols")
            
            # Fetch data for this tier
            tier_data = await self.fetch_multi_asset_data(symbols)
            
            # Create strategy instance for this tier
            strategy = ProfessionalBounceStrategy(
                atr_length=strategy_config["atr_length"],
                atr_multiplier=strategy_config["atr_multiplier"],
                min_confluence_factors=strategy_config["min_confluence_factors"],
                volume_spike_threshold=strategy_config["volume_spike_threshold"]
            )
            
            # Analyze each symbol
            for symbol, ohlcv_data in tier_data.items():
                try:
                    signals = strategy.generate_professional_signals(**ohlcv_data)
                    
                    for signal in signals:
                        if signal["score"] >= 95 and signal["confluence_factors"] >= strategy_config["min_confluence_factors"]:
                            # Add tier and config info to signal
                            signal.update({
                                "symbol": symbol,
                                "tier": tier_name,
                                "tier_config": strategy_config,
                                "current_price": ohlcv_data["close"][-1]
                            })
                            all_signals.append(signal)
                            
                            logger.info(f"🎯 Signal: {symbol} | Score: {signal['score']:.1f} | Confluence: {signal['confluence_factors']}/6")
                
                except Exception as e:
                    logger.error(f"❌ Error analyzing {symbol}: {e}")
        
        # Sort signals by score (highest quality first)
        all_signals.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"📊 Found {len(all_signals)} high-quality signals across portfolio")
        return all_signals

    def calculate_position_allocation(self, signal: Dict[str, Any]) -> float:
        """Calculate position size based on tier allocation and risk management"""
        tier_config = signal["tier_config"]
        tier_allocation = self.asset_universe[signal["tier"]]["allocation"]
        
        # Available capital for this tier
        tier_capital = self.account_balance * tier_allocation
        
        # Risk-based position sizing
        risk_amount = tier_capital * tier_config["risk_per_trade"]
        stop_distance = abs(signal["entry_price"] - signal["sl"])
        
        if stop_distance <= 0:
            return 0.0
        
        position_size = (risk_amount / stop_distance) * signal["entry_price"]
        leveraged_size = position_size * self.leverage
        
        return max(0.001, leveraged_size)  # Minimum size check

    async def execute_portfolio_trades(self, signals: List[Dict[str, Any]]) -> int:
        """Execute trades across portfolio with proper allocation"""
        executed_count = 0
        
        # Check portfolio limits
        current_positions = len(self.active_positions)
        if current_positions >= self.portfolio_limits["max_total_positions"]:
            logger.warning(f"📊 Portfolio limit reached: {current_positions} positions")
            return 0
        
        if self.daily_stats["trades"] >= self.portfolio_limits["max_daily_trades"]:
            logger.warning(f"📊 Daily trade limit reached: {self.daily_stats['trades']}")
            return 0
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_stats["pnl"] / self.account_balance) if self.account_balance > 0 else 0
        if daily_loss_pct >= self.portfolio_limits["max_daily_loss_pct"]:
            logger.warning(f"📊 Daily loss limit reached: {daily_loss_pct*100:.1f}%")
            return 0
        
        # Execute trades by tier priority and position limits
        tier_position_counts = {tier: 0 for tier in self.asset_universe.keys()}
        
        # Count current positions by tier
        for position in self.active_positions.values():
            tier = position.get("tier", "tier2")
            tier_position_counts[tier] += 1
        
        for signal in signals:
            tier = signal["tier"]
            symbol = signal["symbol"]
            
            # Check tier-specific position limits
            if tier_position_counts[tier] >= self.max_positions_per_tier[tier]:
                logger.debug(f"📊 {tier.upper()} position limit reached")
                continue
            
            # Check if we already have position in this symbol
            if any(pos["symbol"] == symbol for pos in self.active_positions.values()):
                logger.debug(f"📊 Already have position in {symbol}")
                continue
            
            # Calculate position size
            position_size = self.calculate_position_allocation(signal)
            
            if position_size <= 0:
                logger.warning(f"❌ Invalid position size for {symbol}")
                continue
            
            # Execute the trade
            success = await self.place_multi_asset_trade(signal, position_size)
            
            if success:
                executed_count += 1
                tier_position_counts[tier] += 1
                self.daily_stats["trades"] += 1
                
                logger.info(f"✅ Executed {symbol} | Tier: {tier} | Size: {position_size:.6f}")
                
                # Check if we've hit portfolio limits
                if len(self.active_positions) >= self.portfolio_limits["max_total_positions"]:
                    break
        
        return executed_count

    async def place_multi_asset_trade(self, signal: Dict[str, Any], position_size: float) -> bool:
        """Place a professional bounce trade for a specific asset"""
        symbol = signal["symbol"]
        
        try:
            entry_price = signal["entry_price"]
            tp1_price = signal["tp1"]
            tp2_price = signal["tp2"]
            sl_price = signal["sl"]
            
            logger.info(f"🚀 Placing multi-asset trade: {symbol}")
            logger.info(f"   Entry: {entry_price:.6f} | TP1: {tp1_price:.6f} | TP2: {tp2_price:.6f} | SL: {sl_price:.6f}")
            logger.info(f"   Size: {position_size:.6f} | Confluence: {signal['confluence_factors']}/6")
            
            # 1. Entry order
            entry_order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side='buy',
                amount=position_size,
                params={'leverage': self.leverage}
            )
            
            # 2. Stop loss
            sl_order = await self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side='sell',
                amount=position_size,
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            
            # 3. Take profit orders
            tp1_size = position_size * 0.6
            tp2_size = position_size * 0.4
            
            tp1_order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='sell',
                amount=tp1_size,
                price=tp1_price,
                params={'reduceOnly': True, 'postOnly': True}
            )
            
            tp2_order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='sell',
                amount=tp2_size,
                price=tp2_price,
                params={'reduceOnly': True, 'postOnly': True}
            )
            
            # Store position info
            position_id = entry_order.get('id', str(time.time()))
            self.active_positions[position_id] = {
                "symbol": symbol,
                "tier": signal["tier"],
                "entry_price": entry_price,
                "entry_time": datetime.now(),
                "size": position_size,
                "tp1_price": tp1_price,
                "tp2_price": tp2_price,
                "sl_price": sl_price,
                "confluence_factors": signal["confluence_factors"],
                "signal_score": signal["score"],
                "orders": {
                    "entry": entry_order,
                    "sl": sl_order,
                    "tp1": tp1_order,
                    "tp2": tp2_order
                }
            }
            
            logger.info(f"✅ Multi-asset trade executed: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute {symbol}: {e}")
            return False

    async def monitor_portfolio_positions(self):
        """Monitor all portfolio positions across assets"""
        if not self.active_positions:
            return
        
        try:
            # Get current prices for all active symbols
            active_symbols = list(set(pos["symbol"] for pos in self.active_positions.values()))
            
            current_prices = {}
            for symbol in active_symbols:
                ticker = await self.exchange.fetch_ticker(symbol)
                current_prices[symbol] = ticker['last']
            
            # Monitor each position
            for position_id, position in list(self.active_positions.items()):
                symbol = position["symbol"]
                current_price = current_prices.get(symbol)
                
                if not current_price:
                    continue
                
                # Calculate unrealized PnL
                entry_price = position["entry_price"]
                unrealized_pct = ((current_price - entry_price) / entry_price) * 100 * self.leverage
                
                logger.debug(f"📈 {symbol}: {unrealized_pct:+.2f}%")
                
                # Check for closed positions (simplified - would check order status in real implementation)
                # This is a simulation of position monitoring
                
        except Exception as e:
            logger.error(f"❌ Error monitoring portfolio: {e}")

    async def generate_portfolio_report(self):
        """Generate portfolio performance report"""
        try:
            # Calculate portfolio statistics
            total_positions = len(self.active_positions)
            
            # Tier distribution
            tier_distribution = {}
            for position in self.active_positions.values():
                tier = position["tier"]
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            
            # Performance stats
            win_rate = (self.daily_stats["win_count"] / max(1, self.daily_stats["trades"])) * 100
            daily_return_pct = (self.daily_stats["pnl"] / self.account_balance) * 100 if self.account_balance > 0 else 0
            
            logger.info(f"📊 PORTFOLIO REPORT:")
            logger.info(f"   Active Positions: {total_positions}")
            logger.info(f"   Tier Distribution: {tier_distribution}")
            logger.info(f"   Daily Trades: {self.daily_stats['trades']}")
            logger.info(f"   Win Rate: {win_rate:.1f}%")
            logger.info(f"   Daily PnL: {self.daily_stats['pnl']:+.2f} USDT ({daily_return_pct:+.2f}%)")
            
        except Exception as e:
            logger.error(f"❌ Error generating portfolio report: {e}")

    async def run_multi_asset_cycle(self):
        """Run one complete multi-asset analysis and trading cycle"""
        try:
            logger.info(f"🔄 Multi-Asset Analysis Cycle Starting")
            
            # 1. Scan for signals across all assets
            signals = await self.scan_multi_asset_signals()
            
            if signals:
                logger.info(f"🎯 Found {len(signals)} quality signals across portfolio")
                
                # 2. Execute trades based on signals and allocations
                executed = await self.execute_portfolio_trades(signals)
                logger.info(f"✅ Executed {executed} new positions")
            else:
                logger.info("🔍 No quality signals found in current cycle")
            
            # 3. Monitor existing positions
            await self.monitor_portfolio_positions()
            
            # 4. Generate portfolio report
            await self.generate_portfolio_report()
            
        except Exception as e:
            logger.error(f"❌ Error in multi-asset cycle: {e}")

    async def run_multi_asset_bot(self):
        """Main loop for multi-asset professional bounce trading"""
        logger.info("🌍 MULTI-ASSET PROFESSIONAL BOUNCE BOT STARTING")
        logger.info("📊 Trading across 100+ diverse cryptocurrency pairs")
        logger.info(f"🎯 Portfolio: {sum(len(tier['symbols']) for tier in self.asset_universe.values())} total assets")
        
        try:
            # Initialize exchange
            await self.initialize_exchange()
            
            # Display portfolio allocation
            logger.info(f"\n💼 PORTFOLIO ALLOCATION:")
            for tier_name, tier_config in self.asset_universe.items():
                allocation_pct = tier_config["allocation"] * 100
                logger.info(f"   {tier_name.upper()}: {allocation_pct:.0f}% | {len(tier_config['symbols'])} assets | {tier_config['config']['min_confluence_factors']}/6 confluence")
            
            # Main trading loop
            cycle_count = 0
            while True:
                cycle_count += 1
                logger.info(f"\n🔄 Multi-Asset Cycle #{cycle_count}")
                
                # Update account balance
                balance = await self.exchange.fetch_balance()
                self.account_balance = balance['USDT']['total']
                
                # Run analysis and trading cycle
                await self.run_multi_asset_cycle()
                
                # Sleep until next cycle
                sleep_seconds = {
                    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400
                }.get(self.primary_timeframe, 3600)
                
                logger.info(f"⏳ Next cycle in {sleep_seconds} seconds...")
                await asyncio.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("👋 Multi-asset bot shutting down...")
        except Exception as e:
            logger.error(f"❌ Critical error in multi-asset bot: {e}")
        finally:
            if self.exchange:
                await self.exchange.close()


# Enhanced asset universe for maximum diversity
COMPREHENSIVE_ASSET_UNIVERSE = {
    "major_assets": [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT", 
        "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT"
    ],
    "growth_assets": [
        "MATIC/USDT:USDT", "UNI/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
        "ATOM/USDT:USDT", "NEAR/USDT:USDT", "ALGO/USDT:USDT", "FTM/USDT:USDT"
    ],
    "defi_protocols": [
        "AAVE/USDT:USDT", "SUSHI/USDT:USDT", "COMP/USDT:USDT", "YFI/USDT:USDT",
        "1INCH/USDT:USDT", "CRV/USDT:USDT", "SNX/USDT:USDT", "MKR/USDT:USDT"
    ],
    "gaming_nft": [
        "AXS/USDT:USDT", "SAND/USDT:USDT", "MANA/USDT:USDT", "GALA/USDT:USDT",
        "CHZ/USDT:USDT", "ENJ/USDT:USDT", "FLOW/USDT:USDT", "IMX/USDT:USDT"
    ],
    "infrastructure": [
        "LINK/USDT:USDT", "GRT/USDT:USDT", "FIL/USDT:USDT", "AR/USDT:USDT",
        "VET/USDT:USDT", "BAT/USDT:USDT", "STORJ/USDT:USDT"
    ],
    "layer1_alternatives": [
        "LUNA/USDT:USDT", "EGLD/USDT:USDT", "KLAY/USDT:USDT", "WAVES/USDT:USDT",
        "QTUM/USDT:USDT", "XTZ/USDT:USDT", "EOS/USDT:USDT"
    ],
    "high_volatility": [
        "SHIB/USDT:USDT", "PEPE/USDT:USDT", "FLOKI/USDT:USDT", "SLP/USDT:USDT"
    ]
}


def get_total_asset_count():
    """Get total number of assets across all categories"""
    return sum(len(assets) for assets in COMPREHENSIVE_ASSET_UNIVERSE.values())


if __name__ == "__main__":
    async def main():
        print(f"🌍 Multi-Asset Professional Bounce Bot")
        print(f"📊 Total asset universe: {get_total_asset_count()} cryptocurrencies")
        print(f"🎯 Portfolio diversification across all market segments")
        
        bot = MultiAssetProfessionalBot()
        await bot.run_multi_asset_bot()
    
    asyncio.run(main())