#!/usr/bin/env python3
"""
AGGRESSIVE LIVE TRADING - 50X LEVERAGE
Ready-to-activate system with optimized settings for immediate execution
"""

import os
import asyncio
import json
import time
import logging
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import ccxt.async_support as ccxt

# Load environment
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/aggressive_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AggressiveLiveTrading:
    def __init__(self):
        self.exchange = None
        self.symbols_50x = []
        self.running = False
        
        # Load verified 50x symbols
        try:
            with open('/workspace/corrected_symbols.json', 'r') as f:
                data = json.load(f)
                self.symbols_50x = data.get('leverage_50x_symbols', [])
            logger.info(f"🔥 Loaded {len(self.symbols_50x)} verified 50x symbols")
        except Exception as e:
            logger.error(f"❌ Failed to load symbols: {e}")
            self.symbols_50x = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        
        # AGGRESSIVE SETTINGS for immediate execution
        self.live_trade = os.getenv("LIVE_TRADE", "true").lower() in ("1", "true", "yes")
        self.leverage_max = float(os.getenv("LEVERAGE_MAX", "50"))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))  # Increased to 1%
        self.max_positions = int(os.getenv("MAX_POSITIONS", "8"))  # Increased to 8
        self.score_min = int(os.getenv("SCORE_MIN", "65"))  # Lowered to 65 for more trades
        self.min_trade_usdt = float(os.getenv("MIN_TRADE_USDT", "5"))  # Minimum $5 per trade
        
        logger.info(f"🔥 AGGRESSIVE LIVE TRADING SETTINGS:")
        logger.info(f"   Live Trading: {'ENABLED' if self.live_trade else 'DISABLED'}")
        logger.info(f"   Max Leverage: {self.leverage_max}x")
        logger.info(f"   Risk per trade: {self.risk_per_trade}%")
        logger.info(f"   Max positions: {self.max_positions}")
        logger.info(f"   Score threshold: {self.score_min} (LOWERED for more trades)")
        logger.info(f"   Min trade size: ${self.min_trade_usdt}")
    
    async def initialize_exchange(self):
        """Initialize Phemex exchange connection"""
        try:
            self.exchange = ccxt.phemex({
                'apiKey': os.getenv("PHEMEX_API_KEY", ""),
                'secret': os.getenv("PHEMEX_API_SECRET", ""),
                'sandbox': False,  # Live trading
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # Futures trading
                    'recvWindow': 10000
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            
            total_usdt = balance.get('USDT', {}).get('total', 0)
            free_usdt = balance.get('USDT', {}).get('free', 0)
            
            logger.info(f"✅ Exchange connected successfully")
            logger.info(f"💰 Account Balance: ${total_usdt:.2f} USDT")
            logger.info(f"💰 Available Balance: ${free_usdt:.2f} USDT")
            
            if total_usdt < 10:
                logger.error("❌ INSUFFICIENT FUNDS: Need at least $10 USDT")
                logger.error("   Please deposit funds to continue live trading")
                return False
            else:
                logger.info(f"✅ SUFFICIENT FUNDS: Ready for aggressive trading")
                return True
            
        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
            return False
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '5m', limit: int = 50):
        """Fetch market data for a symbol - optimized for speed"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if len(ohlcv) < 10:
                return None
            
            return {
                'open': [x[1] for x in ohlcv],
                'high': [x[2] for x in ohlcv], 
                'low': [x[3] for x in ohlcv],
                'close': [x[4] for x in ohlcv],
                'volume': [x[5] for x in ohlcv]
            }
            
        except Exception as e:
            logger.error(f"❌ Data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_aggressive_score(self, data: Dict) -> int:
        """Aggressive scoring algorithm - designed to find more opportunities"""
        if not data or len(data['close']) < 10:
            return 0
        
        try:
            closes = data['close']
            highs = data['high']
            lows = data['low']
            volumes = data['volume']
            
            current = closes[-1]
            prev_10 = closes[-10] if len(closes) >= 10 else closes[0]
            prev_3 = closes[-3] if len(closes) >= 3 else closes[-1]
            
            # Aggressive momentum calculation
            momentum_10 = ((current - prev_10) / prev_10) * 100
            momentum_3 = ((current - prev_3) / prev_3) * 100
            
            # Volume analysis
            vol_avg = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else volumes[-1]
            vol_current = volumes[-1]
            vol_surge = vol_current / vol_avg if vol_avg > 0 else 1
            
            # Volatility (good for 50x leverage)
            recent_high = max(highs[-5:])
            recent_low = min(lows[-5:])
            volatility = ((recent_high - recent_low) / current) * 100
            
            # BASE SCORE - Start higher for more opportunities
            score = 60
            
            # MOMENTUM BONUSES (aggressive)
            if momentum_10 > 1:  # 1%+ move in 10 periods
                score += 15
            elif momentum_10 > 0:
                score += 10
            
            if momentum_3 > 0.5:  # 0.5%+ move in 3 periods  
                score += 10
            elif momentum_3 > 0:
                score += 5
            
            # VOLUME BONUSES (aggressive)
            if vol_surge > 2:  # 2x volume surge
                score += 15
            elif vol_surge > 1.5:
                score += 10
            elif vol_surge > 1.2:
                score += 5
            
            # VOLATILITY BONUS (good for leveraged trading)
            if volatility > 3:  # 3%+ volatility
                score += 10
            elif volatility > 1.5:
                score += 5
            
            # BREAKOUT DETECTION
            if current > max(closes[-5:-1]):  # Breaking above recent highs
                score += 10
            
            return min(max(int(score), 0), 100)
            
        except Exception as e:
            logger.error(f"Score calculation error: {e}")
            return 0
    
    async def place_live_order(self, symbol: str, side: str, amount_usdt: float):
        """Place actual live order with 50x leverage"""
        try:
            if not self.live_trade:
                logger.info(f"📝 DEMO MODE: Would {side.upper()} ${amount_usdt:.2f} of {symbol} at 50x")
                return {'id': 'demo_order', 'status': 'filled'}
            
            # Get symbol info
            market = self.exchange.market(symbol)
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position size with 50x leverage
            leveraged_amount = amount_usdt * self.leverage_max  # $5 * 50x = $250 position
            quantity = leveraged_amount / current_price
            
            # Round to market precision
            quantity = self.exchange.amount_to_precision(symbol, quantity)
            
            logger.info(f"🎯 PLACING LIVE ORDER:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side.upper()}")
            logger.info(f"   Capital: ${amount_usdt:.2f}")
            logger.info(f"   Leverage: {self.leverage_max}x")
            logger.info(f"   Position Size: ${leveraged_amount:.2f}")
            logger.info(f"   Quantity: {quantity}")
            logger.info(f"   Price: ${current_price:.4f}")
            
            # Place market order
            order = await self.exchange.create_market_order(
                symbol, side, quantity, None, None, {
                    'leverage': self.leverage_max
                }
            )
            
            logger.info(f"✅ ORDER PLACED: {order['id']}")
            logger.info(f"   Status: {order['status']}")
            logger.info(f"   Filled: {order.get('filled', 0)}")
            
            # Place stop-loss at 2% (liquidation protection)
            try:
                sl_price = current_price * 0.98 if side == 'buy' else current_price * 1.02
                sl_order = await self.exchange.create_order(
                    symbol, 'stop_market', side, quantity, None, {
                        'stopPrice': sl_price,
                        'reduceOnly': True
                    }
                )
                logger.info(f"🛡️ STOP LOSS SET: ${sl_price:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Stop loss failed: {e}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            return None
    
    async def aggressive_scan(self):
        """Aggressive scanning for immediate opportunities"""
        opportunities = []
        
        logger.info(f"🔍 AGGRESSIVE SCAN: Looking for 50x opportunities...")
        
        # Scan top 15 most liquid pairs for speed
        scan_symbols = self.symbols_50x[:15]
        
        for symbol in scan_symbols:
            try:
                # Fast data fetch
                data = await self.fetch_market_data(symbol, '5m', 20)
                if not data:
                    continue
                
                # Aggressive scoring
                score = self.calculate_aggressive_score(data)
                
                # Get current price
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                volume_24h = ticker.get('quoteVolume', 0)
                
                logger.info(f"📊 {symbol:<20} Score: {score:3d} Price: ${current_price:>10.4f} Vol: ${volume_24h/1000000:>6.1f}M")
                
                # AGGRESSIVE CRITERIA - Lower threshold
                if score >= self.score_min and volume_24h > 1000000:  # Min $1M daily volume
                    opportunities.append({
                        'symbol': symbol,
                        'score': score,
                        'price': current_price,
                        'volume_24h': volume_24h,
                        'data': data
                    })
                    logger.info(f"🎯 OPPORTUNITY FOUND: {symbol} (Score: {score})")
                
                await asyncio.sleep(0.1)  # Fast scanning
                
            except Exception as e:
                logger.error(f"❌ Scan error for {symbol}: {e}")
                continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ AGGRESSIVE SCAN COMPLETE: {len(opportunities)} opportunities found")
        return opportunities
    
    async def check_account_status(self):
        """Quick account status check"""
        try:
            balance = await self.exchange.fetch_balance()
            positions = await self.exchange.fetch_positions()
            
            total_usdt = balance.get('USDT', {}).get('total', 0)
            free_usdt = balance.get('USDT', {}).get('free', 0)
            active_positions = [p for p in positions if float(p.get('size', 0)) != 0]
            
            logger.info(f"💰 Account: ${total_usdt:.2f} USDT | Available: ${free_usdt:.2f} | Positions: {len(active_positions)}")
            
            return total_usdt, free_usdt, active_positions
            
        except Exception as e:
            logger.error(f"❌ Account check failed: {e}")
            return 0, 0, []
    
    async def trading_loop(self):
        """Aggressive trading loop - faster execution"""
        logger.info("🚀 STARTING AGGRESSIVE 50X TRADING LOOP")
        logger.info("=" * 60)
        
        self.running = True
        loop_count = 0
        
        while self.running:
            try:
                loop_count += 1
                logger.info(f"\n{'🔥'*20} AGGRESSIVE SCAN #{loop_count} {'🔥'*20}")
                logger.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                
                # Quick account check
                total_usdt, free_usdt, positions = await self.check_account_status()
                
                if total_usdt < 5:
                    logger.error("❌ INSUFFICIENT FUNDS: Need funds to continue")
                    logger.error("   Please deposit USDT to your Phemex account")
                    await asyncio.sleep(30)
                    continue
                
                # Skip if max positions reached
                if len(positions) >= self.max_positions:
                    logger.info(f"📊 Max positions reached ({len(positions)}/{self.max_positions})")
                    await asyncio.sleep(15)  # Shorter wait
                    continue
                
                # AGGRESSIVE SCAN
                opportunities = await self.aggressive_scan()
                
                # EXECUTE TRADES
                if opportunities:
                    for opp in opportunities[:3]:  # Take top 3 opportunities
                        symbol = opp['symbol']
                        score = opp['score']
                        
                        # Calculate trade size (1% of account, min $5)
                        trade_size = max(total_usdt * (self.risk_per_trade / 100), self.min_trade_usdt)
                        trade_size = min(trade_size, free_usdt * 0.8)  # Don't use all margin
                        
                        if trade_size >= self.min_trade_usdt and free_usdt > trade_size:
                            logger.info(f"🎯 EXECUTING TRADE: {symbol} (Score: {score})")
                            
                            # Place buy order
                            order = await self.place_live_order(symbol, 'buy', trade_size)
                            
                            if order:
                                logger.info(f"✅ TRADE EXECUTED: {symbol}")
                                break  # One trade per cycle
                        else:
                            logger.warning(f"⚠️ Insufficient margin for {symbol}: ${trade_size:.2f} needed, ${free_usdt:.2f} available")
                
                # Faster scan cycle - every 15 seconds
                wait_time = 15
                logger.info(f"⏰ Next aggressive scan in {wait_time}s...")
                
                for i in range(wait_time):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start aggressive trading system"""
        logger.info("🔥 AGGRESSIVE LIVE TRADING - 50X LEVERAGE")
        logger.info("=" * 60)
        logger.info("⚠️  HIGH RISK - HIGH REWARD TRADING MODE")
        logger.info("🎯 Optimized for immediate execution")
        logger.info("=" * 60)
        
        # Initialize exchange
        if not await self.initialize_exchange():
            logger.error("❌ Cannot start - exchange initialization failed")
            return False
        
        # Start trading
        await self.trading_loop()
        
        # Cleanup
        if self.exchange:
            await self.exchange.close()
        
        logger.info("✅ Aggressive trading system stopped")
        return True
    
    def stop(self):
        """Stop trading system"""
        logger.info("🛑 Stopping aggressive trading...")
        self.running = False

async def main():
    trader = AggressiveLiveTrading()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
        trader.stop()
    except Exception as e:
        logger.error(f"❌ System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())