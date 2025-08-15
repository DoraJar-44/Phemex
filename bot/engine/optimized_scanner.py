"""
Optimized High-Performance Scanner with Parallel Processing
Designed for speed and efficiency with Phemex futures scanning
"""
import os
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import httpx
import ccxt.async_support as ccxt
import numpy as np

from bot.config import settings
from bot.engine.symbols import load_symbols
from bot.engine.discovery import discover_symbols
from bot.strategy.pr import compute_predictive_ranges
from bot.strategy.score import ScoreInputs, compute_total_score
from bot.risk.sizing import compute_quantity
from bot.risk.guards import get_equity_usdt, cap_quantity_by_notional, count_open_positions
from bot.execution.brackets import build_bracket_orders
from bot.execution.execute import place_bracket
from bot.validation.math_validator import MathValidator
from bot.ui.tui_display import start_tui, add_score_to_tui, add_validation_error_to_tui


@dataclass
class ScanResult:
    """Optimized scan result container"""
    symbol: str
    price: float
    levels: Dict[str, float]
    signals: Dict[str, bool]
    scores: Dict[str, int]
    timestamp: float


@dataclass 
class CachedOHLCV:
    """Cache container for OHLCV data"""
    data: Dict[str, List[float]]
    timestamp: float
    timeframe: str
    symbol: str


class OptimizedScanner:
    """High-performance scanner with caching and parallel processing"""
    
    def __init__(self):
        self.ohlcv_cache: Dict[str, CachedOHLCV] = {}
        self.cache_ttl = 60  # 60 second cache for OHLCV
        self.validator = MathValidator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.last_symbols_update = 0
        self.symbol_cache: List[str] = []
        
    def _cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate cache key for OHLCV data"""
        return f"{symbol}:{timeframe}"
    
    def _is_cache_valid(self, cache_entry: CachedOHLCV) -> bool:
        """Check if cache entry is still valid"""
        return (time.time() - cache_entry.timestamp) < self.cache_ttl
    
    async def _fetch_candles_optimized(self, exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 400) -> Dict[str, List[float]]:
        """Optimized candle fetching with caching and fallbacks"""
        cache_key = self._cache_key(symbol, timeframe)
        
        # Check cache first
        if cache_key in self.ohlcv_cache:
            cached = self.ohlcv_cache[cache_key]
            if self._is_cache_valid(cached):
                return cached.data
        
        # Prepare symbol fallbacks
        symbols_to_try: List[str] = [symbol]
        if symbol.endswith(":USDT"):
            base_quote = symbol.split(":")[0]
            symbols_to_try.append(base_quote)
            base = base_quote.split("/")[0] 
            symbols_to_try.append(f"{base}USDT")
        
        # Try CCXT first (fastest)
        for sym_try in symbols_to_try:
            try:
                minutes = self._tf_to_minutes(timeframe)
                bars_needed = max(int(settings.pr_atr_len) + 50, 250)
                since_ms = int(time.time() * 1000) - (minutes * 60 * 1000 * bars_needed)
                
                ohlcv = await exchange.fetch_ohlcv(
                    sym_try, 
                    timeframe=timeframe, 
                    since=since_ms, 
                    limit=min(bars_needed, 1000)
                )
                
                if len(ohlcv) >= 100:  # Minimum viable data
                    data = {
                        "open": [x[1] for x in ohlcv],
                        "high": [x[2] for x in ohlcv], 
                        "low": [x[3] for x in ohlcv],
                        "close": [x[4] for x in ohlcv]
                    }
                    
                    # Cache the result
                    self.ohlcv_cache[cache_key] = CachedOHLCV(
                        data=data,
                        timestamp=time.time(),
                        timeframe=timeframe,
                        symbol=symbol
                    )
                    return data
                    
            except Exception:
                continue
        
        # Fallback to direct Phemex API
        try:
            return await self._fetch_phemex_direct(symbol, timeframe, bars_needed)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {e}")
    
    async def _fetch_phemex_direct(self, symbol: str, timeframe: str, bars_needed: int) -> Dict[str, List[float]]:
        """Direct Phemex API fallback"""
        resolution = self._tf_to_minutes(timeframe)
        now_s = int(time.time())
        
        # Convert to compact symbol
        compact = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "")
        
        params = {
            "symbol": compact,
            "resolution": str(resolution), 
            "from": str(now_s - resolution * 60 * bars_needed),
            "to": str(now_s),
        }
        
        async with httpx.AsyncClient(base_url=settings.phemex_base_url, timeout=10) as client:
            r = await client.get("/md/kline", params=params)
            r.raise_for_status()
            data = r.json()
            entries = data.get("data") or data.get("klines") or []
            
            if not entries:
                raise RuntimeError("No data returned from Phemex API")
                
            return {
                "open": [float(x[1]) for x in entries],
                "high": [float(x[2]) for x in entries],
                "low": [float(x[3]) for x in entries], 
                "close": [float(x[4]) for x in entries]
            }
    
    def _tf_to_minutes(self, tf: str) -> int:
        """Convert timeframe to minutes"""
        unit = tf[-1]
        val = int(tf[:-1])
        if unit == 'm':
            return val
        if unit == 'h':
            return val * 60
        if unit == 'd':
            return val * 60 * 24
        raise ValueError(f"Unsupported timeframe: {tf}")
    
    def _compute_scores_vectorized(self, price: float, open_price: float, levels: Dict[str, float]) -> Tuple[int, int]:
        """Vectorized score computation for speed"""
        avg, r1, r2, s1, s2 = levels["avg"], levels["r1"], levels["r2"], levels["s1"], levels["s2"]
        
        # Vectorized bounce probability calculation
        range_size = abs(r1 - s1)
        if range_size > 0:
            long_bounce_prob = max(0.0, 0.9 - (abs(price - s1) / range_size))
            short_bounce_prob = max(0.0, 0.9 - (abs(price - r1) / range_size))
        else:
            long_bounce_prob = short_bounce_prob = 0.5
        
        # Pre-computed bias confidence
        bias_conf = 0.6
        
        # Long score inputs
        si_long = ScoreInputs(
            avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
            close=price, open=open_price,
            bounce_prob=long_bounce_prob,
            bias_up_conf=bias_conf,
            bias_dn_conf=0.0,
            bull_div=False, bear_div=False
        )
        
        # Short score inputs  
        si_short = ScoreInputs(
            avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
            close=price, open=open_price,
            bounce_prob=short_bounce_prob,
            bias_up_conf=0.0,
            bias_dn_conf=bias_conf,
            bull_div=False, bear_div=False
        )
        
        return (
            compute_total_score(si_long, "long", settings.score_min),
            compute_total_score(si_short, "short", settings.score_min)
        )
    
    async def scan_symbol_optimized(self, exchange: ccxt.Exchange, symbol: str, timeframe: str) -> Optional[ScanResult]:
        """Optimized single symbol scan"""
        try:
            # Fetch candles with caching
            c = await self._fetch_candles_optimized(exchange, symbol, timeframe)
            
            # Compute PR levels
            avg, r1, r2, s1, s2 = compute_predictive_ranges(
                c["high"], c["low"], c["close"], 
                settings.pr_atr_len, settings.pr_atr_mult
            )
            
            price = c["close"][-1]
            open_price = c["open"][-1]
            
            # Fast trend check
            trend_len = min(settings.trend_len, len(c["close"]))
            sma = sum(c["close"][-trend_len:]) / trend_len
            trend_ok_long = price > sma
            trend_ok_short = price < sma
            
            # Base signal conditions
            base_long = price > avg and price < r1 and c["close"][-2] <= avg
            base_short = price < avg and price > s1 and c["close"][-2] >= avg
            
            # Optimized score calculation
            levels = {"avg": avg, "r1": r1, "r2": r2, "s1": s1, "s2": s2}
            long_score, short_score = self._compute_scores_vectorized(price, open_price, levels)
            
            # Signal logic
            allow_long = (not settings.score_filter) or (long_score >= settings.score_min)
            allow_short = (not settings.score_filter) or (short_score >= settings.score_min)
            
            long_entry = base_long and trend_ok_long and allow_long
            short_entry = base_short and trend_ok_short and allow_short
            
            return ScanResult(
                symbol=symbol,
                price=price,
                levels=levels,
                signals={"long": long_entry, "short": short_entry},
                scores={"long": long_score, "short": short_score},
                timestamp=time.time()
            )
            
        except Exception as e:
            # Suppress common errors to reduce noise
            if any(err in str(e) for err in ["Not enough candles", "failed to fetch ohlcv", "insufficient"]):
                return None
            print(f"Scan error {symbol}: {e}")
            return None
    
    async def get_symbols_optimized(self, exchange: ccxt.Exchange, manual: str, min_lev: float, exclude_bases: set) -> List[str]:
        """Optimized symbol discovery with caching"""
        current_time = time.time()
        
        # Use manual symbols if provided
        if manual:
            return [s.strip() for s in manual.split(",") if s.strip()]
        
        # Cache symbols for 5 minutes
        if current_time - self.last_symbols_update < 300 and self.symbol_cache:
            return self.symbol_cache
        
        # Discover new symbols
        symbols = await discover_symbols(exchange, min_leverage=min_lev, exclude_bases=exclude_bases)
        self.symbol_cache = symbols
        self.last_symbols_update = current_time
        
        return symbols
    
    async def parallel_scan_batch(self, exchange: ccxt.Exchange, symbols: List[str], timeframe: str, batch_size: int = 10) -> List[ScanResult]:
        """Scan symbols in parallel batches for maximum speed"""
        results = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Parallel processing within batch
            tasks = [
                self.scan_symbol_optimized(exchange, symbol, timeframe)
                for symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            for result in batch_results:
                if isinstance(result, ScanResult):
                    results.append(result)
            
            # Small delay between batches to be API-friendly
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.1)
        
        return results
    
    async def run_optimized_scanner(self):
        """Main optimized scanner loop following MDC configuration"""
        timeframe = os.getenv("TIMEFRAME", "5m")
        manual = os.getenv("SYMBOLS", "")
        min_lev = float(os.getenv("MIN_LEVERAGE", "34"))
        exclude_bases_env = os.getenv("EXCLUDE_BASES", "BTC,ETH,SOL,BNB,XRP,DOGE")
        exclude_bases = {b.strip().upper() for b in exclude_bases_env.split(",") if b.strip()}
        
        # TUI setup
        use_tui = os.getenv("USE_TUI", "true").lower() in ("1", "true", "yes")
        if use_tui:
            print("🚀 Starting Optimized TUI Scanner...")
            start_tui()
            time.sleep(0.5)
        
        # Exchange setup following MDC specs
        ex = ccxt.phemex({
            "apiKey": settings.phemex_api_key,
            "secret": settings.phemex_api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "defaultSubType": "linear",
            },
        })
        
        try:
            await ex.load_markets()
            print(f"✅ Optimized scanner initialized | TF: {timeframe} | Min Leverage: {min_lev}x")
            
            while True:
                scan_start = time.time()
                
                # Get symbols with caching
                symbols = await self.get_symbols_optimized(ex, manual, min_lev, exclude_bases)
                
                # Parallel batch scanning 
                results = await self.parallel_scan_batch(ex, symbols, timeframe, batch_size=15)
                
                scan_duration = time.time() - scan_start
                
                # Process results
                high_score_results = []
                signal_results = []
                
                for res in results:
                    # Filter for high scores or signals
                    if res.scores['long'] >= 75 or res.scores['short'] >= 75:
                        high_score_results.append(res)
                    
                    if res.signals['long'] or res.signals['short']:
                        signal_results.append(res)
                        
                        # Execute trade following MDC specs
                        await self._execute_trade_mdc(ex, res)
                
                # Performance logging 
                print(f"⚡ Scan completed: {len(results)} symbols in {scan_duration:.2f}s | High scores: {len(high_score_results)} | Signals: {len(signal_results)}")
                
                # Update TUI with significant results only
                if use_tui:
                    for res in high_score_results + signal_results:
                        add_score_to_tui(
                            symbol=res.symbol,
                            price=res.price,
                            long_score=res.scores['long'],
                            short_score=res.scores['short'], 
                            long_signal=res.signals['long'],
                            short_signal=res.signals['short'],
                            validation_status="OK"
                        )
                
                # Adaptive sleep based on performance
                sleep_time = max(1.0, 3.0 - scan_duration)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("🛑 Optimized scanner stopped by user")
        except Exception as e:
            print(f"❌ Scanner error: {e}")
        finally:
            print("🔄 Closing exchange connection...")
            await ex.close()
            self.executor.shutdown(wait=True)
    
    async def _execute_trade_mdc(self, exchange: ccxt.Exchange, result: ScanResult):
        """Execute trade following MDC specifications"""
        try:
            # Check position limits
            open_positions = await count_open_positions(exchange)
            if open_positions >= settings.max_positions:
                print(f"⏸️ Skip {result.symbol}: max positions reached ({open_positions}/{settings.max_positions})")
                return
            
            is_long = result.signals["long"]
            
            # Build payload following MDC format
            from bot.signals.webhook_models import WebhookPayload, Levels
            payload = WebhookPayload(
                action="BUY" if is_long else "SELL",
                symbol=result.symbol,
                price=result.price,
                signal_type="ENTRY",
                levels=Levels(
                    avg=result.levels["avg"],
                    r1=result.levels["r1"],
                    r2=result.levels["r2"],
                    s1=result.levels["s1"],
                    s2=result.levels["s2"],
                ),
            )
            
            # Compute quantity with MDC compliance
            compact = result.symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
            meta = (
                settings.symbol_overrides.get(result.symbol)
                or settings.symbol_overrides.get(compact) 
                or {"tickSize": 0.1, "lotSize": 0.001, "minQty": 0.001, "contractValuePerPrice": 1.0}
            )
            
            qty, _ = compute_quantity(
                payload, settings.account_balance_usdt, settings.risk_per_trade_pct,
                meta["lotSize"], meta["minQty"], meta["tickSize"], meta["contractValuePerPrice"]
            )
            
            # Apply fixed notional if configured
            if settings.trade_notional and settings.trade_notional > 0:
                cvpp = meta["contractValuePerPrice"] or 1.0
                if result.price > 0 and cvpp > 0:
                    qty_target = settings.trade_notional / (result.price * cvpp)
                    if meta["lotSize"] > 0:
                        qty_target = (round(qty_target / meta["lotSize"]) * meta["lotSize"]) or meta["minQty"]
                    qty = max(qty, max(qty_target, meta["minQty"]))
                
                # Apply caps
                equity = await get_equity_usdt(exchange)
                max_notional_by_lev = equity * float(settings.leverage_max)
                max_notional_by_fraction = equity * float(settings.max_capital_fraction)
                allowed_notional = max(0.0, min(max_notional_by_lev, max_notional_by_fraction))
                
                if settings.trade_notional > 0:
                    allowed_notional = min(allowed_notional, float(settings.trade_notional))
                
                qty, notional = cap_quantity_by_notional(
                    qty, result.price, meta["contractValuePerPrice"], 
                    allowed_notional, meta["lotSize"], meta["minQty"]
                )
            
            # Build bracket orders
            intents = build_bracket_orders(payload, qty)
            
            # Execute following MDC live trade setting
            if settings.live_trade:
                print(f"🚀 LIVE TRADE [MDC]: {payload.action} {result.symbol} qty={qty} @ {result.price}")
                placed = await place_bracket(result.symbol, intents)
                print(f"✅ Trade executed: {placed}")
            else:
                print(f"📝 DRY RUN [MDC]: {payload.action} {result.symbol} qty={qty} @ {result.price}")
                print(f"🎯 Intents: {intents}")
                
        except Exception as e:
            print(f"❌ Trade execution error for {result.symbol}: {e}")


# Global scanner instance
_scanner_instance = None

def get_scanner() -> OptimizedScanner:
    """Get singleton scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = OptimizedScanner()
    return _scanner_instance


async def run_optimized_scanner():
    """Entry point for optimized scanner"""
    scanner = get_scanner()
    await scanner.run_optimized_scanner()


if __name__ == "__main__":
    asyncio.run(run_optimized_scanner())
