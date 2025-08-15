"""
Intelligent Symbol Filtering for High-Performance Scanning
Reduces scan targets by focusing on most viable trading pairs
"""
import asyncio
import time
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from statistics import median, mean
import ccxt.async_support as ccxt

from bot.config import settings


@dataclass
class SymbolMetrics:
    """Metrics for symbol viability scoring"""
    symbol: str
    volume_24h_usdt: float
    price_change_24h: float
    leverage_available: float
    spread_bps: float  # basis points
    last_updated: float
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    viability_score: float = 0.0


class SmartSymbolFilter:
    """Intelligent symbol filtering based on trading viability"""
    
    def __init__(self):
        self.metrics_cache: Dict[str, SymbolMetrics] = {}
        self.cache_ttl = 300  # 5 minute cache
        self.min_volume_threshold = 1_000_000  # $1M daily volume minimum
        self.max_symbols_per_scan = 50  # Limit active symbols
        
    def _is_cache_valid(self, metrics: SymbolMetrics) -> bool:
        """Check if cached metrics are still valid"""
        return (time.time() - metrics.last_updated) < self.cache_ttl
    
    async def _fetch_symbol_metrics(self, exchange: ccxt.Exchange, symbol: str) -> Optional[SymbolMetrics]:
        """Fetch comprehensive metrics for a symbol"""
        try:
            # Get ticker data
            ticker = await exchange.fetch_ticker(symbol)
            
            # Get orderbook for spread calculation
            try:
                orderbook = await exchange.fetch_order_book(symbol, limit=5)
                if orderbook['bids'] and orderbook['asks']:
                    bid = orderbook['bids'][0][0]
                    ask = orderbook['asks'][0][0]
                    spread_bps = ((ask - bid) / bid) * 10000 if bid > 0 else 999
                else:
                    spread_bps = 999  # Very high spread if no orderbook
            except Exception:
                spread_bps = 999
            
            # Extract key metrics
            volume_24h = ticker.get('quoteVolume', 0) or 0
            price_change_24h = abs(ticker.get('percentage', 0) or 0)
            
            # Get leverage info from market data
            market = exchange.markets.get(symbol, {})
            leverage_available = market.get('limits', {}).get('leverage', {}).get('max', 1)
            
            return SymbolMetrics(
                symbol=symbol,
                volume_24h_usdt=volume_24h,
                price_change_24h=price_change_24h,
                leverage_available=leverage_available,
                spread_bps=spread_bps,
                last_updated=time.time()
            )
            
        except Exception as e:
            print(f"Error fetching metrics for {symbol}: {e}")
            return None
    
    def _calculate_viability_scores(self, metrics_list: List[SymbolMetrics]) -> List[SymbolMetrics]:
        """Calculate viability scores for all symbols"""
        if not metrics_list:
            return []
        
        # Extract values for percentile calculations
        volumes = [m.volume_24h_usdt for m in metrics_list]
        volatilities = [m.price_change_24h for m in metrics_list]
        spreads = [m.spread_bps for m in metrics_list]
        
        # Calculate percentiles for scoring
        vol_median = median(volumes) if volumes else 1
        vol_75th = sorted(volumes)[int(len(volumes) * 0.75)] if len(volumes) > 4 else vol_median
        
        volatility_median = median(volatilities) if volatilities else 1
        spread_median = median(spreads) if spreads else 50
        
        # Score each symbol
        for metrics in metrics_list:
            # Volume/Liquidity Score (0-40 points)
            if metrics.volume_24h_usdt >= vol_75th:
                liquidity_score = 40
            elif metrics.volume_24h_usdt >= vol_median:
                liquidity_score = 25
            elif metrics.volume_24h_usdt >= self.min_volume_threshold:
                liquidity_score = 15
            else:
                liquidity_score = 0
            
            # Volatility Score (0-30 points) - more volatility = better for PR strategy
            if metrics.price_change_24h >= volatility_median * 1.5:
                volatility_score = 30
            elif metrics.price_change_24h >= volatility_median:
                volatility_score = 20
            elif metrics.price_change_24h >= volatility_median * 0.5:
                volatility_score = 10
            else:
                volatility_score = 5
            
            # Spread Score (0-20 points) - tighter spreads are better
            if metrics.spread_bps <= 5:
                spread_score = 20
            elif metrics.spread_bps <= 10:
                spread_score = 15
            elif metrics.spread_bps <= 25:
                spread_score = 10
            elif metrics.spread_bps <= 50:
                spread_score = 5
            else:
                spread_score = 0
            
            # Leverage Score (0-10 points)
            if metrics.leverage_available >= 50:
                leverage_score = 10
            elif metrics.leverage_available >= 25:
                leverage_score = 7
            elif metrics.leverage_available >= 10:
                leverage_score = 5
            else:
                leverage_score = 2
            
            # Total viability score
            metrics.liquidity_score = liquidity_score
            metrics.volatility_score = volatility_score
            metrics.viability_score = liquidity_score + volatility_score + spread_score + leverage_score
        
        return metrics_list
    
    async def get_top_symbols(self, exchange: ccxt.Exchange, candidate_symbols: List[str], max_symbols: Optional[int] = None) -> List[str]:
        """Get top trading symbols based on viability metrics"""
        max_symbols = max_symbols or self.max_symbols_per_scan
        
        # Check cache first
        cached_metrics = []
        symbols_to_fetch = []
        
        for symbol in candidate_symbols:
            if symbol in self.metrics_cache and self._is_cache_valid(self.metrics_cache[symbol]):
                cached_metrics.append(self.metrics_cache[symbol])
            else:
                symbols_to_fetch.append(symbol)
        
        # Fetch metrics for uncached symbols in batches
        new_metrics = []
        batch_size = 10
        
        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i:i + batch_size]
            
            # Parallel fetch within batch
            tasks = [self._fetch_symbol_metrics(exchange, symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, SymbolMetrics):
                    new_metrics.append(result)
                    self.metrics_cache[result.symbol] = result
            
            # Small delay between batches
            if i + batch_size < len(symbols_to_fetch):
                await asyncio.sleep(0.1)
        
        # Combine cached and new metrics
        all_metrics = cached_metrics + new_metrics
        
        # Filter out low-volume symbols immediately
        viable_metrics = [
            m for m in all_metrics 
            if m.volume_24h_usdt >= self.min_volume_threshold and m.spread_bps < 100
        ]
        
        if not viable_metrics:
            print(f"⚠️ No viable symbols found from {len(candidate_symbols)} candidates")
            return candidate_symbols[:max_symbols]  # Fallback to original list
        
        # Calculate viability scores
        scored_metrics = self._calculate_viability_scores(viable_metrics)
        
        # Sort by viability score and take top symbols
        top_metrics = sorted(scored_metrics, key=lambda x: x.viability_score, reverse=True)[:max_symbols]
        
        # Log top performers
        print(f"🎯 Top symbols selected ({len(top_metrics)}/{len(candidate_symbols)}):")
        for i, metrics in enumerate(top_metrics[:10]):  # Show top 10
            print(f"  {i+1}. {metrics.symbol}: Score={metrics.viability_score:.0f} "
                  f"Vol=${metrics.volume_24h_usdt/1e6:.1f}M "
                  f"Change={metrics.price_change_24h:.1f}% "
                  f"Spread={metrics.spread_bps:.1f}bps")
        
        return [m.symbol for m in top_metrics]
    
    async def filter_symbols_by_performance(self, exchange: ccxt.Exchange, symbols: List[str], timeframe: str = "1h") -> List[str]:
        """Additional filtering based on recent price action performance"""
        if len(symbols) <= 20:  # Already small list
            return symbols
        
        performing_symbols = []
        
        try:
            # Quick volatility check using recent candles
            for symbol in symbols[:50]:  # Limit to avoid rate limits
                try:
                    # Get last 24 candles
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=24)
                    
                    if len(ohlcv) >= 12:
                        # Calculate recent volatility
                        highs = [x[2] for x in ohlcv[-12:]]
                        lows = [x[3] for x in ohlcv[-12:]]
                        closes = [x[4] for x in ohlcv[-12:]]
                        
                        # Average true range approximation
                        ranges = [(h - l) / c for h, l, c in zip(highs, lows, closes) if c > 0]
                        avg_range = mean(ranges) if ranges else 0
                        
                        # Include symbols with decent volatility (>0.5% average range)
                        if avg_range > 0.005:
                            performing_symbols.append(symbol)
                            
                except Exception:
                    continue
                    
                # Rate limiting
                await asyncio.sleep(0.05)
        
        except Exception as e:
            print(f"Performance filtering error: {e}")
            return symbols[:30]  # Fallback
        
        return performing_symbols if performing_symbols else symbols[:20]
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics for monitoring"""
        valid_entries = sum(1 for m in self.metrics_cache.values() if self._is_cache_valid(m))
        
        return {
            "total_cached": len(self.metrics_cache),
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / len(self.metrics_cache) if self.metrics_cache else 0,
            "min_volume_threshold": self.min_volume_threshold,
            "max_symbols_per_scan": self.max_symbols_per_scan
        }


# Global filter instance
_filter_instance = None

def get_smart_filter() -> SmartSymbolFilter:
    """Get singleton filter instance"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = SmartSymbolFilter()
    return _filter_instance
