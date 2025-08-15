import os
import asyncio
import time
from typing import Dict, Any, List
import httpx

import ccxt.async_support as ccxt

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


def _tf_to_minutes(tf: str) -> int:
	unit = tf[-1]
	val = int(tf[:-1])
	if unit == 'm':
		return val
	if unit == 'h':
		return val * 60
	if unit == 'd':
		return val * 60 * 24
	raise ValueError(f"Unsupported timeframe: {tf}")


async def _fetch_candles(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 400) -> Dict[str, List[float]]:
    """Fetch candles with Phemex symbol fallbacks.

    Tries unified symbol as-is, then tries without ":USDT", then tries compact "BASEUSDT".
    """
    symbols_to_try: List[str] = [symbol]
    if symbol.endswith(":USDT"):
        base_quote = symbol.split(":")[0]  # e.g. BTC/USDT
        symbols_to_try.append(base_quote)
        base = base_quote.split("/")[0]
        symbols_to_try.append(f"{base}USDT")
    last_err: Exception | None = None
    # compute since to help exchanges that need explicit time range
    try:
        minutes = _tf_to_minutes(timeframe)
        bars_needed = max(int(settings.pr_atr_len) + 50, 250)
        since_ms = int(time.time() * 1000) - (minutes * 60 * 1000 * bars_needed)
    except Exception:
        since_ms = None
    for sym_try in symbols_to_try:
        try:
            limit_req = min(max(int(settings.pr_atr_len) + 50, 250), 1000)
            if since_ms is not None:
                ohlcv = await exchange.fetch_ohlcv(sym_try, timeframe=timeframe, since=since_ms, limit=limit_req)
            else:
                ohlcv = await exchange.fetch_ohlcv(sym_try, timeframe=timeframe, limit=limit_req)
            open_ = [x[1] for x in ohlcv]
            high = [x[2] for x in ohlcv]
            low = [x[3] for x in ohlcv]
            close = [x[4] for x in ohlcv]
            return {"open": open_, "high": high, "low": low, "close": close}
        except Exception as e:
            last_err = e
            continue
    # Fallback: direct Phemex public kline API
    try:
        # Map timeframe to minutes integer
        res_map_unit = _tf_to_minutes(timeframe)
        resolution = max(1, int(res_map_unit))
        now_s = int(time.time())
        # Convert any candidate to compact symbol id like BTCUSDT
        for sym_try in symbols_to_try:
            compact = sym_try.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "")
            bars_needed = max(int(settings.pr_atr_len) + 50, 250)
            params = {
                "symbol": compact,
                "resolution": str(resolution),
                "from": str(now_s - resolution * 60 * bars_needed),
                "to": str(now_s),
            }
            async with httpx.AsyncClient(base_url=settings.phemex_base_url, timeout=15) as client:
                r = await client.get("/md/kline", params=params)
                r.raise_for_status()
                data = r.json()
                entries = data.get("data") or data.get("klines") or []
                if not entries:
                    continue
                # Phemex returns [timestamp, open, high, low, close, volume] or similar; normalize
                open_ = [float(x[1]) for x in entries]
                high = [float(x[2]) for x in entries]
                low = [float(x[3]) for x in entries]
                close = [float(x[4]) for x in entries]
                return {"open": open_, "high": high, "low": low, "close": close}
    except Exception as e:
        last_err = e
    raise last_err or RuntimeError(f"failed to fetch ohlcv for {symbol}")


async def scan_once(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> Dict[str, Any]:
	c = await _fetch_candles(exchange, symbol, timeframe)
	avg, r1, r2, s1, s2 = compute_predictive_ranges(c["high"], c["low"], c["close"], settings.pr_atr_len, settings.pr_atr_mult)
	price = c["close"][-1]
	# Basic trend SMA check (approx):
	sma = sum(c["close"][-settings.trend_len:]) / max(1, settings.trend_len)
	trend_ok_long = price > sma
	trend_ok_short = price < sma
	# Base PR signals (close vs avg band)
	base_long = price > avg and price < r1 and c["close"][-2] <= avg
	base_short = price < avg and price > s1 and c["close"][-2] >= avg

	# Enhanced Score calculation with proper inputs
	# Calculate bounce probability based on price position
	range_size = abs(r1 - s1)
	if range_size > 0:
		# For longs, closer to support = higher bounce prob
		long_bounce_prob = max(0, 0.9 - (abs(price - s1) / range_size))
		# For shorts, closer to resistance = higher bounce prob  
		short_bounce_prob = max(0, 0.9 - (abs(price - r1) / range_size))
	else:
		long_bounce_prob = 0.5
		short_bounce_prob = 0.5
	
	# Basic trend confidence based on SMA position
	trend_strength = abs(price - sma) / sma if sma > 0 else 0
	trend_conf = min(0.8, trend_strength * 10)  # Scale to 0-0.8 range
	
	# Create score inputs for long
	si_long = ScoreInputs(
		avg=avg, r1=r1, r2=r2, s1=s1, s2=s2, 
		close=price, open=c["open"][-1],
		bounce_prob=long_bounce_prob,
		bias_up_conf=trend_conf if trend_ok_long else 0.2,
		bias_dn_conf=0.0,
		bull_div=False,  # Could add divergence detection
		bear_div=False
	)
	
	# Create score inputs for short
	si_short = ScoreInputs(
		avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
		close=price, open=c["open"][-1],
		bounce_prob=short_bounce_prob,
		bias_up_conf=0.0,
		bias_dn_conf=trend_conf if trend_ok_short else 0.2,
		bull_div=False,
		bear_div=False
	)
	
	long_score = compute_total_score(si_long, "long", settings.score_min)
	short_score = compute_total_score(si_short, "short", settings.score_min)
	allow_long = (not settings.score_filter) or (long_score >= settings.score_min)
	allow_short = (not settings.score_filter) or (short_score >= settings.score_min)

	long_entry = base_long and trend_ok_long and allow_long
	short_entry = base_short and trend_ok_short and allow_short

	return {
		"symbol": symbol,
		"price": price,
		"levels": {"avg": avg, "r1": r1, "r2": r2, "s1": s1, "s2": s2},
		"signals": {"long": long_entry, "short": short_entry},
		"scores": {"long": long_score, "short": short_score},
	}


async def run_scanner():
	timeframe = os.getenv("TIMEFRAME", "5m")
	manual = os.getenv("SYMBOLS", "")
	min_lev = float(os.getenv("MIN_LEVERAGE", "34"))
	exclude_bases_env = os.getenv("EXCLUDE_BASES", "BTC,ETH,SOL,BNB,XRP,DOGE")
	exclude_bases = {b.strip().upper() for b in exclude_bases_env.split(",") if b.strip()}
	
	# Initialize validation and TUI
	validator = MathValidator()
	use_tui = os.getenv("USE_TUI", "true").lower() in ("1", "true", "yes")
	
	if use_tui:
		print("Starting TUI... Press 'q' to quit TUI interface.")
		start_tui()
		time.sleep(0.5)  # Give TUI time to initialize
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
		# Ensure markets are loaded for unified symbol handling
		await ex.load_markets()
		# Discover symbols with leverage >= 34, excluding configured bases
		if manual:
			symbols = [s.strip() for s in manual.split(",") if s.strip()]
		else:
			symbols = await discover_symbols(ex, min_leverage=min_lev, exclude_bases=exclude_bases)
		while True:
			for sym in symbols:
				try:
					res = await scan_once(ex, sym, timeframe)
					
					# Validate scoring mathematics
					validation_status = "OK"
					try:
						si = ScoreInputs(
							avg=res["levels"]["avg"],
							r1=res["levels"]["r1"],
							r2=res["levels"]["r2"],
							s1=res["levels"]["s1"],
							s2=res["levels"]["s2"],
							close=res["price"],
							open=res["price"],  # Approximate
						)
						
						# Validate long side
						long_validation = validator.validate_scoring_system(si, "long")
						if not long_validation.is_valid:
							validation_status = "LONG_ERROR"
							if use_tui:
								add_validation_error_to_tui(f"{sym} Long: {long_validation.errors[0]}")
								
						# Validate short side
						short_validation = validator.validate_scoring_system(si, "short")
						if not short_validation.is_valid:
							validation_status = "SHORT_ERROR"
							if use_tui:
								add_validation_error_to_tui(f"{sym} Short: {short_validation.errors[0]}")
								
					except Exception as e:
						validation_status = "VALIDATION_FAILED"
						if use_tui:
							add_validation_error_to_tui(f"{sym} Validation: {str(e)}")
					
					# Send to TUI only for high scores or signals to reduce spam
					if use_tui and (res['scores']['long'] >= 75 or res['scores']['short'] >= 75 or res['signals']['long'] or res['signals']['short']):
						add_score_to_tui(
							symbol=sym,
							price=res["price"],
							long_score=res["scores"]["long"],
							short_score=res["scores"]["short"],
							long_signal=res["signals"]["long"],
							short_signal=res["signals"]["short"],
							validation_status=validation_status
						)
					
					# Only print high scores or signals to reduce spam
					if res['scores']['long'] >= 80 or res['scores']['short'] >= 80 or res['signals']['long'] or res['signals']['short']:
						validation_indicator = "✅" if validation_status == "OK" else "❌"
						print(f"{validation_indicator} {res['symbol']} {res['price']:.4f} score L/S: {res['scores']['long']}/{res['scores']['short']} long? {res['signals']['long']} short? {res['signals']['short']}")
					# Execute when true
					if res["signals"]["long"] or res["signals"]["short"]:
						# Enforce max open positions
						open_positions = await count_open_positions(ex)
						if open_positions >= settings.max_positions:
							print(f"skip {sym}: max positions reached ({open_positions}/{settings.max_positions})")
							continue
						is_long = res["signals"]["long"]
						from bot.signals.webhook_models import WebhookPayload, Levels
						payload = WebhookPayload(
							action="BUY" if is_long else "SELL",
							symbol=sym,
							price=res["price"],
							signal_type="ENTRY",
							levels=Levels(
								avg=res["levels"]["avg"],
								r1=res["levels"]["r1"],
								r2=res["levels"]["r2"],
								s1=res["levels"]["s1"],
								s2=res["levels"]["s2"],
							),
						)
						# Compute meta and size now that we have a signal
						compact = sym.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
						meta = (
							settings.symbol_overrides.get(sym)
							or settings.symbol_overrides.get(compact)
							or {"tickSize": 0.1, "lotSize": 0.001, "minQty": 0.001, "contractValuePerPrice": 1.0}
						)
						qty, _ = compute_quantity(payload, settings.account_balance_usdt, settings.risk_per_trade_pct, meta["lotSize"], meta["minQty"], meta["tickSize"], meta["contractValuePerPrice"])
						# If a fixed per-trade notional is configured, upsize to target notional first (respecting lot/min), then apply caps
						if settings.trade_notional and settings.trade_notional > 0:
							cvpp = meta["contractValuePerPrice"] or 1.0
							if res["price"] > 0 and cvpp > 0:
								qty_target = settings.trade_notional / (res["price"] * cvpp)
								# round to step and ensure >= min
								if meta["lotSize"] > 0:
									qty_target = (round(qty_target / meta["lotSize"]) * meta["lotSize"]) or meta["minQty"]
								qty = max(qty, max(qty_target, meta["minQty"]))
							# Caps: leverage, max capital fraction, per-trade notional target if provided
							equity = await get_equity_usdt(ex)
							max_notional_by_lev = equity * float(settings.leverage_max)
							max_notional_by_fraction = equity * float(settings.max_capital_fraction)
							allowed_notional = max(0.0, min(max_notional_by_lev, max_notional_by_fraction))
							# If trade_notional > 0, further clamp to that fixed notional per trade
							if settings.trade_notional and settings.trade_notional > 0:
								allowed_notional = min(allowed_notional, float(settings.trade_notional))
							qty, notional = cap_quantity_by_notional(qty, res["price"], meta["contractValuePerPrice"], allowed_notional, meta["lotSize"], meta["minQty"])
						intents = build_bracket_orders(payload, qty)
						if settings.live_trade:
							print(f"🚀 LIVE TRADE: Executing {payload.action} {sym} qty={qty}")
							placed = await place_bracket(sym, intents)
							print({"placed": placed})
						else:
							print(f"📝 DRY RUN: Would execute {payload.action} {sym} qty={qty}")
							print({"dryRun": True, "intents": intents})
				except Exception as e:
					msg = str(e)
					# Skip noisy prints for shallow history symbols
					if "Not enough candles" in msg or "failed to fetch ohlcv" in msg:
						continue
					print(f"scan error {sym}: {e}")
			await asyncio.sleep(2)  # Slower scan to reduce spam
	except KeyboardInterrupt:
		print("Scanner interrupted by user")
	except Exception as e:
		print(f"Scanner error: {e}")
	finally:
		print("Closing exchange connection...")
		try:
			await ex.close()
		except Exception:
			pass


if __name__ == "__main__":
	# Auto-reload test - this change should trigger restart
	asyncio.run(run_scanner())
