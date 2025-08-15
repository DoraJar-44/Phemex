from typing import List, Set


async def discover_symbols(exchange, min_leverage: float = 34.0, exclude_bases: Set[str] | None = None) -> List[str]:
	"""Discover Phemex linear USDT perpetuals with leverage >= min_leverage.

	Returns symbols formatted for ccxt, e.g., "ABC/USDT:USDT".
	"""
	exclude = {b.upper() for b in (exclude_bases or set())}
	markets = await exchange.load_markets()
	results: List[str] = []
	for symbol, m in markets.items():
		if not m.get("contract"):
			continue
		if not m.get("linear"):
			continue
		if (m.get("quote") or m.get("settle")) not in ("USDT",):
			continue
		if not m.get("active", True):
			continue
		base = (m.get("base") or "").upper()
		if base in exclude:
			continue
		# Leverage
		max_lev = None
		limits = m.get("limits") or {}
		lev = limits.get("leverage") or {}
		max_lev = lev.get("max")
		if max_lev is None:
			info = m.get("info") or {}
			# Try common fields
			max_lev = (
				info.get("maxLeverage")
				or info.get("maxLeverageRate")
				or info.get("leverageUpper")
			)
		try:
			max_lev = float(max_lev) if max_lev is not None else None
		except Exception:
			max_lev = None
		if max_lev is None or max_lev < float(min_leverage):
			continue
		# Use ccxt unified symbol (already in markets key), e.g., "ABC/USDT:USDT"
		results.append(symbol)
	return sorted(results)


