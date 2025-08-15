from typing import Tuple
import math


async def get_equity_usdt(exchange) -> float:
	"""Fetch USDT equity from exchange; fallback to 0.0 on error."""
	try:
		bal = await exchange.fetch_balance()
		usdt = bal.get("USDT") or {}
		total = usdt.get("total")
		if total is None:
			total = usdt.get("free") or 0.0
		return float(total or 0.0)
	except Exception:
		return 0.0


async def count_open_positions(exchange) -> int:
	"""Return count of open positions with non-zero size/notional."""
	try:
		positions = await exchange.fetch_positions()
		count = 0
		for p in positions or []:
			# try multiple fields as ccxt varies per exchange
			contracts = p.get("contracts") or p.get("contractSize") or 0
			size = p.get("size") or 0
			notional = p.get("notional") or p.get("info", {}).get("unrealisedPnl", 0)  # fallback to any numeric
			is_open = False
			try:
				is_open = (float(contracts) or float(size) or float(notional)) != 0.0
			except Exception:
				is_open = False
			if is_open:
				count += 1
		return count
	except Exception:
		return 0


def cap_quantity_by_notional(
	qty: float,
	price: float,
	contract_value_per_price: float,
	max_notional: float,
	lot_size: float,
	min_qty: float,
) -> Tuple[float, float]:
	"""Return (qty_capped, notional_capped)."""
	if price <= 0 or contract_value_per_price <= 0:
		return qty, qty * price
	notional = qty * price * contract_value_per_price
	if max_notional <= 0 or notional <= max_notional:
		return qty, notional
		# downsize
	qty_cap = max_notional / (price * contract_value_per_price)
	# round to lot size
	if lot_size > 0:
		qty_cap = math.floor(qty_cap / lot_size) * lot_size
	qty_cap = max(qty_cap, min_qty)
	return qty_cap, qty_cap * price * contract_value_per_price


