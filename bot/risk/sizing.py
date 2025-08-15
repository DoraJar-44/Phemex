from typing import Tuple
from bot.signals.webhook_models import WebhookPayload


def round_to_step(value: float, step: float) -> float:
	if step <= 0:
		return value
	return round(value / step) * step


def clamp_min(value: float, minimum: float) -> float:
	return value if value >= minimum else minimum


def choose_sl_price(payload: WebhookPayload) -> float:
	lvl = payload.levels
	# Prefer far band (s2/r2). Fallback to near band (s1/r1) if missing.
	if payload.action == "BUY":
		return (lvl.s2 if lvl.s2 is not None else lvl.s1)
	else:
		return (lvl.r2 if lvl.r2 is not None else lvl.r1)


def compute_quantity(
	payload: WebhookPayload,
	account_balance_usdt: float,
	risk_per_trade_pct: float,
	lot_size: float,
	min_qty: float,
	price_tick: float,
	contract_value_per_price: float = 1.0,
	existing_position_size: float = 0.0,
) -> Tuple[float, float]:
	entry_price = payload.price
	sl_price = choose_sl_price(payload)
	distance = abs(entry_price - sl_price)
	distance = max(distance, price_tick)
	risk_amount = account_balance_usdt * (risk_per_trade_pct / 100.0)
	if contract_value_per_price <= 0:
		contract_value_per_price = 1.0
	qty_raw = risk_amount / (distance * contract_value_per_price)
	qty = round_to_step(qty_raw, lot_size)
	qty = clamp_min(qty, min_qty)
	
	# Position awareness: if we have existing position in same direction, reduce new order size
	if existing_position_size != 0.0:
		same_direction = (
			(payload.action == "BUY" and existing_position_size > 0) or
			(payload.action == "SELL" and existing_position_size < 0)
		)
		if same_direction:
			# Reduce size by 50% if adding to existing position
			qty = qty * 0.5
			qty = round_to_step(qty, lot_size)
			qty = clamp_min(qty, min_qty)
	
	return qty, distance


