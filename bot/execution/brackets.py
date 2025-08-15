from typing import Dict, Any
from bot.signals.webhook_models import WebhookPayload


TP1_FRACTION = 0.5  # 50% of total position
TP2_FRACTION = 0.5  # 50% of remaining position after TP1


def build_bracket_orders(payload: WebhookPayload, quantity: float) -> Dict[str, Any]:
	side = "buy" if payload.action == "BUY" else "sell"
	is_long = payload.action == "BUY"

	entry_price = payload.price
	lvl = payload.levels

	if is_long:
		tp1_price = lvl.r1
		tp2_price = lvl.r2 if lvl.r2 is not None else lvl.r1
		sl_price = lvl.s2 if lvl.s2 is not None else lvl.s1
	else:
		tp1_price = lvl.s1
		tp2_price = lvl.s2 if lvl.s2 is not None else lvl.s1
		sl_price = lvl.r2 if lvl.r2 is not None else lvl.r1

	orders = {
		"entry": {
			"side": side,
			"type": "market",
			"price": entry_price,
			"quantity": quantity,
		},
		"tp1": {
			"side": "sell" if is_long else "buy",
			"type": "limit",
			"price": tp1_price,
			"quantity": round(quantity * TP1_FRACTION, 8),
			"reduceOnly": True,
		},
		"tp2": {
			"side": "sell" if is_long else "buy",
			"type": "limit",
			"price": tp2_price,
			"quantity": round(quantity * (1 - TP1_FRACTION) * TP2_FRACTION, 8),
			"reduceOnly": True,
		},
		"sl": {
			"side": "sell" if is_long else "buy",
			"type": "stop_market",
			"stopPrice": sl_price,
			"quantity": quantity,
			"reduceOnly": True,
		},
	}

	return orders


