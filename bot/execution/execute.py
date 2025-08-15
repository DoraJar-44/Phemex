import uuid
from typing import Dict, Any
from bot.config import settings
from bot.exchange.phemex_client import get_client
from bot.exchange.ccxt_client import get_ccxt_client
from bot.utils.symbol_conversion import ccxt_to_phemex_symbol


def _key(prefix: str) -> str:
	return f"{prefix}-{uuid.uuid4().hex[:12]}"



async def place_bracket(symbol: str, intents: Dict[str, Any]) -> Dict[str, Any]:
	use_ccxt = getattr(settings, "use_ccxt", False)
	client = get_ccxt_client() if use_ccxt else get_client()
	res = {"placed": []}
	if not settings.live_trade:
		return {"dryRun": True, "intents": intents}

	def _normalize_symbol(sym: str) -> str:
		# Convert unified or TradingView formats to Phemex compact, e.g. BTC/USDT:USDT -> BTCUSDT
		return ccxt_to_phemex_symbol(sym)

	p_symbol = _normalize_symbol(symbol)

	entry = intents["entry"]
	tp1 = intents["tp1"]
	tp2 = intents["tp2"]
	sl = intents["sl"]

	# Entry
	if use_ccxt:
		placed_entry = await client.create_order(
			symbol=p_symbol,
			side=entry["side"],
			ord_type="market",
			amount=entry["quantity"],
			price=None,
			params={"clientOrderId": _key("entry"), "reduceOnly": False},
		)
	else:
		placed_entry = await client.place_order(
			symbol=p_symbol,
			side=entry["side"],
			ord_type="market",
			qty=entry["quantity"],
			client_order_id=_key("entry"),
			pos_side="Merged",
		)
	res["placed"].append({"entry": placed_entry})

	# Take Profits (reduceOnly limit)
	total_tp_qty = 0
	for i, leg in enumerate((tp1, tp2), start=1):
		if leg["quantity"] <= 0:
			continue
		# Validate TP quantity doesn't exceed position
		total_tp_qty += leg["quantity"]
		if total_tp_qty > entry["quantity"]:
			print(f"Warning: TP{i} quantity {leg['quantity']} would exceed position size. Skipping.")
			continue
		if use_ccxt:
			placed_tp = await client.create_order(
				symbol=p_symbol,
				side=leg["side"],
				ord_type="limit",
				amount=leg["quantity"],
				price=leg["price"],
				params={"reduceOnly": True, "clientOrderId": _key(f"tp{i}")},
			)
		else:
			placed_tp = await client.place_order(
				symbol=p_symbol,
				side=leg["side"],
				ord_type="limit",
				qty=leg["quantity"],
				price=leg["price"],
				reduce_only=True,
				client_order_id=_key(f"tp{i}"),
				pos_side="Merged",
			)
		res["placed"].append({f"tp{i}": placed_tp})

	# Stop Loss (reduceOnly stop market)
	if use_ccxt:
		placed_sl = await client.create_order(
			symbol=p_symbol,
			side=sl["side"],
			ord_type="market",
			amount=sl["quantity"],
			price=None,
			params={"stop": True, "triggerPrice": sl["stopPrice"], "reduceOnly": True, "clientOrderId": _key("sl")},
		)
	else:
		placed_sl = await client.place_order(
			symbol=p_symbol,
			side=sl["side"],
			ord_type="Stop",  # Changed from "stop" to match Phemex API
			qty=sl["quantity"],
			stop_px=sl["stopPrice"],
			reduce_only=True,
			client_order_id=_key("sl"),
			pos_side="Merged",
		)
	res["placed"].append({"sl": placed_sl})

	return res


