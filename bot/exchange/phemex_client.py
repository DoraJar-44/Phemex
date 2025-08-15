import hmac
import time
from typing import Any, Dict, Optional, Tuple
from hashlib import sha256
import httpx
import json
import os
from bot.config import settings


class PhemexClient:
	def __init__(self, api_key: str, api_secret: str, base_url: str):
		self.api_key = api_key
		self.api_secret = api_secret.encode()
		self.base_url = base_url.rstrip("/")
		self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
		self._scales_cache: Dict[str, Tuple[int, int]] = {}

	async def close(self):
		await self.client.aclose()

	def _sign(self, path: str, query: str = "", body: str = "") -> Dict[str, str]:
		# MDC: expiry in seconds, future timestamp; signature = HMAC(secret, path+query+expiry+body)
		expiry = str(int(time.time()) + 60)
		msg = f"{path}{query}{expiry}{body}"
		sig = hmac.new(self.api_secret, msg.encode(), sha256).hexdigest()
		return {
			"x-phemex-access-token": self.api_key,
			"x-phemex-request-expiry": expiry,
			"x-phemex-request-signature": sig,
			"Content-Type": "application/json",
		}

	def _get_symbol_scales(self, symbol: str) -> Tuple[int, int]:
		# Return (priceScale, qtyScale). Cache per process. Try local products.json, else infer from overrides.
		sym = symbol.upper()
		if sym in self._scales_cache:
			return self._scales_cache[sym]
		# Try local products.json
		try:
			root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
			with open(os.path.join(root, "products.json"), "r", encoding="utf-8") as f:
				data = json.load(f)
				# Accept common shapes
				products = data.get("products") or data.get("data") or data
				if isinstance(products, dict):
					products = products.get("products") or products.get("rows") or []
				for p in products or []:
					psym = (p.get("symbol") or p.get("symbolCode") or "").upper()
					if psym == sym:
						price_scale = int(p.get("priceScale") or p.get("pricePrecision") or 0)
						qty_scale = int(p.get("qtyScale") or p.get("qtyPrecision") or 0)
						self._scales_cache[sym] = (price_scale, qty_scale)
						return price_scale, qty_scale
		except Exception:
			pass
		# Fallback: infer from overrides tick/lot
		ovr = settings.symbol_overrides.get(sym) or {}
		def _infer_scale(step: float) -> int:
			try:
				text = ("%f" % float(step)).rstrip("0").split(".")
				return len(text[1]) if len(text) > 1 else 0
			except Exception:
				return 0
		price_scale = _infer_scale(ovr.get("tickSize", 0.01))
		qty_scale = _infer_scale(ovr.get("lotSize", 0.001))
		self._scales_cache[sym] = (price_scale, qty_scale)
		return price_scale, qty_scale

	async def get_products(self) -> Any:
		r = await self.client.get("/public/products")
		r.raise_for_status()
		return r.json()

	async def place_order(
		self,
		symbol: str,
		side: str,
		ord_type: str,
		qty: float,
		price: Optional[float] = None,
		client_order_id: Optional[str] = None,
		reduce_only: Optional[bool] = None,
		stop_px: Optional[float] = None,
		pos_side: Optional[str] = None,
	) -> Any:
		# MDC-compliant: PUT /g-orders/create with scaled integers (Rp/Rq)
		path = "/g-orders/create"
		price_scale, qty_scale = self._get_symbol_scales(symbol)
		def to_rp(val: Optional[float]) -> Optional[str]:
			if val is None:
				return None
			return str(int(round(float(val) * (10 ** price_scale))))
		def to_rq(val: float) -> str:
			return str(int(round(float(val) * (10 ** qty_scale))))
		# Map ordType to Phemex enum case
		ord_type_map = {
			"market": "Market",
			"limit": "Limit",
			"stop": "Stop",
			"stop_limit": "StopLimit",
			"take_profit": "TakeProfit",
			"take_profit_limit": "TakeProfitLimit",
		}
		body: Dict[str, Any] = {
			"symbol": symbol,
			"side": side.capitalize(),
			"ordType": ord_type_map.get(ord_type.lower(), ord_type),
			"orderQtyRq": to_rq(qty),
			"timeInForce": "GoodTillCancel",
			"posSide": (pos_side or "Merged"),
		}
		if price is not None:
			body["priceRp"] = to_rp(price)
		if client_order_id:
			body["clOrdID"] = client_order_id
		if reduce_only is not None:
			body["reduceOnly"] = bool(reduce_only)
		if stop_px is not None:
			body["stopPxRp"] = to_rp(stop_px)

		body_json = json.dumps(body, separators=(",", ":"))
		headers = self._sign(path, "", body_json)
		r = await self.client.put(path, headers=headers, content=body_json)
		r.raise_for_status()
		return r.json()


_client_singleton: Optional[PhemexClient] = None


def get_client() -> PhemexClient:
	global _client_singleton
	if _client_singleton is None:
		_client_singleton = PhemexClient(settings.phemex_api_key, settings.phemex_api_secret, settings.phemex_base_url)
	return _client_singleton


