import asyncio
from typing import Any, Dict, Optional

# Use ccxt async support so we can await calls
import ccxt.async_support as ccxt  # type: ignore

from bot.config import settings


def _to_ccxt_symbol(usdt_contract_symbol: str) -> str:
	"""Convert symbols like BTCUSDT -> BTC/USDT:USDT for CCXT linear swaps."""
	s = usdt_contract_symbol.upper().strip()
	if s.endswith("USDT") and "/" not in s:
		base = s[:-4]
		return f"{base}/USDT:USDT"
	return s


class CCXTPhemexClient:
	def __init__(self) -> None:
		self.exchange = ccxt.phemex({
			"apiKey": settings.phemex_api_key,
			"secret": settings.phemex_api_secret,
			"enableRateLimit": True,
			"options": {
				"defaultType": "swap",
			},
		})

	async def close(self) -> None:
		try:
			await self.exchange.close()
		except Exception:
			pass

	async def create_order(
		self,
		symbol: str,
		side: str,
		ord_type: str,
		amount: float,
		price: Optional[float] = None,
		params: Optional[Dict[str, Any]] = None,
	) -> Any:
		ccxt_symbol = _to_ccxt_symbol(symbol)
		params = params or {}
		# CCXT expects lower-case side and type
		return await self.exchange.create_order(ccxt_symbol, ord_type.lower(), side.lower(), amount, price, params)


_client_singleton: Optional[CCXTPhemexClient] = None


def get_ccxt_client() -> CCXTPhemexClient:
	global _client_singleton
	if _client_singleton is None:
		_client_singleton = CCXTPhemexClient()
	return _client_singleton


