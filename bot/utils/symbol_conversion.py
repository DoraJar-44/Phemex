"""Symbol format conversion utilities for different exchanges."""


def ccxt_to_phemex_symbol(ccxt_symbol: str) -> str:
	"""Convert CCXT unified symbol to Phemex API format.
	
	Examples:
	- 'BTC/USDT:USDT' -> 'BTCUSDT'
	- 'ETH/USDT:USDT' -> 'ETHUSDT'
	- 'ALGO/USDT:USDT' -> 'ALGOUSDT'
	"""
	if '/' not in ccxt_symbol:
		return ccxt_symbol  # Already in compact format
	
	parts = ccxt_symbol.split('/')
	if len(parts) != 2:
		return ccxt_symbol
	
	base = parts[0]
	quote_part = parts[1]
	
	# Handle ':USDT' suffix in perpetual contracts
	if ':' in quote_part:
		quote = quote_part.split(':')[0]
	else:
		quote = quote_part
	
	return f"{base}{quote}"


def phemex_to_ccxt_symbol(phemex_symbol: str, contract_type: str = "swap") -> str:
	"""Convert Phemex API symbol to CCXT unified format.
	
	Examples:
	- 'BTCUSDT' -> 'BTC/USDT:USDT' (for perpetuals)
	- 'ETHUSDT' -> 'ETH/USDT:USDT' (for perpetuals)
	"""
	if '/' in phemex_symbol:
		return phemex_symbol  # Already in CCXT format
	
	# For USDT perpetuals, extract base by removing USDT suffix
	if phemex_symbol.endswith('USDT'):
		base = phemex_symbol[:-4]
		if contract_type == "swap":
			return f"{base}/USDT:USDT"
		else:
			return f"{base}/USDT"
	
	return phemex_symbol  # Return as-is if can't parse


def normalize_symbol_for_api(symbol: str, target_format: str = "phemex") -> str:
	"""Normalize symbol for specific API format.
	
	Args:
		symbol: Input symbol in any format
		target_format: 'phemex' or 'ccxt'
	"""
	if target_format == "phemex":
		return ccxt_to_phemex_symbol(symbol)
	elif target_format == "ccxt":
		return phemex_to_ccxt_symbol(symbol)
	else:
		return symbol
