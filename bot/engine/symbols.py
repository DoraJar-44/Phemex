import os
from typing import List


def load_symbols() -> List[str]:
	val = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT")
	return [s.strip().upper() for s in val.split(",") if s.strip()]


