from typing import Optional


class Levels:
    def __init__(self, avg: float, r1: float, r2: Optional[float] = None, s1: float = 0.0, s2: Optional[float] = None):
        self.avg = avg
        self.r1 = r1
        self.r2 = r2
        self.s1 = s1
        self.s2 = s2


class Stats:
    def __init__(self, atr: Optional[float] = None, rsi: Optional[float] = None, range_strength: Optional[float] = None):
        self.atr = atr
        self.rsi = rsi
        self.range_strength = range_strength


class WebhookPayload:
    def __init__(
        self,
        action: str,
        symbol: str,
        price: float,
        signal_type: str,
        levels: Levels,
        is_strong: Optional[bool] = False,
        timestamp: Optional[str] = None,
        stats: Optional[Stats] = None,
        token: Optional[str] = None,
    ):
        if action not in ("BUY", "SELL"):
            raise ValueError("action must be BUY or SELL")
        self.action = action
        self.symbol = symbol
        self.price = price
        self.signal_type = signal_type
        self.is_strong = is_strong
        self.timestamp = timestamp
        self.levels = levels
        self.stats = stats
        self.token = token


