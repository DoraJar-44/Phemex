from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreInputs:
	# Levels
	avg: float
	r1: float
	r2: float
	s1: float
	s2: float
	# Price/indicators
	close: float
	open: float
	rsi: Optional[float] = None
	trend_sma: Optional[float] = None
	# Higher-timeframe bias/confidence
	bias_up_conf: float = 0.0  # [0,1]
	bias_dn_conf: float = 0.0  # [0,1]
	# Bounce prob and divergences
	bounce_prob: float = 0.0   # [0,0.9]
	bull_div: bool = False
	bear_div: bool = False


def clamp(v: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, v))


def compute_total_score(inp: ScoreInputs, side: str, score_min: int = 85) -> int:
	base = 50.0
	# Range score
	if side == "long":
		span = max(abs(inp.avg - inp.s1), 1e-9)
		proximity = clamp((inp.r1 - inp.close) / (inp.r1 - inp.s1 + 1e-9), 0.0, 1.0)
		range_score = 30.0 * proximity
	else:
		span = max(abs(inp.avg - inp.r1), 1e-9)
		proximity = clamp((inp.close - inp.s1) / (inp.r1 - inp.s1 + 1e-9), 0.0, 1.0)
		range_score = 30.0 * proximity

	# Bounce score
	bounce_score = 20.0 * clamp(inp.bounce_prob / 0.9, 0.0, 1.0)

	# Divergence score
	div_score = 0.0
	if side == "long" and inp.bull_div:
		div_score = 20.0
	elif side == "short" and inp.bear_div:
		div_score = 20.0

	# Bias score from multi-TF confidence
	bias_conf = inp.bias_up_conf if side == "long" else inp.bias_dn_conf
	bias_score = 30.0 * clamp(bias_conf, 0.0, 1.0)

	total = int(base + range_score + bounce_score + div_score + bias_score)
	return total


