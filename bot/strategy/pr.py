from typing import List, Tuple


def compute_atr(high: List[float], low: List[float], close: List[float], length: int) -> List[float]:
	"""Simple RMA ATR (Wilder) to approximate Pine ta.atr."""
	atr: List[float] = []
	prev_close = close[0]
	trs: List[float] = []
	for i in range(len(close)):
		tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
		trs.append(tr)
		prev_close = close[i]
	# Wilder's smoothing
	avg = None
	for i, tr in enumerate(trs):
		if i == 0:
			avg = tr
		elif i < length:
			avg = ((avg * i) + tr) / (i + 1)
		else:
			avg = (avg * (length - 1) + tr) / length
		atr.append(avg)
	return atr


def compute_predictive_ranges(
	high: List[float],
	low: List[float],
	close: List[float],
	atr_len: int = 200,
	atr_mult: float = 6.0,
) -> Tuple[float, float, float, float, float]:
	"""Port of Pine core: step-following avg with ATR-based bands. Returns (avg, r1, r2, s1, s2) for last bar."""
	assert len(close) >= max(2, atr_len), "Not enough candles"
	atr_series = compute_atr(high, low, close, atr_len)
	# align to close length
	shift = len(close) - len(atr_series)
	atr_series = ([atr_series[0]] * shift) + atr_series
	atr_mult_val_series = [a * atr_mult for a in atr_series]

	avg = close[0]
	hold_atr = atr_series[0] * 0.5
	for i in range(1, len(close)):
		atr_val = atr_series[i]
		atr_mult_val = atr_mult_val_series[i]
		prev_avg = avg
		c = close[i]
		if c - avg > atr_mult_val:
			avg = avg + atr_mult_val
		elif avg - c > atr_mult_val:
			avg = avg - atr_mult_val
		# holdAtr adaptive blend
		if avg != prev_avg:
			target_hold = atr_mult_val * 0.5
			hold_atr = (hold_atr * 0.8) + (target_hold * 0.2)
		else:
			default_hold = atr_mult_val * 0.5
			hold_atr = (hold_atr * 0.9) + (default_hold * 0.1)
		# clamp hold_atr between 0.25*atr and atr_mult*atr
		hold_atr = max(min(hold_atr, atr_mult_val), atr_val * 0.25)

	pr_avg = avg
	pr_r1 = pr_avg + hold_atr
	pr_s1 = pr_avg - hold_atr
	pr_r2 = pr_avg + 2.0 * hold_atr
	pr_s2 = pr_avg - 2.0 * hold_atr
	return pr_avg, pr_r1, pr_r2, pr_s1, pr_s2


