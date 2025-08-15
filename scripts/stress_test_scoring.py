import asyncio
import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt

# Robust settings import (handles module vs package naming conflicts)
try:
    from bot.config import settings  # type: ignore
except Exception:
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = root / "bot" / "config.py"
    spec = importlib.util.spec_from_file_location("bot_config_file", str(cfg_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    settings = mod.settings  # type: ignore
from bot.strategy.pr import compute_predictive_ranges
from bot.strategy.score import ScoreInputs, compute_total_score


# Windows asyncio compatibility for ccxt.async_support
try:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass


@dataclass
class TradeResult:
    symbol: str
    side: str  # "long" | "short"
    entry_index: int
    exit_index: int
    entry_price: float
    tp1_price: float
    tp2_price: float
    sl_price: float
    score: int
    outcome: str  # "tp2", "tp1_sl", "sl", "ambiguous", "no_exit"
    pnl_pct: float
    bars_held: int


async def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Dict[str, List[float]]:
    data = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not data or len(data) < 300:
        raise RuntimeError(f"insufficient OHLCV for {symbol}")
    open_ = [float(x[1]) for x in data]
    high = [float(x[2]) for x in data]
    low = [float(x[3]) for x in data]
    close = [float(x[4]) for x in data]
    return {"open": open_, "high": high, "low": low, "close": close}


def _base_signals(price: float, avg: float, r1: float, s1: float, prev_close: float, prev_avg: float) -> Tuple[bool, bool]:
    base_long = (price > avg) and (price < r1) and (prev_close <= prev_avg)
    base_short = (price < avg) and (price > s1) and (prev_close >= prev_avg)
    return base_long, base_short


def _calc_scores(
    close_price: float,
    open_price: float,
    avg: float,
    r1: float,
    r2: Optional[float],
    s1: float,
    s2: Optional[float],
    sma: Optional[float],
    score_min: int,
) -> Tuple[int, int]:
    # Bounce probability based on position within band (mirrors scanner/server)
    range_size = abs(r1 - s1)
    if range_size > 0:
        long_bounce_prob = max(0.0, 0.9 - (abs(close_price - s1) / range_size))
        short_bounce_prob = max(0.0, 0.9 - (abs(close_price - r1) / range_size))
    else:
        long_bounce_prob = 0.5
        short_bounce_prob = 0.5

    # Bias confidence placeholders
    bias_up_conf = 0.6
    bias_dn_conf = 0.6

    si_long = ScoreInputs(
        avg=avg,
        r1=r1,
        r2=(r2 if r2 is not None else r1),
        s1=s1,
        s2=(s2 if s2 is not None else s1),
        close=close_price,
        open=open_price,
        trend_sma=sma,
        bias_up_conf=bias_up_conf,
        bias_dn_conf=0.0,
        bounce_prob=long_bounce_prob,
        bull_div=False,
        bear_div=False,
    )
    si_short = ScoreInputs(
        avg=avg,
        r1=r1,
        r2=(r2 if r2 is not None else r1),
        s1=s1,
        s2=(s2 if s2 is not None else s1),
        close=close_price,
        open=open_price,
        trend_sma=sma,
        bias_up_conf=0.0,
        bias_dn_conf=bias_dn_conf,
        bounce_prob=short_bounce_prob,
        bull_div=False,
        bear_div=False,
    )

    long_score = compute_total_score(si_long, "long", score_min)
    short_score = compute_total_score(si_short, "short", score_min)
    return long_score, short_score


def _simulate_bracket(
    side: str,
    entry_index: int,
    entry_price: float,
    tp1_price: float,
    tp2_price: float,
    sl_price: float,
    highs: List[float],
    lows: List[float],
) -> Tuple[str, int, float]:
    """
    Returns (outcome, exit_index, pnl_pct)
    outcome in {"tp2", "tp1_sl", "sl", "ambiguous", "no_exit"}
    pnl_pct computed with 50% at TP1 and 25% at TP2 as defined in brackets.
    Remaining 25% assumed to exit at SL if hit after TP1.
    """
    qty_tp1 = 0.5
    qty_tp2 = 0.25
    qty_left = 0.25

    def _ret(p0: float, p1: float) -> float:
        if side == "long":
            return (p1 - p0) / p0
        else:
            return (p0 - p1) / p0

    first_exit_index: Optional[int] = None
    first_event: Optional[str] = None

    # Scan forward bar-by-bar
    for j in range(entry_index + 1, len(highs)):
        hi = highs[j]
        lo = lows[j]
        # Determine triggers for this bar
        tp_hit = (hi >= tp1_price) if side == "long" else (lo <= tp1_price)
        tp2_hit = (hi >= tp2_price) if side == "long" else (lo <= tp2_price)
        sl_hit = (lo <= sl_price) if side == "long" else (hi >= sl_price)

        if sl_hit and (tp_hit or tp2_hit):
            # Ambiguous intrabar sequence
            return ("ambiguous", j, 0.0)

        if sl_hit:
            # No targets hit first -> full stop
            return ("sl", j, _ret(entry_price, sl_price))

        if tp_hit:
            # Lock TP1, continue scanning for TP2 vs SL for remainder
            first_exit_index = j
            first_event = "tp1"
            break

    if first_event != "tp1":
        # never hit anything
        return ("no_exit", len(highs) - 1, 0.0)

    # After TP1: look for TP2 or SL
    for j in range(first_exit_index + 1, len(highs)):
        hi = highs[j]
        lo = lows[j]
        tp2_hit = (hi >= tp2_price) if side == "long" else (lo <= tp2_price)
        sl_hit = (lo <= sl_price) if side == "long" else (hi >= sl_price)

        if tp2_hit and sl_hit:
            return ("ambiguous", j, 0.0)

        if tp2_hit:
            pnl = qty_tp1 * _ret(entry_price, tp1_price) + qty_tp2 * _ret(entry_price, tp2_price)
            # leave last 25% open? assume flat at entry (neutral) to be conservative
            pnl += qty_left * 0.0
            return ("tp2", j, pnl)

        if sl_hit:
            pnl = qty_tp1 * _ret(entry_price, tp1_price) + (qty_tp2 + qty_left) * _ret(entry_price, sl_price)
            return ("tp1_sl", j, pnl)

    # Neither TP2 nor SL after TP1 within sample
    pnl = qty_tp1 * _ret(entry_price, tp1_price)
    return ("no_exit", len(highs) - 1, pnl)


async def stress_symbol(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int, score_min: int) -> Dict[str, any]:
    ohlcv = await fetch_ohlcv(ex, symbol, timeframe, limit)
    open_ = ohlcv["open"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    warmup = max(settings.pr_atr_len + 5, settings.trend_len + 5)
    in_position = False
    results: List[TradeResult] = []

    i = warmup
    while i < len(close) - 2:
        price = close[i]
        prev_close = close[i - 1]

        # Compute PR on data up to i (no lookahead) and previous bar PR for crossing condition
        prev_avg, prev_r1, prev_r2, prev_s1, prev_s2 = compute_predictive_ranges(
            high[: i], low[: i], close[: i], settings.pr_atr_len, settings.pr_atr_mult
        )
        avg, r1, r2, s1, s2 = compute_predictive_ranges(
            high[: i + 1], low[: i + 1], close[: i + 1], settings.pr_atr_len, settings.pr_atr_mult
        )

        # Trend filter (SMA)
        sma_window = max(1, settings.trend_len)
        sma = sum(close[max(0, i - sma_window + 1) : i + 1]) / sma_window

        base_long, base_short = _base_signals(price, avg, r1, s1, prev_close, prev_avg)
        long_score, short_score = _calc_scores(
            close_price=price,
            open_price=open_[i],
            avg=avg,
            r1=r1,
            r2=r2,
            s1=s1,
            s2=s2,
            sma=sma,
            score_min=score_min,
        )
        allow_long = long_score >= score_min
        allow_short = short_score >= score_min

        decided = False
        if base_long and price > sma and allow_long:
            tp1 = r1
            tp2 = r2 if r2 is not None else r1
            sl = s2 if s2 is not None else s1
            outcome, exit_idx, pnl = _simulate_bracket(
                side="long",
                entry_index=i,
                entry_price=price,
                tp1_price=tp1,
                tp2_price=tp2,
                sl_price=sl,
                highs=high,
                lows=low,
            )
            results.append(
                TradeResult(
                    symbol=symbol,
                    side="long",
                    entry_index=i,
                    exit_index=exit_idx,
                    entry_price=price,
                    tp1_price=tp1,
                    tp2_price=tp2,
                    sl_price=sl,
                    score=long_score,
                    outcome=outcome,
                    pnl_pct=pnl * 100.0,
                    bars_held=max(0, exit_idx - i),
                )
            )
            decided = True

        elif base_short and price < sma and allow_short:
            tp1 = s1
            tp2 = s2 if s2 is not None else s1
            sl = r2 if r2 is not None else r1
            outcome, exit_idx, pnl = _simulate_bracket(
                side="short",
                entry_index=i,
                entry_price=price,
                tp1_price=tp1,
                tp2_price=tp2,
                sl_price=sl,
                highs=high,
                lows=low,
            )
            results.append(
                TradeResult(
                    symbol=symbol,
                    side="short",
                    entry_index=i,
                    exit_index=exit_idx,
                    entry_price=price,
                    tp1_price=tp1,
                    tp2_price=tp2,
                    sl_price=sl,
                    score=short_score,
                    outcome=outcome,
                    pnl_pct=pnl * 100.0,
                    bars_held=max(0, exit_idx - i),
                )
            )
            decided = True

        # Advance; if a trade was taken, skip forward to exit index to avoid overlapping trades
        if decided and results:
            i = max(i + 1, results[-1].exit_index)
        else:
            i += 1

    # Aggregate metrics
    wins = sum(1 for r in results if r.outcome in ("tp2", "tp1_sl") and r.pnl_pct > 0)
    losses = sum(1 for r in results if r.outcome in ("sl", "tp1_sl") and r.pnl_pct < 0)
    ambiguous = sum(1 for r in results if r.outcome == "ambiguous")
    no_exit = sum(1 for r in results if r.outcome == "no_exit")
    pnl_list = [r.pnl_pct for r in results]
    avg_pnl = statistics.mean(pnl_list) if pnl_list else 0.0
    med_pnl = statistics.median(pnl_list) if pnl_list else 0.0
    avg_hold = statistics.mean([r.bars_held for r in results]) if results else 0.0

    return {
        "symbol": symbol,
        "trades": len(results),
        "wins": wins,
        "losses": losses,
        "ambiguous": ambiguous,
        "no_exit": no_exit,
        "avg_pnl_pct": round(avg_pnl, 4),
        "median_pnl_pct": round(med_pnl, 4),
        "avg_bars_held": round(avg_hold, 2),
        "examples": [results[i] for i in range(min(5, len(results)))],
    }


async def main():
    parser = argparse.ArgumentParser(description="Stress test PR scoring with bracket TP/SL (no webhooks)")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated list of Phemex symbols (e.g. BTC/USDT:USDT,ETH/USDT:USDT)")
    parser.add_argument("--timeframe", type=str, default=os.getenv("TIMEFRAME", "5m"))
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--score-min", type=int, default=int(os.getenv("SCORE_MIN", str(settings.score_min))))
    args = parser.parse_args()

    print("Initializing Phemex client (read-only OHLCV)...")
    ex = ccxt.phemex({
        "apiKey": settings.phemex_api_key,
        "secret": settings.phemex_api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSubType": "linear"},
    })
    await ex.load_markets()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        # Auto-discover a few USDT linear swaps
        symbols = []
        for m in ex.markets.values():
            if (m.get("type") == "swap" and m.get("linear") and m.get("quote") == "USDT" and m.get("active")):
                symbols.append(m["symbol"])  # e.g. BTC/USDT:USDT
        symbols = sorted(set(symbols))[:5]

    print(f"Scoring stress test | timeframe={args.timeframe} limit={args.limit} score_min={args.score_min}")
    print(f"Symbols: {', '.join(symbols)}")

    summaries: List[Dict[str, any]] = []
    for sym in symbols:
        try:
            res = await stress_symbol(ex, sym, args.timeframe, args.limit, args.score_min)
            summaries.append(res)
            print(f"- {sym}: trades={res['trades']} wins={res['wins']} losses={res['losses']} avg_pnl={res['avg_pnl_pct']}%")
        except Exception as e:
            print(f"- {sym}: error: {e}")

    # Overall
    total_trades = sum(s["trades"] for s in summaries)
    total_wins = sum(s["wins"] for s in summaries)
    total_losses = sum(s["losses"] for s in summaries)
    overall_avg_pnl = statistics.mean([s["avg_pnl_pct"] for s in summaries]) if summaries else 0.0
    print("=" * 60)
    print(f"TOTAL: trades={total_trades} wins={total_wins} losses={total_losses} avg_pnl={round(overall_avg_pnl, 4)}%")

    await ex.close()


if __name__ == "__main__":
    asyncio.run(main())


