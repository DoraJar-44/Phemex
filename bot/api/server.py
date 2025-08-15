import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from json import JSONDecodeError
import structlog
from bot.config import settings
from bot.signals.webhook_models import WebhookPayload, Levels
from bot.execution.brackets import build_bracket_orders
from bot.risk.sizing import compute_quantity
from bot.execution.execute import place_bracket
from bot.exchange.phemex_client import get_client
from bot.strategy.score import ScoreInputs, compute_total_score


logger = structlog.get_logger()

app = FastAPI(title="Phemex PR Bot", version="0.2.0")

static_dir = os.path.join(os.path.dirname(__file__), "../ui/static")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/health")
async def health():
	return {"status": "ok"}


@app.get("/")
async def index():
	index_html = os.path.join(static_dir, "index.html")
	with open(index_html, "r", encoding="utf-8") as f:
		return HTMLResponse(f.read())


def _symbol_meta(symbol: str):
	meta = settings.symbol_overrides.get(symbol.upper())
	if not meta:
		meta = {"tickSize": 0.1, "lotSize": 0.001, "minQty": 0.001, "contractValuePerPrice": 1.0}
	return meta


@app.post("/simulate")
async def simulate(request: Request):
	try:
		body = await request.json()
		lvl = body.get("levels") or {}
		payload = WebhookPayload(
			action=body.get("action"),
			symbol=body.get("symbol"),
			price=float(body.get("price")),
			signal_type=body.get("signal_type"),
			levels=Levels(
				avg=float(lvl.get("avg")),
				r1=float(lvl.get("r1")),
				r2=(float(lvl.get("r2")) if lvl.get("r2") is not None else None),
				s1=float(lvl.get("s1")),
				s2=(float(lvl.get("s2")) if lvl.get("s2") is not None else None),
			),
		)
	except (JSONDecodeError, Exception) as e:
		raise HTTPException(status_code=400, detail=str(e))
	meta = _symbol_meta(payload.symbol)
	qty, distance = compute_quantity(
		payload,
		settings.account_balance_usdt,
		settings.risk_per_trade_pct,
		meta["lotSize"],
		meta["minQty"],
		meta["tickSize"],
		meta["contractValuePerPrice"],
	)
	intents = build_bracket_orders(payload, qty)
	return {"qty": qty, "stop_distance": distance, "intents": intents}


@app.post("/webhook/tradingview")
async def webhook_tv(request: Request):
	token = request.headers.get("x-webhook-token", "")
	body = await request.json()
	try:
		lvl = body.get("levels") or {}
		payload = WebhookPayload(
			action=body.get("action"),
			symbol=body.get("symbol"),
			price=float(body.get("price")),
			signal_type=body.get("signal_type"),
			levels=Levels(
				avg=float(lvl.get("avg")),
				r1=float(lvl.get("r1")),
				r2=(float(lvl.get("r2")) if lvl.get("r2") is not None else None),
				s1=float(lvl.get("s1")),
				s2=(float(lvl.get("s2")) if lvl.get("s2") is not None else None),
			),
			is_strong=body.get("is_strong"),
			timestamp=body.get("timestamp"),
			token=body.get("token"),
		)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

	if settings.webhook_token and token != settings.webhook_token and payload.token != settings.webhook_token:
		raise HTTPException(status_code=401, detail="invalid token")

	meta = _symbol_meta(payload.symbol)
	qty, _ = compute_quantity(
		payload,
		settings.account_balance_usdt,
		settings.risk_per_trade_pct,
		meta["lotSize"],
		meta["minQty"],
		meta["tickSize"],
		meta["contractValuePerPrice"],
	)
	intents = build_bracket_orders(payload, qty)
	# Compute score with enhanced inputs
	side = "long" if payload.action == "BUY" else "short"
	
	# Calculate bounce probability based on price proximity to support/resistance
	if side == "long":
		# For longs, check proximity to support
		dist_to_s1 = abs(payload.price - payload.levels.s1)
		range_size = abs(payload.levels.r1 - payload.levels.s1)
		bounce_prob = max(0, 0.9 - (dist_to_s1 / range_size)) if range_size > 0 else 0.5
	else:
		# For shorts, check proximity to resistance
		dist_to_r1 = abs(payload.price - payload.levels.r1)
		range_size = abs(payload.levels.r1 - payload.levels.s1)
		bounce_prob = max(0, 0.9 - (dist_to_r1 / range_size)) if range_size > 0 else 0.5
	
	# Set a default bias confidence based on signal strength
	bias_conf = 0.7 if payload.is_strong else 0.4
	
	si = ScoreInputs(
		avg=payload.levels.avg,
		r1=payload.levels.r1,
		r2=payload.levels.r2 or payload.levels.r1,  # fallback
		s1=payload.levels.s1,
		s2=payload.levels.s2 or payload.levels.s1,  # fallback
		close=payload.price,
		open=payload.price,
		rsi=(payload.stats.rsi if payload.stats else None),
		bounce_prob=bounce_prob,
		bias_up_conf=bias_conf if side == "long" else 0.0,
		bias_dn_conf=bias_conf if side == "short" else 0.0,
		# Could add divergence detection here if we had OHLC history
		bull_div=False,
		bear_div=False,
	)
	score = compute_total_score(si, side)
	logger.info("signal_received", symbol=payload.symbol, side=payload.action, score=score, intents=intents)
	
	# Check if score meets minimum requirement
	if settings.score_filter and score < settings.score_min:
		logger.warning("signal_rejected", symbol=payload.symbol, side=payload.action, score=score, min_required=settings.score_min)
		return {"ok": False, "score": score, "reason": f"Score {score} below minimum {settings.score_min}", "intents": intents}
	
	# Only place trades if live trading is enabled
	if settings.live_trade:
		placed = await place_bracket(payload.symbol, intents)
		return {"ok": True, "score": score, "intents": intents, "placed": placed}
	else:
		return {"ok": True, "score": score, "intents": intents, "dryRun": True}


@app.on_event("shutdown")
async def _shutdown_client():
	# Ensure we close the async http client used for Phemex
	try:
		client = get_client()
		await client.close()
	except Exception:
		pass


