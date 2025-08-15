import os
from dotenv import load_dotenv
from typing import Dict, Any


class Settings:
	def __init__(self):
		# Load environment from .env once at startup (does not override real env)
		load_dotenv(override=False)
		# Exchange credentials
		self.phemex_api_key: str = os.getenv("PHEMEX_API_KEY", "")
		self.phemex_api_secret: str = os.getenv("PHEMEX_API_SECRET", "")
		self.phemex_base_url: str = os.getenv("PHEMEX_BASE_URL", "https://testnet-api.phemex.com")
		self.phemex_testnet: bool = os.getenv("PHEMEX_TESTNET", "false").lower() in ("1", "true", "yes")

		# Execution backend
		self.use_ccxt: bool = os.getenv("USE_CCXT", "1").lower() in ("1", "true", "yes")

		# Bitget placeholders (parity)
		self.bitget_api_key: str = os.getenv("BITGET_API_KEY", "")
		self.bitget_api_secret: str = os.getenv("BITGET_API_SECRET", "")
		self.bitget_api_password: str = os.getenv("BITGET_API_PASSWORD", os.getenv("BITGET_PASSPHRASE", ""))

		# Core runtime / risk caps
		# Risk per trade in percent (RISK_PCT or fallback RISK_PER_TRADE_PCT)
		self.risk_per_trade_pct: float = float(os.getenv("RISK_PCT", os.getenv("RISK_PER_TRADE_PCT", "0.5")))
		self.leverage_max: float = float(os.getenv("LEVERAGE", os.getenv("LEVERAGE_MAX", "5")))
		self.max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "3"))
		self.max_capital_fraction: float = float(os.getenv("MAX_CAPITAL_FRACTION", "0.6"))
		self.trade_dollars_min: float = float(os.getenv("TRADE_DOLLARS", "0"))
		self.trade_notional: float = float(os.getenv("TRADE_NOTIONAL", "0"))
		# Position limits
		self.max_positions: int = int(os.getenv("MAX_POSITIONS", "5"))
		self.entry_cooldown_s: int = int(os.getenv("ENTRY_COOLDOWN_S", "30"))

		# Strategy / Scoring
		self.score_filter: bool = os.getenv("SCORE_FILTER", "1").lower() in ("1", "true", "yes")
		self.score_min: int = int(os.getenv("SCORE_MIN", "85"))
		self.trend_len: int = int(os.getenv("TREND_LEN", "50"))
		self.use_enhanced_entry: bool = os.getenv("USE_ENHANCED_ENTRY", "1").lower() in ("1", "true", "yes")
		self.use_rsi: bool = os.getenv("USE_RSI", "0").lower() in ("1", "true", "yes")
		self.rsi_len: int = int(os.getenv("RSI_LEN", "14"))
		self.min_body_atr: float = float(os.getenv("MIN_BODY_ATR", "0.20"))
		self.buffer_percent: float = float(os.getenv("BUFFER_PERCENT", "0.10"))
		self.buffer_atr_mult: float = float(os.getenv("BUFFER_ATR_MULT", "0.25"))
		self.strong_strict: bool = os.getenv("STRONG_STRICT", "1").lower() in ("1", "true", "yes")

		# Predictive Ranges
		self.pr_atr_len: int = int(os.getenv("PR_ATR_LEN", "200"))
		self.pr_atr_mult: float = float(os.getenv("PR_ATR_MULT", "6.0"))

		# MTF confirm
		self.mtf_confirm: bool = os.getenv("MTF_CONFIRM", "0").lower() in ("1", "true", "yes")
		self.mtf_tfs: str = os.getenv("MTF_TFS", "5m,10m,15m")
		self.mtf_require: int = int(os.getenv("MTF_REQUIRE", "2"))

		# Account / balances
		self.account_balance_usdt: float = float(os.getenv("ACCOUNT_BALANCE_USDT", "1000"))

		# Temporary symbol metadata overrides until fetched from exchange
		self.symbol_overrides: Dict[str, Dict[str, Any]] = {
			"BTCUSDT": {"tickSize": 0.5, "lotSize": 0.001, "minQty": 0.001, "contractValuePerPrice": 1.0},
			"ETHUSDT": {"tickSize": 0.05, "lotSize": 0.01,  "minQty": 0.01,  "contractValuePerPrice": 1.0},
		}

		# Webhook
		self.webhook_token: str = os.getenv("WEBHOOK_TOKEN", "2267")

		# Live trading toggle
		self.live_trade: bool = os.getenv("LIVE_TRADE", "true").lower() in ("1", "true", "yes")

		# UI / theme
		self.cosmic_theme: bool = os.getenv("COSMIC_THEME", "1").lower() in ("1", "true", "yes")
		self.dashboard: bool = os.getenv("DASHBOARD", "1").lower() in ("1", "true", "yes")
		self.dashboard_interval: int = int(os.getenv("DASHBOARD_INTERVAL", "1"))


settings = Settings()
