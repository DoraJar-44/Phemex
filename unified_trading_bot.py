#!/usr/bin/env python3
"""
UNIFIED PHEMEX TRADING BOT
Complete trading bot with TUI, scanning, scoring, and automated trading with TP/SL
All-in-one script that does everything you need.
ROBUST VERSION WITH CHECKPOINTS AND ERROR RECOVERY
"""

import os
import sys
import asyncio
import time
import threading
import uuid
import httpx
import logging
import traceback
import signal
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import curses
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

import ccxt.async_support as ccxt

# Configure logging with rotation to prevent log file bloat
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global shutdown flag for graceful termination
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_REQUESTED
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    SHUTDOWN_REQUESTED = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log_exception(func_name: str, e: Exception):
    """Log detailed exception information"""
    logger.error(f"Exception in {func_name}: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return str(e)

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

class CheckpointManager:
    """Manages checkpoints and recovery for the trading bot"""
    
    def __init__(self, checkpoint_file: str = "bot_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoints = {}
        self.load_checkpoints()
    
    def load_checkpoints(self):
        """Load existing checkpoints"""
        try:
            if os.path.exists(self.checkpoint_file):
                import json
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoints = json.load(f)
                logger.info(f"Loaded {len(self.checkpoints)} checkpoints")
        except Exception as e:
            logger.warning(f"Failed to load checkpoints: {e}")
            self.checkpoints = {}
    
    def save_checkpoint(self, name: str, data: Dict[str, Any]):
        """Save a checkpoint"""
        try:
            self.checkpoints[name] = {
                "timestamp": time.time(),
                "data": data
            }
            import json
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoints, f, indent=2)
            logger.debug(f"Checkpoint saved: {name}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint {name}: {e}")
    
    def get_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a checkpoint"""
        return self.checkpoints.get(name, {}).get("data")
    
    def clear_checkpoint(self, name: str):
        """Clear a checkpoint"""
        if name in self.checkpoints:
            del self.checkpoints[name]
            self.save_checkpoint("", {})  # Trigger save

# Global checkpoint manager
checkpoint_manager = CheckpointManager()

# ============================================================================
# ERROR RECOVERY AND TIMEOUT SYSTEM
# ============================================================================

class TimeoutManager:
    """Manages timeouts and prevents hanging operations"""
    
    def __init__(self):
        self.timeouts = {
            "api_call": 30,      # 30 seconds for API calls
            "symbol_scan": 10,    # 10 seconds per symbol scan
            "tui_startup": 5,     # 5 seconds for TUI startup
            "client_init": 15,    # 15 seconds for client initialization
        }
    
    async def with_timeout(self, coro, timeout_name: str, default_return=None):
        """Execute coroutine with timeout"""
        try:
            timeout_seconds = self.timeouts.get(timeout_name, 30)
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out: {timeout_name}")
            return default_return
        except Exception as e:
            logger.error(f"Operation failed: {timeout_name} - {e}")
            return default_return

# Global timeout manager
timeout_manager = TimeoutManager()

# ============================================================================
# CONFIGURATION & SETTINGS
# ============================================================================

class Settings:
    def __init__(self):
        # Load environment from .env once at startup (does not override real env)
        load_dotenv(override=True)
        
        # Exchange credentials
        self.phemex_api_key: str = os.getenv("PHEMEX_API_KEY", "")
        self.phemex_api_secret: str = os.getenv("PHEMEX_API_SECRET", "")
        self.phemex_base_url: str = os.getenv("PHEMEX_BASE_URL", "https://api.phemex.com")
        self.phemex_testnet: bool = os.getenv("PHEMEX_TESTNET", "false").lower() in ("1", "true", "yes")

        # Execution backend
        self.use_ccxt: bool = os.getenv("USE_CCXT", "1").lower() in ("1", "true", "yes")

        # Live trading toggle - CRITICAL SAFETY CHECK
        self.live_trade: bool = os.getenv("LIVE_TRADE", "false").lower() in ("1", "true", "yes")
        
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

        # Discovery / symbol filters
        self.min_leverage: float = float(os.getenv("MIN_LEVERAGE", "1"))
        exclude_bases_env = os.getenv("EXCLUDE_BASES", "BTC,ETH,SOL,BNB,XRP,DOGE")
        self.exclude_bases = {b.strip().upper() for b in exclude_bases_env.split(",") if b.strip()}
        self.symbols_manual: str = os.getenv("SYMBOLS", "")

        # Temporary symbol metadata overrides until fetched from exchange
        self.symbol_overrides: Dict[str, Dict[str, Any]] = {
            "BTCUSDT": {"tickSize": 0.5, "lotSize": 0.001, "minQty": 0.001, "contractValuePerPrice": 1.0},
            "ETHUSDT": {"tickSize": 0.05, "lotSize": 0.01,  "minQty": 0.01,  "contractValuePerPrice": 1.0},
        }

        # Webhook
        self.webhook_token: str = os.getenv("WEBHOOK_TOKEN", "2267")

        # UI / theme
        self.cosmic_theme: bool = os.getenv("COSMIC_THEME", "1").lower() in ("1", "true", "yes")
        self.dashboard: bool = os.getenv("DASHBOARD", "1").lower() in ("1", "true", "yes")
        self.dashboard_interval: int = int(os.getenv("DASHBOARD_INTERVAL", "1"))

settings = Settings()

# ============================================================================
# SAFETY VALIDATION
# ============================================================================

def validate_safety_settings():
    """Validate critical safety settings before starting"""
    errors = []
    
    # Check LIVE_TRADE setting
    if settings.live_trade:
        if not settings.phemex_api_key or not settings.phemex_api_secret:
            errors.append("LIVE_TRADE=true but missing API credentials")
        if settings.risk_per_trade_pct > 5.0:
            errors.append(f"Risk per trade too high: {settings.risk_per_trade_pct}%")
        if settings.max_daily_loss_pct > 10.0:
            errors.append(f"Max daily loss too high: {settings.max_daily_loss_pct}%")
    
    # Check risk parameters
    if settings.leverage_max > 100:
        errors.append(f"Leverage too high: {settings.leverage_max}x")
    if settings.max_positions > 20:
        errors.append(f"Max positions too high: {settings.max_positions}")
    
    if errors:
        logger.error("Safety validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Safety validation passed")
    return True

# ============================================================================
# ROBUST TUI MANAGEMENT
# ============================================================================

def safe_start_tui():
    """Safely start TUI with error handling and fallback"""
    try:
        # Try to start TUI with timeout
        logger.info("Attempting to start TUI...")
        
        # Check if we're in a proper terminal
        if not sys.stdout.isatty():
            logger.info("Not in terminal, skipping TUI")
            return None
        
        # Try to start TUI with timeout
        tui_result = asyncio.run(timeout_manager.with_timeout(
            start_tui(), "tui_startup", None
        ))
        
        if tui_result:
            logger.info("TUI started successfully")
            return tui_result
        else:
            logger.warning("TUI failed to start, continuing without it")
            return None
            
    except Exception as e:
        error_msg = log_exception("safe_start_tui", e)
        logger.warning(f"TUI startup failed: {error_msg}")
        print(f"TUI failed to start: {error_msg}")
        print("Continuing without TUI...")
        return None

# ============================================================================
# ROBUST CLIENT INITIALIZATION
# ============================================================================

async def safe_initialize_client():
    """Safely initialize exchange client with error handling"""
    try:
        logger.info("Initializing exchange client...")
        
        if settings.live_trade:
            # Live trading - initialize real client
            client = await timeout_manager.with_timeout(
                get_phemex_client(), "client_init", None
            )
            
            if client:
                logger.info("Live trading client initialized successfully")
                checkpoint_manager.save_checkpoint("client_status", {"status": "connected", "mode": "live"})
                return client
            else:
                logger.error("Failed to initialize live trading client")
                return None
        else:
            # Paper trading - no client needed
            logger.info("Paper trading mode - no exchange client needed")
            checkpoint_manager.save_checkpoint("client_status", {"status": "paper", "mode": "paper"})
            return None
            
    except Exception as e:
        error_msg = log_exception("safe_initialize_client", e)
        logger.error(f"Client initialization failed: {error_msg}")
        
        if settings.live_trade:
            logger.error("Live trading client failed, cannot continue")
            return None
        else:
            logger.info("Continuing with paper trading mode")
            return None

# ============================================================================
# TUI DISPLAY SYSTEM
# ============================================================================

@dataclass
class ScoreEntry:
    symbol: str
    price: float
    long_score: int
    short_score: int
    long_signal: bool
    short_signal: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TradeEntry:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    tp1_price: float
    tp2_price: float
    sl_price: float
    timestamp: datetime = field(default_factory=datetime.now)

class TradingTUI:
    def __init__(self):
        self.stdscr = None
        self.running = False
        self.scores: deque = deque(maxlen=100)
        self.high_scores: deque = deque(maxlen=50)
        self.trades: deque = deque(maxlen=20)
        self.errors: deque = deque(maxlen=10)
        self.positions: deque = deque(maxlen=10)
        self.logs: deque = deque(maxlen=50)
        self.stats = {
            "scanned": 0, "signals": 0, "trades": 0, "pnl": 0.0,
            "account_balance": 0.0, "equity": 0.0, "margin_used": 0.0,
            "win_rate": 0.0, "avg_profit": 0.0, "max_drawdown": 0.0,
            "daily_pnl": 0.0, "total_volume": 0.0, "open_positions": 0,
            "winning_trades": 0, "losing_trades": 0, "total_trades": 0,
            "best_trade": 0.0, "worst_trade": 0.0, "current_streak": 0
        }
        self.color_pairs = {}
        self.unicode_support = True
        self.animation_frame = 0
        self.market_trend = "neutral"
        
    def init_colors(self):
        """Initialize enhanced color scheme"""
        curses.start_color()
        curses.use_default_colors()
        
        # Enhanced color definitions for better contrast
        color_definitions = [
            (1, 46, -1),    # Bright green
            (2, 226, -1),   # Yellow
            (3, 196, -1),   # Red
            (4, 51, -1),    # Cyan
            (5, 201, -1),   # Magenta
            (6, 255, -1),   # White
            (7, 244, -1),   # Gray
            (8, 82, -1),    # Light green
            (9, 208, -1),   # Orange
            (10, 21, -1),   # Blue
            (11, 118, -1),  # Bright green for profit
            (12, 160, -1),  # Dark red for loss
            (13, 220, -1),  # Gold for headers
            (14, 87, -1),   # Light cyan for info
            (15, 93, -1),   # Purple for special
        ]
        
        for i, (pair_num, fg, bg) in enumerate(color_definitions, 1):
            try:
                if curses.COLORS >= 256:
                    curses.init_pair(pair_num, fg, bg)
                else:
                    basic_colors = [
                        curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED,
                        curses.COLOR_CYAN, curses.COLOR_MAGENTA, curses.COLOR_WHITE
                    ]
                    curses.init_pair(pair_num, basic_colors[min(i-1, 5)], curses.COLOR_BLACK)
            except:
                curses.init_pair(pair_num, curses.COLOR_WHITE, curses.COLOR_BLACK)
                
        self.color_pairs = {
            'profit': curses.color_pair(11),
            'loss': curses.color_pair(12),
            'header': curses.color_pair(13),
            'info': curses.color_pair(14),
            'special': curses.color_pair(15),
            'success': curses.color_pair(1),
            'warning': curses.color_pair(2),
            'danger': curses.color_pair(3),
            'primary': curses.color_pair(4),
            'secondary': curses.color_pair(5),
            'normal': curses.color_pair(6),
            'muted': curses.color_pair(7),
        }

    def get_score_color(self, score: int) -> int:
        """Get color based on score value with gradient"""
        if score >= 95:
            return self.color_pairs['profit'] | curses.A_BOLD
        elif score >= 90:
            return self.color_pairs['success'] | curses.A_BOLD
        elif score >= 85:
            return self.color_pairs['success']
        elif score >= 80:
            return curses.color_pair(8)
        elif score >= 75:
            return self.color_pairs['warning']
        elif score >= 70:
            return curses.color_pair(9)
        else:
            return self.color_pairs['danger']

    def draw_border(self, win, title: str, color=None, style="double"):
        """Draw enhanced border with title"""
        if color is None:
            color = self.color_pairs.get('header', curses.color_pair(4))
            
        h, w = win.getmaxyx()
        
        # Use Unicode borders if supported
        if style == "double" and self.unicode_support:
            chars = {'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝', 'h': '═', 'v': '║'}
        elif style == "rounded" and self.unicode_support:
            chars = {'tl': '╭', 'tr': '╮', 'bl': '╰', 'br': '╯', 'h': '─', 'v': '│'}
        else:
            win.box()
            win.addstr(0, 2, f" {title} ", color | curses.A_BOLD)
            return
            
        try:
            # Draw corners
            win.addstr(0, 0, chars['tl'], color)
            win.addstr(0, w-1, chars['tr'], color)
            win.addstr(h-1, 0, chars['bl'], color)
            win.addstr(h-1, w-1, chars['br'], color)
            
            # Draw lines
            for x in range(1, w-1):
                win.addstr(0, x, chars['h'], color)
                win.addstr(h-1, x, chars['h'], color)
            for y in range(1, h-1):
                win.addstr(y, 0, chars['v'], color)
                win.addstr(y, w-1, chars['v'], color)
                
            # Add centered title
            title_with_padding = f" {title} "
            title_start = (w - len(title_with_padding)) // 2
            win.addstr(0, title_start, title_with_padding, color | curses.A_BOLD)
        except curses.error:
            pass

    def draw_progress_bar(self, win, y: int, x: int, width: int, value: float, max_value: float, 
                         label: str = "", color=None):
        """Draw a progress bar with percentage"""
        if color is None:
            color = self.color_pairs.get('success', curses.color_pair(1))
            
        if max_value <= 0:
            percentage = 0
        else:
            percentage = min(100, max(0, (value / max_value) * 100))
            
        filled = int((width - 2) * percentage / 100)
        
        try:
            win.addstr(y, x, "[", self.color_pairs.get('muted', curses.color_pair(7)))
            win.addstr(y, x + width - 1, "]", self.color_pairs.get('muted', curses.color_pair(7)))
            
            # Draw filled portion
            if self.unicode_support:
                for i in range(filled):
                    win.addstr(y, x + 1 + i, "█", color)
                for i in range(filled, width - 2):
                    win.addstr(y, x + 1 + i, "░", self.color_pairs.get('muted', curses.color_pair(7)))
            else:
                for i in range(filled):
                    win.addstr(y, x + 1 + i, "=", color)
                for i in range(filled, width - 2):
                    win.addstr(y, x + 1 + i, "-", self.color_pairs.get('muted', curses.color_pair(7)))
                    
            if label:
                label_text = f" {label}: {percentage:.1f}%"
                win.addstr(y, x + width + 1, label_text, self.color_pairs.get('normal', curses.color_pair(6)))
        except curses.error:
            pass

    def format_number(self, value: float, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        """Format number with proper spacing and symbols"""
        if abs(value) >= 1_000_000:
            return f"{prefix}{value/1_000_000:.{decimals}f}M{suffix}"
        elif abs(value) >= 1_000:
            return f"{prefix}{value/1_000:.{decimals}f}K{suffix}"
        else:
            return f"{prefix}{value:.{decimals}f}{suffix}"

    def draw_scores_pane(self, win):
        win.clear()
        self.draw_border(win, "🎯 HIGH SCORES")
        
        h, w = win.getmaxyx()
        y = 1
        
        # Show recent high scores (≥75)
        high_scores = [entry for entry in self.scores if max(entry.long_score, entry.short_score) >= 75]
        
        for entry in list(high_scores)[-8:]:  # Last 8 entries
            if y >= h - 1:
                break
                
            max_score = max(entry.long_score, entry.short_score)
            side = "L" if entry.long_score > entry.short_score else "S"
            signal_indicator = "🚀" if (side == "L" and entry.long_signal) or (side == "S" and entry.short_signal) else ""
            
            score_color = self.get_score_color(max_score)
            
            try:
                line = f"{entry.symbol:8} {side}{max_score:3d} ${entry.price:8.3f} {signal_indicator}"
                win.addstr(y, 1, line[:w-2], score_color)
            except curses.error:
                pass
            y += 1

    def draw_market_overview(self, win):
        win.clear()
        self.draw_border(win, "📊 MARKET OVERVIEW")
        
        h, w = win.getmaxyx()
        y = 1
        
        # Market summary info
        market_info = [
            f"Total Symbols: {len(self.scores)}",
            f"Active Signals: {self.stats['signals']}",
            f"High Scores (≥80): {len([s for s in self.scores if max(s.long_score, s.short_score) >= 80])}",
            f"Recent Scans: {self.stats['scanned']}",
            "",
            "Top Performers:",
        ]
        
        # Add top scoring symbols
        top_scores = sorted(self.scores, key=lambda x: max(x.long_score, x.short_score), reverse=True)[:3]
        for entry in top_scores:
            max_score = max(entry.long_score, entry.short_score)
            side = "L" if entry.long_score > entry.short_score else "S"
            market_info.append(f"{entry.symbol:8} {side}{max_score:3d}")
        
        for i, line in enumerate(market_info):
            if y >= h - 1:
                break
            try:
                color = curses.color_pair(1) if "High Scores" in line else curses.color_pair(6)
                win.addstr(y, 1, line[:w-2], color)
            except curses.error:
                pass
            y += 1

    def draw_signals_pane(self, win):
        win.clear()
        self.draw_border(win, "⚡ SIGNALS")
        
        h, w = win.getmaxyx()
        y = 1
        
        # Show active signals
        signals = [entry for entry in self.scores if entry.long_signal or entry.short_signal]
        
        for entry in list(signals)[-8:]:
            if y >= h - 1:
                break
                
            side = "LONG" if entry.long_signal else "SHORT"
            score = entry.long_score if entry.long_signal else entry.short_score
            
            try:
                line = f"{entry.symbol:8} {side:5} {score:3d}"
                win.addstr(y, 1, line[:w-2], curses.color_pair(5) | curses.A_BOLD)
            except curses.error:
                pass
            y += 1

    def draw_trades_pane(self, win):
        win.clear()
        self.draw_border(win, "💰 TRADES")
        
        h, w = win.getmaxyx()
        y = 1
        
        for trade in list(self.trades)[-8:]:
            if y >= h - 1:
                break
                
            try:
                line = f"{trade.symbol:8} {trade.side:4} ${trade.entry_price:.3f}"
                win.addstr(y, 1, line[:w-2], curses.color_pair(1))
            except curses.error:
                pass
            y += 1

    def draw_account_info(self, win):
        win.clear()
        self.draw_border(win, "💰 ACCOUNT OVERVIEW")
        
        h, w = win.getmaxyx()
        y = 2
        
        # Calculate metrics
        balance = self.stats['account_balance']
        equity = self.stats['equity']
        margin = self.stats['margin_used']
        free_margin = equity - margin
        margin_level = (equity / margin * 100) if margin > 0 else 999
        
        # Display with enhanced formatting
        metrics = [
            ("Balance", self.format_number(balance, "$"), self.color_pairs.get('normal', curses.color_pair(6))),
            ("Equity", self.format_number(equity, "$"), 
             self.color_pairs.get('profit', curses.color_pair(1)) if equity > balance else self.color_pairs.get('loss', curses.color_pair(3))),
            ("Margin Used", self.format_number(margin, "$"), self.color_pairs.get('warning', curses.color_pair(2))),
            ("Free Margin", self.format_number(free_margin, "$"), 
             self.color_pairs.get('success', curses.color_pair(1)) if free_margin > margin else self.color_pairs.get('warning', curses.color_pair(2))),
            ("Margin Level", f"{margin_level:.1f}%", 
             self.color_pairs.get('success', curses.color_pair(1)) if margin_level > 200 else self.color_pairs.get('danger', curses.color_pair(3))),
        ]
        
        for label, value, color in metrics:
            if y >= h - 2:
                break
            try:
                win.addstr(y, 2, f"{label}:", self.color_pairs.get('muted', curses.color_pair(7)))
                win.addstr(y, 15, value, color | curses.A_BOLD)
            except curses.error:
                pass
            y += 1
        
        # Add margin level progress bar
        if y + 1 < h - 1 and w > 20:
            self.draw_progress_bar(win, y + 1, 2, min(20, w - 4), 
                                 min(margin_level, 500), 500, "Margin", 
                                 self.color_pairs.get('success', curses.color_pair(1)) if margin_level > 200 else self.color_pairs.get('danger', curses.color_pair(3)))

    def draw_performance_metrics(self, win):
        win.clear()
        self.draw_border(win, "📊 PERFORMANCE")
        
        h, w = win.getmaxyx()
        y = 2
        
        # Calculate win rate
        total_trades = self.stats.get('total_trades', 0) or self.stats.get('trades', 0)
        winning_trades = self.stats.get('winning_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        try:
            # Daily PnL with color
            daily_pnl = self.stats['daily_pnl']
            daily_pnl_pct = (daily_pnl / self.stats['account_balance'] * 100) if self.stats['account_balance'] > 0 else 0
            pnl_color = self.color_pairs.get('profit', curses.color_pair(1)) if daily_pnl > 0 else self.color_pairs.get('loss', curses.color_pair(3)) if daily_pnl < 0 else self.color_pairs.get('normal', curses.color_pair(6))
            
            win.addstr(y, 2, "Daily P&L:", self.color_pairs.get('muted', curses.color_pair(7)))
            win.addstr(y, 13, f"{self.format_number(daily_pnl, '$')} ({daily_pnl_pct:+.2f}%)", pnl_color | curses.A_BOLD)
            y += 1
            
            # Total PnL
            total_pnl = self.stats['pnl']
            total_pnl_pct = (total_pnl / self.stats['account_balance'] * 100) if self.stats['account_balance'] > 0 else 0
            pnl_color = self.color_pairs.get('profit', curses.color_pair(1)) if total_pnl > 0 else self.color_pairs.get('loss', curses.color_pair(3)) if total_pnl < 0 else self.color_pairs.get('normal', curses.color_pair(6))
            
            win.addstr(y, 2, "Total P&L:", self.color_pairs.get('muted', curses.color_pair(7)))
            win.addstr(y, 13, f"{self.format_number(total_pnl, '$')} ({total_pnl_pct:+.2f}%)", pnl_color | curses.A_BOLD)
            y += 2
            
            # Win rate with progress bar
            if y < h - 1:
                win.addstr(y, 2, f"Win Rate: {win_rate:.1f}%", self.color_pairs.get('normal', curses.color_pair(6)))
                y += 1
            if y < h - 1 and w > 20:
                self.draw_progress_bar(win, y, 2, min(20, w - 4), win_rate, 100, "", 
                                     self.color_pairs.get('success', curses.color_pair(1)) if win_rate > 50 else self.color_pairs.get('warning', curses.color_pair(2)))
                y += 2
            
            # Trade stats
            if y < h - 1:
                win.addstr(y, 2, f"Trades: {total_trades}", self.color_pairs.get('info', curses.color_pair(4)))
                y += 1
                
            # Trading mode
            if y < h - 1:
                mode = "🔴 LIVE" if settings.live_trade else "🟡 PAPER"
                mode_color = self.color_pairs.get('danger', curses.color_pair(3)) if settings.live_trade else self.color_pairs.get('warning', curses.color_pair(2))
                win.addstr(y, 2, f"Mode: {mode}", mode_color | curses.A_BOLD)
                
        except curses.error:
            pass

    def draw_risk_monitor(self, win):
        win.clear()
        self.draw_border(win, "⚠️ RISK MONITOR")
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            # Calculate risk metrics
            margin_ratio = (self.stats['margin_used'] / self.stats['equity'] * 100) if self.stats['equity'] > 0 else 0
            daily_loss_pct = abs(self.stats['daily_pnl'] / self.stats['account_balance'] * 100) if self.stats['account_balance'] > 0 and self.stats['daily_pnl'] < 0 else 0
            position_usage = (self.stats['open_positions'] / settings.max_positions * 100) if settings.max_positions > 0 else 0
            
            # Risk status
            risk_level = "LOW"
            risk_color = self.color_pairs.get('success', curses.color_pair(1))
            if margin_ratio > 80 or daily_loss_pct > 2:
                risk_level = "HIGH"
                risk_color = self.color_pairs.get('danger', curses.color_pair(3))
            elif margin_ratio > 50 or daily_loss_pct > 1:
                risk_level = "MEDIUM"
                risk_color = self.color_pairs.get('warning', curses.color_pair(2))
                
            win.addstr(y, 2, f"Risk Level: {risk_level}", risk_color | curses.A_BOLD)
            y += 2
            
            # Margin usage with progress bar
            if y < h - 1:
                win.addstr(y, 2, "Margin Usage:", self.color_pairs.get('normal', curses.color_pair(6)))
                y += 1
            if y < h - 1 and w > 20:
                bar_color = self.color_pairs.get('danger', curses.color_pair(3)) if margin_ratio > 80 else self.color_pairs.get('warning', curses.color_pair(2)) if margin_ratio > 50 else self.color_pairs.get('success', curses.color_pair(1))
                self.draw_progress_bar(win, y, 2, min(20, w - 15), margin_ratio, 100, f"{margin_ratio:.1f}%", bar_color)
                y += 2
            
            # Daily loss with progress bar
            if y < h - 1:
                win.addstr(y, 2, "Daily Loss:", self.color_pairs.get('normal', curses.color_pair(6)))
                y += 1
            if y < h - 1 and w > 20:
                bar_color = self.color_pairs.get('danger', curses.color_pair(3)) if daily_loss_pct > 2 else self.color_pairs.get('warning', curses.color_pair(2)) if daily_loss_pct > 1 else self.color_pairs.get('success', curses.color_pair(1))
                self.draw_progress_bar(win, y, 2, min(20, w - 15), daily_loss_pct, settings.max_daily_loss_pct, f"{daily_loss_pct:.1f}%", bar_color)
                y += 2
            
            # Position usage
            if y < h - 1:
                win.addstr(y, 2, f"Positions: {self.stats['open_positions']}/{settings.max_positions}", self.color_pairs.get('normal', curses.color_pair(6)))
                
        except curses.error:
            pass

    def draw_active_positions(self, win):
        win.clear()
        self.draw_border(win, "🎯 ACTIVE POSITIONS")
        
        h, w = win.getmaxyx()
        y = 1
        
        if not self.positions:
            try:
                win.addstr(y, 1, "No active positions", curses.color_pair(6))
            except curses.error:
                pass
        else:
            for pos in list(self.positions)[-6:]:  # Show last 6 positions
                if y >= h - 1:
                    break
                try:
                    line = f"{pos.symbol:8} {pos.side:4} {pos.quantity:.3f}"
                    color = curses.color_pair(1) if pos.side == "LONG" else curses.color_pair(3)
                    win.addstr(y, 1, line[:w-2], color)
                except curses.error:
                    pass
                y += 1

    def draw_system_info(self, win):
        win.clear()
        self.draw_border(win, "🎛️ SYSTEM")
        
        h, w = win.getmaxyx()
        y = 1
        
        system_lines = [
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"Exchange: Phemex",
            f"Min Score: {settings.score_min}",
            f"Timeframe: 5m",
            f"Symbols: {len(self.scores)}",
            "",
            "Controls:",
            "Press 'q' to quit",
        ]
        
        for line in system_lines:
            if y >= h - 1:
                break
            try:
                color = curses.color_pair(4) if "Controls:" in line else curses.color_pair(6)
                win.addstr(y, 1, line[:w-2], color)
            except curses.error:
                pass
            y += 1

    def draw_errors_pane(self, win):
        win.clear()
        self.draw_border(win, "⚠️ VALIDATION")
        
        h, w = win.getmaxyx()
        y = 1
        
        for error in list(self.errors)[-8:]:
            if y >= h - 1:
                break
            try:
                win.addstr(y, 1, error[:w-2], curses.color_pair(3))
            except curses.error:
                pass
            y += 1

    def update_display(self):
        if not self.stdscr:
            return
            
        try:
            # Get screen dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Calculate pane dimensions (3x3 grid)
            pane_h = max_y // 3
            pane_w = max_x // 3
            
            # Clear screen once
            self.stdscr.clear()
            
            # Create 3x3 grid of windows
            panes = {}
            for row in range(3):
                for col in range(3):
                    y = row * pane_h
                    x = col * pane_w
                    h = pane_h if row < 2 else max_y - y
                    w = pane_w if col < 2 else max_x - x
                    panes[(row, col)] = curses.newwin(h, w, y, x)
            
            # Draw each pane with useful information
            self.draw_scores_pane(panes[(0, 0)])
            self.draw_market_overview(panes[(0, 1)])
            self.draw_signals_pane(panes[(0, 2)])
            
            self.draw_account_info(panes[(1, 0)])
            self.draw_performance_metrics(panes[(1, 1)])
            self.draw_risk_monitor(panes[(1, 2)])
            
            self.draw_active_positions(panes[(2, 0)])
            self.draw_trades_pane(panes[(2, 1)])
            self.draw_system_info(panes[(2, 2)])
            
            # Single refresh at the end
            self.stdscr.refresh()
            
        except curses.error as e:
            logger.debug(f"Curses error in update_display: {e}")
        except Exception as e:
            log_exception("update_display", e)

    def run(self):
        try:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            curses.curs_set(0)  # Hide cursor
            
            # Check for Unicode support
            try:
                self.stdscr.addstr(0, 0, "█")
                self.unicode_support = True
            except:
                self.unicode_support = False
            self.stdscr.clear()
            
            self.init_colors()
            self.running = True
            
            while self.running:
                # Check for 'q' key press
                try:
                    key = self.stdscr.getch()
                    if key == ord('q') or key == ord('Q'):
                        self.running = False
                        break
                except curses.error:
                    pass
                
                self.update_display()
                time.sleep(1.0)  # Update every second instead of 0.1 to reduce choppiness
                
        except Exception as e:
            print(f"TUI Error: {e}")
        finally:
            if self.stdscr:
                curses.endwin()

    def stop(self):
        self.running = False

    def add_score_entry(self, entry: ScoreEntry):
        self.scores.append(entry)
        self.stats["scanned"] += 1
        if entry.long_signal or entry.short_signal:
            self.stats["signals"] += 1

    def add_trade_entry(self, trade: TradeEntry):
        self.trades.append(trade)
        self.stats["trades"] += 1

    def add_error(self, error: str):
        self.errors.append(error)

# Global TUI instance
_tui_instance: Optional[TradingTUI] = None
_tui_thread: Optional[threading.Thread] = None

def get_tui_instance() -> TradingTUI:
    global _tui_instance
    if _tui_instance is None:
        _tui_instance = TradingTUI()
    return _tui_instance

def start_tui():
    global _tui_thread
    if _tui_thread is None or not _tui_thread.is_alive():
        tui = get_tui_instance()
        _tui_thread = threading.Thread(target=tui.run, daemon=True)
        _tui_thread.start()

# ============================================================================
# TRADING STRATEGY & SCORING
# ============================================================================

@dataclass
class ScoreInputs:
    avg: float
    r1: float
    r2: float
    s1: float
    s2: float
    close: float
    open: float
    bounce_prob: float
    bias_up_conf: float = 0.0
    bias_dn_conf: float = 0.0
    bull_div: bool = False
    bear_div: bool = False

def compute_predictive_ranges(candles: Dict[str, List[float]]) -> Tuple[float, float, float, float, float]:
    """Compute predictive ranges - returns avg, r1, r2, s1, s2"""
    try:
        logger.debug(f"Computing predictive ranges for {len(candles.get('close', []))} candles")
        
        highs = candles["high"]
        lows = candles["low"]
        closes = candles["close"]
        
        if len(closes) < settings.pr_atr_len:
            logger.warning(f"Insufficient candles: {len(closes)} < {settings.pr_atr_len}")
            return 0, 0, 0, 0, 0
        
        # Simple ATR calculation
        atr_sum = 0
        for i in range(1, min(len(closes), settings.pr_atr_len + 1)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            atr_sum += tr
        
        atr = atr_sum / min(len(closes) - 1, settings.pr_atr_len)
        
        # Calculate ranges
        avg = sum(closes[-50:]) / min(50, len(closes))  # 50-period SMA
        atr_mult = settings.pr_atr_mult
        
        r1 = avg + atr * atr_mult * 0.5
        r2 = avg + atr * atr_mult
        s1 = avg - atr * atr_mult * 0.5
        s2 = avg - atr * atr_mult
        
        logger.debug(f"Computed ranges: avg={avg:.4f}, r1={r1:.4f}, r2={r2:.4f}, s1={s1:.4f}, s2={s2:.4f}")
        return avg, r1, r2, s1, s2
        
    except Exception as e:
        error_msg = log_exception("compute_predictive_ranges", e)
        return 0, 0, 0, 0, 0

def compute_total_score(inputs: ScoreInputs, direction: str, min_score: int = 85) -> int:
    """Compute trading score based on multiple factors"""
    try:
        logger.debug(f"Computing {direction} score for price {inputs.close}")
        
        price = inputs.close
        
        # Add price validation to prevent division by zero
        if price <= 0:
            logger.error(f"Invalid price for scoring: {price}")
            return 0
        
        if direction == "long":
            # Long scoring logic
            # Fixed: Return 0 distance if levels are invalid instead of 1
            support_distance = abs(price - inputs.s1) / price if inputs.s1 > 0 and price > 0 else 0
            resistance_distance = abs(inputs.r1 - price) / price if inputs.r1 > 0 and price > 0 else 0
            
            # Base score from price position
            if price <= inputs.s1:
                base_score = 85  # Near support
            elif price <= inputs.avg:
                base_score = 70  # Below average
            else:
                base_score = 50  # Above average
                
            # Trend bonus
            trend_bonus = min(20, inputs.bias_up_conf * 30)
            
            # Bounce probability bonus
            bounce_bonus = min(15, inputs.bounce_prob * 20)
            
            total = int(base_score + trend_bonus + bounce_bonus)
            
        else:  # short
            # Short scoring logic
            # Fixed: Return 0 distance if levels are invalid instead of 1
            resistance_distance = abs(price - inputs.r1) / price if inputs.r1 > 0 and price > 0 else 0
            support_distance = abs(inputs.s1 - price) / price if inputs.s1 > 0 and price > 0 else 0
            
            # Base score from price position
            if price >= inputs.r1:
                base_score = 85  # Near resistance
            elif price >= inputs.avg:
                base_score = 70  # Above average
            else:
                base_score = 50  # Below average
                
            # Trend bonus
            trend_bonus = min(20, inputs.bias_dn_conf * 30)
            
            # Bounce probability bonus
            bounce_bonus = min(15, inputs.bounce_prob * 20)
            
            total = int(base_score + trend_bonus + bounce_bonus)
        
        final_score = max(0, min(100, total))
        logger.debug(f"Final {direction} score: {final_score}")
        return final_score
        
    except Exception as e:
        error_msg = log_exception("compute_total_score", e)
        return 0

# ============================================================================
# EXCHANGE CLIENT & TRADING
# ============================================================================

def _key(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

async def get_phemex_client():
    """Get Phemex CCXT client"""
    try:
        logger.info("Initializing Phemex client...")
        
        if not settings.phemex_api_key or not settings.phemex_api_secret:
            raise ValueError("Missing Phemex API credentials")
            
        client = ccxt.phemex({
            'apiKey': settings.phemex_api_key,
            'secret': settings.phemex_api_secret,
            'sandbox': False,  # Set to True for testnet
            'enableRateLimit': True,
        })
        
        logger.info("Phemex client initialized successfully")
        return client
        
    except Exception as e:
        error_msg = log_exception("get_phemex_client", e)
        raise

def compute_quantity(price: float, stop_distance: float, account_balance: float, risk_pct: float) -> float:
    """Compute position size based on fixed $1 risk per trade"""
    # Use fixed $1 per trade regardless of account balance
    risk_amount = 1.0  # Fixed $1 risk per trade
    quantity = risk_amount / stop_distance
    return round(quantity, 6)

def build_bracket_orders(symbol: str, side: str, quantity: float, entry_price: float, 
                        tp1_price: float, tp2_price: float, sl_price: float) -> Dict[str, Any]:
    """Build bracket order structure"""
    
    opposite_side = "sell" if side == "buy" else "buy"
    
    # Split quantity for TP levels
    tp1_qty = quantity * 0.5  # 50% at TP1
    tp2_qty = quantity * 0.5  # 50% at TP2
    
    return {
        "entry": {
            "side": side,
            "quantity": quantity,
            "price": entry_price,
            "type": "market"
        },
        "tp1": {
            "side": opposite_side,
            "quantity": tp1_qty,
            "price": tp1_price,
            "type": "limit"
        },
        "tp2": {
            "side": opposite_side,
            "quantity": tp2_qty,
            "price": tp2_price,
            "type": "limit"
        },
        "sl": {
            "side": opposite_side,
            "quantity": quantity,
            "stopPrice": sl_price,
            "type": "stop_market"
        }
    }

async def place_bracket_trade(client, symbol: str, intents: Dict[str, Any]) -> Dict[str, Any]:
    """Place bracket trade with entry, TP1, TP2, and SL"""
    try:
        logger.info(f"Placing bracket trade for {symbol}")
        
        if not settings.live_trade:
            logger.info(f"Dry run mode for {symbol}")
            return {"dryRun": True, "intents": intents}
        
        results = {"placed": []}
        
        try:
            # Place entry order
            entry = intents["entry"]
            logger.debug(f"Placing entry order for {symbol}: {entry}")
            
            entry_order = await client.create_order(
                symbol=symbol,
                type=entry["type"],
                side=entry["side"],
                amount=entry["quantity"],
                price=entry.get("price"),
                params={
                    "clientOrderId": _key("entry"),
                    "posSide": "Merged",  # One-way mode as per rules
                    "reduceOnly": False
                }
            )
            results["placed"].append({"entry": entry_order})
            logger.info(f"Entry order placed for {symbol}: {entry_order.get('id', 'unknown')}")
            
            # Wait a moment for entry to fill
            await asyncio.sleep(1)
            
            # Place TP1 order (as per memory: limit orders, reduce-only)
            tp1 = intents["tp1"]
            logger.debug(f"Placing TP1 order for {symbol}: {tp1}")
            
            try:
                tp1_order = await client.create_order(
                    symbol=symbol,
                    type="limit",  # As per memory: limit orders for TP
                    side=tp1["side"],
                    amount=tp1["quantity"],
                    price=tp1["price"],
                    params={
                        "clientOrderId": _key("tp1"),
                        "posSide": "Merged",
                        "reduceOnly": True,  # Reduce-only as per rules
                        "postOnly": True  # As per memory: post-only
                    }
                )
                results["placed"].append({"tp1": tp1_order})
                logger.info(f"TP1 order placed for {symbol}: {tp1_order.get('id', 'unknown')}")
            except Exception as e:
                # Rollback: Close position if TP placement fails
                logger.error(f"Failed to place TP1 order, attempting to close position: {e}")
                try:
                    # Place market order to close position
                    close_order = await client.create_order(
                        symbol=symbol,
                        type="market",
                        side="sell" if entry["side"] == "buy" else "buy",
                        amount=entry["quantity"],
                        params={"reduceOnly": True}
                    )
                    return {"error": f"Failed to place TP1, position closed: {e}", "closed": close_order}
                except Exception as close_error:
                    return {"error": f"Failed to place TP1 and failed to close position: {e}, {close_error}"}
            
            # Place TP2 order (as per memory: limit orders, reduce-only)
            tp2 = intents["tp2"]
            logger.debug(f"Placing TP2 order for {symbol}: {tp2}")
            
            tp2_order = await client.create_order(
                symbol=symbol,
                type="limit",  # As per memory: limit orders for TP
                side=tp2["side"],
                amount=tp2["quantity"],
                price=tp2["price"],
                params={
                    "clientOrderId": _key("tp2"),
                    "posSide": "Merged",
                    "reduceOnly": True,  # Reduce-only as per rules
                    "postOnly": True  # As per memory: post-only
                }
            )
            results["placed"].append({"tp2": tp2_order})
            logger.info(f"TP2 order placed for {symbol}: {tp2_order.get('id', 'unknown')}")
            
            # Place SL order (FIXED: Use stop_market for proper stop loss execution)
            sl = intents["sl"]
            logger.debug(f"Placing SL order for {symbol}: {sl}")
            
            sl_order = await client.create_order(
                symbol=symbol,
                type="stop_market",  # FIXED: Changed from limit to stop_market for proper SL execution
                side=sl["side"],
                amount=sl["quantity"],
                stopPrice=sl.get("stopPrice"),  # Use stopPrice for stop orders
                params={
                    "clientOrderId": _key("sl"),
                    "posSide": "Merged",
                    "reduceOnly": True,  # Reduce-only as per rules
                    # Removed postOnly as it's not compatible with stop orders
                }
            )
            results["placed"].append({"sl": sl_order})
            logger.info(f"SL order placed for {symbol}: {sl_order.get('id', 'unknown')}")
            
            logger.info(f"Bracket trade completed successfully for {symbol}")
            return results
            
        except Exception as e:
            error_msg = log_exception(f"place_bracket_trade_orders_{symbol}", e)
            tui = get_tui_instance()
            tui.add_error(f"Trade error: {error_msg}")
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = log_exception(f"place_bracket_trade_{symbol}", e)
        return {"error": error_msg}

# ============================================================================
# MARKET SCANNING ENGINE
# ============================================================================

def _tf_to_minutes(tf: str) -> int:
    unit = tf[-1]
    val = int(tf[:-1])
    if unit == 'm':
        return val
    if unit == 'h':
        return val * 60
    if unit == 'd':
        return val * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")

async def fetch_candles(client, symbol: str, timeframe: str = "5m") -> Dict[str, List[float]]:
    """Fetch OHLCV candles for a symbol"""
    try:
        logger.debug(f"Fetching candles for {symbol} on {timeframe}")
        
        # For paper trading or when no client, use mock data
        if not client or not settings.live_trade:
            try:
                import random
                
                # Generate some realistic mock data for now
                limit = max(settings.pr_atr_len + 50, 250)
                base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
                
                candles = []
                current_price = base_price
                
                logger.debug(f"Generating {limit} mock candles for {symbol}")
                
                for i in range(limit):
                    change = random.uniform(-0.01, 0.01)
                    current_price *= (1 + change)
                    
                    volatility = current_price * 0.005
                    high = current_price + random.uniform(0, volatility)
                    low = current_price - random.uniform(0, volatility)
                    open_price = current_price + random.uniform(-volatility/2, volatility/2)
                    close = current_price
                    
                    candles.append([open_price, high, low, close])
                
                result = {
                    "open": [x[0] for x in candles],
                    "high": [x[1] for x in candles],
                    "low": [x[2] for x in candles],
                    "close": [x[3] for x in candles]
                }
                
                logger.debug(f"Generated mock data for {symbol}: {len(result['close'])} candles")
                return result
                
            except Exception as e:
                error_msg = log_exception(f"fetch_candles_mock_{symbol}", e)
                tui = get_tui_instance()
                tui.add_error(f"Mock data generation error {symbol}: {error_msg}")
                return {"open": [], "high": [], "low": [], "close": []}
        
        # Live trading - fetch real data
        try:
            limit = max(settings.pr_atr_len + 50, 250)
            logger.debug(f"Fetching {limit} real candles for {symbol}")
            
            ohlcv = await client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            result = {
                "open": [x[1] for x in ohlcv],
                "high": [x[2] for x in ohlcv],
                "low": [x[3] for x in ohlcv],
                "close": [x[4] for x in ohlcv]
            }
            
            logger.debug(f"Fetched real data for {symbol}: {len(result['close'])} candles")
            return result
            
        except Exception as e:
            error_msg = log_exception(f"fetch_candles_live_{symbol}", e)
            tui = get_tui_instance()
            tui.add_error(f"Candle fetch error {symbol}: {error_msg}")
            return {"open": [], "high": [], "low": [], "close": []}
            
    except Exception as e:
        error_msg = log_exception(f"fetch_candles_{symbol}", e)
        return {"open": [], "high": [], "low": [], "close": []}

async def scan_symbol(client, symbol: str, timeframe: str = "5m") -> Optional[Dict[str, Any]]:
    """Scan a single symbol for trading opportunities"""
    try:
        logger.debug(f"Scanning symbol {symbol}")
        
        # Fetch candles
        candles = await fetch_candles(client, symbol, timeframe)
        
        if not candles["close"] or len(candles["close"]) < 50:
            logger.warning(f"Insufficient candle data for {symbol}: {len(candles.get('close', []))} candles")
            return None
        
        # Get current price
        price = candles["close"][-1]
        logger.debug(f"Current price for {symbol}: {price}")
        
        # Compute predictive ranges
        avg, r1, r2, s1, s2 = compute_predictive_ranges(candles)
        
        if avg == 0:  # Failed to compute ranges
            logger.warning(f"Failed to compute predictive ranges for {symbol}")
            return None
        
        # Calculate trend
        sma = sum(candles["close"][-settings.trend_len:]) / min(settings.trend_len, len(candles["close"]))
        trend_ok_long = price > sma
        trend_ok_short = price < sma
        
        logger.debug(f"Trend analysis for {symbol}: SMA={sma:.4f}, long_ok={trend_ok_long}, short_ok={trend_ok_short}")
        
        # Calculate bounce probabilities (simplified)
        long_bounce_prob = max(0, (s1 - price) / s1) if s1 > 0 else 0
        short_bounce_prob = max(0, (price - r1) / r1) if r1 > 0 else 0
        
        # Basic entry conditions
        base_long = price <= s1 * 1.01  # Near support
        base_short = price >= r1 * 0.99  # Near resistance
        
        # Trend strength
        trend_strength = abs(price - sma) / sma if sma > 0 else 0
        trend_conf = min(0.8, trend_strength * 10)
        
        # Create score inputs for long
        si_long = ScoreInputs(
            avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
            close=price, open=candles["open"][-1],
            bounce_prob=long_bounce_prob,
            bias_up_conf=trend_conf if trend_ok_long else 0.2,
            bias_dn_conf=0.0
        )
        
        # Create score inputs for short
        si_short = ScoreInputs(
            avg=avg, r1=r1, r2=r2, s1=s1, s2=s2,
            close=price, open=candles["open"][-1],
            bounce_prob=short_bounce_prob,
            bias_up_conf=0.0,
            bias_dn_conf=trend_conf if trend_ok_short else 0.2
        )
        
        # Compute scores
        long_score = compute_total_score(si_long, "long", settings.score_min)
        short_score = compute_total_score(si_short, "short", settings.score_min)
        
        # Entry signals
        long_entry = base_long and trend_ok_long and long_score >= settings.score_min
        short_entry = base_short and trend_ok_short and short_score >= settings.score_min
        
        logger.debug(f"Scan results for {symbol}: long_score={long_score}, short_score={short_score}, long_signal={long_entry}, short_signal={short_entry}")
        
        return {
            "symbol": symbol,
            "price": price,
            "levels": {"avg": avg, "r1": r1, "r2": r2, "s1": s1, "s2": s2},
            "signals": {"long": long_entry, "short": short_entry},
            "scores": {"long": long_score, "short": short_score},
            "candles": candles
        }
        
    except Exception as e:
        error_msg = log_exception(f"scan_symbol_{symbol}", e)
        tui = get_tui_instance()
        tui.add_error(f"Scan error {symbol}: {error_msg}")
        return None

async def update_account_data(client, tui: TradingTUI):
    """Update account data and positions in TUI"""
    try:
        logger.debug("Updating account data")
        
        if not settings.live_trade:
            # Paper trading - use mock data
            logger.debug("Using mock account data for paper trading")
            tui.stats.update({
                "account_balance": settings.account_balance_usdt,
                "equity": settings.account_balance_usdt + tui.stats["pnl"],
                "margin_used": 0.0,
                "open_positions": len(tui.positions)
            })
            return
        
        # Live trading - fetch real account data
        try:
            logger.debug("Fetching live account balance")
            balance_info = await client.fetch_balance()
            
            logger.debug("Fetching live positions")
            positions = await client.fetch_positions()
            
            # Update account stats
            usdt_balance = balance_info.get('USDT', {})
            tui.stats.update({
                "account_balance": float(usdt_balance.get('total', 0)),
                "equity": float(usdt_balance.get('total', 0)),
                "margin_used": float(usdt_balance.get('used', 0)),
                "open_positions": len([p for p in positions if float(p.get('contracts', 0)) > 0])
            })
            
            logger.debug(f"Account balance updated: {tui.stats['account_balance']}, open positions: {tui.stats['open_positions']}")
            
            # Update positions in TUI
            tui.positions.clear()
            for pos in positions:
                try:
                    if float(pos.get('contracts', 0)) > 0:
                        position_entry = TradeEntry(
                            symbol=pos['symbol'],
                            side="LONG" if float(pos['contracts']) > 0 else "SHORT",
                            quantity=abs(float(pos['contracts'])),
                            entry_price=float(pos.get('entryPrice', 0)),
                            tp1_price=0,
                            tp2_price=0,
                            sl_price=0
                        )
                        tui.positions.append(position_entry)
                except Exception as e:
                    log_exception(f"update_position_{pos.get('symbol', 'unknown')}", e)
                    continue
            
        except Exception as e:
            error_msg = log_exception("fetch_account_data", e)
            tui.add_error(f"Account fetch error: {error_msg}")
        
    except Exception as e:
        error_msg = log_exception("update_account_data", e)
        tui.add_error(f"Account update error: {error_msg}")

async def execute_trade(client, scan_result: Dict[str, Any]):
    """Execute a trade based on scan result"""
    try:
        symbol = scan_result["symbol"]
        price = scan_result["price"]
        levels = scan_result["levels"]
        signals = scan_result["signals"]
        
        logger.info(f"Attempting to execute trade for {symbol}")
        
        if not (signals["long"] or signals["short"]):
            logger.debug(f"No trading signals for {symbol}")
            return
        
        tui = get_tui_instance()
        
        try:
            # Check position limits
            if tui.stats["open_positions"] >= settings.max_positions:
                msg = f"Max positions ({settings.max_positions}) reached"
                logger.warning(msg)
                tui.add_error(msg)
                return
            
            # Check daily loss limit
            daily_loss_pct = (tui.stats['daily_pnl'] / tui.stats['account_balance'] * 100) if tui.stats['account_balance'] > 0 else 0
            if abs(daily_loss_pct) >= settings.max_daily_loss_pct:
                msg = f"Daily loss limit ({settings.max_daily_loss_pct}%) reached"
                logger.warning(msg)
                tui.add_error(msg)
                return
            
            # Determine trade direction and calculate levels
            if signals["long"]:
                side = "buy"
                entry_price = price
                # Fixed: Ensure TP is always above entry for longs
                tp1_price = max(levels["avg"], price * 1.01) if levels["avg"] > price else price * 1.01
                tp2_price = max(levels["r1"], price * 1.02) if levels["r1"] > price else price * 1.02
                sl_price = levels["s2"] if levels["s2"] > 0 and levels["s2"] < price else price * 0.98
                
            else:  # short signal
                side = "sell"
                entry_price = price
                # Fixed: Ensure TP is always below entry for shorts
                tp1_price = min(levels["avg"], price * 0.99) if levels["avg"] < price else price * 0.99
                tp2_price = min(levels["s1"], price * 0.98) if levels["s1"] > 0 and levels["s1"] < price else price * 0.98
                sl_price = levels["r2"] if levels["r2"] > 0 and levels["r2"] > price else price * 1.02
            
            logger.debug(f"Trade levels for {symbol}: entry={entry_price}, tp1={tp1_price}, tp2={tp2_price}, sl={sl_price}")
            
            # Calculate position size based on risk
            stop_distance = abs(entry_price - sl_price)
            if stop_distance <= 0:
                msg = f"Invalid stop distance for {symbol}: {stop_distance}"
                logger.error(msg)
                tui.add_error(msg)
                return
            
            # Validate stop distance isn't too large (max 5%)
            max_stop_distance = entry_price * 0.05
            if stop_distance > max_stop_distance:
                logger.warning(f"Stop distance too large for {symbol}: {stop_distance/entry_price*100:.2f}%, capping at 5%")
                stop_distance = max_stop_distance
                # Recalculate stop price based on capped distance
                if signals["long"]:
                    sl_price = entry_price - stop_distance
                else:
                    sl_price = entry_price + stop_distance
                
            # Use live equity if available, otherwise fallback to configured account balance
            effective_equity = tui.stats.get("equity") or tui.stats.get("account_balance") or settings.account_balance_usdt
            risk_amount = float(effective_equity) * (settings.risk_per_trade_pct / 100)
            quantity = risk_amount / stop_distance
            
            # Apply minimum quantity and lot size constraints
            min_qty = 0.001  # Default minimum
            lot_size = 0.001  # Default lot size
            
            if symbol in settings.symbol_overrides:
                min_qty = settings.symbol_overrides[symbol].get("minQty", 0.001)
                lot_size = settings.symbol_overrides[symbol].get("lotSize", 0.001)
            
            quantity = max(min_qty, round(quantity / lot_size) * lot_size)
            
            # Add maximum position size check
            max_position_value = effective_equity * settings.max_capital_fraction
            max_quantity = max_position_value / entry_price
            quantity = min(quantity, max_quantity)
            
            logger.debug(f"Position sizing for {symbol}: risk_amount={risk_amount}, quantity={quantity}")
            
            # Build bracket orders
            intents = build_bracket_orders(symbol, side, quantity, entry_price, tp1_price, tp2_price, sl_price)
            
            # Execute trade
            if settings.live_trade:
                logger.info(f"Placing live trade for {symbol}")
                result = await place_bracket_trade(client, symbol, intents)
                
                if "error" in result:
                    msg = f"Trade failed {symbol}: {result['error']}"
                    logger.error(msg)
                    tui.add_error(msg)
                    return
                    
                msg = f"✅ Trade placed: {symbol} {side.upper()} {quantity}"
                logger.info(msg)
                tui.add_error(msg)
            else:
                # Paper trading - simulate execution
                result = {"dryRun": True, "intents": intents}
                msg = f"📝 Paper trade: {symbol} {side.upper()} {quantity}"
                logger.info(msg)
                tui.add_error(msg)
            
            # Add to TUI trades
            trade_entry = TradeEntry(
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                entry_price=entry_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                sl_price=sl_price
            )
            
            tui.add_trade_entry(trade_entry)
            
            # Update stats
            tui.stats["trades"] += 1
            if not settings.live_trade:
                # For paper trading, simulate some PnL
                simulated_pnl = quantity * (tp1_price - entry_price) * 0.5 if side == "buy" else quantity * (entry_price - tp1_price) * 0.5
                tui.stats["pnl"] += simulated_pnl
                tui.stats["daily_pnl"] += simulated_pnl
            
            print(f"✅ Trade executed: {symbol} {side.upper()} {quantity:.6f} @ ${entry_price:.4f}")
            
        except Exception as e:
            error_msg = log_exception(f"execute_trade_inner_{symbol}", e)
            tui.add_error(f"Execute error {symbol}: {error_msg}")
            print(f"❌ Trade error {symbol}: {error_msg}")
            
    except Exception as e:
        error_msg = log_exception(f"execute_trade_{scan_result.get('symbol', 'unknown')}", e)
        print(f"❌ Trade execution failed: {error_msg}")

async def get_trading_symbols(client) -> List[str]:
    """Get list of symbols to trade"""
    try:
        logger.info("Getting trading symbols")
        
        if settings.symbols_manual:
            # Use manually specified symbols
            symbols = [s.strip().upper() for s in settings.symbols_manual.split(",") if s.strip()]
            logger.info(f"Using manual symbols: {symbols}")
            return symbols
        
        # Paper trading fallback symbols
        if not client or not settings.live_trade:
            fallback_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "AVAX/USDT:USDT", "ADA/USDT:USDT"]
            logger.info(f"Using fallback symbols for paper trading: {fallback_symbols}")
            return fallback_symbols
        
        try:
            # Fetch all markets
            logger.debug("Fetching markets from exchange")
            markets = await client.fetch_markets()
            logger.debug(f"Fetched {len(markets)} markets")
            
            # Filter for USDT perpetual futures with sufficient leverage
            symbols = []
            for market in markets:
                try:
                    if (market.get('type') == 'swap' and 
                        market.get('quote') == 'USDT' and 
                        market.get('active', False)):
                        
                        # Check leverage
                        leverage_info = market.get('limits', {}).get('leverage', {})
                        max_leverage = leverage_info.get('max', 1)
                        
                        if max_leverage >= 34:  # Use 34x minimum leverage
                            # Include ALL pairs - no exclusions for maximum coverage
                            symbols.append(market['symbol'])
                                
                except Exception as e:
                    log_exception(f"filter_market_{market.get('symbol', 'unknown')}", e)
                    continue
            
            filtered_symbols = symbols  # Use ALL symbols that meet 34x leverage requirement
            logger.info(f"Filtered {len(filtered_symbols)} trading symbols from {len(markets)} markets")
            return filtered_symbols
            
        except Exception as e:
            error_msg = log_exception("fetch_markets", e)
            tui = get_tui_instance()
            tui.add_error(f"Symbol fetch error: {error_msg}")
            fallback = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
            logger.warning(f"Using fallback symbols due to error: {fallback}")
            return fallback
            
    except Exception as e:
        error_msg = log_exception("get_trading_symbols", e)
        fallback = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        logger.error(f"Failed to get symbols, using minimal fallback: {fallback}")
        return fallback

# ============================================================================
# MAIN SCANNING LOOP WITH CHECKPOINTS
# ============================================================================

async def run_trading_bot():
    """Main trading bot loop with checkpoints and error recovery"""
    global SHUTDOWN_REQUESTED
    
    try:
        logger.info("Starting Unified Phemex Trading Bot")
        
        print("Starting Unified Phemex Trading Bot")
        print("=" * 50)
        print(f"Exchange: Phemex")
        print(f"Live Trading: {'YES' if settings.live_trade else 'NO (Paper Mode)'}")
        print(f"Leverage: {settings.leverage_max}x")
        print(f"Min Score: {settings.score_min}")
        print(f"Risk per Trade: {settings.risk_per_trade_pct}%")
        print("=" * 50)
        
        # Quick run mode for immediate completion
        quick_run = os.getenv("QUICK_RUN", "false").lower() in ("1", "true", "yes")
        
        if quick_run:
            logger.info("QUICK RUN MODE enabled")
            print("QUICK RUN MODE - Single scan then exit")
        
        # Checkpoint: Bot startup
        checkpoint_manager.save_checkpoint("bot_startup", {
            "timestamp": time.time(),
            "mode": "live" if settings.live_trade else "paper",
            "quick_run": quick_run
        })
        
        # Try to start TUI safely
        tui = None
        if not quick_run:
            tui = safe_start_tui()
            if tui:
                print("TUI started successfully")
            else:
                print("TUI failed, continuing in console mode")
        
        # Initialize exchange client safely
        client = await safe_initialize_client()
        if not client and settings.live_trade:
            logger.error("Cannot continue without live trading client")
            return
        
        # Checkpoint: Client initialized
        checkpoint_manager.save_checkpoint("client_ready", {
            "client_status": "ready" if client else "paper",
            "timestamp": time.time()
        })
        
        # Get trading symbols with fallback
        try:
            logger.info("Loading trading symbols...")
            symbols = await timeout_manager.with_timeout(
                get_trading_symbols(client), "api_call", ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            )
            
            if not symbols:
                logger.warning("No symbols loaded, using fallback")
                symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            
            print(f"Loaded {len(symbols)} symbols")
            logger.info(f"Loaded {len(symbols)} symbols for trading")
            
            # Checkpoint: Symbols loaded
            checkpoint_manager.save_checkpoint("symbols_loaded", {
                "count": len(symbols),
                "symbols": symbols[:10]  # Save first 10 for reference
            })
            
        except Exception as e:
            error_msg = log_exception("get_trading_symbols", e)
            print(f"Failed to load symbols: {error_msg}")
            print("Using fallback symbols...")
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        
        scan_count = 0
        
        try:
            max_scans = 1 if quick_run else 999999
            logger.info(f"Starting scan loop: max_scans={max_scans}")
            
            for scan_cycle in range(max_scans):
                # Check for shutdown request
                if SHUTDOWN_REQUESTED:
                    logger.info("Shutdown requested, stopping scan loop")
                    break
                
                scan_count += 1
                logger.debug(f"Starting scan cycle {scan_count}")
                
                # Checkpoint: Scan cycle start
                checkpoint_manager.save_checkpoint("scan_cycle", {
                    "cycle": scan_count,
                    "timestamp": time.time(),
                    "symbols_count": len(symbols)
                })
                
                if quick_run:
                    print(f"Quick scan of {len(symbols)} symbols...")
                
                # Scan all symbols with timeout protection
                for i, symbol in enumerate(symbols):
                    # Check for shutdown request
                    if SHUTDOWN_REQUESTED:
                        break
                    
                    try:
                        # Progress indicator
                        if i % 10 == 0:
                            print(f"Scanning... {i+1}/{len(symbols)} symbols")
                        
                        # Scan symbol with timeout
                        result = await timeout_manager.with_timeout(
                            scan_symbol(client, symbol), "symbol_scan", None
                        )
                        
                        if result:
                            # Process result
                            max_score = max(result["scores"]["long"], result["scores"]["short"])
                            
                            # Console output for high scores
                            if max_score >= 80:
                                side = "LONG" if result["scores"]["long"] > result["scores"]["short"] else "SHORT"
                                signal_str = "SIGNAL" if result["signals"]["long"] or result["signals"]["short"] else ""
                                print(f"{result['symbol']:12} {side:5} Score:{max_score:3d} Price:${result['price']:8.3f} {signal_str}")
                            
                            # Execute trade if signal and score is high enough
                            if (result["signals"]["long"] or result["signals"]["short"]) and max_score >= settings.score_min:
                                if quick_run:
                                    print(f"SIGNAL FOUND: {symbol} {max_score}")
                                logger.info(f"Trade signal found: {symbol} score={max_score}")
                                
                                # Execute trade with timeout
                                await timeout_manager.with_timeout(
                                    execute_trade(client, result), "api_call", None
                                )
                        
                        # Small delay between symbols to prevent overwhelming
                        await asyncio.sleep(0.05)
                        
                    except Exception as e:
                        error_msg = log_exception(f"scan_symbol_{symbol}", e)
                        print(f"Error scanning {symbol}: {error_msg}")
                        continue
                
                # Checkpoint: Scan cycle completed
                checkpoint_manager.save_checkpoint("scan_completed", {
                    "cycle": scan_count,
                    "timestamp": time.time(),
                    "status": "success"
                })
                
                if quick_run:
                    print(f"Quick scan completed! Scanned {len(symbols)} symbols")
                    logger.info(f"Quick scan completed successfully")
                    break
                else:
                    # Wait before next scan cycle
                    print(f"\nScan #{scan_count} completed. Waiting 30s for next cycle...")
                    logger.info(f"Scan cycle {scan_count} completed, waiting for next cycle")
                    
                    # Wait with shutdown check
                    for _ in range(30):
                        if SHUTDOWN_REQUESTED:
                            break
                        await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
            print("\nShutting down...")
        except Exception as e:
            error_msg = log_exception("main_scan_loop", e)
            print(f"Fatal error: {error_msg}")
            
            # Checkpoint: Error occurred
            checkpoint_manager.save_checkpoint("error_occurred", {
                "error": error_msg,
                "timestamp": time.time(),
                "scan_count": scan_count
            })
        finally:
            # Cleanup
            logger.info("Starting cleanup")
            try:
                if client:
                    logger.debug("Closing exchange client")
                    await timeout_manager.with_timeout(client.close(), "api_call", None)
            except Exception as e:
                log_exception("client_cleanup", e)
            
            if tui:
                logger.debug("Stopping TUI")
                try:
                    tui.stop()
                except:
                    pass
            
            # Checkpoint: Cleanup completed
            checkpoint_manager.save_checkpoint("cleanup_completed", {
                "timestamp": time.time(),
                "scan_count": scan_count
            })
            
            print("Bot stopped")
            logger.info("Bot stopped successfully")
            
    except Exception as e:
        error_msg = log_exception("run_trading_bot", e)
        print(f"Fatal error in main bot function: {error_msg}")
        
        # Checkpoint: Fatal error
        checkpoint_manager.save_checkpoint("fatal_error", {
            "error": error_msg,
            "timestamp": time.time()
        })

# ============================================================================
# MAIN ENTRY POINT WITH SAFETY CHECKS
# ============================================================================

def main():
    """Main entry point with comprehensive safety checks"""
    try:
        logger.info("Starting main function")
        
        # Quick run mode check
        quick_run = os.getenv("QUICK_RUN", "false").lower() in ("1", "true", "yes")
        
        # CRITICAL SAFETY CHECK - Validate settings before starting
        if not validate_safety_settings():
            print("❌ Safety validation failed! Check your configuration.")
            print("Set LIVE_TRADE=false for paper trading mode.")
            sys.exit(1)
        
        # Validate required environment variables only for live trading
        if settings.live_trade and (not settings.phemex_api_key or not settings.phemex_api_secret):
            error_msg = "Missing Phemex API credentials for live trading!"
            logger.error(error_msg)
            print("❌ Missing Phemex API credentials for live trading!")
            print("Set PHEMEX_API_KEY and PHEMEX_API_SECRET environment variables")
            print("Or set LIVE_TRADE=false for paper trading")
            sys.exit(1)
        
        if not settings.live_trade:
            logger.info("PAPER TRADING mode enabled")
            print("PAPER TRADING mode - no API keys required")
        
        if quick_run:
            logger.info("QUICK RUN mode enabled")
            print("QUICK RUN mode enabled - will complete immediately")
        
        # Checkpoint: Main function started
        checkpoint_manager.save_checkpoint("main_started", {
            "timestamp": time.time(),
            "live_trade": settings.live_trade,
            "quick_run": quick_run
        })
        
        try:
            logger.info("Starting asyncio event loop")
            asyncio.run(run_trading_bot())
            
            if quick_run:
                logger.info("Quick run completed successfully")
                print("✅ Quick run completed successfully!")
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            print("\n👋 Goodbye!")
        except Exception as e:
            error_msg = log_exception("main_asyncio_run", e)
            print(f"❌ Fatal error: {error_msg}")
            
            # Checkpoint: Asyncio error
            checkpoint_manager.save_checkpoint("asyncio_error", {
                "error": error_msg,
                "timestamp": time.time()
            })
            
            sys.exit(1)
            
    except Exception as e:
        error_msg = log_exception("main_function", e)
        print(f"❌ Fatal error in main: {error_msg}")
        
        # Checkpoint: Main function error
        checkpoint_manager.save_checkpoint("main_error", {
            "error": error_msg,
            "timestamp": time.time()
        })
        
        sys.exit(1)

if __name__ == "__main__":
    main()
