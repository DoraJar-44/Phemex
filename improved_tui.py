#!/usr/bin/env python3
"""
IMPROVED TRADING BOT TUI
Enhanced Terminal User Interface with better design, colors, and layout
"""

import curses
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

@dataclass
class ScoreEntry:
    symbol: str
    long_score: int
    short_score: int
    price: float
    long_signal: bool
    short_signal: bool
    timestamp: datetime

@dataclass
class TradeEntry:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    timestamp: datetime

@dataclass
class PositionEntry:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float

class ImprovedTradingTUI:
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
            "account_balance": 10000.0, "equity": 10000.0, "margin_used": 0.0,
            "win_rate": 0.0, "avg_profit": 0.0, "max_drawdown": 0.0,
            "daily_pnl": 0.0, "total_volume": 0.0, "open_positions": 0,
            "winning_trades": 0, "losing_trades": 0, "total_trades": 0,
            "best_trade": 0.0, "worst_trade": 0.0, "current_streak": 0
        }
        self.color_pairs = {}
        self.unicode_support = True
        self.animation_frame = 0
        self.market_trend = "neutral"  # "bullish", "bearish", "neutral"
        
    def init_colors(self):
        """Initialize enhanced color scheme"""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs with better contrast
        color_definitions = [
            (1, 46, -1),    # Bright green on default
            (2, 226, -1),   # Yellow on default
            (3, 196, -1),   # Red on default
            (4, 51, -1),    # Cyan on default
            (5, 201, -1),   # Magenta on default
            (6, 255, -1),   # White on default
            (7, 244, -1),   # Gray on default
            (8, 82, -1),    # Light green on default
            (9, 208, -1),   # Orange on default
            (10, 21, -1),   # Blue on default
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
                    # Fallback to basic colors
                    basic_colors = [
                        curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED,
                        curses.COLOR_CYAN, curses.COLOR_MAGENTA, curses.COLOR_WHITE
                    ]
                    curses.init_pair(pair_num, basic_colors[min(i-1, 5)], curses.COLOR_BLACK)
            except:
                curses.init_pair(pair_num, curses.COLOR_WHITE, curses.COLOR_BLACK)
                
        # Store color pairs for easy access
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

    def draw_double_border(self, win, title: str, color=None, style="double"):
        """Draw enhanced border with title"""
        if color is None:
            color = self.color_pairs['header']
            
        h, w = win.getmaxyx()
        
        # Border characters
        if style == "double" and self.unicode_support:
            chars = {
                'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
                'h': '═', 'v': '║'
            }
        elif style == "rounded" and self.unicode_support:
            chars = {
                'tl': '╭', 'tr': '╮', 'bl': '╰', 'br': '╯',
                'h': '─', 'v': '│'
            }
        else:
            # Fallback to simple box
            win.box()
            win.addstr(0, 2, f" {title} ", color | curses.A_BOLD)
            return
            
        try:
            # Draw corners
            win.addstr(0, 0, chars['tl'], color)
            win.addstr(0, w-1, chars['tr'], color)
            win.addstr(h-1, 0, chars['bl'], color)
            win.addstr(h-1, w-1, chars['br'], color)
            
            # Draw horizontal lines
            for x in range(1, w-1):
                win.addstr(0, x, chars['h'], color)
                win.addstr(h-1, x, chars['h'], color)
                
            # Draw vertical lines
            for y in range(1, h-1):
                win.addstr(y, 0, chars['v'], color)
                win.addstr(y, w-1, chars['v'], color)
                
            # Add title with padding
            title_with_padding = f" {title} "
            title_start = (w - len(title_with_padding)) // 2
            win.addstr(0, title_start, title_with_padding, color | curses.A_BOLD)
        except curses.error:
            pass

    def draw_progress_bar(self, win, y: int, x: int, width: int, value: float, max_value: float, 
                         label: str = "", color=None):
        """Draw a progress bar with percentage"""
        if color is None:
            color = self.color_pairs['success']
            
        if max_value <= 0:
            percentage = 0
        else:
            percentage = min(100, max(0, (value / max_value) * 100))
            
        filled = int((width - 2) * percentage / 100)
        
        try:
            # Draw bar background
            win.addstr(y, x, "[", self.color_pairs['muted'])
            win.addstr(y, x + width - 1, "]", self.color_pairs['muted'])
            
            # Draw filled portion
            if self.unicode_support:
                for i in range(filled):
                    win.addstr(y, x + 1 + i, "█", color)
                for i in range(filled, width - 2):
                    win.addstr(y, x + 1 + i, "░", self.color_pairs['muted'])
            else:
                for i in range(filled):
                    win.addstr(y, x + 1 + i, "=", color)
                for i in range(filled, width - 2):
                    win.addstr(y, x + 1 + i, "-", self.color_pairs['muted'])
                    
            # Add percentage label
            if label:
                label_text = f" {label}: {percentage:.1f}%"
                win.addstr(y, x + width + 1, label_text, self.color_pairs['normal'])
        except curses.error:
            pass

    def draw_sparkline(self, win, y: int, x: int, data: List[float], width: int, color=None):
        """Draw a mini chart using Unicode characters"""
        if not data or not self.unicode_support:
            return
            
        if color is None:
            color = self.color_pairs['info']
            
        # Unicode spark characters
        sparks = " ▁▂▃▄▅▆▇█"
        
        # Normalize data
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Sample data if too long
        if len(data) > width:
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = data
            
        try:
            for i, value in enumerate(sampled):
                if i >= width:
                    break
                normalized = (value - min_val) / range_val
                spark_idx = int(normalized * (len(sparks) - 1))
                win.addstr(y, x + i, sparks[spark_idx], color)
        except curses.error:
            pass

    def format_number(self, value: float, prefix: str = "", suffix: str = "", 
                     decimals: int = 2) -> str:
        """Format number with proper spacing and symbols"""
        if abs(value) >= 1_000_000:
            return f"{prefix}{value/1_000_000:.{decimals}f}M{suffix}"
        elif abs(value) >= 1_000:
            return f"{prefix}{value/1_000:.{decimals}f}K{suffix}"
        else:
            return f"{prefix}{value:.{decimals}f}{suffix}"

    def draw_header_pane(self, win):
        """Draw main header with title and status"""
        win.clear()
        h, w = win.getmaxyx()
        
        # Title
        title = "╔═══════════════════════════════════════════════════════════════════╗"
        subtitle = "║         PHEMEX TRADING BOT - PROFESSIONAL DASHBOARD v2.0         ║"
        bottom = "╚═══════════════════════════════════════════════════════════════════╝"
        
        if w >= len(title):
            x_offset = (w - len(title)) // 2
            try:
                win.addstr(0, x_offset, title, self.color_pairs['header'])
                win.addstr(1, x_offset, subtitle, self.color_pairs['header'] | curses.A_BOLD)
                win.addstr(2, x_offset, bottom, self.color_pairs['header'])
            except curses.error:
                pass
        
        # Status indicators
        try:
            status_y = 3
            
            # Connection status
            conn_status = "● CONNECTED" if self.running else "○ DISCONNECTED"
            conn_color = self.color_pairs['success'] if self.running else self.color_pairs['danger']
            win.addstr(status_y, 2, conn_status, conn_color | curses.A_BOLD)
            
            # Trading mode
            mode = "LIVE" if self.stats.get('live_mode', False) else "PAPER"
            mode_color = self.color_pairs['danger'] if mode == "LIVE" else self.color_pairs['warning']
            win.addstr(status_y, 20, f"[{mode}]", mode_color | curses.A_BOLD)
            
            # Time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            win.addstr(status_y, w - 22, current_time, self.color_pairs['info'])
            
            # Market trend indicator
            trend_icon = {"bullish": "↑", "bearish": "↓", "neutral": "→"}.get(self.market_trend, "→")
            trend_color = {"bullish": self.color_pairs['success'], 
                          "bearish": self.color_pairs['danger'],
                          "neutral": self.color_pairs['warning']}.get(self.market_trend, self.color_pairs['normal'])
            win.addstr(status_y, 35, f"Market: {trend_icon}", trend_color | curses.A_BOLD)
            
        except curses.error:
            pass

    def draw_account_overview(self, win):
        """Draw account overview with key metrics"""
        win.clear()
        self.draw_double_border(win, "💰 ACCOUNT OVERVIEW", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        # Calculate metrics
        balance = self.stats['account_balance']
        equity = self.stats['equity']
        margin = self.stats['margin_used']
        free_margin = equity - margin
        margin_level = (equity / margin * 100) if margin > 0 else 999
        
        # Display metrics with proper formatting
        metrics = [
            ("Balance", self.format_number(balance, "$"), self.color_pairs['normal']),
            ("Equity", self.format_number(equity, "$"), 
             self.color_pairs['profit'] if equity > balance else self.color_pairs['loss']),
            ("Margin Used", self.format_number(margin, "$"), self.color_pairs['warning']),
            ("Free Margin", self.format_number(free_margin, "$"), 
             self.color_pairs['success'] if free_margin > margin else self.color_pairs['warning']),
            ("Margin Level", f"{margin_level:.1f}%", 
             self.color_pairs['success'] if margin_level > 200 else self.color_pairs['danger']),
        ]
        
        for label, value, color in metrics:
            if y >= h - 1:
                break
            try:
                win.addstr(y, 2, f"{label}:", self.color_pairs['muted'])
                win.addstr(y, 15, value, color | curses.A_BOLD)
            except curses.error:
                pass
            y += 1
        
        # Add margin level progress bar
        if y + 1 < h - 1:
            self.draw_progress_bar(win, y + 1, 2, min(30, w - 4), 
                                 min(margin_level, 500), 500, "Margin", 
                                 self.color_pairs['success'] if margin_level > 200 else self.color_pairs['danger'])

    def draw_performance_dashboard(self, win):
        """Draw performance metrics with visual indicators"""
        win.clear()
        self.draw_double_border(win, "📊 PERFORMANCE METRICS", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        # Calculate win rate
        total_trades = self.stats.get('total_trades', 0)
        winning_trades = self.stats.get('winning_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Performance metrics
        try:
            # Daily PnL with color coding
            daily_pnl = self.stats['daily_pnl']
            daily_pnl_pct = (daily_pnl / self.stats['account_balance'] * 100) if self.stats['account_balance'] > 0 else 0
            pnl_color = self.color_pairs['profit'] if daily_pnl > 0 else self.color_pairs['loss'] if daily_pnl < 0 else self.color_pairs['normal']
            
            win.addstr(y, 2, "Daily P&L:", self.color_pairs['muted'])
            win.addstr(y, 15, f"{self.format_number(daily_pnl, '$')} ({daily_pnl_pct:+.2f}%)", pnl_color | curses.A_BOLD)
            y += 1
            
            # Total PnL
            total_pnl = self.stats['pnl']
            total_pnl_pct = (total_pnl / self.stats['account_balance'] * 100) if self.stats['account_balance'] > 0 else 0
            pnl_color = self.color_pairs['profit'] if total_pnl > 0 else self.color_pairs['loss'] if total_pnl < 0 else self.color_pairs['normal']
            
            win.addstr(y, 2, "Total P&L:", self.color_pairs['muted'])
            win.addstr(y, 15, f"{self.format_number(total_pnl, '$')} ({total_pnl_pct:+.2f}%)", pnl_color | curses.A_BOLD)
            y += 2
            
            # Win rate with progress bar
            win.addstr(y, 2, f"Win Rate: {win_rate:.1f}%", self.color_pairs['normal'])
            y += 1
            self.draw_progress_bar(win, y, 2, min(30, w - 4), win_rate, 100, "", 
                                 self.color_pairs['success'] if win_rate > 50 else self.color_pairs['warning'])
            y += 2
            
            # Trade statistics
            win.addstr(y, 2, f"Trades: {total_trades} (W:{winning_trades} L:{self.stats.get('losing_trades', 0)})", 
                      self.color_pairs['info'])
            y += 1
            
            # Best/Worst trade
            if self.stats.get('best_trade', 0) != 0:
                win.addstr(y, 2, f"Best: {self.format_number(self.stats['best_trade'], '$')}", self.color_pairs['profit'])
                y += 1
            if self.stats.get('worst_trade', 0) != 0:
                win.addstr(y, 2, f"Worst: {self.format_number(self.stats['worst_trade'], '$')}", self.color_pairs['loss'])
                y += 1
                
            # Current streak
            streak = self.stats.get('current_streak', 0)
            if streak != 0:
                streak_text = f"Streak: {abs(streak)} {'wins' if streak > 0 else 'losses'}"
                streak_color = self.color_pairs['success'] if streak > 0 else self.color_pairs['danger']
                win.addstr(y, 2, streak_text, streak_color)
                
        except curses.error:
            pass

    def draw_market_scanner(self, win):
        """Draw market scanner with top opportunities"""
        win.clear()
        self.draw_double_border(win, "🔍 MARKET SCANNER", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            # Scanner stats
            win.addstr(y, 2, f"Scanned: {self.stats['scanned']} | Signals: {self.stats['signals']}", 
                      self.color_pairs['info'])
            y += 1
            
            # Separator
            win.addstr(y, 2, "─" * (w - 4), self.color_pairs['muted'])
            y += 1
            
            # Column headers
            headers = f"{'Symbol':<10} {'Side':<6} {'Score':<6} {'Price':<12} {'Signal':<8}"
            win.addstr(y, 2, headers, self.color_pairs['header'] | curses.A_BOLD)
            y += 1
            
            # Top scoring symbols
            high_scores = sorted(self.scores, key=lambda x: max(x.long_score, x.short_score), reverse=True)[:10]
            
            for entry in high_scores:
                if y >= h - 1:
                    break
                    
                max_score = max(entry.long_score, entry.short_score)
                side = "LONG" if entry.long_score > entry.short_score else "SHORT"
                signal = "✓" if (side == "LONG" and entry.long_signal) or (side == "SHORT" and entry.short_signal) else ""
                
                # Format row
                row = f"{entry.symbol:<10} {side:<6} {max_score:<6} ${entry.price:<11.4f} {signal:<8}"
                
                # Color based on score and signal
                if signal:
                    color = self.color_pairs['special'] | curses.A_BOLD
                else:
                    color = self.get_score_color(max_score)
                    
                win.addstr(y, 2, row[:w-3], color)
                y += 1
                
        except curses.error:
            pass

    def draw_active_positions(self, win):
        """Draw active positions with P&L"""
        win.clear()
        self.draw_double_border(win, "📈 ACTIVE POSITIONS", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            if not self.positions:
                win.addstr(y, 2, "No active positions", self.color_pairs['muted'])
            else:
                # Headers
                headers = f"{'Symbol':<10} {'Side':<6} {'Size':<10} {'Entry':<10} {'P&L':<12} {'%':<8}"
                win.addstr(y, 2, headers, self.color_pairs['header'] | curses.A_BOLD)
                y += 1
                win.addstr(y, 2, "─" * (w - 4), self.color_pairs['muted'])
                y += 1
                
                total_pnl = 0
                for pos in self.positions:
                    if y >= h - 1:
                        break
                        
                    # Calculate P&L
                    pnl = pos.pnl
                    pnl_pct = pos.pnl_percent
                    total_pnl += pnl
                    
                    # Format row
                    row = f"{pos.symbol:<10} {pos.side:<6} {pos.quantity:<10.4f} ${pos.entry_price:<9.2f}"
                    
                    # P&L with color
                    pnl_text = f"{pnl:+.2f}"
                    pct_text = f"{pnl_pct:+.2f}%"
                    
                    win.addstr(y, 2, row, self.color_pairs['normal'])
                    
                    # Add P&L with appropriate color
                    pnl_color = self.color_pairs['profit'] if pnl > 0 else self.color_pairs['loss']
                    win.addstr(y, 2 + len(row), f" {pnl_text:<11}", pnl_color | curses.A_BOLD)
                    win.addstr(y, 2 + len(row) + 12, pct_text, pnl_color)
                    y += 1
                
                # Total P&L
                if y < h - 1:
                    win.addstr(y, 2, "─" * (w - 4), self.color_pairs['muted'])
                    y += 1
                    total_color = self.color_pairs['profit'] if total_pnl > 0 else self.color_pairs['loss']
                    win.addstr(y, 2, f"Total P&L: {self.format_number(total_pnl, '$')}", 
                              total_color | curses.A_BOLD)
                    
        except curses.error:
            pass

    def draw_recent_trades(self, win):
        """Draw recent trades history"""
        win.clear()
        self.draw_double_border(win, "💹 RECENT TRADES", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            if not self.trades:
                win.addstr(y, 2, "No recent trades", self.color_pairs['muted'])
            else:
                # Headers
                headers = f"{'Time':<9} {'Symbol':<10} {'Side':<6} {'Price':<10} {'Size':<10}"
                win.addstr(y, 2, headers, self.color_pairs['header'] | curses.A_BOLD)
                y += 1
                
                for trade in list(self.trades)[-8:]:
                    if y >= h - 1:
                        break
                        
                    time_str = trade.timestamp.strftime("%H:%M:%S")
                    row = f"{time_str:<9} {trade.symbol:<10} {trade.side:<6} ${trade.entry_price:<9.4f} {trade.quantity:<10.4f}"
                    
                    # Color based on side
                    color = self.color_pairs['success'] if trade.side == "LONG" else self.color_pairs['danger']
                    win.addstr(y, 2, row[:w-3], color)
                    y += 1
                    
        except curses.error:
            pass

    def draw_risk_monitor(self, win):
        """Draw risk monitoring panel"""
        win.clear()
        self.draw_double_border(win, "⚠️ RISK MONITOR", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            # Risk metrics
            margin_ratio = (self.stats['margin_used'] / self.stats['equity'] * 100) if self.stats['equity'] > 0 else 0
            daily_loss_pct = abs(self.stats['daily_pnl'] / self.stats['account_balance'] * 100) if self.stats['account_balance'] > 0 and self.stats['daily_pnl'] < 0 else 0
            position_usage = (self.stats['open_positions'] / 5 * 100)  # Assuming max 5 positions
            
            # Risk status
            risk_level = "LOW"
            risk_color = self.color_pairs['success']
            if margin_ratio > 80 or daily_loss_pct > 2:
                risk_level = "HIGH"
                risk_color = self.color_pairs['danger']
            elif margin_ratio > 50 or daily_loss_pct > 1:
                risk_level = "MEDIUM"
                risk_color = self.color_pairs['warning']
                
            win.addstr(y, 2, f"Risk Level: {risk_level}", risk_color | curses.A_BOLD)
            y += 2
            
            # Risk meters with progress bars
            # Margin usage
            win.addstr(y, 2, "Margin Usage:", self.color_pairs['normal'])
            y += 1
            self.draw_progress_bar(win, y, 2, min(30, w - 15), margin_ratio, 100, f"{margin_ratio:.1f}%",
                                 self.color_pairs['danger'] if margin_ratio > 80 else self.color_pairs['warning'] if margin_ratio > 50 else self.color_pairs['success'])
            y += 2
            
            # Daily loss
            win.addstr(y, 2, "Daily Loss:", self.color_pairs['normal'])
            y += 1
            self.draw_progress_bar(win, y, 2, min(30, w - 15), daily_loss_pct, 3, f"{daily_loss_pct:.1f}%",
                                 self.color_pairs['danger'] if daily_loss_pct > 2 else self.color_pairs['warning'] if daily_loss_pct > 1 else self.color_pairs['success'])
            y += 2
            
            # Position usage
            win.addstr(y, 2, f"Positions: {self.stats['open_positions']}/5", self.color_pairs['normal'])
            y += 1
            self.draw_progress_bar(win, y, 2, min(30, w - 15), position_usage, 100, "",
                                 self.color_pairs['warning'] if position_usage > 80 else self.color_pairs['success'])
            y += 2
            
            # Max drawdown
            drawdown = self.stats.get('max_drawdown', 0)
            win.addstr(y, 2, f"Max Drawdown: {drawdown:.1f}%", 
                      self.color_pairs['danger'] if drawdown > 10 else self.color_pairs['warning'] if drawdown > 5 else self.color_pairs['success'])
                      
        except curses.error:
            pass

    def draw_system_status(self, win):
        """Draw system status and controls"""
        win.clear()
        self.draw_double_border(win, "🎛️ SYSTEM STATUS", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            # System info
            info_lines = [
                ("Exchange:", "Phemex", self.color_pairs['info']),
                ("Timeframe:", "5m", self.color_pairs['info']),
                ("Min Score:", str(self.stats.get('min_score', 85)), self.color_pairs['info']),
                ("Leverage:", f"{self.stats.get('leverage', 25)}x", self.color_pairs['warning']),
                ("Risk/Trade:", f"{self.stats.get('risk_pct', 0.5)}%", self.color_pairs['warning']),
            ]
            
            for label, value, color in info_lines:
                if y >= h - 4:
                    break
                win.addstr(y, 2, f"{label:<12} {value}", color)
                y += 1
                
            # Separator
            y += 1
            win.addstr(y, 2, "─" * (w - 4), self.color_pairs['muted'])
            y += 1
            
            # Keyboard shortcuts
            win.addstr(y, 2, "CONTROLS:", self.color_pairs['header'] | curses.A_BOLD)
            y += 1
            
            shortcuts = [
                ("Q", "Quit"),
                ("R", "Refresh"),
                ("P", "Pause/Resume"),
                ("L", "Toggle Logs"),
                ("H", "Help"),
            ]
            
            for key, action in shortcuts:
                if y >= h - 1:
                    break
                win.addstr(y, 2, f"[{key}]", self.color_pairs['special'] | curses.A_BOLD)
                win.addstr(y, 6, action, self.color_pairs['normal'])
                y += 1
                
        except curses.error:
            pass

    def draw_log_panel(self, win):
        """Draw scrollable log panel"""
        win.clear()
        self.draw_double_border(win, "📜 SYSTEM LOGS", self.color_pairs['header'])
        
        h, w = win.getmaxyx()
        y = 2
        
        try:
            if not self.logs:
                win.addstr(y, 2, "No logs available", self.color_pairs['muted'])
            else:
                for log_entry in list(self.logs)[-(h-3):]:
                    if y >= h - 1:
                        break
                        
                    # Truncate log if too long
                    if len(log_entry) > w - 4:
                        log_entry = log_entry[:w-7] + "..."
                        
                    # Color based on log level
                    color = self.color_pairs['normal']
                    if "ERROR" in log_entry:
                        color = self.color_pairs['danger']
                    elif "WARNING" in log_entry:
                        color = self.color_pairs['warning']
                    elif "SUCCESS" in log_entry or "PROFIT" in log_entry:
                        color = self.color_pairs['success']
                    elif "INFO" in log_entry:
                        color = self.color_pairs['info']
                        
                    win.addstr(y, 2, log_entry, color)
                    y += 1
                    
        except curses.error:
            pass

    def update_display(self):
        """Update the entire display with new layout"""
        if not self.stdscr:
            return
            
        try:
            # Get screen dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Clear screen
            self.stdscr.clear()
            
            # Calculate layout (responsive design)
            header_height = 5
            
            # Main content area (below header)
            content_height = max_y - header_height
            
            # Create layout regions
            # Top row: 3 equal panels
            panel_width = max_x // 3
            panel1_height = content_height // 2
            panel2_height = content_height - panel1_height
            
            # Create windows
            header_win = curses.newwin(header_height, max_x, 0, 0)
            
            # First row of panels
            account_win = curses.newwin(panel1_height, panel_width, header_height, 0)
            performance_win = curses.newwin(panel1_height, panel_width, header_height, panel_width)
            scanner_win = curses.newwin(panel1_height, max_x - 2 * panel_width, header_height, 2 * panel_width)
            
            # Second row of panels
            positions_win = curses.newwin(panel2_height, panel_width, header_height + panel1_height, 0)
            trades_win = curses.newwin(panel2_height, panel_width, header_height + panel1_height, panel_width)
            
            # Split last column into two
            risk_height = panel2_height // 2
            system_height = panel2_height - risk_height
            
            risk_win = curses.newwin(risk_height, max_x - 2 * panel_width, 
                                    header_height + panel1_height, 2 * panel_width)
            system_win = curses.newwin(system_height, max_x - 2 * panel_width, 
                                     header_height + panel1_height + risk_height, 2 * panel_width)
            
            # Draw all panels
            self.draw_header_pane(header_win)
            self.draw_account_overview(account_win)
            self.draw_performance_dashboard(performance_win)
            self.draw_market_scanner(scanner_win)
            self.draw_active_positions(positions_win)
            self.draw_recent_trades(trades_win)
            self.draw_risk_monitor(risk_win)
            self.draw_system_status(system_win)
            
            # Refresh all windows
            header_win.refresh()
            account_win.refresh()
            performance_win.refresh()
            scanner_win.refresh()
            positions_win.refresh()
            trades_win.refresh()
            risk_win.refresh()
            system_win.refresh()
            
            # Update animation frame
            self.animation_frame = (self.animation_frame + 1) % 60
            
        except curses.error as e:
            pass
        except Exception as e:
            self.logs.append(f"ERROR: Display update failed: {str(e)}")

    def run(self):
        """Main TUI loop"""
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
            
            # Add some sample data for testing
            self.add_sample_data()
            
            while self.running:
                # Check for key press
                try:
                    key = self.stdscr.getch()
                    if key == ord('q') or key == ord('Q'):
                        self.running = False
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.stdscr.clear()
                    elif key == ord('p') or key == ord('P'):
                        # Toggle pause
                        pass
                    elif key == ord('h') or key == ord('H'):
                        # Show help
                        pass
                except curses.error:
                    pass
                
                self.update_display()
                time.sleep(0.5)  # Update every 500ms for smooth display
                
        except Exception as e:
            print(f"TUI Error: {e}")
        finally:
            if self.stdscr:
                curses.endwin()

    def add_sample_data(self):
        """Add sample data for testing the display"""
        import random
        
        # Add sample scores
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
                  "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT"]
        
        for symbol in symbols:
            score = ScoreEntry(
                symbol=symbol,
                long_score=random.randint(60, 100),
                short_score=random.randint(60, 100),
                price=random.uniform(0.5, 50000),
                long_signal=random.choice([True, False]),
                short_signal=random.choice([True, False]),
                timestamp=datetime.now()
            )
            self.scores.append(score)
            
        # Add sample positions
        for i in range(3):
            pos = PositionEntry(
                symbol=random.choice(symbols),
                side=random.choice(["LONG", "SHORT"]),
                quantity=random.uniform(0.001, 1.0),
                entry_price=random.uniform(100, 50000),
                current_price=random.uniform(100, 50000),
                pnl=random.uniform(-100, 200),
                pnl_percent=random.uniform(-5, 10)
            )
            self.positions.append(pos)
            
        # Add sample trades
        for i in range(5):
            trade = TradeEntry(
                symbol=random.choice(symbols),
                side=random.choice(["LONG", "SHORT"]),
                entry_price=random.uniform(100, 50000),
                quantity=random.uniform(0.001, 1.0),
                timestamp=datetime.now()
            )
            self.trades.append(trade)
            
        # Add sample logs
        log_messages = [
            "INFO: System initialized successfully",
            "SUCCESS: Connected to Phemex API",
            "INFO: Scanning 100 symbols",
            "WARNING: High volatility detected on BTC/USDT",
            "SUCCESS: Trade executed - LONG ETH/USDT @ $3,245.50",
            "INFO: Position closed with +2.5% profit",
            "ERROR: Failed to place order - insufficient margin",
            "INFO: Risk management triggered - pausing new trades",
        ]
        
        for msg in log_messages:
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            
        # Update stats
        self.stats.update({
            "scanned": 100,
            "signals": 8,
            "trades": 25,
            "winning_trades": 15,
            "losing_trades": 10,
            "total_trades": 25,
            "open_positions": len(self.positions),
            "best_trade": 450.25,
            "worst_trade": -125.50,
            "current_streak": 3,
            "live_mode": False,
            "min_score": 85,
            "leverage": 25,
            "risk_pct": 0.5
        })

    def stop(self):
        """Stop the TUI"""
        self.running = False


if __name__ == "__main__":
    # Test the improved TUI
    tui = ImprovedTradingTUI()
    try:
        tui.run()
    except KeyboardInterrupt:
        tui.stop()
    finally:
        print("\nTUI stopped successfully")