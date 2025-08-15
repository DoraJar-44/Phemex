"""Terminal User Interface for trading bot with 3x3 grid layout."""
import os
import sys
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import curses
from datetime import datetime


@dataclass
class ScoreEntry:
    """Individual score entry for display."""
    symbol: str
    price: float
    long_score: int
    short_score: int
    long_signal: bool
    short_signal: bool
    timestamp: datetime = field(default_factory=datetime.now)
    validation_status: str = "OK"


@dataclass
class GridPane:
    """Individual pane in the 3x3 grid."""
    title: str
    content: List[str] = field(default_factory=list)
    highlight: bool = False
    border_color: int = curses.COLOR_WHITE
    max_lines: int = 10


class TradingTUI:
    """3x3 Grid Terminal User Interface for trading bot."""
    
    def __init__(self):
        self.stdscr = None
        self.running = False
        self.scores_history: deque = deque(maxlen=1000)
        self.high_scores: deque = deque(maxlen=50)
        self.validation_errors: deque = deque(maxlen=20)
        self.live_feed: deque = deque(maxlen=100)
        
        # 3x3 Grid layout
        self.panes: Dict[Tuple[int, int], GridPane] = {
            (0, 0): GridPane("🎯 HIGH SCORES", highlight=True),
            (0, 1): GridPane("📊 LIVE FEED"),
            (0, 2): GridPane("⚡ SIGNALS"),
            (1, 0): GridPane("📈 LONG ANALYSIS"),
            (1, 1): GridPane("🎛️ SYSTEM STATUS"),
            (1, 2): GridPane("📉 SHORT ANALYSIS"),
            (2, 0): GridPane("⚠️ VALIDATION"),
            (2, 1): GridPane("💰 TRADES"),
            (2, 2): GridPane("📋 STATS"),
        }
        
        # Color scheme
        self.colors = {}
        self.score_threshold_high = 90
        self.score_threshold_medium = 75
        
    def init_colors(self):
        """Initialize color pairs for the TUI."""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)    # High scores
        curses.init_pair(2, curses.COLOR_YELLOW, -1)   # Medium scores
        curses.init_pair(3, curses.COLOR_RED, -1)      # Low scores / errors
        curses.init_pair(4, curses.COLOR_CYAN, -1)     # Headers / highlights
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Signals
        curses.init_pair(6, curses.COLOR_WHITE, -1)    # Normal text
        curses.init_pair(7, curses.COLOR_BLUE, -1)     # Info
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Highlight background
        
        self.colors = {
            'high': curses.color_pair(1) | curses.A_BOLD,
            'medium': curses.color_pair(2),
            'low': curses.color_pair(3),
            'header': curses.color_pair(4) | curses.A_BOLD,
            'signal': curses.color_pair(5) | curses.A_BOLD,
            'normal': curses.color_pair(6),
            'info': curses.color_pair(7),
            'highlight_bg': curses.color_pair(8),
        }
        
    def get_score_color(self, score: int) -> int:
        """Get color attribute based on score value."""
        if score >= self.score_threshold_high:
            return self.colors['high']
        elif score >= self.score_threshold_medium:
            return self.colors['medium']
        else:
            return self.colors['low']
            
    def add_score_entry(self, entry: ScoreEntry):
        """Add a new score entry to the system."""
        self.scores_history.append(entry)
        
        # Track high scores
        max_score = max(entry.long_score, entry.short_score)
        if max_score >= self.score_threshold_high:
            self.high_scores.append(entry)
            
        # Update live feed
        signal_text = ""
        if entry.long_signal:
            signal_text = "🔥 LONG"
        elif entry.short_signal:
            signal_text = "🔥 SHORT"
        else:
            signal_text = "⏸️"
            
        feed_line = f"{entry.timestamp.strftime('%H:%M:%S')} {entry.symbol:>12} ${entry.price:>8.4f} L:{entry.long_score:>3} S:{entry.short_score:>3} {signal_text}"
        self.live_feed.append(feed_line)
        
    def add_validation_error(self, error: str):
        """Add a validation error to display."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.validation_errors.append(f"{timestamp} ❌ {error}")
        
    def calculate_pane_dimensions(self, height: int, width: int) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
        """Calculate dimensions for each pane in the 3x3 grid."""
        pane_height = height // 3
        pane_width = width // 3
        
        dimensions = {}
        for row in range(3):
            for col in range(3):
                y = row * pane_height
                x = col * pane_width
                h = pane_height - 1  # Leave space for borders
                w = pane_width - 1
                
                # Adjust last row/column to fill remaining space
                if row == 2:
                    h = height - y - 1
                if col == 2:
                    w = width - x - 1
                    
                dimensions[(row, col)] = (y, x, h, w)
                
        return dimensions
        
    def draw_pane(self, row: int, col: int, y: int, x: int, height: int, width: int):
        """Draw an individual pane with its content."""
        pane = self.panes[(row, col)]
        
        # Draw border
        border_attr = self.colors['header'] if pane.highlight else self.colors['normal']
        
        try:
            # Top border
            self.stdscr.addstr(y, x, "┌" + "─" * (width - 2) + "┐", border_attr)
            
            # Title in top border
            title_text = f" {pane.title} "
            title_x = x + (width - len(title_text)) // 2
            if title_x > x and title_x + len(title_text) < x + width:
                self.stdscr.addstr(y, title_x, title_text, border_attr)
                
            # Side borders and content area
            for i in range(1, height - 1):
                if y + i < curses.LINES - 1:
                    self.stdscr.addstr(y + i, x, "│", border_attr)
                    if x + width - 1 < curses.COLS:
                        self.stdscr.addstr(y + i, x + width - 1, "│", border_attr)
                        
            # Bottom border
            if y + height - 1 < curses.LINES:
                self.stdscr.addstr(y + height - 1, x, "└" + "─" * (width - 2) + "┘", border_attr)
                
        except curses.error:
            pass  # Ignore drawing errors at screen edges
            
        # Draw content
        content_start_y = y + 1
        content_start_x = x + 1
        content_width = width - 2
        content_height = height - 2
        
        self.draw_pane_content(row, col, content_start_y, content_start_x, content_width, content_height)
        
    def draw_pane_content(self, row: int, col: int, y: int, x: int, width: int, height: int):
        """Draw content for specific pane based on its purpose."""
        pane = self.panes[(row, col)]
        
        try:
            if (row, col) == (0, 0):  # High Scores
                self.draw_high_scores(y, x, width, height)
            elif (row, col) == (0, 1):  # Live Feed
                self.draw_live_feed(y, x, width, height)
            elif (row, col) == (0, 2):  # Signals
                self.draw_signals(y, x, width, height)
            elif (row, col) == (1, 0):  # Long Analysis
                self.draw_long_analysis(y, x, width, height)
            elif (row, col) == (1, 1):  # System Status
                self.draw_system_status(y, x, width, height)
            elif (row, col) == (1, 2):  # Short Analysis
                self.draw_short_analysis(y, x, width, height)
            elif (row, col) == (2, 0):  # Validation
                self.draw_validation(y, x, width, height)
            elif (row, col) == (2, 1):  # Trades
                self.draw_trades(y, x, width, height)
            elif (row, col) == (2, 2):  # Stats
                self.draw_stats(y, x, width, height)
        except curses.error:
            pass
            
    def draw_high_scores(self, y: int, x: int, width: int, height: int):
        """Draw high scores pane - HIGHLIGHTED."""
        line = 0
        
        if not self.high_scores:
            self.stdscr.addstr(y + line, x, "No high scores yet...", self.colors['info'])
            return
            
        # Sort by highest score
        sorted_scores = sorted(self.high_scores, 
                             key=lambda e: max(e.long_score, e.short_score), 
                             reverse=True)
                             
        for entry in sorted_scores[:height]:
            if line >= height:
                break
                
            max_score = max(entry.long_score, entry.short_score)
            side = "L" if entry.long_score > entry.short_score else "S"
            
            # Truncate symbol if needed
            symbol_display = entry.symbol[:8] if len(entry.symbol) > 8 else entry.symbol
            
            score_text = f"{symbol_display:>8} {side}:{max_score:>3}"
            color = self.get_score_color(max_score)
            
            if entry.long_signal or entry.short_signal:
                color |= curses.A_REVERSE  # Highlight active signals
                
            self.stdscr.addstr(y + line, x, score_text[:width], color)
            line += 1
            
    def draw_live_feed(self, y: int, x: int, width: int, height: int):
        """Draw live feed of recent activity."""
        line = 0
        
        # Show most recent entries first
        recent_feed = list(self.live_feed)[-height:]
        
        for feed_line in recent_feed:
            if line >= height:
                break
                
            # Truncate line to fit width
            display_line = feed_line[:width] if len(feed_line) > width else feed_line
            
            # Color based on content
            color = self.colors['normal']
            if "🔥" in feed_line:
                color = self.colors['signal']
            elif any(score in feed_line for score in ["L:9", "S:9"]):  # High scores
                color = self.colors['high']
                
            self.stdscr.addstr(y + line, x, display_line, color)
            line += 1
            
    def draw_signals(self, y: int, x: int, width: int, height: int):
        """Draw active trading signals."""
        line = 0
        
        # Find recent signals
        active_signals = [entry for entry in list(self.scores_history)[-50:] 
                         if entry.long_signal or entry.short_signal]
                         
        if not active_signals:
            self.stdscr.addstr(y + line, x, "No active signals", self.colors['info'])
            return
            
        for entry in active_signals[-height:]:
            if line >= height:
                break
                
            signal_type = "LONG" if entry.long_signal else "SHORT"
            score = entry.long_score if entry.long_signal else entry.short_score
            
            signal_text = f"{entry.symbol[:6]:>6} {signal_type} {score:>3}"
            
            self.stdscr.addstr(y + line, x, signal_text[:width], self.colors['signal'])
            line += 1
            
    def draw_system_status(self, y: int, x: int, width: int, height: int):
        """Draw system status information."""
        line = 0
        
        status_info = [
            f"Uptime: {time.time() - getattr(self, 'start_time', time.time()):.0f}s",
            f"Symbols: {len(set(e.symbol for e in self.scores_history)) if self.scores_history else 0}",
            f"Scores: {len(self.scores_history)}",
            f"High: {len(self.high_scores)}",
            f"Errors: {len(self.validation_errors)}",
            f"Feed: {len(self.live_feed)}",
        ]
        
        for info in status_info:
            if line >= height:
                break
            self.stdscr.addstr(y + line, x, info[:width], self.colors['info'])
            line += 1
            
    def draw_validation(self, y: int, x: int, width: int, height: int):
        """Draw validation errors and warnings."""
        line = 0
        
        if not self.validation_errors:
            self.stdscr.addstr(y + line, x, "All validations OK ✅", self.colors['high'])
            return
            
        recent_errors = list(self.validation_errors)[-height:]
        for error in recent_errors:
            if line >= height:
                break
            self.stdscr.addstr(y + line, x, error[:width], self.colors['low'])
            line += 1
            
    def draw_long_analysis(self, y: int, x: int, width: int, height: int):
        """Draw long position analysis."""
        if not self.scores_history:
            return
            
        recent = list(self.scores_history)[-20:]
        long_scores = [e.long_score for e in recent]
        
        if long_scores:
            avg_long = sum(long_scores) / len(long_scores)
            max_long = max(long_scores)
            
            line = 0
            self.stdscr.addstr(y + line, x, f"Avg: {avg_long:.1f}", self.colors['info'])
            line += 1
            self.stdscr.addstr(y + line, x, f"Max: {max_long}", self.get_score_color(max_long))
            
    def draw_short_analysis(self, y: int, x: int, width: int, height: int):
        """Draw short position analysis."""
        if not self.scores_history:
            return
            
        recent = list(self.scores_history)[-20:]
        short_scores = [e.short_score for e in recent]
        
        if short_scores:
            avg_short = sum(short_scores) / len(short_scores)
            max_short = max(short_scores)
            
            line = 0
            self.stdscr.addstr(y + line, x, f"Avg: {avg_short:.1f}", self.colors['info'])
            line += 1
            self.stdscr.addstr(y + line, x, f"Max: {max_short}", self.get_score_color(max_short))
            
    def draw_trades(self, y: int, x: int, width: int, height: int):
        """Draw recent trades information."""
        # Placeholder for trade execution data
        line = 0
        self.stdscr.addstr(y + line, x, "Trade exec:", self.colors['info'])
        line += 1
        self.stdscr.addstr(y + line, x, "Dry run mode", self.colors['normal'])
        
    def draw_stats(self, y: int, x: int, width: int, height: int):
        """Draw statistics summary."""
        if not self.scores_history:
            return
            
        total_scores = len(self.scores_history)
        high_count = len(self.high_scores)
        success_rate = (high_count / total_scores * 100) if total_scores > 0 else 0
        
        line = 0
        self.stdscr.addstr(y + line, x, f"Total: {total_scores}", self.colors['info'])
        line += 1
        self.stdscr.addstr(y + line, x, f"High: {high_count}", self.colors['high'])
        line += 1
        if line < height:
            self.stdscr.addstr(y + line, x, f"Rate: {success_rate:.1f}%", 
                             self.colors['high'] if success_rate > 20 else self.colors['normal'])
            
    def refresh_display(self):
        """Refresh the entire display."""
        if not self.stdscr:
            return
            
        try:
            self.stdscr.clear()
            height, width = self.stdscr.getmaxyx()
            
            # Calculate pane dimensions
            dimensions = self.calculate_pane_dimensions(height, width)
            
            # Draw each pane
            for (row, col), (y, x, h, w) in dimensions.items():
                self.draw_pane(row, col, y, x, h, w)
                
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                self.stdscr.addstr(height - 1, 0, f"Phemex Trading Bot - {timestamp}", self.colors['header'])
            except curses.error:
                pass
                
            self.stdscr.refresh()
            
        except curses.error:
            pass  # Ignore refresh errors
            
    def run(self):
        """Run the TUI main loop."""
        self.start_time = time.time()
        
        try:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)  # Non-blocking input
            curses.curs_set(0)  # Hide cursor
            
            self.init_colors()
            self.running = True
            
            while self.running:
                self.refresh_display()
                
                # Check for quit key
                try:
                    key = self.stdscr.getch()
                    if key == ord('q') or key == ord('Q'):
                        self.running = False
                except curses.error:
                    pass
                    
                time.sleep(0.1)  # 10 FPS
                
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up curses interface."""
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
            
    def stop(self):
        """Stop the TUI."""
        self.running = False


# Global TUI instance
_tui_instance: Optional[TradingTUI] = None
_tui_thread: Optional[threading.Thread] = None


def get_tui_instance() -> TradingTUI:
    """Get or create the global TUI instance."""
    global _tui_instance
    if _tui_instance is None:
        _tui_instance = TradingTUI()
    return _tui_instance


def start_tui():
    """Start the TUI in a separate thread."""
    global _tui_thread
    if _tui_thread is None or not _tui_thread.is_alive():
        tui = get_tui_instance()
        _tui_thread = threading.Thread(target=tui.run, daemon=True)
        _tui_thread.start()


def stop_tui():
    """Stop the TUI."""
    global _tui_instance
    if _tui_instance:
        _tui_instance.stop()


def add_score_to_tui(symbol: str, price: float, long_score: int, short_score: int, 
                    long_signal: bool, short_signal: bool, validation_status: str = "OK"):
    """Add a score entry to the TUI."""
    tui = get_tui_instance()
    entry = ScoreEntry(
        symbol=symbol,
        price=price,
        long_score=long_score,
        short_score=short_score,
        long_signal=long_signal,
        short_signal=short_signal,
        validation_status=validation_status
    )
    tui.add_score_entry(entry)


def add_validation_error_to_tui(error: str):
    """Add a validation error to the TUI."""
    tui = get_tui_instance()
    tui.add_validation_error(error)
