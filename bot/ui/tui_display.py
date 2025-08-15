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
        
        # Detect terminal environment
        self.terminal_env = self._detect_terminal_environment()
        
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
        
    def _detect_terminal_environment(self) -> Dict[str, Any]:
        """Detect terminal capabilities and environment."""
        env_info = {
            'is_windows': sys.platform.startswith('win'),
            'is_powershell': False,
            'is_cmd': False,
            'is_wsl': False,
            'term_program': os.environ.get('TERM_PROGRAM', ''),
            'term': os.environ.get('TERM', ''),
            'colorterm': os.environ.get('COLORTERM', ''),
            'supports_256_color': False,
            'supports_truecolor': False,
        }
        
        # Detect Windows terminal types
        if env_info['is_windows']:
            if 'PowerShell' in os.environ.get('PSModulePath', ''):
                env_info['is_powershell'] = True
            elif os.environ.get('ComSpec', '').endswith('cmd.exe'):
                env_info['is_cmd'] = True
        
        # Check for WSL
        if 'microsoft' in os.environ.get('WSL_DISTRO_NAME', '').lower():
            env_info['is_wsl'] = True
            
        # Check color support
        term = env_info['term'].lower()
        colorterm = env_info['colorterm'].lower()
        
        # Common 256-color terminal indicators
        if any(indicator in term for indicator in ['256', 'xterm-256']):
            env_info['supports_256_color'] = True
        if any(indicator in colorterm for indicator in ['256', 'truecolor', '24bit']):
            env_info['supports_256_color'] = True
            env_info['supports_truecolor'] = True
            
        # Windows Terminal and modern terminals
        if env_info['term_program'] in ['vscode', 'cursor']:
            env_info['supports_256_color'] = True
            env_info['supports_truecolor'] = True
            
        # Check for Windows Terminal specifically
        if env_info['is_windows']:
            wt_session = os.environ.get('WT_SESSION')
            if wt_session:
                env_info['supports_256_color'] = True
                env_info['supports_truecolor'] = True
                
        # Force enable colors in known good environments
        if any(prog in env_info['term_program'].lower() for prog in ['cursor', 'vscode', 'terminal']):
            env_info['supports_256_color'] = True
            
        return env_info
        
    def init_colors(self):
        """Initialize color pairs for the TUI with enhanced terminal compatibility."""
        try:
            curses.start_color()
            
            # Enable all available color features
            has_colors = curses.has_colors()
            if not has_colors:
                # Fallback for terminals without color support
                self.colors = {
                    'high': curses.A_BOLD | curses.A_REVERSE,
                    'medium': curses.A_BOLD,
                    'low': curses.A_DIM,
                    'header': curses.A_BOLD | curses.A_UNDERLINE,
                    'signal': curses.A_BOLD | curses.A_BLINK,
                    'normal': curses.A_NORMAL,
                    'info': curses.A_DIM,
                    'highlight_bg': curses.A_REVERSE,
                }
                return
            
            # Use default colors to respect terminal theme
            curses.use_default_colors()
            
            # Get terminal capabilities
            color_count = curses.COLORS
            can_change = hasattr(curses, 'can_change_color') and curses.can_change_color()
            
            # Initialize color pairs based on terminal capabilities
            if color_count >= 256 and (self.terminal_env['supports_256_color'] or color_count >= 256):
                # 256-color terminal support
                try:
                    # High contrast colors optimized for different terminal environments
                    if self.terminal_env['is_windows'] and self.terminal_env['is_powershell']:
                        # Windows PowerShell optimized colors
                        curses.init_pair(1, 10, -1)     # Bright green
                        curses.init_pair(2, 11, -1)     # Bright yellow
                        curses.init_pair(3, 9, -1)      # Bright red
                        curses.init_pair(4, 14, -1)     # Bright cyan
                        curses.init_pair(5, 13, -1)     # Bright magenta
                        curses.init_pair(6, 15, -1)     # Bright white
                        curses.init_pair(7, 12, -1)     # Bright blue
                        curses.init_pair(8, 0, 10)      # Black on bright green
                        curses.init_pair(9, 11, -1)     # Bright yellow
                        curses.init_pair(10, 9, -1)     # Bright red
                    else:
                        # Standard 256-color palette
                        curses.init_pair(1, 46, -1)     # Bright green on default background
                        curses.init_pair(2, 214, -1)    # Orange on default background  
                        curses.init_pair(3, 196, -1)    # Bright red on default background
                        curses.init_pair(4, 51, -1)     # Bright cyan on default background
                        curses.init_pair(5, 129, -1)    # Purple on default background
                        curses.init_pair(6, 15, -1)     # Bright white on default background
                        curses.init_pair(7, 33, -1)     # Blue on default background
                        curses.init_pair(8, 0, 46)      # Black on bright green for highlights
                        curses.init_pair(9, 226, -1)    # Bright yellow on default background
                        curses.init_pair(10, 208, -1)   # Dark orange on default background
                except curses.error:
                    # Fallback to 8-color if 256-color fails
                    self._init_8_colors()
            elif color_count >= 16:
                # 16-color terminal support
                try:
                    curses.init_pair(1, curses.COLOR_GREEN, -1)    # Green on default
                    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # Yellow on default
                    curses.init_pair(3, curses.COLOR_RED, -1)      # Red on default
                    curses.init_pair(4, curses.COLOR_CYAN, -1)     # Cyan on default
                    curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Magenta on default
                    curses.init_pair(6, curses.COLOR_WHITE, -1)    # White on default
                    curses.init_pair(7, curses.COLOR_BLUE, -1)     # Blue on default
                    curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Highlight
                    curses.init_pair(9, curses.COLOR_YELLOW, -1)   # Bright yellow
                    curses.init_pair(10, curses.COLOR_RED, -1)     # Alternative red
                except curses.error:
                    self._init_8_colors()
            else:
                # 8-color terminal support
                self._init_8_colors()
                
            # Create color mappings with enhanced attributes
            self.colors = {
                'high': curses.color_pair(1) | curses.A_BOLD,      # Bright green + bold
                'medium': curses.color_pair(2) | curses.A_BOLD,    # Orange/Yellow + bold
                'low': curses.color_pair(3),                       # Red
                'header': curses.color_pair(4) | curses.A_BOLD,    # Cyan + bold
                'signal': curses.color_pair(5) | curses.A_BOLD,    # Purple/Magenta + bold
                'normal': curses.color_pair(6),                    # White/Normal
                'info': curses.color_pair(7),                      # Blue
                'highlight_bg': curses.color_pair(8),              # Reverse for highlights
                'warning': curses.color_pair(9),                   # Bright yellow
                'error': curses.color_pair(10) | curses.A_BOLD,    # Bright red + bold
            }
            
        except curses.error as e:
            # Ultimate fallback - no colors, just attributes
            self.colors = {
                'high': curses.A_BOLD | curses.A_REVERSE,
                'medium': curses.A_BOLD,
                'low': curses.A_DIM,
                'header': curses.A_BOLD | curses.A_UNDERLINE,
                'signal': curses.A_BOLD | curses.A_BLINK,
                'normal': curses.A_NORMAL,
                'info': curses.A_DIM,
                'highlight_bg': curses.A_REVERSE,
                'warning': curses.A_BOLD,
                'error': curses.A_BOLD | curses.A_REVERSE,
            }
    
    def _init_8_colors(self):
        """Initialize basic 8-color pairs as fallback."""
        try:
            curses.init_pair(1, curses.COLOR_GREEN, -1)     # Green
            curses.init_pair(2, curses.COLOR_YELLOW, -1)    # Yellow  
            curses.init_pair(3, curses.COLOR_RED, -1)       # Red
            curses.init_pair(4, curses.COLOR_CYAN, -1)      # Cyan
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)   # Magenta
            curses.init_pair(6, curses.COLOR_WHITE, -1)     # White
            curses.init_pair(7, curses.COLOR_BLUE, -1)      # Blue
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Highlight
            curses.init_pair(9, curses.COLOR_YELLOW, -1)    # Bright yellow
            curses.init_pair(10, curses.COLOR_RED, -1)      # Alternative red
        except curses.error:
            # If even basic colors fail, we'll use the attribute-only fallback
            pass
    
    def test_terminal_capabilities(self) -> Dict[str, Any]:
        """Test actual terminal capabilities after curses initialization."""
        capabilities = {
            'has_colors': False,
            'color_count': 0,
            'can_change_color': False,
            'pairs_available': 0,
        }
        
        try:
            capabilities['has_colors'] = curses.has_colors()
            capabilities['color_count'] = curses.COLORS
            capabilities['can_change_color'] = hasattr(curses, 'can_change_color') and curses.can_change_color()
            capabilities['pairs_available'] = curses.COLOR_PAIRS
        except (AttributeError, curses.error):
            pass
            
        return capabilities
        
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
        """Calculate responsive dimensions for each pane in the 3x3 grid."""
        # Ensure minimum dimensions
        min_height, min_width = 24, 80
        if height < min_height or width < min_width:
            # For small screens, use single column layout
            return self.calculate_mobile_dimensions(height, width)
        
        # Reserve space for bottom status bar
        usable_height = height - 1
        
        # Calculate responsive pane sizes
        pane_height = usable_height // 3
        pane_width = width // 3
        
        dimensions = {}
        for row in range(3):
            for col in range(3):
                y = row * pane_height
                x = col * pane_width
                h = pane_height
                w = pane_width
                
                # Adjust last row/column to fill remaining space perfectly
                if row == 2:
                    h = usable_height - y
                if col == 2:
                    w = width - x
                    
                dimensions[(row, col)] = (y, x, h, w)
                
        return dimensions
    
    def calculate_mobile_dimensions(self, height: int, width: int) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
        """Calculate dimensions for mobile/small screen layout."""
        usable_height = height - 1
        pane_height = usable_height // 6  # Stack 6 panes vertically
        
        # Priority panes for mobile view
        mobile_panes = [(0, 0), (0, 1), (1, 1), (0, 2), (2, 0), (2, 2)]
        dimensions = {}
        
        for i, (row, col) in enumerate(mobile_panes):
            y = i * pane_height
            h = pane_height if i < 5 else usable_height - y
            dimensions[(row, col)] = (y, 0, h, width)
        
        # Hide other panes
        for row in range(3):
            for col in range(3):
                if (row, col) not in mobile_panes:
                    dimensions[(row, col)] = (0, 0, 0, 0)  # Hidden
        
        return dimensions
        
    def draw_pane(self, row: int, col: int, y: int, x: int, height: int, width: int):
        """Draw an individual pane with its content - fully responsive."""
        # Skip drawing if pane is hidden (mobile mode)
        if height <= 0 or width <= 0:
            return
            
        pane = self.panes[(row, col)]
        
        # Ensure we don't draw outside screen boundaries
        max_y, max_x = self.stdscr.getmaxyx()
        if y >= max_y or x >= max_x:
            return
            
        # Adjust dimensions to fit screen
        height = min(height, max_y - y)
        width = min(width, max_x - x)
        
        if height < 3 or width < 3:  # Too small to be useful
            return
        
        # Draw border
        border_attr = self.colors['header'] if pane.highlight else self.colors['normal']
        
        try:
            # Top border
            if width >= 2:
                border_line = "┌" + "─" * max(0, width - 2) + ("┐" if width > 1 else "")
                self.stdscr.addstr(y, x, border_line[:width], border_attr)
            
            # Title in top border - responsive truncation
            title_text = f" {pane.title} "
            if len(title_text) > width - 4:
                # Truncate title for narrow screens
                title_text = f" {pane.title[:width-7]}... "
            
            title_x = x + max(1, (width - len(title_text)) // 2)
            if title_x > x and title_x + len(title_text) <= x + width:
                self.stdscr.addstr(y, title_x, title_text[:width-title_x+x], border_attr)
                
            # Side borders and content area
            for i in range(1, height - 1):
                if y + i < max_y:
                    self.stdscr.addstr(y + i, x, "│", border_attr)
                    if x + width - 1 < max_x and width > 1:
                        self.stdscr.addstr(y + i, x + width - 1, "│", border_attr)
                        
            # Bottom border
            if y + height - 1 < max_y and height > 1:
                border_line = "└" + "─" * max(0, width - 2) + ("┘" if width > 1 else "")
                self.stdscr.addstr(y + height - 1, x, border_line[:width], border_attr)
                
        except curses.error:
            pass  # Ignore drawing errors at screen edges
            
        # Draw content with responsive dimensions
        content_start_y = y + 1
        content_start_x = x + 1
        content_width = max(1, width - 2)
        content_height = max(1, height - 2)
        
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
        """Draw high scores pane - HIGHLIGHTED and responsive."""
        line = 0
        
        if not self.high_scores:
            msg = "No high scores yet..." if width > 20 else "No scores"
            self.stdscr.addstr(y + line, x, msg[:width], self.colors['info'])
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
            
            # Responsive symbol display
            if width < 15:
                # Ultra narrow: just score
                score_text = f"{side}:{max_score}"
            elif width < 25:
                # Narrow: short symbol + score
                symbol_display = entry.symbol[:4]
                score_text = f"{symbol_display} {side}:{max_score}"
            else:
                # Full display
                symbol_display = entry.symbol[:min(12, width-8)]
                score_text = f"{symbol_display:>12} {side}:{max_score:>3}"
            
            color = self.get_score_color(max_score)
            
            if entry.long_signal or entry.short_signal:
                color |= curses.A_REVERSE  # Highlight active signals
                
            self.stdscr.addstr(y + line, x, score_text[:width], color)
            line += 1
            
    def draw_live_feed(self, y: int, x: int, width: int, height: int):
        """Draw live feed of recent activity - responsive."""
        line = 0
        
        # Show most recent entries first
        recent_feed = list(self.live_feed)[-height:]
        
        for feed_line in recent_feed:
            if line >= height:
                break
            
            # Responsive line formatting
            if width < 30:
                # Ultra compact: just time + symbol + score
                parts = feed_line.split()
                if len(parts) >= 4:
                    time_part = parts[0]
                    symbol_part = parts[1].split('/')[0][:6]  # First part of symbol
                    score_parts = [p for p in parts if 'L:' in p or 'S:' in p]
                    if score_parts:
                        display_line = f"{time_part[-5:]} {symbol_part} {score_parts[0]}"
                    else:
                        display_line = f"{time_part[-5:]} {symbol_part}"
                else:
                    display_line = feed_line
            elif width < 50:
                # Medium: time + symbol + scores
                parts = feed_line.split()
                if len(parts) >= 6:
                    display_line = f"{parts[0][-8:]} {parts[1][:10]} {parts[4]} {parts[5]}"
                else:
                    display_line = feed_line
            else:
                # Full display
                display_line = feed_line
                
            # Truncate to fit width
            display_line = display_line[:width]
            
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
        """Draw system status information - responsive."""
        line = 0
        
        uptime = time.time() - getattr(self, 'start_time', time.time())
        symbols_count = len(set(e.symbol for e in self.scores_history)) if self.scores_history else 0
        
        if width < 20:
            # Ultra compact status
            status_info = [
                f"Up:{uptime:.0f}s",
                f"Sym:{symbols_count}",
                f"Scr:{len(self.scores_history)}",
                f"Hi:{len(self.high_scores)}",
            ]
        elif width < 35:
            # Compact status
            status_info = [
                f"Uptime: {uptime:.0f}s",
                f"Symbols: {symbols_count}",
                f"Scores: {len(self.scores_history)}",
                f"High: {len(self.high_scores)}",
                f"Errors: {len(self.validation_errors)}",
            ]
        else:
            # Full status with terminal info
            term_info = getattr(self, 'terminal_env', {})
            color_count = getattr(curses, 'COLORS', 0) if hasattr(curses, 'COLORS') else 0
            
            status_info = [
                f"Uptime: {uptime:.0f}s",
                f"Symbols: {symbols_count}",
                f"Scores: {len(self.scores_history)}",
                f"High: {len(self.high_scores)}",
                f"Errors: {len(self.validation_errors)}",
                f"Feed: {len(self.live_feed)}",
                f"Colors: {color_count}",
                f"Term: {term_info.get('term_program', 'unknown')}",
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
        """Refresh the entire display - 100% responsive and static."""
        if not self.stdscr:
            return
            
        try:
            # Get current terminal size
            height, width = self.stdscr.getmaxyx()
            
            # Respect terminal's default background
            self.stdscr.bkgd(' ', self.colors.get('normal', curses.A_NORMAL))
            
            # Clear screen efficiently
            self.stdscr.erase()
            
            # Calculate responsive pane dimensions
            dimensions = self.calculate_pane_dimensions(height, width)
            
            # Draw each visible pane
            for (row, col), (y, x, h, w) in dimensions.items():
                if h > 0 and w > 0:  # Only draw visible panes
                    self.draw_pane(row, col, y, x, h, w)
                
            # Add responsive bottom status bar
            self.draw_status_bar(height, width)
                
            self.stdscr.refresh()
            
        except (curses.error, ValueError):
            pass  # Ignore refresh errors and continue
    
    def draw_status_bar(self, height: int, width: int):
        """Draw responsive status bar at bottom."""
        if height < 2:
            return
            
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status_text = f"Phemex Trading Bot - {timestamp}"
            
            # Truncate for narrow screens
            if len(status_text) > width:
                timestamp_short = datetime.now().strftime('%H:%M:%S')
                status_text = f"Phemex Bot - {timestamp_short}"
                
            if len(status_text) > width:
                status_text = f"Bot - {timestamp_short}"
                
            # Center the status text
            x_pos = max(0, (width - len(status_text)) // 2)
            self.stdscr.addstr(height - 1, x_pos, status_text[:width], self.colors['header'])
            
        except curses.error:
            pass
            
    def run(self):
        """Run the TUI main loop - 100% responsive with 3s updates."""
        self.start_time = time.time()
        
        try:
            # Windows-specific terminal setup
            if self.terminal_env['is_windows']:
                # Set environment variables for better Windows color support
                os.environ.setdefault('TERM', 'xterm-256color')
                if self.terminal_env['is_powershell']:
                    os.environ.setdefault('COLORTERM', 'truecolor')
            
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)  # Non-blocking input
            curses.curs_set(0)  # Hide cursor
            
            # Initialize colors first
            self.init_colors()
            
            # Test terminal capabilities and log them
            capabilities = self.test_terminal_capabilities()
            
            # Set background to respect terminal theme
            self.stdscr.bkgd(' ', self.colors.get('normal', curses.A_NORMAL))
            self.stdscr.clear()
            
            # Enable resize detection
            if hasattr(curses, 'SIGWINCH'):
                import signal
                signal.signal(signal.SIGWINCH, self.handle_resize)
            
            self.running = True
            
            # Force initial display
            self.refresh_display()
            
            while self.running:
                # Check for quit key or resize
                try:
                    key = self.stdscr.getch()
                    if key == ord('q') or key == ord('Q'):
                        self.running = False
                    elif key == curses.KEY_RESIZE:
                        self.handle_resize_key()
                except curses.error:
                    pass
                
                # Real-time update every 3 seconds
                self.refresh_display()
                time.sleep(3.0)
                
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.cleanup()
    
    def handle_resize(self, signum=None, frame=None):
        """Handle terminal resize signal."""
        try:
            curses.endwin()
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            curses.curs_set(0)
            self.init_colors()
            self.refresh_display()
        except:
            pass
    
    def handle_resize_key(self):
        """Handle resize key event."""
        try:
            curses.update_lines_cols()
            self.refresh_display()
        except:
            pass
            
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
