#!/usr/bin/env python3
"""
Demo script to showcase the improved trading bot display
"""

from improved_tui import ImprovedTradingTUI
import threading
import time
import random
from datetime import datetime

def simulate_trading_data(tui):
    """Simulate live trading data updates"""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    
    while tui.running:
        # Update market trend
        if random.random() > 0.9:
            tui.market_trend = random.choice(["bullish", "bearish", "neutral"])
        
        # Update account stats
        tui.stats['equity'] = tui.stats['account_balance'] + random.uniform(-500, 1000)
        tui.stats['margin_used'] = random.uniform(0, tui.stats['equity'] * 0.3)
        tui.stats['daily_pnl'] = random.uniform(-200, 400)
        tui.stats['pnl'] += random.uniform(-10, 20)
        
        # Add new log entry
        if random.random() > 0.7:
            log_types = [
                f"INFO: Scanning {random.choice(symbols)}",
                f"SUCCESS: Signal detected on {random.choice(symbols)}",
                f"WARNING: High volatility on {random.choice(symbols)}",
                f"INFO: Position updated for {random.choice(symbols)}",
            ]
            tui.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {random.choice(log_types)}")
        
        # Update positions P&L
        for pos in tui.positions:
            pos.current_price = pos.entry_price * random.uniform(0.98, 1.02)
            if pos.side == "LONG":
                pos.pnl = (pos.current_price - pos.entry_price) * pos.quantity
            else:
                pos.pnl = (pos.entry_price - pos.current_price) * pos.quantity
            pos.pnl_percent = (pos.pnl / (pos.entry_price * pos.quantity)) * 100
        
        # Update scores
        if random.random() > 0.8:
            from improved_tui import ScoreEntry
            score = ScoreEntry(
                symbol=random.choice(symbols),
                long_score=random.randint(70, 100),
                short_score=random.randint(70, 100),
                price=random.uniform(100, 50000),
                long_signal=random.random() > 0.7,
                short_signal=random.random() > 0.7,
                timestamp=datetime.now()
            )
            tui.scores.append(score)
            tui.stats['scanned'] += 1
            if score.long_signal or score.short_signal:
                tui.stats['signals'] += 1
        
        time.sleep(2)

def main():
    print("Starting Improved Trading Bot Display Demo...")
    print("Press 'Q' to quit")
    print("-" * 50)
    
    tui = ImprovedTradingTUI()
    
    # Start data simulation thread
    sim_thread = threading.Thread(target=simulate_trading_data, args=(tui,))
    sim_thread.daemon = True
    sim_thread.start()
    
    try:
        # Run the TUI
        tui.run()
    except KeyboardInterrupt:
        tui.stop()
    finally:
        print("\nDemo stopped successfully")
        print("The improved display features:")
        print("✅ Professional header with status indicators")
        print("✅ Color-coded metrics (profit/loss)")
        print("✅ Progress bars for risk monitoring")
        print("✅ Organized layout with clear sections")
        print("✅ Real-time updates with smooth refresh")
        print("✅ Unicode borders and symbols")
        print("✅ Keyboard shortcuts for control")
        print("✅ Responsive design that adapts to terminal size")

if __name__ == "__main__":
    main()