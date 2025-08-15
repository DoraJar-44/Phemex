#!/usr/bin/env python3
"""
1-Day Trading Simulation
Parameters:
- 3% position size per trade
- Max 10 concurrent positions
- 344x leverage
- Based on the unified trading bot logic
"""

import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class SimulatedTrade:
    """Represents a single trade in the simulation"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    position_size: float  # in USD
    leverage: float
    tp1_price: float
    tp2_price: float
    sl_price: float
    exit_time: datetime = None
    exit_price: float = None
    exit_reason: str = None  # 'tp1', 'tp2', 'sl', 'manual'
    pnl_usd: float = 0.0
    pnl_percent: float = 0.0
    fees_usd: float = 0.0

class TradingSimulator:
    def __init__(self, initial_balance: float = 10000.0):
        """Initialize the trading simulator"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.leverage = 344
        self.position_size_pct = 3.0  # 3% per position
        self.max_positions = 10
        self.maker_fee = 0.01 / 100  # 0.01% maker fee (Phemex)
        self.taker_fee = 0.06 / 100  # 0.06% taker fee (Phemex)
        
        # Trading statistics
        self.trades: List[SimulatedTrade] = []
        self.open_positions: List[SimulatedTrade] = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Market simulation parameters
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT",
            "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
            "XLMUSDT", "NEARUSDT", "ALGOUSDT", "FTMUSDT", "VETUSDT"
        ]
        
        # Realistic win rates and R:R based on scoring system
        self.base_win_rate = 0.58  # 58% win rate with good scoring
        self.avg_winner_rr = 1.8   # Average R:R for winners
        self.avg_loser_rr = 1.0    # Average R:R for losers
        
    def generate_market_opportunity(self, hour: int) -> Dict[str, Any]:
        """Generate a trading opportunity based on market conditions"""
        # Market activity varies by hour (UTC)
        activity_multiplier = 1.0
        if 8 <= hour <= 16:  # European/US overlap
            activity_multiplier = 1.5
        elif 0 <= hour <= 8:  # Asian session
            activity_multiplier = 1.2
        else:
            activity_multiplier = 0.8
            
        # Generate opportunities
        opportunities = []
        base_opportunities = int(random.uniform(0, 3) * activity_multiplier)
        
        for _ in range(base_opportunities):
            symbol = random.choice(self.symbols)
            
            # Score distribution (85-100 for valid signals)
            score = random.uniform(85, 100)
            
            # Higher scores have better win probability
            win_probability = self.base_win_rate + (score - 85) * 0.02
            
            # Determine direction based on "market conditions"
            is_long = random.random() > 0.5
            
            opportunities.append({
                'symbol': symbol,
                'score': score,
                'is_long': is_long,
                'win_probability': min(win_probability, 0.75)  # Cap at 75%
            })
            
        return opportunities
    
    def calculate_position_size(self) -> float:
        """Calculate position size based on current balance"""
        return self.current_balance * (self.position_size_pct / 100)
    
    def simulate_trade_outcome(self, trade: SimulatedTrade, win_probability: float) -> SimulatedTrade:
        """Simulate the outcome of a trade"""
        # Determine if trade wins
        is_winner = random.random() < win_probability
        
        # Calculate fees (entry is market order - taker fee)
        entry_notional = trade.position_size * trade.leverage
        entry_fee = entry_notional * self.taker_fee
        
        if is_winner:
            # Determine which TP is hit
            tp_level = random.choices([1, 2], weights=[0.6, 0.4])[0]
            
            if tp_level == 1:
                exit_price = trade.tp1_price
                exit_reason = "tp1"
                # Only 50% of position exits at TP1
                exit_notional = (entry_notional * 0.5)
            else:
                exit_price = trade.tp2_price
                exit_reason = "tp2"
                exit_notional = entry_notional
                
            # Calculate PnL
            if trade.side == "long":
                price_change_pct = (exit_price - trade.entry_price) / trade.entry_price
            else:  # short
                price_change_pct = (trade.entry_price - exit_price) / trade.entry_price
                
        else:
            # Stop loss hit
            exit_price = trade.sl_price
            exit_reason = "sl"
            exit_notional = entry_notional
            
            # Calculate PnL
            if trade.side == "long":
                price_change_pct = (exit_price - trade.entry_price) / trade.entry_price
            else:  # short
                price_change_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        # Calculate fees (exit is limit order - maker fee for TP, taker for SL)
        if exit_reason in ["tp1", "tp2"]:
            exit_fee = exit_notional * self.maker_fee
        else:
            exit_fee = exit_notional * self.taker_fee
            
        total_fees = entry_fee + exit_fee
        
        # Calculate final PnL
        gross_pnl = trade.position_size * trade.leverage * price_change_pct
        net_pnl = gross_pnl - total_fees
        pnl_percent = (net_pnl / trade.position_size) * 100
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl_usd = net_pnl
        trade.pnl_percent = pnl_percent
        trade.fees_usd = total_fees
        
        # Set exit time (random between 5 minutes to 4 hours)
        minutes_to_exit = random.uniform(5, 240)
        trade.exit_time = trade.entry_time + timedelta(minutes=minutes_to_exit)
        
        return trade
    
    def run_simulation(self, hours: int = 24) -> Dict[str, Any]:
        """Run the trading simulation for specified hours"""
        print(f"\n{'='*60}")
        print(f"STARTING 24-HOUR TRADING SIMULATION")
        print(f"{'='*60}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"Position Size: {self.position_size_pct}% per trade")
        print(f"Max Positions: {self.max_positions}")
        print(f"{'='*60}\n")
        
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=hours)
        
        hourly_stats = []
        
        while current_time < end_time:
            hour = current_time.hour
            
            # Check for closed positions
            for trade in self.open_positions[:]:
                if trade.exit_time and trade.exit_time <= current_time:
                    self.open_positions.remove(trade)
                    self.current_balance += trade.pnl_usd
                    self.total_fees += trade.fees_usd
                    
                    if trade.pnl_usd > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    # Update peak and drawdown
                    if self.current_balance > self.peak_balance:
                        self.peak_balance = self.current_balance
                    drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
                    if drawdown > self.max_drawdown:
                        self.max_drawdown = drawdown
            
            # Generate new opportunities
            opportunities = self.generate_market_opportunity(hour)
            
            for opp in opportunities:
                # Check if we can take the position
                if len(self.open_positions) >= self.max_positions:
                    continue
                    
                # Check if we have sufficient balance
                position_size = self.calculate_position_size()
                if position_size < 10:  # Minimum position size
                    continue
                
                # Create and execute trade
                trade = SimulatedTrade(
                    symbol=opp['symbol'],
                    side="long" if opp['is_long'] else "short",
                    entry_time=current_time,
                    entry_price=100.0,  # Normalized price
                    position_size=position_size,
                    leverage=self.leverage,
                    tp1_price=101.0 if opp['is_long'] else 99.0,
                    tp2_price=102.0 if opp['is_long'] else 98.0,
                    sl_price=98.0 if opp['is_long'] else 102.0
                )
                
                # Simulate outcome
                trade = self.simulate_trade_outcome(trade, opp['win_probability'])
                
                self.trades.append(trade)
                self.open_positions.append(trade)
            
            # Record hourly stats
            hourly_stats.append({
                'hour': current_time.strftime('%Y-%m-%d %H:00'),
                'balance': self.current_balance,
                'open_positions': len(self.open_positions),
                'total_trades': len(self.trades)
            })
            
            # Move to next hour
            current_time += timedelta(hours=1)
        
        # Close any remaining positions at market
        for trade in self.open_positions:
            if not trade.exit_time:
                trade.exit_time = end_time
                trade.exit_price = trade.entry_price  # Exit at breakeven for simplicity
                trade.exit_reason = "end_of_simulation"
                self.current_balance += trade.pnl_usd
        
        return self.generate_report(hourly_stats)
    
    def generate_report(self, hourly_stats: List[Dict]) -> Dict[str, Any]:
        """Generate a comprehensive report of the simulation"""
        total_trades = len(self.trades)
        
        if total_trades == 0:
            return {"error": "No trades executed"}
        
        # Calculate statistics
        total_pnl = self.current_balance - self.initial_balance
        total_return_pct = (total_pnl / self.initial_balance) * 100
        
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average winner and loser
        winners = [t for t in self.trades if t.pnl_usd > 0]
        losers = [t for t in self.trades if t.pnl_usd < 0]
        
        avg_winner = np.mean([t.pnl_usd for t in winners]) if winners else 0
        avg_loser = np.mean([abs(t.pnl_usd) for t in losers]) if losers else 0
        
        profit_factor = (sum([t.pnl_usd for t in winners]) / 
                        abs(sum([t.pnl_usd for t in losers]))) if losers else 0
        
        # Best and worst trades
        best_trade = max(self.trades, key=lambda t: t.pnl_usd) if self.trades else None
        worst_trade = min(self.trades, key=lambda t: t.pnl_usd) if self.trades else None
        
        report = {
            "simulation_parameters": {
                "initial_balance": self.initial_balance,
                "leverage": self.leverage,
                "position_size_pct": self.position_size_pct,
                "max_positions": self.max_positions
            },
            "results": {
                "final_balance": round(self.current_balance, 2),
                "total_pnl_usd": round(total_pnl, 2),
                "total_return_pct": round(total_return_pct, 2),
                "total_fees_usd": round(self.total_fees, 2)
            },
            "trade_statistics": {
                "total_trades": total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate_pct": round(win_rate, 2),
                "avg_winner_usd": round(avg_winner, 2),
                "avg_loser_usd": round(avg_loser, 2),
                "profit_factor": round(profit_factor, 2)
            },
            "risk_metrics": {
                "max_drawdown_pct": round(self.max_drawdown, 2),
                "peak_balance": round(self.peak_balance, 2)
            },
            "best_trade": {
                "symbol": best_trade.symbol if best_trade else None,
                "pnl_usd": round(best_trade.pnl_usd, 2) if best_trade else None,
                "pnl_pct": round(best_trade.pnl_percent, 2) if best_trade else None
            },
            "worst_trade": {
                "symbol": worst_trade.symbol if worst_trade else None,
                "pnl_usd": round(worst_trade.pnl_usd, 2) if worst_trade else None,
                "pnl_pct": round(worst_trade.pnl_percent, 2) if worst_trade else None
            },
            "hourly_progression": hourly_stats
        }
        
        return report

def run_multiple_simulations(num_simulations: int = 100):
    """Run multiple simulations to get statistical distribution"""
    results = []
    
    print(f"\nRunning {num_simulations} simulations...")
    print("This will show the range of possible outcomes\n")
    
    for i in range(num_simulations):
        sim = TradingSimulator(initial_balance=10000)
        report = sim.run_simulation(hours=24)
        results.append(report["results"]["total_return_pct"])
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_simulations} simulations...")
    
    # Calculate statistics
    results_array = np.array(results)
    
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION RESULTS ({num_simulations} runs)")
    print(f"{'='*60}")
    print(f"\n📊 RETURN DISTRIBUTION (24 hours):")
    print(f"   Best Case:     {np.max(results_array):.2f}%")
    print(f"   75th Percentile: {np.percentile(results_array, 75):.2f}%")
    print(f"   Median:        {np.median(results_array):.2f}%")
    print(f"   Average:       {np.mean(results_array):.2f}%")
    print(f"   25th Percentile: {np.percentile(results_array, 25):.2f}%")
    print(f"   Worst Case:    {np.min(results_array):.2f}%")
    print(f"   Std Deviation: {np.std(results_array):.2f}%")
    
    # Calculate risk metrics
    profitable_runs = sum(1 for r in results if r > 0)
    loss_runs = sum(1 for r in results if r < 0)
    
    print(f"\n📈 OUTCOME PROBABILITIES:")
    print(f"   Profitable days: {profitable_runs}/{num_simulations} ({profitable_runs/num_simulations*100:.1f}%)")
    print(f"   Loss days:       {loss_runs}/{num_simulations} ({loss_runs/num_simulations*100:.1f}%)")
    
    # Risk categories
    severe_loss = sum(1 for r in results if r < -50)
    moderate_loss = sum(1 for r in results if -50 <= r < -20)
    small_loss = sum(1 for r in results if -20 <= r < 0)
    small_gain = sum(1 for r in results if 0 <= r < 20)
    moderate_gain = sum(1 for r in results if 20 <= r < 50)
    large_gain = sum(1 for r in results if r >= 50)
    
    print(f"\n⚠️ RISK BREAKDOWN:")
    print(f"   Severe Loss (< -50%):    {severe_loss} ({severe_loss/num_simulations*100:.1f}%)")
    print(f"   Moderate Loss (-50 to -20%): {moderate_loss} ({moderate_loss/num_simulations*100:.1f}%)")
    print(f"   Small Loss (-20 to 0%):   {small_loss} ({small_loss/num_simulations*100:.1f}%)")
    print(f"   Small Gain (0 to 20%):    {small_gain} ({small_gain/num_simulations*100:.1f}%)")
    print(f"   Moderate Gain (20 to 50%): {moderate_gain} ({moderate_gain/num_simulations*100:.1f}%)")
    print(f"   Large Gain (> 50%):       {large_gain} ({large_gain/num_simulations*100:.1f}%)")
    
    return results

def main():
    """Run the main simulation"""
    print("="*60)
    print("24-HOUR TRADING SIMULATION")
    print("Configuration: 3% position size, 344x leverage, max 10 positions")
    print("="*60)
    
    # Run single detailed simulation
    print("\n1️⃣ SINGLE DETAILED SIMULATION:")
    sim = TradingSimulator(initial_balance=10000)
    report = sim.run_simulation(hours=24)
    
    # Print detailed results
    print(f"\n📊 DETAILED RESULTS:")
    print(f"   Initial Balance:  ${report['simulation_parameters']['initial_balance']:,.2f}")
    print(f"   Final Balance:    ${report['results']['final_balance']:,.2f}")
    print(f"   Total P&L:        ${report['results']['total_pnl_usd']:,.2f}")
    print(f"   Return:           {report['results']['total_return_pct']:.2f}%")
    print(f"   Total Fees:       ${report['results']['total_fees_usd']:,.2f}")
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"   Total Trades:     {report['trade_statistics']['total_trades']}")
    print(f"   Win Rate:         {report['trade_statistics']['win_rate_pct']:.2f}%")
    print(f"   Winners/Losers:   {report['trade_statistics']['winning_trades']}/{report['trade_statistics']['losing_trades']}")
    print(f"   Avg Winner:       ${report['trade_statistics']['avg_winner_usd']:,.2f}")
    print(f"   Avg Loser:        ${report['trade_statistics']['avg_loser_usd']:,.2f}")
    print(f"   Profit Factor:    {report['trade_statistics']['profit_factor']:.2f}")
    
    print(f"\n⚠️ RISK METRICS:")
    print(f"   Max Drawdown:     {report['risk_metrics']['max_drawdown_pct']:.2f}%")
    print(f"   Peak Balance:     ${report['risk_metrics']['peak_balance']:,.2f}")
    
    if report['best_trade']['symbol']:
        print(f"\n🏆 BEST TRADE:")
        print(f"   Symbol: {report['best_trade']['symbol']}")
        print(f"   P&L: ${report['best_trade']['pnl_usd']:,.2f} ({report['best_trade']['pnl_pct']:.2f}%)")
    
    if report['worst_trade']['symbol']:
        print(f"\n💔 WORST TRADE:")
        print(f"   Symbol: {report['worst_trade']['symbol']}")
        print(f"   P&L: ${report['worst_trade']['pnl_usd']:,.2f} ({report['worst_trade']['pnl_pct']:.2f}%)")
    
    # Save detailed report
    with open('/workspace/simulation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Detailed report saved to simulation_report.json")
    
    # Run Monte Carlo simulation
    print(f"\n2️⃣ MONTE CARLO ANALYSIS:")
    results = run_multiple_simulations(100)
    
    # Warning about leverage
    print(f"\n{'='*60}")
    print(f"⚠️  EXTREME LEVERAGE WARNING")
    print(f"{'='*60}")
    print(f"""
🔴 CRITICAL RISK FACTORS WITH 344x LEVERAGE:

1. LIQUIDATION RISK: With 344x leverage, a mere 0.29% adverse move 
   will liquidate your entire position (100% / 344 = 0.29%)

2. MARGIN CALLS: Even smaller moves can trigger margin calls,
   forcing position closure at losses

3. FUNDING RATES: At 344x, funding costs are multiplied 344 times,
   potentially eating significant profits even on winning trades

4. SLIPPAGE IMPACT: A 0.1% slippage becomes 34.4% loss on your margin

5. EXCHANGE LIMITS: Most exchanges limit leverage to 100-125x for retail

6. PSYCHOLOGICAL PRESSURE: Managing 344x positions requires extreme
   emotional control - most traders cannot handle this stress

📌 RECOMMENDED MAXIMUM LEVERAGE: 10-20x for experienced traders
📌 SAFE LEVERAGE FOR BEGINNERS: 2-5x

The simulation assumes perfect execution, no slippage, and no funding.
Real-world results will likely be significantly worse.
""")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()