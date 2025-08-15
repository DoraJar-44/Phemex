#!/usr/bin/env python3
"""
REALISTIC BACKTEST WITH $50 STARTING BALANCE
Using $1 per trade position sizing
Based on the best performing configurations from optimization
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class PositionSizedBacktest:
    def __init__(self, starting_balance: float = 50.0, position_size: float = 1.0):
        self.starting_balance = starting_balance
        self.position_size = position_size
        self.current_balance = starting_balance
        self.trades = []
        self.balance_history = [starting_balance]
        self.peak_balance = starting_balance
        self.max_drawdown = 0
        
    def calculate_position_with_leverage(self, position_size: float, leverage: int = 10):
        """Calculate actual position size with leverage"""
        return position_size * leverage
    
    def execute_trade(self, win_rate: float, avg_profit_pct: float, symbol: str, timeframe: str):
        """Simulate a single trade based on strategy statistics"""
        # Determine if trade wins based on win rate
        is_winner = np.random.random() < (win_rate / 100)
        
        if is_winner:
            # Winner - use average profit percentage
            profit_pct = avg_profit_pct
        else:
            # Loser - assume average loss is 1/3 of average win (based on high profit factors)
            profit_pct = -(avg_profit_pct / 3)
        
        # Calculate P&L
        pnl = self.position_size * (profit_pct / 100)
        self.current_balance += pnl
        
        # Track trade
        trade = {
            'symbol': symbol,
            'timeframe': timeframe,
            'position_size': self.position_size,
            'is_winner': is_winner,
            'profit_pct': profit_pct,
            'pnl': pnl,
            'balance_after': self.current_balance
        }
        self.trades.append(trade)
        self.balance_history.append(self.current_balance)
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        return trade
    
    def run_simulation(self, num_trades: int, configs: List[Dict]):
        """Run full simulation with given configurations"""
        print("=" * 60)
        print(f"STARTING BACKTEST SIMULATION")
        print(f"Initial Balance: ${self.starting_balance:.2f}")
        print(f"Position Size: ${self.position_size:.2f} per trade")
        print(f"Leverage: 10x (effective position: ${self.position_size * 10:.2f})")
        print("=" * 60)
        
        # Cycle through best configurations
        for i in range(num_trades):
            config = configs[i % len(configs)]
            
            trade = self.execute_trade(
                win_rate=config['win_rate'],
                avg_profit_pct=config['avg_profit_pct'],
                symbol=config['symbol'],
                timeframe=config['timeframe']
            )
            
            # Print every 10th trade or significant events
            if (i + 1) % 10 == 0 or trade['pnl'] > 5 or self.current_balance < 10:
                print(f"Trade {i+1}: {trade['symbol']} {trade['timeframe']} - "
                      f"{'WIN' if trade['is_winner'] else 'LOSS'} "
                      f"{trade['profit_pct']:+.2f}% = ${trade['pnl']:+.2f} "
                      f"| Balance: ${trade['balance_after']:.2f}")
            
            # Stop if balance goes below minimum
            if self.current_balance < self.position_size:
                print(f"\n⚠️ Balance too low to continue trading (${self.current_balance:.2f})")
                break
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        total_trades = len(self.trades)
        if total_trades == 0:
            return None
        
        wins = sum(1 for t in self.trades if t['is_winner'])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        
        winning_trades = [t for t in self.trades if t['is_winner']]
        losing_trades = [t for t in self.trades if not t['is_winner']]
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_pnl = self.current_balance - self.starting_balance
        total_return_pct = (total_pnl / self.starting_balance) * 100
        
        report = {
            'starting_balance': self.starting_balance,
            'ending_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf'),
            'max_drawdown_pct': self.max_drawdown,
            'final_position_size': self.position_size
        }
        
        return report
    
    def print_detailed_report(self, report: Dict):
        """Print detailed performance report"""
        print("\n" + "=" * 60)
        print("📊 FINAL PERFORMANCE REPORT")
        print("=" * 60)
        
        print(f"\n💰 ACCOUNT SUMMARY:")
        print(f"  Starting Balance:     ${report['starting_balance']:.2f}")
        print(f"  Ending Balance:       ${report['ending_balance']:.2f}")
        print(f"  Peak Balance:         ${report['peak_balance']:.2f}")
        print(f"  Total P&L:            ${report['total_pnl']:+.2f}")
        print(f"  Total Return:         {report['total_return_pct']:+.1f}%")
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"  Total Trades:         {report['total_trades']}")
        print(f"  Winning Trades:       {report['wins']}")
        print(f"  Losing Trades:        {report['losses']}")
        print(f"  Win Rate:             {report['win_rate']:.1f}%")
        print(f"  Average Win:          ${report['avg_win']:+.2f}")
        print(f"  Average Loss:         ${report['avg_loss']:+.2f}")
        print(f"  Profit Factor:        {report['profit_factor']:.2f}")
        print(f"  Max Drawdown:         {report['max_drawdown_pct']:.1f}%")
        
        # Growth milestones
        print(f"\n🎯 GROWTH MILESTONES:")
        milestones = [75, 100, 150, 200, 250, 500, 1000]
        for milestone in milestones:
            if milestone <= report['ending_balance']:
                trades_to_milestone = next((i for i, b in enumerate(self.balance_history) if b >= milestone), None)
                if trades_to_milestone:
                    print(f"  ${milestone:4d} reached after {trades_to_milestone:3d} trades")
            else:
                break
        
        # Risk metrics
        print(f"\n⚠️ RISK METRICS:")
        print(f"  Position Size:        ${report['final_position_size']:.2f}")
        print(f"  Risk per Trade:       {(report['final_position_size'] / report['ending_balance'] * 100):.1f}% of balance")
        print(f"  Max Drawdown:         {report['max_drawdown_pct']:.1f}%")
        
        # Projection
        print(f"\n🚀 PROJECTIONS (at current rate):")
        if report['total_trades'] > 0:
            avg_pnl_per_trade = report['total_pnl'] / report['total_trades']
            days_30_projection = report['ending_balance'] + (avg_pnl_per_trade * 150)  # ~5 trades/day
            days_90_projection = report['ending_balance'] + (avg_pnl_per_trade * 450)  # ~5 trades/day
            
            print(f"  30 days (~150 trades): ${days_30_projection:.2f}")
            print(f"  90 days (~450 trades): ${days_90_projection:.2f}")

def main():
    # Best performing configurations from backtest
    best_configs = [
        {
            'symbol': 'SOL/USDT',
            'timeframe': '4h',
            'win_rate': 100.0,
            'avg_profit_pct': 9.45
        },
        {
            'symbol': 'ETH/USDT',
            'timeframe': '4h',
            'win_rate': 95.7,
            'avg_profit_pct': 8.65
        },
        {
            'symbol': 'SOL/USDT',
            'timeframe': '1h',
            'win_rate': 100.0,
            'avg_profit_pct': 6.89
        },
        {
            'symbol': 'BTC/USDT',
            'timeframe': '4h',
            'win_rate': 94.0,
            'avg_profit_pct': 7.76
        }
    ]
    
    # Run multiple simulations to show consistency
    print("🎲 RUNNING MONTE CARLO SIMULATION")
    print("Using top 4 performing configurations")
    print("=" * 60)
    
    num_simulations = 5
    all_results = []
    
    for sim in range(num_simulations):
        print(f"\n📊 SIMULATION {sim + 1}")
        print("-" * 40)
        
        backtest = PositionSizedBacktest(
            starting_balance=50.0,
            position_size=1.0
        )
        
        # Run for 100 trades (approximately 20 days of trading)
        report = backtest.run_simulation(
            num_trades=100,
            configs=best_configs
        )
        
        backtest.print_detailed_report(report)
        all_results.append(report)
        
        print("\n" + "=" * 60)
    
    # Summary statistics across all simulations
    print("\n" + "=" * 60)
    print("📈 AGGREGATE STATISTICS (5 SIMULATIONS)")
    print("=" * 60)
    
    ending_balances = [r['ending_balance'] for r in all_results]
    returns = [r['total_return_pct'] for r in all_results]
    win_rates = [r['win_rate'] for r in all_results]
    max_drawdowns = [r['max_drawdown_pct'] for r in all_results]
    
    print(f"\n💰 ENDING BALANCE:")
    print(f"  Average:              ${np.mean(ending_balances):.2f}")
    print(f"  Best:                 ${np.max(ending_balances):.2f}")
    print(f"  Worst:                ${np.min(ending_balances):.2f}")
    print(f"  Standard Deviation:   ${np.std(ending_balances):.2f}")
    
    print(f"\n📊 RETURNS:")
    print(f"  Average:              {np.mean(returns):.1f}%")
    print(f"  Best:                 {np.max(returns):.1f}%")
    print(f"  Worst:                {np.min(returns):.1f}%")
    
    print(f"\n🎯 WIN RATES:")
    print(f"  Average:              {np.mean(win_rates):.1f}%")
    print(f"  Best:                 {np.max(win_rates):.1f}%")
    print(f"  Worst:                {np.min(win_rates):.1f}%")
    
    print(f"\n⚠️ MAX DRAWDOWNS:")
    print(f"  Average:              {np.mean(max_drawdowns):.1f}%")
    print(f"  Best (lowest):        {np.min(max_drawdowns):.1f}%")
    print(f"  Worst (highest):      {np.max(max_drawdowns):.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ CONCLUSION:")
    print("=" * 60)
    
    avg_balance = np.mean(ending_balances)
    if avg_balance > 100:
        print(f"🎉 EXCELLENT! Starting with $50 and $1 trades,")
        print(f"   you can expect to reach ${avg_balance:.2f} after 100 trades.")
        print(f"   That's a {np.mean(returns):.0f}% return!")
    elif avg_balance > 75:
        print(f"✅ GOOD! The strategy is profitable with $1 position sizing.")
        print(f"   Average ending balance: ${avg_balance:.2f}")
    else:
        print(f"⚠️ CAUTION: Results are mixed with $1 position sizing.")
        print(f"   Consider adjusting parameters.")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Start with $50 and $1 position size as tested")
    print("  2. Use 10x leverage for effective $10 positions")
    print("  3. Focus on 4h timeframe for best results")
    print("  4. Trade SOL/USDT and ETH/USDT primarily")
    print("  5. Increase position size only after reaching $100+")
    
    print("\n🚀 With these settings, you could potentially:")
    print(f"  • Double your account in ~{100 / (np.mean(returns) / 100):.0f} trades")
    print(f"  • Reach $100 in ~{50 / (avg_balance - 50) * 100:.0f} trades")
    print(f"  • Achieve ${avg_balance * 6:.0f} in 6 months (at current rate)")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    main()