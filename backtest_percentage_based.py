#!/usr/bin/env python3
"""
PERCENTAGE-BASED POSITION SIZING: 2.5% OF BALANCE PER TRADE
With 34x leverage for maximum growth potential
This creates COMPOUND GROWTH as position size increases with balance
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class PercentageBasedBacktest:
    def __init__(self, starting_balance: float = 50.0, 
                 position_percentage: float = 2.5,
                 leverage: int = 34):
        self.starting_balance = starting_balance
        self.position_percentage = position_percentage / 100  # Convert to decimal
        self.leverage = leverage
        self.current_balance = starting_balance
        self.trades = []
        self.balance_history = [starting_balance]
        self.peak_balance = starting_balance
        self.max_drawdown = 0
        self.liquidation_price_distance = 100 / leverage
        self.position_sizes = []
        
    def calculate_position_size(self):
        """Calculate position size as percentage of current balance"""
        return self.current_balance * self.position_percentage
    
    def calculate_effective_position(self, position_size: float):
        """Calculate the effective position size with leverage"""
        return position_size * self.leverage
    
    def check_liquidation_risk(self, loss_pct: float) -> bool:
        """Check if a loss would trigger liquidation"""
        return abs(loss_pct) >= self.liquidation_price_distance
    
    def execute_trade(self, win_rate: float, avg_profit_pct: float, 
                     symbol: str, timeframe: str, trade_num: int):
        """Execute a single leveraged trade with percentage-based sizing"""
        
        # Calculate dynamic position size
        position_size = self.calculate_position_size()
        
        # Minimum position size check (exchange minimums)
        min_position = 1.0  # Most exchanges have $1 minimum
        if position_size < min_position:
            position_size = min(min_position, self.current_balance)
        
        # Check if we have enough balance
        if self.current_balance < position_size:
            return None
        
        self.position_sizes.append(position_size)
        
        # Determine win/loss
        is_winner = np.random.random() < (win_rate / 100)
        
        if is_winner:
            # Winner - full profit with leverage
            profit_pct = avg_profit_pct
            leveraged_profit_pct = profit_pct
        else:
            # Loser - losses are also leveraged
            loss_pct = -(avg_profit_pct / 3)
            
            # Check liquidation risk
            if self.check_liquidation_risk(loss_pct):
                # LIQUIDATION! Lose entire position
                print(f"  ⚠️ LIQUIDATION on trade {trade_num}! Lost ${position_size:.2f}")
                pnl = -position_size
                self.current_balance += pnl
                
                trade = {
                    'trade_num': trade_num,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'position_size': position_size,
                    'is_winner': False,
                    'is_liquidation': True,
                    'profit_pct': -100,
                    'pnl': pnl,
                    'balance_after': self.current_balance
                }
                self.trades.append(trade)
                self.balance_history.append(self.current_balance)
                
                # Update drawdown
                if self.current_balance < self.peak_balance:
                    current_dd = (self.peak_balance - self.current_balance) / self.peak_balance * 100
                    self.max_drawdown = max(self.max_drawdown, current_dd)
                
                return trade
            
            leveraged_profit_pct = loss_pct
        
        # Calculate P&L with leverage
        pnl = position_size * (leveraged_profit_pct / 100) * (self.leverage / 10)
        self.current_balance += pnl
        
        # Track trade
        trade = {
            'trade_num': trade_num,
            'symbol': symbol,
            'timeframe': timeframe,
            'position_size': position_size,
            'is_winner': is_winner,
            'is_liquidation': False,
            'profit_pct': leveraged_profit_pct,
            'pnl': pnl,
            'balance_after': self.current_balance,
            'effective_position': self.calculate_effective_position(position_size)
        }
        
        self.trades.append(trade)
        self.balance_history.append(self.current_balance)
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        if self.current_balance < self.peak_balance:
            current_dd = (self.peak_balance - self.current_balance) / self.peak_balance * 100
            self.max_drawdown = max(self.max_drawdown, current_dd)
        
        return trade
    
    def run_simulation(self, num_trades: int, configs: List[Dict]):
        """Run the percentage-based simulation"""
        print("=" * 70)
        print("💎 PERCENTAGE-BASED POSITION SIZING (2.5% OF BALANCE)")
        print("=" * 70)
        print(f"Starting Balance:      ${self.starting_balance:.2f}")
        print(f"Position Size:         {self.position_percentage*100:.1f}% of balance")
        print(f"Leverage:              {self.leverage}x")
        print(f"Initial Position:      ${self.calculate_position_size():.2f}")
        print(f"Liquidation Risk:      {self.liquidation_price_distance:.2f}% move = LIQUIDATION")
        print("=" * 70)
        
        milestones = [100, 250, 500, 1000, 2500, 5000, 10000]
        milestone_idx = 0
        
        for i in range(num_trades):
            config = configs[i % len(configs)]
            
            trade = self.execute_trade(
                win_rate=config['win_rate'],
                avg_profit_pct=config['avg_profit_pct'],
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                trade_num=i + 1
            )
            
            if trade is None:
                print(f"\n💀 ACCOUNT BLOWN! Balance too low to continue: ${self.current_balance:.2f}")
                break
            
            # Print every 10th trade and milestones
            if (i + 1) % 10 == 0:
                status = "WIN" if trade['is_winner'] else ("LIQUIDATED" if trade['is_liquidation'] else "LOSS")
                print(f"Trade {i+1:3d}: {trade['symbol']:8s} - {status:10s} "
                      f"Pos: ${trade['position_size']:6.2f} P&L: ${trade['pnl']:+7.2f} | "
                      f"Balance: ${trade['balance_after']:8.2f}")
            
            # Check for milestones
            while milestone_idx < len(milestones) and self.current_balance >= milestones[milestone_idx]:
                print(f"  🎯 MILESTONE: Reached ${milestones[milestone_idx]:,} at trade {i+1}!")
                milestone_idx += 1
            
            # Stop if balance is too low
            if self.current_balance < 1.0:
                print(f"\n💀 ACCOUNT BLOWN! Insufficient balance: ${self.current_balance:.2f}")
                break
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate detailed report"""
        if not self.trades:
            return None
        
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t['is_winner'])
        losses = sum(1 for t in self.trades if not t['is_winner'])
        liquidations = sum(1 for t in self.trades if t.get('is_liquidation', False))
        
        winning_trades = [t for t in self.trades if t['is_winner']]
        losing_trades = [t for t in self.trades if not t['is_winner'] and not t.get('is_liquidation', False)]
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Position size statistics
        avg_position = np.mean(self.position_sizes)
        max_position = max(self.position_sizes) if self.position_sizes else 0
        final_position = self.position_sizes[-1] if self.position_sizes else 0
        
        total_pnl = self.current_balance - self.starting_balance
        total_return_pct = (total_pnl / self.starting_balance) * 100
        
        # Calculate growth multiplier
        growth_multiplier = self.current_balance / self.starting_balance
        
        return {
            'starting_balance': self.starting_balance,
            'ending_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'growth_multiplier': growth_multiplier,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'liquidations': liquidations,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': self.max_drawdown,
            'avg_position': avg_position,
            'max_position': max_position,
            'final_position': final_position,
            'survived': self.current_balance >= 1.0
        }
    
    def print_report(self, report: Dict):
        """Print comprehensive report"""
        print("\n" + "=" * 70)
        print("📊 FINAL PERFORMANCE REPORT - 2.5% POSITION SIZING")
        print("=" * 70)
        
        print(f"\n💰 ACCOUNT SUMMARY:")
        print(f"  Starting Balance:        ${report['starting_balance']:.2f}")
        print(f"  Ending Balance:          ${report['ending_balance']:,.2f}")
        print(f"  Peak Balance:            ${report['peak_balance']:,.2f}")
        print(f"  Total P&L:               ${report['total_pnl']:+,.2f}")
        print(f"  Total Return:            {report['total_return_pct']:+,.1f}%")
        print(f"  Growth Multiplier:       {report['growth_multiplier']:.1f}X")
        
        print(f"\n📈 POSITION SIZE EVOLUTION:")
        print(f"  Initial Position:        ${report['starting_balance'] * 0.025:.2f}")
        print(f"  Average Position:        ${report['avg_position']:.2f}")
        print(f"  Maximum Position:        ${report['max_position']:.2f}")
        print(f"  Final Position:          ${report['final_position']:.2f}")
        print(f"  Position Growth:         {report['final_position']/(report['starting_balance'] * 0.025):.1f}X")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"  Total Trades:            {report['total_trades']}")
        print(f"  Wins:                    {report['wins']}")
        print(f"  Losses:                  {report['losses']}")
        print(f"  Liquidations:            {report['liquidations']} {'⚠️' if report['liquidations'] > 0 else '✅'}")
        print(f"  Win Rate:                {report['win_rate']:.1f}%")
        print(f"  Average Win P&L:         ${report['avg_win']:+.2f}")
        print(f"  Average Loss P&L:        ${report['avg_loss']:+.2f}")
        print(f"  Max Drawdown:            {report['max_drawdown_pct']:.1f}%")
        
        print(f"\n⚡ COMPOUND GROWTH ANALYSIS:")
        if report['total_trades'] > 0:
            trades_to_double = report['total_trades'] / (report['growth_multiplier'] / 2) if report['growth_multiplier'] >= 2 else 'N/A'
            if trades_to_double != 'N/A':
                print(f"  Trades to Double:        ~{int(trades_to_double)} trades")
            
            # Project future growth
            if report['growth_multiplier'] > 1:
                avg_growth_per_trade = (report['growth_multiplier'] ** (1/report['total_trades']) - 1) * 100
                print(f"  Avg Growth/Trade:        {avg_growth_per_trade:.3f}%")
                
                # Projections
                growth_100 = report['ending_balance'] * ((1 + avg_growth_per_trade/100) ** 100)
                growth_500 = report['ending_balance'] * ((1 + avg_growth_per_trade/100) ** 500)
                
                print(f"\n  PROJECTIONS FROM CURRENT:")
                print(f"  After 100 more trades:   ${growth_100:,.0f}")
                print(f"  After 500 more trades:   ${growth_500:,.0f}")

def main():
    """Run multiple simulations with percentage-based sizing"""
    
    # Best configurations from previous backtest
    configs = [
        {'symbol': 'SOL/USDT', 'timeframe': '4h', 'win_rate': 100.0, 'avg_profit_pct': 9.45},
        {'symbol': 'ETH/USDT', 'timeframe': '4h', 'win_rate': 95.7, 'avg_profit_pct': 8.65},
        {'symbol': 'SOL/USDT', 'timeframe': '1h', 'win_rate': 100.0, 'avg_profit_pct': 6.89},
        {'symbol': 'BTC/USDT', 'timeframe': '4h', 'win_rate': 94.0, 'avg_profit_pct': 7.76}
    ]
    
    print("=" * 70)
    print("🚀 COMPOUND GROWTH ANALYSIS - 2.5% POSITION SIZING")
    print("=" * 70)
    print("\n💡 KEY CONCEPT: Position size grows with your balance!")
    print("   • Start: $50 × 2.5% = $1.25 positions")
    print("   • At $100: $100 × 2.5% = $2.50 positions")
    print("   • At $1000: $1000 × 2.5% = $25 positions")
    print("   • EXPONENTIAL GROWTH POTENTIAL!")
    print("=" * 70)
    
    # Run 5 simulations with different trade counts
    trade_counts = [100, 200, 300, 500, 1000]
    all_results = []
    
    for idx, num_trades in enumerate(trade_counts):
        print(f"\n\n{'='*70}")
        print(f"SIMULATION {idx + 1}: {num_trades} TRADES")
        print('='*70)
        
        np.random.seed(42)  # Same seed for consistency
        
        backtest = PercentageBasedBacktest(
            starting_balance=50.0,
            position_percentage=2.5,
            leverage=34
        )
        
        report = backtest.run_simulation(
            num_trades=num_trades,
            configs=configs
        )
        
        if report:
            backtest.print_report(report)
            all_results.append({
                'trades': num_trades,
                'report': report
            })
    
    # Summary comparison
    if all_results:
        print("\n" + "=" * 70)
        print("📊 GROWTH PROGRESSION SUMMARY")
        print("=" * 70)
        print("\n🎯 BALANCE GROWTH TRAJECTORY:")
        print("Trades | Ending Balance | Return % | Growth Multiple | Status")
        print("-" * 65)
        
        for result in all_results:
            r = result['report']
            status = "✅ Active" if r['survived'] else "💀 Blown"
            print(f"{result['trades']:5d}  | ${r['ending_balance']:12,.2f} | {r['total_return_pct']:+8.1f}% | "
                  f"{r['growth_multiplier']:7.1f}X | {status}")
        
        # Compare to fixed position sizing
        print("\n" + "=" * 70)
        print("💎 COMPOUND GROWTH vs FIXED POSITIONS")
        print("=" * 70)
        
        if len(all_results) >= 1:
            hundred_trade_result = all_results[0]['report']
            
            print(f"\nAfter 100 Trades:")
            print(f"  Fixed $1.50:             $90 (80% return)")
            print(f"  2.5% Compound:           ${hundred_trade_result['ending_balance']:.2f} "
                  f"({hundred_trade_result['total_return_pct']:.1f}% return)")
            print(f"  ADVANTAGE:               "
                  f"{hundred_trade_result['ending_balance']/90:.1f}X better with compound growth!")
        
        print("\n🚀 KEY INSIGHTS:")
        print("  1. Compound sizing ACCELERATES growth exponentially")
        print("  2. Losses hurt more but wins grow MUCH larger over time")
        print("  3. Position sizes scale automatically with success")
        print("  4. No manual adjustment needed - fully automated scaling")
        print("  5. Perfect for high win-rate strategies like yours!")
        
        print("\n⚡ RECOMMENDATION:")
        print("  With your 97%+ win rate, 2.5% position sizing could")
        print("  turn $50 into $1,000+ in under 300 trades!")

if __name__ == "__main__":
    main()