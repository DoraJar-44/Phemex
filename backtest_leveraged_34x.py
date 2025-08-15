#!/usr/bin/env python3
"""
HIGH LEVERAGE BACKTEST: 34x LEVERAGE WITH $1.50 POSITIONS
Starting balance: $50 USD
WARNING: Higher leverage = Higher risk AND higher reward
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class HighLeverageBacktest:
    def __init__(self, starting_balance: float = 50.0, 
                 position_size: float = 1.5, 
                 leverage: int = 34):
        self.starting_balance = starting_balance
        self.position_size = position_size
        self.leverage = leverage
        self.current_balance = starting_balance
        self.trades = []
        self.balance_history = [starting_balance]
        self.peak_balance = starting_balance
        self.max_drawdown = 0
        self.liquidation_price_distance = 100 / leverage  # Approximate liquidation distance
        
    def calculate_effective_position(self):
        """Calculate the effective position size with leverage"""
        return self.position_size * self.leverage
    
    def check_liquidation_risk(self, loss_pct: float) -> bool:
        """Check if a loss would trigger liquidation"""
        # With 34x leverage, a ~2.94% move against you = liquidation
        return abs(loss_pct) >= self.liquidation_price_distance
    
    def execute_trade(self, win_rate: float, avg_profit_pct: float, 
                     symbol: str, timeframe: str, trade_num: int):
        """Execute a single leveraged trade"""
        
        # Check if we have enough balance
        if self.current_balance < self.position_size:
            return None
        
        # Determine win/loss
        is_winner = np.random.random() < (win_rate / 100)
        
        if is_winner:
            # Winner - full profit with leverage
            profit_pct = avg_profit_pct
            leveraged_profit_pct = profit_pct  # The profit % is already on the leveraged position
        else:
            # Loser - losses are also leveraged
            # Using 1/3 of avg win as loss (based on high profit factors)
            loss_pct = -(avg_profit_pct / 3)
            
            # Check liquidation risk
            if self.check_liquidation_risk(loss_pct):
                # LIQUIDATION! Lose entire position
                print(f"  ⚠️ LIQUIDATION on trade {trade_num}! Lost ${self.position_size:.2f}")
                pnl = -self.position_size
                self.current_balance += pnl
                
                trade = {
                    'trade_num': trade_num,
                    'symbol': symbol,
                    'timeframe': timeframe,
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
        # P&L = position_size * (profit_pct / 100) * leverage
        pnl = self.position_size * (leveraged_profit_pct / 100) * (self.leverage / 10)  # Normalized for 10x base
        self.current_balance += pnl
        
        # Track trade
        trade = {
            'trade_num': trade_num,
            'symbol': symbol,
            'timeframe': timeframe,
            'is_winner': is_winner,
            'is_liquidation': False,
            'profit_pct': leveraged_profit_pct,
            'pnl': pnl,
            'balance_after': self.current_balance,
            'effective_position': self.calculate_effective_position()
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
        """Run the high leverage simulation"""
        print("=" * 70)
        print("🚨 HIGH LEVERAGE BACKTEST SIMULATION")
        print("=" * 70)
        print(f"Starting Balance:      ${self.starting_balance:.2f}")
        print(f"Position Size:         ${self.position_size:.2f} per trade")
        print(f"Leverage:              {self.leverage}x")
        print(f"Effective Position:    ${self.calculate_effective_position():.2f}")
        print(f"Liquidation Risk:      {self.liquidation_price_distance:.2f}% move = LIQUIDATION")
        print("=" * 70)
        
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
            
            # Print milestones and every 10th trade
            if (i + 1) % 10 == 0 or trade['balance_after'] >= 100 * ((trade['balance_after'] // 100)):
                status = "WIN" if trade['is_winner'] else ("LIQUIDATED" if trade['is_liquidation'] else "LOSS")
                print(f"Trade {i+1:3d}: {trade['symbol']:8s} {trade['timeframe']:2s} - {status:10s} "
                      f"P&L: ${trade['pnl']:+6.2f} | Balance: ${trade['balance_after']:7.2f}")
            
            # Check for major milestones
            if self.current_balance >= 100 and self.starting_balance < 100:
                if not any(b >= 100 for b in self.balance_history[:-1]):
                    print(f"  🎯 MILESTONE: Reached $100 at trade {i+1}!")
            
            if self.current_balance >= 500 and max(self.balance_history[:-1]) < 500:
                print(f"  🎯 MILESTONE: Reached $500 at trade {i+1}!")
            
            if self.current_balance >= 1000 and max(self.balance_history[:-1]) < 1000:
                print(f"  🚀 MILESTONE: Reached $1,000 at trade {i+1}!")
            
            # Stop if balance is too low
            if self.current_balance < self.position_size:
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
        
        total_pnl = self.current_balance - self.starting_balance
        total_return_pct = (total_pnl / self.starting_balance) * 100
        
        return {
            'starting_balance': self.starting_balance,
            'ending_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'liquidations': liquidations,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': self.max_drawdown,
            'survived': self.current_balance >= self.position_size
        }
    
    def print_report(self, report: Dict):
        """Print comprehensive report"""
        print("\n" + "=" * 70)
        print("📊 FINAL PERFORMANCE REPORT - 34x LEVERAGE")
        print("=" * 70)
        
        print(f"\n💰 ACCOUNT SUMMARY:")
        print(f"  Starting Balance:        ${report['starting_balance']:.2f}")
        print(f"  Ending Balance:          ${report['ending_balance']:.2f}")
        print(f"  Peak Balance:            ${report['peak_balance']:.2f}")
        print(f"  Total P&L:               ${report['total_pnl']:+.2f}")
        print(f"  Total Return:            {report['total_return_pct']:+.1f}%")
        
        # Calculate daily/monthly projections
        if report['total_trades'] > 0:
            avg_pnl_per_trade = report['total_pnl'] / report['total_trades']
            trades_per_day = 5
            daily_pnl = avg_pnl_per_trade * trades_per_day
            monthly_pnl = daily_pnl * 30
            
            print(f"\n📈 PROJECTIONS (at current rate):")
            print(f"  Per Trade Average:       ${avg_pnl_per_trade:+.2f}")
            print(f"  Daily (5 trades):        ${daily_pnl:+.2f}")
            print(f"  Monthly (150 trades):    ${monthly_pnl:+.2f}")
            print(f"  Projected 30-day balance: ${report['ending_balance'] + monthly_pnl:.2f}")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"  Total Trades:            {report['total_trades']}")
        print(f"  Wins:                    {report['wins']}")
        print(f"  Losses:                  {report['losses']}")
        print(f"  Liquidations:            {report['liquidations']} {'⚠️' if report['liquidations'] > 0 else '✅'}")
        print(f"  Win Rate:                {report['win_rate']:.1f}%")
        print(f"  Average Win:             ${report['avg_win']:+.2f}")
        print(f"  Average Loss:            ${report['avg_loss']:+.2f}")
        print(f"  Max Drawdown:            {report['max_drawdown_pct']:.1f}%")
        
        print(f"\n⚠️ RISK ANALYSIS:")
        print(f"  Leverage Used:           {self.leverage}x")
        print(f"  Position Size:           ${self.position_size:.2f}")
        print(f"  Effective Position:      ${self.calculate_effective_position():.2f}")
        print(f"  Liquidation Distance:    {self.liquidation_price_distance:.2f}% move")
        print(f"  Account Survived:        {'YES ✅' if report['survived'] else 'NO 💀'}")
        
        # Risk warnings
        if report['liquidations'] > 0:
            print(f"\n🚨 WARNING: {report['liquidations']} liquidation(s) occurred!")
            print(f"   Consider reducing leverage to avoid liquidations.")
        
        if report['max_drawdown_pct'] > 20:
            print(f"\n🚨 WARNING: High drawdown of {report['max_drawdown_pct']:.1f}%")
            print(f"   This indicates significant risk to capital.")

def main():
    """Run multiple simulations with 34x leverage"""
    
    # Best configurations from previous backtest
    configs = [
        {'symbol': 'SOL/USDT', 'timeframe': '4h', 'win_rate': 100.0, 'avg_profit_pct': 9.45},
        {'symbol': 'ETH/USDT', 'timeframe': '4h', 'win_rate': 95.7, 'avg_profit_pct': 8.65},
        {'symbol': 'SOL/USDT', 'timeframe': '1h', 'win_rate': 100.0, 'avg_profit_pct': 6.89},
        {'symbol': 'BTC/USDT', 'timeframe': '4h', 'win_rate': 94.0, 'avg_profit_pct': 7.76}
    ]
    
    print("=" * 70)
    print("🔥 34X LEVERAGE ANALYSIS - $50 START, $1.50 POSITIONS")
    print("=" * 70)
    print("\n⚠️ WARNING: High leverage trading is extremely risky!")
    print("   A 2.94% move against you = LIQUIDATION with 34x leverage")
    print("=" * 70)
    
    # Run 5 simulations
    all_results = []
    
    for sim in range(5):
        print(f"\n\n{'='*70}")
        print(f"SIMULATION {sim + 1} OF 5")
        print('='*70)
        
        np.random.seed(42 + sim)  # Different seed for each simulation
        
        backtest = HighLeverageBacktest(
            starting_balance=50.0,
            position_size=1.50,
            leverage=34
        )
        
        report = backtest.run_simulation(
            num_trades=100,
            configs=configs
        )
        
        if report:
            backtest.print_report(report)
            all_results.append(report)
    
    # Aggregate statistics
    if all_results:
        print("\n" + "=" * 70)
        print("📊 AGGREGATE RESULTS - ALL SIMULATIONS")
        print("=" * 70)
        
        survived = sum(1 for r in all_results if r['survived'])
        survival_rate = (survived / len(all_results)) * 100
        
        print(f"\n🎲 SURVIVAL STATISTICS:")
        print(f"  Simulations Run:         {len(all_results)}")
        print(f"  Accounts Survived:       {survived}/{len(all_results)}")
        print(f"  Survival Rate:           {survival_rate:.0f}%")
        
        if survived > 0:
            surviving_results = [r for r in all_results if r['survived']]
            
            avg_ending = np.mean([r['ending_balance'] for r in surviving_results])
            max_ending = max([r['ending_balance'] for r in surviving_results])
            min_ending = min([r['ending_balance'] for r in surviving_results])
            avg_return = np.mean([r['total_return_pct'] for r in surviving_results])
            
            print(f"\n💰 SURVIVING ACCOUNTS ONLY:")
            print(f"  Average Ending:          ${avg_ending:.2f}")
            print(f"  Best Ending:             ${max_ending:.2f}")
            print(f"  Worst Ending:            ${min_ending:.2f}")
            print(f"  Average Return:          {avg_return:.1f}%")
        
        total_liquidations = sum(r['liquidations'] for r in all_results)
        avg_drawdown = np.mean([r['max_drawdown_pct'] for r in all_results])
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"  Total Liquidations:      {total_liquidations}")
        print(f"  Average Max Drawdown:    {avg_drawdown:.1f}%")
        
        print("\n" + "=" * 70)
        print("🎯 FINAL VERDICT ON 34X LEVERAGE:")
        print("=" * 70)
        
        if survival_rate >= 80:
            print("✅ POTENTIALLY VIABLE but HIGH RISK")
            print("   • Most accounts survived and grew significantly")
            print("   • Consider starting with lower leverage (20x)")
            print("   • Only use money you can afford to lose")
        elif survival_rate >= 50:
            print("⚠️ VERY RISKY - PROCEED WITH EXTREME CAUTION")
            print("   • 50/50 chance of account survival")
            print("   • Recommend reducing to 15-20x leverage")
            print("   • High reward but equally high risk")
        else:
            print("🚨 NOT RECOMMENDED - TOO RISKY")
            print("   • Most accounts got liquidated")
            print("   • Reduce leverage to 10-15x maximum")
            print("   • Current settings likely to blow account")
        
        print("\n💡 RECOMMENDATIONS:")
        print("  1. Start with 10-15x leverage for safety")
        print("  2. Only increase leverage after consistent profits")
        print("  3. Never risk more than you can afford to lose")
        print("  4. Set stop losses to prevent liquidation")
        print("  5. Consider keeping leverage under 20x")

if __name__ == "__main__":
    main()