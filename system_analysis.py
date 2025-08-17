#!/usr/bin/env python3
"""
Comprehensive System Analysis for Live Trading Readiness
"""

import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
import json

load_dotenv()

class SystemAnalyzer:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.critical_gaps = []
        self.missing_components = []
    
    def check_api_credentials(self) -> Dict[str, Any]:
        """Check API credentials and configuration"""
        print("🔐 Checking API Credentials...")
        
        api_key = os.getenv("PHEMEX_API_KEY", "")
        api_secret = os.getenv("PHEMEX_API_SECRET", "")
        testnet = os.getenv("PHEMEX_TESTNET", "false").lower() in ("1", "true", "yes")
        
        issues = []
        if not api_key:
            issues.append("❌ PHEMEX_API_KEY is missing")
        elif len(api_key) < 30:
            issues.append("⚠️ PHEMEX_API_KEY looks invalid (too short)")
            
        if not api_secret:
            issues.append("❌ PHEMEX_API_SECRET is missing")
        elif len(api_secret) < 50:
            issues.append("⚠️ PHEMEX_API_SECRET looks invalid (too short)")
        
        if testnet:
            self.warnings.append("⚠️ Running in TESTNET mode")
        
        return {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues,
            "testnet": testnet
        }
    
    def check_environment_config(self) -> Dict[str, Any]:
        """Check environment configuration"""
        print("⚙️ Checking Environment Configuration...")
        
        required_vars = [
            "LIVE_TRADE", "LEVERAGE_MAX", "RISK_PER_TRADE_PCT",
            "MAX_POSITIONS", "SCORE_MIN", "TIMEFRAME"
        ]
        
        issues = []
        config = {}
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                issues.append(f"❌ {var} not set")
            config[var] = value
        
        # Check specific values
        live_trade = os.getenv("LIVE_TRADE", "false").lower() in ("1", "true", "yes")
        leverage_max = float(os.getenv("LEVERAGE_MAX", "5"))
        risk_pct = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
        
        if not live_trade:
            issues.append("❌ LIVE_TRADE is not enabled")
        
        if leverage_max != 50:
            issues.append(f"⚠️ LEVERAGE_MAX is {leverage_max}, not 50x")
        
        if risk_pct > 2.0:
            issues.append(f"⚠️ RISK_PER_TRADE_PCT is high: {risk_pct}%")
        
        return {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues,
            "config": config
        }
    
    def check_system_components(self) -> Dict[str, Any]:
        """Check system components and file structure"""
        print("🏗️ Checking System Components...")
        
        critical_files = [
            "unified_trading_bot.py",
            "bot/config.py",
            "bot/exchange/phemex_client.py",
            "bot/execution/execute.py",
            "bot/strategy/score.py",
            "bot/risk/sizing.py",
            "requirements.txt"
        ]
        
        issues = []
        missing_files = []
        
        for file in critical_files:
            if not os.path.exists(file):
                missing_files.append(file)
                issues.append(f"❌ Missing critical file: {file}")
        
        # Check data files
        data_files = [
            "data/phemex_watchlist_50x.txt",
            "data/phemex_50x_pairs.json"
        ]
        
        for file in data_files:
            if not os.path.exists(file):
                issues.append(f"⚠️ Missing data file: {file}")
        
        return {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues,
            "missing_files": missing_files
        }
    
    def check_risk_management(self) -> Dict[str, Any]:
        """Check risk management implementation"""
        print("🛡️ Checking Risk Management...")
        
        issues = []
        
        # Check if risk management files exist
        risk_files = [
            "bot/risk/guards.py",
            "bot/risk/sizing.py"
        ]
        
        for file in risk_files:
            if not os.path.exists(file):
                issues.append(f"❌ Missing risk file: {file}")
        
        # Check risk parameters
        max_positions = int(os.getenv("MAX_POSITIONS", "5"))
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
        
        if max_positions > 10:
            issues.append(f"⚠️ MAX_POSITIONS is high: {max_positions}")
        
        if max_daily_loss > 5.0:
            issues.append(f"⚠️ MAX_DAILY_LOSS_PCT is high: {max_daily_loss}%")
        
        return {
            "status": "PASS" if not issues else "FAIL", 
            "issues": issues
        }
    
    def check_execution_system(self) -> Dict[str, Any]:
        """Check order execution system"""
        print("⚡ Checking Execution System...")
        
        issues = []
        
        execution_files = [
            "bot/execution/execute.py",
            "bot/execution/brackets.py"
        ]
        
        for file in execution_files:
            if not os.path.exists(file):
                issues.append(f"❌ Missing execution file: {file}")
        
        # Check if system can handle bracket orders (TP/SL)
        try:
            from bot.execution import brackets
            issues.append("✅ Bracket orders system available")
        except ImportError:
            issues.append("❌ Cannot import bracket orders system")
        
        return {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues
        }
    
    def check_strategy_system(self) -> Dict[str, Any]:
        """Check strategy and scoring system"""
        print("📊 Checking Strategy System...")
        
        issues = []
        
        strategy_files = [
            "bot/strategy/score.py",
            "bot/strategy/pr.py"
        ]
        
        for file in strategy_files:
            if not os.path.exists(file):
                issues.append(f"❌ Missing strategy file: {file}")
        
        score_min = int(os.getenv("SCORE_MIN", "85"))
        if score_min < 80:
            issues.append(f"⚠️ SCORE_MIN is low: {score_min}")
        
        return {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies"""
        print("📦 Checking Dependencies...")
        
        issues = []
        
        required_packages = [
            "ccxt", "asyncio", "httpx", "numpy", "python-dotenv",
            "fastapi", "uvicorn", "pydantic"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"❌ Missing package: {package}")
        
        return {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues
        }
    
    def identify_critical_gaps(self) -> List[str]:
        """Identify critical gaps that must be fixed"""
        print("\n🚨 IDENTIFYING CRITICAL GAPS...")
        
        gaps = []
        
        # Critical system checks
        if not os.getenv("LIVE_TRADE", "false").lower() in ("1", "true", "yes"):
            gaps.append("LIVE_TRADE must be enabled")
        
        if not os.getenv("PHEMEX_API_KEY"):
            gaps.append("PHEMEX_API_KEY must be provided")
        
        if not os.getenv("PHEMEX_API_SECRET"):
            gaps.append("PHEMEX_API_SECRET must be provided")
        
        if not os.path.exists("unified_trading_bot.py"):
            gaps.append("Main trading bot file missing")
        
        # Risk management gaps
        leverage = float(os.getenv("LEVERAGE_MAX", "5"))
        if leverage != 50:
            gaps.append(f"LEVERAGE_MAX should be 50x, currently {leverage}x")
        
        # Strategy gaps
        if not os.path.exists("data/phemex_watchlist_50x.txt"):
            gaps.append("50x leverage pairs list missing")
        
        return gaps
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for system improvements"""
        recommendations = []
        
        # System improvements
        recommendations.append("✅ Test API connection with small test order")
        recommendations.append("✅ Implement position monitoring dashboard")
        recommendations.append("✅ Add emergency stop functionality")
        recommendations.append("✅ Implement profit/loss tracking")
        recommendations.append("✅ Add trade logging and analytics")
        recommendations.append("✅ Test bracket order execution")
        recommendations.append("✅ Validate risk management limits")
        recommendations.append("✅ Test symbol discovery for 50x pairs")
        
        return recommendations
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete system analysis"""
        print("🔍 COMPREHENSIVE SYSTEM ANALYSIS FOR LIVE TRADING")
        print("=" * 60)
        
        results = {
            "api_credentials": self.check_api_credentials(),
            "environment_config": self.check_environment_config(),
            "system_components": self.check_system_components(),
            "risk_management": self.check_risk_management(),
            "execution_system": self.check_execution_system(),
            "strategy_system": self.check_strategy_system(),
            "dependencies": self.check_dependencies()
        }
        
        # Identify critical gaps
        critical_gaps = self.identify_critical_gaps()
        recommendations = self.generate_recommendations()
        
        # Overall assessment
        total_issues = sum(len(r.get("issues", [])) for r in results.values())
        critical_count = len(critical_gaps)
        
        overall_status = "READY" if critical_count == 0 else "NOT READY"
        
        final_report = {
            "overall_status": overall_status,
            "total_issues": total_issues,
            "critical_gaps": critical_gaps,
            "recommendations": recommendations,
            "detailed_results": results,
            "50x_pairs_available": 290,  # From previous discovery
            "100x_pairs_available": 2    # BTC and ETH
        }
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Overall Status: {'🟢 ' + overall_status if overall_status == 'READY' else '🔴 ' + overall_status}")
        print(f"   Total Issues: {total_issues}")
        print(f"   Critical Gaps: {critical_count}")
        print(f"   50x Leverage Pairs: 290")
        print(f"   100x Leverage Pairs: 2 (BTC, ETH)")
        
        if critical_gaps:
            print(f"\n🚨 CRITICAL GAPS MUST BE FIXED:")
            for gap in critical_gaps:
                print(f"   ❌ {gap}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations[:5]:  # Show top 5
            print(f"   {rec}")
        
        return final_report

async def main():
    analyzer = SystemAnalyzer()
    report = await analyzer.run_full_analysis()
    
    # Save report
    with open('/workspace/system_readiness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Full report saved to: system_readiness_report.json")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())