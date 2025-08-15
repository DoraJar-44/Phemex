#!/usr/bin/env python3
"""Test script for mathematical validation system."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bot.validation.math_validator import MathValidator
from bot.strategy.score import ScoreInputs
import numpy as np


def test_validation_system():
    """Test the validation system with various edge cases."""
    validator = MathValidator()
    print("🧪 Testing Mathematical Validation System\n")
    
    # Test 1: ATR Validation
    print("1️⃣ Testing ATR Calculation...")
    high = [100.5, 101.2, 102.0, 101.8, 103.1]
    low = [99.5, 100.0, 100.8, 100.5, 101.9]
    close = [100.0, 100.8, 101.5, 101.2, 102.5]
    
    atr_result = validator.validate_atr_calculation(high, low, close, 3)
    print(f"   ATR Valid: {atr_result.is_valid}")
    if atr_result.errors:
        print(f"   Errors: {atr_result.errors}")
    if atr_result.warnings:
        print(f"   Warnings: {atr_result.warnings}")
    print(f"   Metrics: {atr_result.metrics}\n")
    
    # Test 2: Predictive Ranges
    print("2️⃣ Testing Predictive Ranges...")
    # Generate synthetic data
    np.random.seed(42)
    n_bars = 250
    base_price = 50000
    prices = [base_price]
    for _ in range(n_bars - 1):
        change = np.random.normal(0, base_price * 0.02)
        prices.append(max(prices[-1] + change, base_price * 0.5))
    
    highs = [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
    
    pr_result = validator.validate_predictive_ranges(highs, lows, prices)
    print(f"   PR Valid: {pr_result.is_valid}")
    if pr_result.errors:
        print(f"   Errors: {pr_result.errors}")
    if pr_result.warnings:
        print(f"   Warnings: {pr_result.warnings}")
    print(f"   Key Metrics: avg={pr_result.metrics.get('avg', 0):.2f}, spread={pr_result.metrics.get('range_spread', 0):.2f}\n")
    
    # Test 3: Scoring System
    print("3️⃣ Testing Scoring System...")
    
    # Normal case
    score_input = ScoreInputs(
        avg=50000,
        r1=50500,
        r2=51000,
        s1=49500,
        s2=49000,
        close=50200,
        open=50100,
        bias_up_conf=0.7,
        bounce_prob=0.5,
        bull_div=True
    )
    
    long_result = validator.validate_scoring_system(score_input, "long")
    short_result = validator.validate_scoring_system(score_input, "short")
    
    print(f"   Long Score Valid: {long_result.is_valid}, Score: {long_result.metrics.get('total_score', 0)}")
    print(f"   Short Score Valid: {short_result.is_valid}, Score: {short_result.metrics.get('total_score', 0)}")
    
    # Edge case - extreme values
    print("\n4️⃣ Testing Edge Cases...")
    extreme_input = ScoreInputs(
        avg=float('inf'),
        r1=50500,
        r2=51000,
        s1=49500,
        s2=49000,
        close=50200,
        open=50100,
        bias_up_conf=1.5,  # Invalid range
        bounce_prob=1.0,   # Invalid range
    )
    
    extreme_result = validator.validate_scoring_system(extreme_input, "long")
    print(f"   Extreme Case Valid: {extreme_result.is_valid}")
    if extreme_result.errors:
        print(f"   Errors: {extreme_result.errors}")
        
    # Test 5: Clamp function
    print("\n5️⃣ Testing Clamp Function...")
    clamp_result = validator.validate_clamp_function()
    print(f"   Clamp Valid: {clamp_result.is_valid}")
    if clamp_result.errors:
        print(f"   Errors: {clamp_result.errors}")
        
    # Test 6: Full validation
    print("\n6️⃣ Running Full Validation...")
    full_results = validator.run_full_validation(highs, lows, prices, score_input, "long")
    
    print("   Component Results:")
    for component, result in full_results.items():
        status = "✅" if result.is_valid else "❌"
        print(f"      {component}: {status}")
        if result.errors:
            print(f"         Errors: {result.errors[:2]}")  # Show first 2 errors
            
    print("\n🎯 Validation Testing Complete!")
    return all(result.is_valid for result in full_results.values())


if __name__ == "__main__":
    success = test_validation_system()
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    sys.exit(0 if success else 1)
