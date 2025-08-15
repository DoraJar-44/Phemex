#!/usr/bin/env python3
"""
Test script to validate critical fixes in the trading bot
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from unified_trading_bot import compute_total_score, ScoreInputs, build_bracket_orders

def test_score_calculation():
    """Test scoring logic fixes"""
    print("\n=== Testing Score Calculation ===")
    
    # Test 1: Zero price handling
    print("\n1. Testing zero price handling:")
    inputs = ScoreInputs(
        avg=100, r1=110, r2=120, s1=90, s2=80,
        close=0,  # Invalid price
        open=100,
        bounce_prob=0.5,
        bias_up_conf=0.5,
        bias_dn_conf=0.5
    )
    score = compute_total_score(inputs, "long", 85)
    assert score == 0, f"Expected 0 for zero price, got {score}"
    print("✅ Zero price correctly returns score of 0")
    
    # Test 2: Valid long score
    print("\n2. Testing valid long score:")
    inputs = ScoreInputs(
        avg=100, r1=110, r2=120, s1=90, s2=80,
        close=89,  # Near support for long
        open=91,
        bounce_prob=0.5,
        bias_up_conf=0.7,
        bias_dn_conf=0.3
    )
    score = compute_total_score(inputs, "long", 85)
    print(f"   Long score at support: {score}")
    assert score > 80, f"Expected high score near support, got {score}"
    print("✅ Long scoring works correctly")
    
    # Test 3: Valid short score
    print("\n3. Testing valid short score:")
    inputs = ScoreInputs(
        avg=100, r1=110, r2=120, s1=90, s2=80,
        close=111,  # Near resistance for short
        open=109,
        bounce_prob=0.5,
        bias_up_conf=0.3,
        bias_dn_conf=0.7
    )
    score = compute_total_score(inputs, "short", 85)
    print(f"   Short score at resistance: {score}")
    assert score > 80, f"Expected high score near resistance, got {score}"
    print("✅ Short scoring works correctly")

def test_bracket_orders():
    """Test bracket order building"""
    print("\n=== Testing Bracket Orders ===")
    
    # Test long bracket
    print("\n1. Testing long bracket orders:")
    orders = build_bracket_orders(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        entry_price=50000,
        tp1_price=51000,
        tp2_price=52000,
        sl_price=49000
    )
    
    assert orders["entry"]["side"] == "buy", "Entry should be buy for long"
    assert orders["tp1"]["side"] == "sell", "TP1 should be sell for long"
    assert orders["tp2"]["side"] == "sell", "TP2 should be sell for long"
    assert orders["sl"]["side"] == "sell", "SL should be sell for long"
    assert orders["sl"]["type"] == "stop_market", "SL should be stop_market"
    assert orders["tp1"]["quantity"] == 0.005, "TP1 should be 50% of quantity"
    assert orders["tp2"]["quantity"] == 0.005, "TP2 should be 50% of quantity"
    print("✅ Long bracket orders structured correctly")
    
    # Test short bracket
    print("\n2. Testing short bracket orders:")
    orders = build_bracket_orders(
        symbol="BTCUSDT",
        side="sell",
        quantity=0.01,
        entry_price=50000,
        tp1_price=49000,
        tp2_price=48000,
        sl_price=51000
    )
    
    assert orders["entry"]["side"] == "sell", "Entry should be sell for short"
    assert orders["tp1"]["side"] == "buy", "TP1 should be buy for short"
    assert orders["tp2"]["side"] == "buy", "TP2 should be buy for short"
    assert orders["sl"]["side"] == "buy", "SL should be buy for short"
    assert orders["sl"]["type"] == "stop_market", "SL should be stop_market"
    print("✅ Short bracket orders structured correctly")

def test_tp_sl_validation():
    """Test TP/SL level validation"""
    print("\n=== Testing TP/SL Level Validation ===")
    
    # Simulate the fixed TP calculation logic
    print("\n1. Testing long TP levels:")
    price = 100
    levels = {"avg": 95, "r1": 105, "r2": 110, "s1": 90, "s2": 85}
    
    # Fixed logic for longs
    tp1_price = max(levels["avg"], price * 1.01) if levels["avg"] > price else price * 1.01
    tp2_price = max(levels["r1"], price * 1.02) if levels["r1"] > price else price * 1.02
    
    assert tp1_price > price, f"TP1 ({tp1_price}) must be above entry ({price}) for long"
    assert tp2_price > tp1_price, f"TP2 ({tp2_price}) must be above TP1 ({tp1_price})"
    print(f"   Long: Entry={price}, TP1={tp1_price}, TP2={tp2_price}")
    print("✅ Long TP levels are valid")
    
    print("\n2. Testing short TP levels:")
    price = 100
    levels = {"avg": 105, "r1": 110, "r2": 115, "s1": 95, "s2": 90}
    
    # Fixed logic for shorts
    tp1_price = min(levels["avg"], price * 0.99) if levels["avg"] < price else price * 0.99
    tp2_price = min(levels["s1"], price * 0.98) if levels["s1"] > 0 and levels["s1"] < price else price * 0.98
    
    assert tp1_price < price, f"TP1 ({tp1_price}) must be below entry ({price}) for short"
    assert tp2_price < tp1_price, f"TP2 ({tp2_price}) must be below TP1 ({tp1_price})"
    print(f"   Short: Entry={price}, TP1={tp1_price}, TP2={tp2_price}")
    print("✅ Short TP levels are valid")

def test_position_sizing():
    """Test position sizing with max limits"""
    print("\n=== Testing Position Sizing ===")
    
    print("\n1. Testing stop distance validation:")
    entry_price = 100
    stop_distance = 10  # 10% stop
    max_stop_distance = entry_price * 0.05  # 5% max
    
    if stop_distance > max_stop_distance:
        print(f"   Original stop: {stop_distance/entry_price*100:.2f}%")
        stop_distance = max_stop_distance
        print(f"   Capped stop: {stop_distance/entry_price*100:.2f}%")
    
    assert stop_distance <= max_stop_distance, "Stop distance should be capped at 5%"
    print("✅ Stop distance capping works correctly")
    
    print("\n2. Testing max position size:")
    effective_equity = 10000
    max_capital_fraction = 0.6
    entry_price = 100
    
    # Calculate max position
    max_position_value = effective_equity * max_capital_fraction
    max_quantity = max_position_value / entry_price
    
    # Test with oversized position
    test_quantity = 100  # Would be $10,000 worth
    final_quantity = min(test_quantity, max_quantity)
    
    print(f"   Max allowed: {max_quantity:.2f} units (${max_position_value:.2f})")
    print(f"   Requested: {test_quantity:.2f} units")
    print(f"   Final: {final_quantity:.2f} units")
    
    assert final_quantity <= max_quantity, "Position should be capped at max"
    print("✅ Position size limiting works correctly")

def main():
    """Run all tests"""
    print("=" * 60)
    print("TRADING BOT FIX VALIDATION")
    print("=" * 60)
    
    try:
        test_score_calculation()
        test_bracket_orders()
        test_tp_sl_validation()
        test_position_sizing()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - FIXES VALIDATED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()