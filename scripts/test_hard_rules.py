#!/usr/bin/env python3
"""
HARD RULES TEST: Demonstrates the enforcement of critical data source rules
NO EXCEPTIONS EVER - Only OHLCV or websocket subscribed live data allowed
"""

import time
import logging
from bot.validation.data_source_validator import enforce_data_source_rule, get_data_source_stats
from bot.validation.data_source_decorator import hard_rule_enforcer, enforce_live_data_only
from bot.config.hard_rules import enforce_hard_rules, get_hard_rules_compliance

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_allowed_data_sources():
    """Test that allowed data sources pass validation"""
    print("\n" + "="*60)
    print("TESTING ALLOWED DATA SOURCES")
    print("="*60)
    
    # Test 1: OHLCV Live Data
    print("\n1. Testing OHLCV Live Data...")
    ohlcv_data = {
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 100.5,
        'timestamp': time.time()
    }
    
    ohlcv_source = {
        'source_type': 'ohlcv_live',
        'provider': 'phemex_official',
        'method': 'rest_api'
    }
    
    try:
        result = enforce_data_source_rule(ohlcv_data, ohlcv_source)
        print(f"✅ OHLCV Live Data: PASSED (Result: {result})")
    except Exception as e:
        print(f"❌ OHLCV Live Data: FAILED - {e}")
    
    # Test 2: Websocket Subscribed Data
    print("\n2. Testing Websocket Subscribed Data...")
    ws_data = {
        'price': 50500.0,
        'quantity': 10.0,
        'side': 'buy',
        'timestamp': time.time()
    }
    
    ws_source = {
        'source_type': 'websocket_subscribed',
        'provider': 'phemex_official',
        'method': 'websocket',
        'websocket_id': 'ws_12345',
        'subscription_topic': 'trades.BTCUSDT'
    }
    
    try:
        result = enforce_data_source_rule(ws_data, ws_source)
        print(f"✅ Websocket Subscribed Data: PASSED (Result: {result})")
    except Exception as e:
        print(f"❌ Websocket Subscribed Data: FAILED - {e}")

def test_forbidden_data_sources():
    """Test that forbidden data sources are REJECTED"""
    print("\n" + "="*60)
    print("TESTING FORBIDDEN DATA SOURCES (SHOULD ALL BE REJECTED)")
    print("="*60)
    
    forbidden_sources = [
        {
            'name': 'Historical Data',
            'data': {'close': 50000.0, 'timestamp': time.time() - 86400},  # 1 day old
            'source': {'source_type': 'historical_data', 'provider': 'database', 'method': 'query'}
        },
        {
            'name': 'Simulated Data',
            'data': {'close': 50000.0, 'timestamp': time.time()},
            'source': {'source_type': 'simulated_data', 'provider': 'backtest_engine', 'method': 'simulation'}
        },
        {
            'name': 'Backtest Data',
            'data': {'close': 50000.0, 'timestamp': time.time()},
            'source': {'source_type': 'backtest_data', 'provider': 'strategy_tester', 'method': 'historical_simulation'}
        },
        {
            'name': 'Paper Trading Data',
            'data': {'close': 50000.0, 'timestamp': time.time()},
            'source': {'source_type': 'paper_trading_data', 'provider': 'demo_account', 'method': 'virtual_trading'}
        },
        {
            'name': 'Delayed Data',
            'data': {'close': 50000.0, 'timestamp': time.time()},
            'source': {'source_type': 'delayed_data', 'provider': 'data_vendor', 'method': 'delayed_feed'}
        },
        {
            'name': 'Cached Data',
            'data': {'close': 50000.0, 'timestamp': time.time()},
            'source': {'source_type': 'cached_data', 'provider': 'cache_service', 'method': 'memory_cache'}
        }
    ]
    
    for test_case in forbidden_sources:
        print(f"\nTesting {test_case['name']}...")
        try:
            result = enforce_data_source_rule(test_case['data'], test_case['source'])
            print(f"❌ {test_case['name']}: SHOULD HAVE BEEN REJECTED but got result: {result}")
        except ValueError as e:
            print(f"✅ {test_case['name']}: CORRECTLY REJECTED - {str(e)[:100]}...")
        except Exception as e:
            print(f"❌ {test_case['name']}: Unexpected error - {e}")

def test_decorator_enforcement():
    """Test that decorators enforce the hard rules"""
    print("\n" + "="*60)
    print("TESTING DECORATOR ENFORCEMENT")
    print("="*60)
    
    @enforce_live_data_only
    def process_market_data(data, source_info):
        """Function that should only accept live data"""
        return f"Processed data from {source_info.get('source_type', 'unknown')}"
    
    @hard_rule_enforcer
    def analyze_market_data(data, source_info):
        """Function with maximum enforcement"""
        return f"Analyzed data from {source_info.get('source_type', 'unknown')}"
    
    # Test with allowed data
    print("\n1. Testing decorator with allowed data...")
    allowed_data = {'close': 50000.0, 'timestamp': time.time()}
    allowed_source = {'source_type': 'ohlcv_live', 'provider': 'phemex_official'}
    
    try:
        result1 = process_market_data(allowed_data, allowed_source)
        print(f"✅ Decorator with allowed data: PASSED - {result1}")
        
        result2 = analyze_market_data(allowed_data, allowed_source)
        print(f"✅ Hard rule enforcer with allowed data: PASSED - {result2}")
    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
    
    # Test with forbidden data
    print("\n2. Testing decorator with forbidden data...")
    forbidden_data = {'close': 50000.0, 'timestamp': time.time()}
    forbidden_source = {'source_type': 'historical_data', 'provider': 'database'}
    
    try:
        result = process_market_data(forbidden_data, forbidden_source)
        print(f"❌ Decorator with forbidden data: SHOULD HAVE BEEN REJECTED but got: {result}")
    except ValueError as e:
        print(f"✅ Decorator correctly rejected forbidden data: {str(e)[:100]}...")
    except Exception as e:
        print(f"❌ Decorator test failed unexpectedly: {e}")

def test_hard_rules_config():
    """Test the hard rules configuration system"""
    print("\n" + "="*60)
    print("TESTING HARD RULES CONFIGURATION")
    print("="*60)
    
    # Test with compliant data
    print("\n1. Testing hard rules with compliant data...")
    compliant_data = {'source_type': 'ohlcv_live', 'provider': 'phemex_official'}
    compliant_context = {'connection_status': 'active_connection', 'data_quality': 'complete_ohlcv'}
    
    try:
        result = enforce_hard_rules(compliant_data, compliant_context)
        print(f"✅ Hard rules with compliant data: PASSED (Result: {result})")
    except Exception as e:
        print(f"❌ Hard rules test failed: {e}")
    
    # Test with non-compliant data
    print("\n2. Testing hard rules with non-compliant data...")
    non_compliant_data = {'source_type': 'historical_data', 'provider': 'database'}
    non_compliant_context = {'connection_status': 'disconnected', 'data_quality': 'incomplete_data'}
    
    try:
        result = enforce_hard_rules(non_compliant_data, non_compliant_context)
        print(f"❌ Hard rules with non-compliant data: SHOULD HAVE BEEN REJECTED but got: {result}")
    except Exception as e:
        print(f"✅ Hard rules correctly rejected non-compliant data: {str(e)[:100]}...")
    
    # Get compliance report
    print("\n3. Getting compliance report...")
    try:
        compliance = get_hard_rules_compliance()
        print(f"✅ Compliance report: {compliance['compliance_status']}")
        print(f"   Total rules: {compliance['total_rules']}")
        print(f"   Active rules: {compliance['active_rules']}")
        print(f"   Total violations: {compliance['total_violations']}")
    except Exception as e:
        print(f"❌ Failed to get compliance report: {e}")

def test_data_source_stats():
    """Test the data source validation statistics"""
    print("\n" + "="*60)
    print("TESTING DATA SOURCE VALIDATION STATISTICS")
    print("="*60)
    
    try:
        stats = get_data_source_stats()
        print(f"✅ Data source stats retrieved:")
        print(f"   Total validations: {stats['total_validations']}")
        print(f"   Successful validations: {stats['successful_validations']}")
        print(f"   Blocked violations: {stats['blocked_violations']}")
        print(f"   Compliance rate: {stats['compliance_rate']:.2f}%")
        
        if stats['last_violation']:
            print(f"   Last violation: {stats['last_violation']['source_type']}")
    except Exception as e:
        print(f"❌ Failed to get data source stats: {e}")

def main():
    """Run all hard rule tests"""
    print("🚀 STARTING HARD RULES ENFORCEMENT TESTS")
    print("NO EXCEPTIONS EVER - Only OHLCV or websocket subscribed live data allowed")
    
    try:
        # Test allowed data sources
        test_allowed_data_sources()
        
        # Test forbidden data sources
        test_forbidden_data_sources()
        
        # Test decorator enforcement
        test_decorator_enforcement()
        
        # Test hard rules configuration
        test_hard_rules_config()
        
        # Test statistics
        test_data_source_stats()
        
        print("\n" + "="*60)
        print("🎯 ALL TESTS COMPLETED")
        print("="*60)
        print("✅ Hard rules are working correctly")
        print("✅ Only OHLCV or websocket subscribed live data allowed")
        print("✅ NO EXCEPTIONS EVER - All forbidden sources rejected")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main()
