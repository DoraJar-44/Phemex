#!/usr/bin/env python3
"""
HARD RULES CONFIGURATION: System-level enforcement of critical rules
NO EXCEPTIONS EVER - These rules are absolute and cannot be bypassed
"""

import os
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class RuleSeverity(Enum):
    """Rule severity levels"""
    CRITICAL = "CRITICAL"      # NO EXCEPTIONS EVER
    HIGH = "HIGH"              # NO EXCEPTIONS EVER  
    MEDIUM = "MEDIUM"          # NO EXCEPTIONS EVER
    LOW = "LOW"                # NO EXCEPTIONS EVER

@dataclass
class HardRule:
    """Definition of a hard rule that cannot be violated"""
    name: str
    description: str
    severity: RuleSeverity
    enforcement_level: str
    allowed_values: List[str] = field(default_factory=list)
    blocked_values: List[str] = field(default_factory=list)
    is_active: bool = True
    violation_count: int = 0
    
    def __post_init__(self):
        if self.severity in [RuleSeverity.CRITICAL, RuleSeverity.HIGH]:
            self.is_active = True  # Critical rules are always active

class HardRulesConfig:
    """
    HARD RULES ENFORCER: System-level configuration that enforces critical rules
    NO EXCEPTIONS EVER - These rules are absolute
    """
    
    def __init__(self):
        self.rules: Dict[str, HardRule] = {}
        self.violation_log: List[Dict[str, Any]] = []
        self._initialize_hard_rules()
    
    def _initialize_hard_rules(self):
        """Initialize all hard rules - NO EXCEPTIONS EVER"""
        
        # RULE 1: DATA SOURCE ENFORCEMENT - CRITICAL
        self.rules['data_source_only'] = HardRule(
            name="Data Source Enforcement",
            description="ONLY OHLCV or websocket subscribed live data is allowed EVER. NO EXCEPTIONS EVER.",
            severity=RuleSeverity.CRITICAL,
            enforcement_level="SYSTEM_LEVEL",
            allowed_values=["ohlcv_live", "websocket_subscribed"],
            blocked_values=["historical_data", "simulated_data", "backtest_data", "paper_trading_data", "delayed_data", "cached_data"]
        )
        
        # RULE 2: LIVE DATA FRESHNESS - CRITICAL
        self.rules['data_freshness'] = HardRule(
            name="Data Freshness Enforcement",
            description="All data must be live and fresh. NO EXCEPTIONS EVER.",
            severity=RuleSeverity.CRITICAL,
            enforcement_level="SYSTEM_LEVEL",
            allowed_values=["live", "real_time", "current"],
            blocked_values=["stale", "old", "delayed", "cached", "historical"]
        )
        
        # RULE 3: WEBSOCKET CONNECTION - HIGH
        self.rules['websocket_connection'] = HardRule(
            name="Websocket Connection Validation",
            description="Websocket data must come from active, subscribed connections. NO EXCEPTIONS EVER.",
            severity=RuleSeverity.HIGH,
            enforcement_level="CONNECTION_LEVEL",
            allowed_values=["active_connection", "subscribed_topic", "live_stream"],
            blocked_values=["disconnected", "unsubscribed", "mock_connection", "test_connection"]
        )
        
        # RULE 4: OHLCV INTEGRITY - HIGH
        self.rules['ohlcv_integrity'] = HardRule(
            name="OHLCV Data Integrity",
            description="OHLCV data must contain all required fields and be properly formatted. NO EXCEPTIONS EVER.",
            severity=RuleSeverity.HIGH,
            enforcement_level="DATA_LEVEL",
            allowed_values=["complete_ohlcv", "valid_timestamp", "proper_scaling"],
            blocked_values=["incomplete_data", "invalid_timestamp", "wrong_scaling", "corrupted_data"]
        )
        
        # RULE 5: PROVIDER VALIDATION - MEDIUM
        self.rules['provider_validation'] = HardRule(
            name="Data Provider Validation",
            description="Only authorized data providers are allowed. NO EXCEPTIONS EVER.",
            severity=RuleSeverity.MEDIUM,
            enforcement_level="PROVIDER_LEVEL",
            allowed_values=["phemex_official", "ccxt_phemex", "direct_api"],
            blocked_values=["unauthorized_provider", "third_party", "unverified_source"]
        )
    
    def enforce_rule(self, rule_name: str, data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        ENFORCE HARD RULE: This method will ALWAYS enforce the specified rule
        NO EXCEPTIONS EVER - Returns False if rule violated
        """
        if rule_name not in self.rules:
            logger.critical(f"Unknown rule '{rule_name}' - REJECTING for safety")
            return False
        
        rule = self.rules[rule_name]
        
        if not rule.is_active:
            logger.warning(f"Rule '{rule_name}' is inactive - but still enforcing for safety")
        
        # Extract relevant values from data and context
        data_values = self._extract_values(data, context)
        
        # Check against allowed values
        for value in data_values:
            if value in rule.allowed_values:
                continue
            
            if value in rule.blocked_values:
                # RULE VIOLATION - LOG AND REJECT
                self._log_violation(rule, value, data, context)
                return False
            
            # Unknown value - REJECT for safety (NO EXCEPTIONS)
            logger.critical(f"Unknown value '{value}' for rule '{rule_name}' - REJECTING for safety")
            self._log_violation(rule, value, data, context)
            return False
        
        return True
    
    def _extract_values(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract relevant values for rule validation"""
        values = []
        
        # Extract from data
        if 'source_type' in data:
            values.append(data['source_type'])
        if 'provider' in data:
            values.append(data['provider'])
        if 'method' in data:
            values.append(data['method'])
        if 'timestamp' in data:
            values.append('live' if self._is_fresh_timestamp(data['timestamp']) else 'stale')
        
        # Extract from context
        if 'connection_status' in context:
            values.append(context['connection_status'])
        if 'subscription_status' in context:
            values.append(context['subscription_status'])
        if 'data_quality' in context:
            values.append(context['data_quality'])
        
        return values
    
    def _is_fresh_timestamp(self, timestamp: Any) -> bool:
        """Check if timestamp is fresh (within acceptable range)"""
        import time
        try:
            ts = float(timestamp)
            current_time = time.time()
            return (current_time - ts) <= 300  # 5 minutes max
        except (ValueError, TypeError):
            return False
    
    def _log_violation(self, rule: HardRule, value: str, data: Dict[str, Any], context: Dict[str, Any]):
        """Log rule violation for audit and security"""
        import time
        violation = {
            'timestamp': time.time(),
            'rule_name': rule.name,
            'rule_severity': rule.severity.value,
            'violated_value': value,
            'data_sample': str(data)[:200],
            'context_sample': str(context)[:200],
            'severity': 'CRITICAL'
        }
        
        self.violation_log.append(violation)
        rule.violation_count += 1
        
        logger.critical(f"HARD RULE VIOLATION: {rule.name} - Value '{value}' is FORBIDDEN")
        logger.critical(f"Violation details: {violation}")
        
        # Could add alerting here (email, Slack, etc.)
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive compliance report"""
        total_violations = sum(rule.violation_count for rule in self.rules.values())
        active_rules = len([rule for rule in self.rules.values() if rule.is_active])
        
        return {
            'total_rules': len(self.rules),
            'active_rules': active_rules,
            'total_violations': total_violations,
            'rule_details': {
                name: {
                    'violation_count': rule.violation_count,
                    'is_active': rule.is_active,
                    'severity': rule.severity.value
                }
                for name, rule in self.rules.items()
            },
            'recent_violations': self.violation_log[-10:] if self.violation_log else [],
            'compliance_status': 'COMPLIANT' if total_violations == 0 else 'VIOLATIONS_DETECTED'
        }
    
    def enforce_all_rules(self, data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        ENFORCE ALL HARD RULES: Validates against every active rule
        NO EXCEPTIONS EVER - Returns False if ANY rule violated
        """
        for rule_name in self.rules:
            if not self.enforce_rule(rule_name, data, context):
                return False
        return True

# Global hard rules configuration
hard_rules_config = HardRulesConfig()

def enforce_hard_rules(data: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    GLOBAL HARD RULES ENFORCER: Use this function to validate against ALL hard rules
    NO EXCEPTIONS EVER - Returns False if ANY rule violated
    """
    return hard_rules_config.enforce_all_rules(data, context)

def get_hard_rules_compliance() -> Dict[str, Any]:
    """Get current hard rules compliance status"""
    return hard_rules_config.get_compliance_report()
