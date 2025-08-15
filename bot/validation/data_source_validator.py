#!/usr/bin/env python3
"""
HARD RULE: ONLY OHLCV or websocket subscribed live data is allowed EVER
NO EXCEPTIONS EVER - This is a critical security and data integrity rule
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Allowed data source types - ONLY these are permitted"""
    OHLCV_LIVE = "ohlcv_live"
    WEBSOCKET_SUBSCRIBED = "websocket_subscribed"
    # NO OTHER TYPES ALLOWED - EVER

@dataclass
class DataSourceValidation:
    """Validation result for data source compliance"""
    is_valid: bool
    source_type: Optional[DataSourceType]
    error_message: str
    timestamp: float
    data_hash: str

class DataSourceValidator:
    """
    HARD RULE ENFORCER: Only OHLCV or websocket subscribed live data allowed
    NO EXCEPTIONS EVER - This validator will REJECT any other data sources
    """
    
    def __init__(self):
        self.allowed_sources = {
            DataSourceType.OHLCV_LIVE,
            DataSourceType.WEBSOCKET_SUBSCRIBED
        }
        self.blocked_attempts = []
        self.validation_log = []
        
    def validate_data_source(self, data: Dict[str, Any], source_info: Dict[str, Any]) -> DataSourceValidation:
        """
        HARD RULE: Validate that data comes ONLY from allowed sources
        REJECTS ALL OTHER SOURCES IMMEDIATELY
        """
        timestamp = time.time()
        
        # Extract source information
        source_type_str = source_info.get('source_type', '').lower()
        data_provider = source_info.get('provider', '')
        data_method = source_info.get('method', '')
        
        # HARD RULE CHECK: Only these exact source types allowed
        if source_type_str == 'ohlcv_live':
            source_type = DataSourceType.OHLCV_LIVE
            is_valid = True
            error_message = ""
        elif source_type_str == 'websocket_subscribed':
            source_type = DataSourceType.WEBSOCKET_SUBSCRIBED
            is_valid = True
            error_message = ""
        else:
            # HARD RULE VIOLATION - REJECT IMMEDIATELY
            source_type = None
            is_valid = False
            error_message = f"HARD RULE VIOLATION: Data source '{source_type_str}' is FORBIDDEN. Only OHLCV_LIVE or WEBSOCKET_SUBSCRIBED allowed. Provider: {data_provider}, Method: {data_method}"
            
            # Log violation for security audit
            self._log_violation(source_type_str, data_provider, data_method, data)
            
            # Raise exception - NO EXCEPTIONS TO THIS RULE
            raise ValueError(f"CRITICAL SECURITY VIOLATION: {error_message}")
        
        # Additional validation for allowed sources
        if is_valid:
            is_valid = self._validate_source_integrity(data, source_info, source_type)
            if not is_valid:
                error_message = f"Source integrity check failed for {source_type.value}"
        
        # Create validation result
        data_hash = self._hash_data(data)
        validation = DataSourceValidation(
            is_valid=is_valid,
            source_type=source_type,
            error_message=error_message,
            timestamp=timestamp,
            data_hash=data_hash
        )
        
        # Log validation
        self.validation_log.append(validation)
        
        return validation
    
    def _validate_source_integrity(self, data: Dict[str, Any], source_info: Dict[str, Any], source_type: DataSourceType) -> bool:
        """Validate integrity of allowed data sources"""
        
        if source_type == DataSourceType.OHLCV_LIVE:
            return self._validate_ohlcv_live(data, source_info)
        elif source_type == DataSourceType.WEBSOCKET_SUBSCRIBED:
            return self._validate_websocket_subscribed(data, source_info)
        
        return False
    
    def _validate_ohlcv_live(self, data: Dict[str, Any], source_info: Dict[str, Any]) -> bool:
        """Validate OHLCV live data integrity"""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required OHLCV fields
        for field in required_fields:
            if field not in data:
                logger.error(f"OHLCV validation failed: missing field '{field}'")
                return False
        
        # Check data freshness (must be recent)
        timestamp = data.get('timestamp', 0)
        current_time = time.time()
        if current_time - timestamp > 300:  # 5 minutes max age
            logger.error(f"OHLCV validation failed: data too old ({current_time - timestamp}s)")
            return False
        
        return True
    
    def _validate_websocket_subscribed(self, data: Dict[str, Any], source_info: Dict[str, Any]) -> bool:
        """Validate websocket subscribed data integrity"""
        
        # Must have websocket connection info
        if 'websocket_id' not in source_info:
            logger.error("Websocket validation failed: missing websocket_id")
            return False
        
        # Must have subscription info
        if 'subscription_topic' not in source_info:
            logger.error("Websocket validation failed: missing subscription_topic")
            return False
        
        # Check data freshness
        timestamp = data.get('timestamp', 0)
        current_time = time.time()
        if current_time - timestamp > 60:  # 1 minute max age for websocket
            logger.error(f"Websocket validation failed: data too old ({current_time - timestamp}s)")
            return False
        
        return True
    
    def _hash_data(self, data: Dict[str, Any]) -> str:
        """Create hash of data for tracking"""
        import hashlib
        data_str = str(sorted(data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _log_violation(self, source_type: str, provider: str, method: str, data: Dict[str, Any]):
        """Log security violations for audit"""
        violation = {
            'timestamp': time.time(),
            'source_type': source_type,
            'provider': provider,
            'method': method,
            'data_sample': str(data)[:200],  # First 200 chars
            'severity': 'CRITICAL'
        }
        
        self.blocked_attempts.append(violation)
        logger.critical(f"HARD RULE VIOLATION BLOCKED: {violation}")
        
        # Could add alerting here (email, Slack, etc.)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_validations = len(self.validation_log)
        successful_validations = len([v for v in self.validation_log if v.is_valid])
        blocked_violations = len(self.blocked_attempts)
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'blocked_violations': blocked_violations,
            'compliance_rate': (successful_validations / total_validations * 100) if total_validations > 0 else 0,
            'last_violation': self.blocked_attempts[-1] if self.blocked_attempts else None
        }
    
    def enforce_hard_rule(self, data: Dict[str, Any], source_info: Dict[str, Any]) -> bool:
        """
        ENFORCE HARD RULE: This method will ALWAYS enforce the rule
        Returns True only if data passes ALL validations
        """
        try:
            validation = self.validate_data_source(data, source_info)
            return validation.is_valid
        except ValueError as e:
            # HARD RULE VIOLATION - NEVER ALLOW
            logger.critical(f"HARD RULE ENFORCED: {e}")
            return False
        except Exception as e:
            # Any other error - REJECT for safety
            logger.critical(f"Unexpected error during validation - REJECTING: {e}")
            return False

# Global validator instance
data_source_validator = DataSourceValidator()

def enforce_data_source_rule(data: Dict[str, Any], source_info: Dict[str, Any]) -> bool:
    """
    GLOBAL HARD RULE ENFORCER: Use this function to validate ALL data sources
    NO EXCEPTIONS EVER - Only OHLCV_LIVE or WEBSOCKET_SUBSCRIBED allowed
    """
    return data_source_validator.enforce_hard_rule(data, source_info)

def get_data_source_stats() -> Dict[str, Any]:
    """Get current validation statistics"""
    return data_source_validator.get_validation_stats()
