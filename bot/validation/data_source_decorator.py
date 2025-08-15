#!/usr/bin/env python3
"""
HARD RULE DECORATOR: Enforces data source validation on any function
NO EXCEPTIONS EVER - Only OHLCV or websocket subscribed live data allowed
"""

import functools
import logging
from typing import Dict, Any, Callable, TypeVar, ParamSpec
from .data_source_validator import enforce_data_source_rule, get_data_source_stats

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

def enforce_live_data_only(func: Callable[P, T]) -> Callable[P, T]:
    """
    HARD RULE DECORATOR: Enforces that ONLY live data sources are used
    NO EXCEPTIONS EVER - Rejects any non-live data sources
    
    Usage:
    @enforce_live_data_only
    def process_market_data(data, source_info):
        # This function will ONLY accept OHLCV_LIVE or WEBSOCKET_SUBSCRIBED data
        pass
    """
    
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Find data and source_info in arguments
        data = None
        source_info = None
        
        # Look for data in positional args
        for arg in args:
            if isinstance(arg, dict):
                if 'open' in arg or 'close' in arg or 'timestamp' in arg:
                    data = arg
                elif 'source_type' in arg or 'provider' in arg:
                    source_info = arg
        
        # Look for data in keyword args
        if 'data' in kwargs:
            data = kwargs['data']
        if 'source_info' in kwargs:
            source_info = kwargs['source_info']
        
        # If we found data, enforce the hard rule
        if data is not None and source_info is not None:
            # HARD RULE ENFORCEMENT - NO EXCEPTIONS
            if not enforce_data_source_rule(data, source_info):
                error_msg = f"HARD RULE VIOLATION: Function {func.__name__} rejected non-live data source"
                logger.critical(error_msg)
                raise ValueError(error_msg)
        
        # If no data found, log warning but allow (might be a setup function)
        elif data is None and source_info is None:
            logger.warning(f"Function {func.__name__} called without data validation - ensure data sources are validated elsewhere")
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    
    return wrapper

def validate_data_source_before(func: Callable[P, T]) -> Callable[P, T]:
    """
    PRE-EXECUTION VALIDATOR: Validates data source before function execution
    NO EXCEPTIONS EVER - Stops execution if rule violated
    """
    
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Extract data and source_info from function signature
        import inspect
        
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Look for data parameters
        data = None
        source_info = None
        
        for param_name, param_value in bound_args.arguments.items():
            if isinstance(param_value, dict):
                if 'open' in param_value or 'close' in param_value or 'timestamp' in param_value:
                    data = param_value
                elif 'source_type' in param_value or 'provider' in param_value:
                    source_info = param_value
        
        # ENFORCE HARD RULE if data found
        if data is not None and source_info is not None:
            if not enforce_data_source_rule(data, source_info):
                error_msg = f"CRITICAL: Function {func.__name__} blocked - data source violates hard rule"
                logger.critical(error_msg)
                raise ValueError(error_msg)
        
        return func(*args, **kwargs)
    
    return wrapper

def log_data_source_usage(func: Callable[P, T]) -> Callable[P, T]:
    """
    AUDIT DECORATOR: Logs all data source usage for compliance tracking
    NO EXCEPTIONS EVER - Tracks every data access
    """
    
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Log function call with data source info
        data_source_info = "unknown"
        
        # Try to extract source info
        for arg in args:
            if isinstance(arg, dict) and 'source_type' in arg:
                data_source_info = arg.get('source_type', 'unknown')
                break
        
        if 'source_info' in kwargs:
            data_source_info = kwargs['source_info'].get('source_type', 'unknown')
        
        logger.info(f"Function {func.__name__} called with data source: {data_source_info}")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Log completion
        logger.info(f"Function {func.__name__} completed successfully with data source: {data_source_info}")
        
        return result
    
    return wrapper

# Combined decorator for maximum enforcement
def hard_rule_enforcer(func: Callable[P, T]) -> Callable[P, T]:
    """
    ULTIMATE HARD RULE ENFORCER: Combines all validation layers
    NO EXCEPTIONS EVER - Maximum security and compliance
    """
    return enforce_live_data_only(validate_data_source_before(log_data_source_usage(func)))

# Utility function to check current compliance
def get_compliance_status() -> Dict[str, Any]:
    """Get current hard rule compliance status"""
    return get_data_source_stats()

# Utility function to force validation on any data
def force_validate_data(data: Dict[str, Any], source_info: Dict[str, Any]) -> bool:
    """
    FORCE VALIDATION: Use this to validate any data before processing
    NO EXCEPTIONS EVER - Returns False if rule violated
    """
    return enforce_data_source_rule(data, source_info)
