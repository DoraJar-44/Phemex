#!/usr/bin/env python3
"""
Configuration module for bot settings and hard rules
"""

# Import settings from main config file (avoiding circular import)
import importlib.util
import os
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
_spec = importlib.util.spec_from_file_location("bot_config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
settings = _config.settings

from .hard_rules import (
    enforce_hard_rules,
    get_hard_rules_compliance,
    HardRulesConfig,
    HardRule,
    RuleSeverity
)

__all__ = [
    'settings',
    'enforce_hard_rules',
    'get_hard_rules_compliance',
    'HardRulesConfig',
    'HardRule',
    'RuleSeverity'
]
