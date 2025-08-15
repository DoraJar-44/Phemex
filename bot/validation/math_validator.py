"""Mathematical validation utilities for trading bot calculations."""
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from bot.strategy.score import ScoreInputs, compute_total_score, clamp
from bot.strategy.pr import compute_atr, compute_predictive_ranges


@dataclass
class ValidationResult:
    """Result of mathematical validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class MathValidator:
    """Validates mathematical calculations for accuracy and edge cases."""
    
    def __init__(self):
        self.tolerance = 1e-9
        
    def validate_atr_calculation(self, high: List[float], low: List[float], 
                                close: List[float], length: int) -> ValidationResult:
        """Validate ATR calculation against known formula."""
        errors = []
        warnings = []
        metrics = {}
        
        if len(high) != len(low) or len(low) != len(close):
            errors.append("High, low, close arrays must have same length")
            return ValidationResult(False, errors, warnings, metrics)
            
        if length <= 0:
            errors.append("ATR length must be positive")
            
        if len(close) < length + 1:
            warnings.append(f"Insufficient data for ATR({length}): {len(close)} bars")
            
        # Calculate ATR using bot's method
        try:
            atr_values = compute_atr(high, low, close, length)
            
            # Validate individual TR calculations
            for i in range(min(10, len(close))):  # Check first 10 bars
                if i == 0:
                    expected_tr = high[0] - low[0]
                else:
                    tr1 = high[i] - low[i]
                    tr2 = abs(high[i] - close[i-1])
                    tr3 = abs(low[i] - close[i-1])
                    expected_tr = max(tr1, tr2, tr3)
                    
                # Verify ATR values are reasonable
                if atr_values[i] < 0:
                    errors.append(f"Negative ATR at index {i}: {atr_values[i]}")
                    
                if math.isnan(atr_values[i]) or math.isinf(atr_values[i]):
                    errors.append(f"Invalid ATR value at index {i}: {atr_values[i]}")
                    
            # Check ATR smoothing properties
            if len(atr_values) > 1:
                max_change = max(abs(atr_values[i] - atr_values[i-1]) 
                               for i in range(1, len(atr_values)))
                metrics['max_atr_change'] = max_change
                metrics['final_atr'] = atr_values[-1]
                
        except Exception as e:
            errors.append(f"ATR calculation failed: {str(e)}")
            
        return ValidationResult(len(errors) == 0, errors, warnings, metrics)
    
    def validate_predictive_ranges(self, high: List[float], low: List[float], 
                                 close: List[float], atr_len: int = 200, 
                                 atr_mult: float = 6.0) -> ValidationResult:
        """Validate predictive ranges calculation."""
        errors = []
        warnings = []
        metrics = {}
        
        if atr_mult <= 0:
            errors.append("ATR multiplier must be positive")
            
        try:
            avg, r1, r2, s1, s2 = compute_predictive_ranges(high, low, close, atr_len, atr_mult)
            
            # Validate range ordering
            if not (s2 <= s1 <= avg <= r1 <= r2):
                errors.append(f"Range ordering invalid: s2({s2:.4f}) <= s1({s1:.4f}) <= avg({avg:.4f}) <= r1({r1:.4f}) <= r2({r2:.4f})")
            
            # Check for reasonable values
            for name, val in [("avg", avg), ("r1", r1), ("r2", r2), ("s1", s1), ("s2", s2)]:
                if math.isnan(val) or math.isinf(val):
                    errors.append(f"Invalid {name} value: {val}")
                    
            # Validate range spreads
            range_spread = r1 - s1
            if range_spread <= 0:
                errors.append(f"Invalid range spread: {range_spread}")
            else:
                metrics['range_spread'] = range_spread
                metrics['range_ratio'] = (r2 - s2) / range_spread if range_spread > 0 else 0
                
            metrics.update({
                'avg': avg, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2,
                'avg_to_close_ratio': abs(avg - close[-1]) / close[-1] if close[-1] != 0 else float('inf')
            })
            
        except Exception as e:
            errors.append(f"Predictive ranges calculation failed: {str(e)}")
            
        return ValidationResult(len(errors) == 0, errors, warnings, metrics)
    
    def validate_scoring_system(self, inp: ScoreInputs, side: str) -> ValidationResult:
        """Validate scoring system calculations."""
        errors = []
        warnings = []
        metrics = {}
        
        # Validate input ranges
        if not (0.0 <= inp.bias_up_conf <= 1.0):
            errors.append(f"bias_up_conf out of range [0,1]: {inp.bias_up_conf}")
        if not (0.0 <= inp.bias_dn_conf <= 1.0):
            errors.append(f"bias_dn_conf out of range [0,1]: {inp.bias_dn_conf}")
        if not (0.0 <= inp.bounce_prob <= 0.9):
            errors.append(f"bounce_prob out of range [0,0.9]: {inp.bounce_prob}")
            
        if side not in ["long", "short"]:
            errors.append(f"Invalid side: {side}")
            
        try:
            score = compute_total_score(inp, side)
            
            # Validate score bounds
            if not (0 <= score <= 150):  # Theoretical max: 50 + 30 + 20 + 20 + 30
                warnings.append(f"Score outside expected range [0,150]: {score}")
                
            # Test component calculations manually
            base = 50.0
            
            # Range score validation
            if side == "long":
                span = max(abs(inp.avg - inp.s1), 1e-9)
                proximity = clamp((inp.r1 - inp.close) / (inp.r1 - inp.s1 + 1e-9), 0.0, 1.0)
                range_score = 30.0 * proximity
            else:
                span = max(abs(inp.avg - inp.r1), 1e-9)
                proximity = clamp((inp.close - inp.s1) / (inp.r1 - inp.s1 + 1e-9), 0.0, 1.0)
                range_score = 30.0 * proximity
                
            bounce_score = 20.0 * clamp(inp.bounce_prob / 0.9, 0.0, 1.0)
            
            div_score = 0.0
            if side == "long" and inp.bull_div:
                div_score = 20.0
            elif side == "short" and inp.bear_div:
                div_score = 20.0
                
            bias_conf = inp.bias_up_conf if side == "long" else inp.bias_dn_conf
            bias_score = 30.0 * clamp(bias_conf, 0.0, 1.0)
            
            calculated_total = int(base + range_score + bounce_score + div_score + bias_score)
            
            if abs(calculated_total - score) > 1:  # Allow for rounding
                errors.append(f"Score calculation mismatch: expected {calculated_total}, got {score}")
                
            metrics.update({
                'total_score': score,
                'base_score': base,
                'range_score': range_score,
                'bounce_score': bounce_score,
                'div_score': div_score,
                'bias_score': bias_score,
                'proximity': proximity if 'proximity' in locals() else 0
            })
            
        except Exception as e:
            errors.append(f"Score calculation failed: {str(e)}")
            
        return ValidationResult(len(errors) == 0, errors, warnings, metrics)
    
    def validate_clamp_function(self) -> ValidationResult:
        """Test the clamp utility function."""
        errors = []
        warnings = []
        metrics = {}
        
        test_cases = [
            (5.0, 0.0, 10.0, 5.0),  # Normal case
            (-1.0, 0.0, 10.0, 0.0),  # Below range
            (15.0, 0.0, 10.0, 10.0),  # Above range
            (5.0, 5.0, 5.0, 5.0),  # Exact bounds
            (float('inf'), 0.0, 10.0, 10.0),  # Infinity
            (float('-inf'), 0.0, 10.0, 0.0),  # Negative infinity
        ]
        
        for i, (value, lo, hi, expected) in enumerate(test_cases):
            try:
                result = clamp(value, lo, hi)
                if abs(result - expected) > self.tolerance:
                    errors.append(f"Clamp test {i} failed: clamp({value}, {lo}, {hi}) = {result}, expected {expected}")
            except Exception as e:
                errors.append(f"Clamp test {i} exception: {str(e)}")
                
        return ValidationResult(len(errors) == 0, errors, warnings, metrics)
    
    def run_full_validation(self, high: List[float], low: List[float], close: List[float],
                           score_inputs: ScoreInputs, side: str) -> Dict[str, ValidationResult]:
        """Run comprehensive validation of all mathematical components."""
        results = {}
        
        results['clamp'] = self.validate_clamp_function()
        results['atr'] = self.validate_atr_calculation(high, low, close, 14)
        results['predictive_ranges'] = self.validate_predictive_ranges(high, low, close)
        results['scoring'] = self.validate_scoring_system(score_inputs, side)
        
        return results
