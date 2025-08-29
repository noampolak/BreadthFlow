"""
Data Validation System

This module provides data validation components for the BreadthFlow abstraction system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class ValidationRule:
    """A validation rule"""
    name: str
    description: str
    validator: Callable
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True

class DataValidator:
    """Data validation system"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        # Data completeness rule
        self.add_rule(
            ValidationRule(
                name="data_completeness",
                description="Check if data is complete",
                validator=self._validate_completeness,
                severity=ValidationSeverity.WARNING
            )
        )
        
        # Data quality rule
        self.add_rule(
            ValidationRule(
                name="data_quality",
                description="Check data quality metrics",
                validator=self._validate_quality,
                severity=ValidationSeverity.ERROR
            )
        )
        
        # Data consistency rule
        self.add_rule(
            ValidationRule(
                name="data_consistency",
                description="Check data consistency",
                validator=self._validate_consistency,
                severity=ValidationSeverity.WARNING
            )
        )
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a validation rule by name"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"Removed validation rule: {rule_name}")
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate data using all enabled rules"""
        results = []
        context = context or {}
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            try:
                result = rule.validator(data, context)
                if isinstance(result, bool):
                    # Simple boolean result
                    results.append(ValidationResult(
                        is_valid=result,
                        severity=rule.severity,
                        message=f"Rule '{rule.name}' validation {'passed' if result else 'failed'}",
                        details={"rule": rule.name, "description": rule.description}
                    ))
                elif isinstance(result, ValidationResult):
                    # Detailed validation result
                    results.append(result)
                else:
                    # Invalid validator return type
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid validator return type for rule '{rule.name}'",
                        details={"rule": rule.name, "return_type": type(result).__name__}
                    ))
                    
            except Exception as e:
                logger.error(f"Validation rule '{rule.name}' failed: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule '{rule.name}' failed: {str(e)}",
                    details={"rule": rule.name, "error": str(e)}
                ))
        
        return results
    
    def _validate_completeness(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate data completeness"""
        if data is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Data is None",
                details={"missing": "all_data"}
            )
        
        if isinstance(data, (list, tuple)) and len(data) == 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Data is empty",
                details={"count": 0}
            )
        
        if isinstance(data, dict) and len(data) == 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Data dictionary is empty",
                details={"keys": 0}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Data completeness check passed",
            details={"data_type": type(data).__name__, "size": len(data) if hasattr(data, '__len__') else "unknown"}
        )
    
    def _validate_quality(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate data quality"""
        if data is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Data is None",
                details={"quality_score": 0}
            )
        
        # Simple quality check - can be enhanced based on data type
        quality_score = 1.0
        
        if isinstance(data, (list, tuple)):
            # Check for None values in list
            none_count = sum(1 for item in data if item is None)
            if none_count > 0:
                quality_score -= (none_count / len(data)) * 0.5
        
        elif isinstance(data, dict):
            # Check for None values in dict
            none_count = sum(1 for value in data.values() if value is None)
            if none_count > 0:
                quality_score -= (none_count / len(data)) * 0.5
        
        if quality_score < 0.5:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Data quality is poor",
                details={"quality_score": quality_score}
            )
        elif quality_score < 0.8:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message="Data quality is acceptable but could be improved",
                details={"quality_score": quality_score}
            )
        else:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Data quality is good",
                details={"quality_score": quality_score}
            )
    
    def _validate_consistency(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate data consistency"""
        if data is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Data is None",
                details={"consistency": "none"}
            )
        
        # Simple consistency check - can be enhanced based on data type
        if isinstance(data, (list, tuple)) and len(data) > 1:
            # Check if all items have the same type
            first_type = type(data[0])
            consistent_types = all(isinstance(item, first_type) for item in data)
            
            if not consistent_types:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Data types are inconsistent",
                    details={"expected_type": first_type.__name__, "inconsistent": True}
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Data consistency check passed",
            details={"consistency": "good"}
        )
    
    def get_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get a summary of validation results"""
        total = len(results)
        passed = sum(1 for r in results if r.is_valid)
        failed = total - passed
        
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(1 for r in results if r.severity == severity)
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "severity_breakdown": severity_counts,
            "overall_valid": all(r.is_valid for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        }
