"""
Enhanced Validator for Qualsynth Framework

This component validates generated samples against:
1. JSON parsing (if LLM returns JSON)
2. Schema validation (types, ranges, categories)
3. Fairness constraint checking
4. Duplicate detection using Gower distance


"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import json
import warnings
from scipy.spatial.distance import cdist


@dataclass
class ValidationResult:
    """Result of validating a single sample."""
    sample_id: str
    is_valid: bool = False
    
    # Validation checks
    json_valid: bool = True
    schema_valid: bool = True
    fairness_valid: bool = True
    duplicate_valid: bool = True
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Parsed sample
    parsed_sample: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """
    Comprehensive validation report for a batch of samples.
    
    This is the output of the Validator and will be used by:
    - Optimizer to filter invalid samples
    - Re-Auditor to check fairness
    - Generator module to improve generation
    """
    total_samples: int
    valid_samples: int = 0
    invalid_samples: int = 0
    
    # Validation results per sample
    results: List[ValidationResult] = field(default_factory=list)
    
    # Error statistics
    json_errors: int = 0
    schema_errors: int = 0
    fairness_errors: int = 0
    duplicate_errors: int = 0
    
    # Valid samples (parsed)
    valid_samples_df: Optional[pd.DataFrame] = None
    
    # Summary
    validation_rate: float = 0.0
    summary: str = ""


class Validator:
    """
    Enhanced Validator - validates generated samples.
    
    This is a TOOL (not an agent) that performs deterministic validation.
    
    Key features:
    1. JSON parsing with error recovery
    2. Schema validation (types, ranges, categories)
    3. Fairness constraint checking
    4. Duplicate detection using Gower distance (handles mixed types)
    """
    
    def __init__(
        self,
        duplicate_threshold: float = 0.05,
        strict_mode: bool = False
    ):
        """
        Initialize Validator.
        
        Args:
            duplicate_threshold: Gower distance threshold for duplicates (0-1)
            strict_mode: If True, reject samples with any warnings
        """
        self.duplicate_threshold = duplicate_threshold
        self.strict_mode = strict_mode
    
    def validate(
        self,
        samples: Union[List[Dict], List[str], pd.DataFrame],
        schema: Any,  # DatasetSchema from SchemaProfiler
        existing_data: Optional[pd.DataFrame] = None,
        fairness_constraints: Optional[List[Any]] = None
    ) -> ValidationReport:
        """
        Validate a batch of generated samples.
        
        Args:
            samples: Generated samples (JSON strings, dicts, or DataFrame)
            schema: DatasetSchema from SchemaProfiler
            existing_data: Existing training data for duplicate detection
            fairness_constraints: Fairness constraints from SchemaProfiler
        
        Returns:
            ValidationReport with detailed results
        """
        report = ValidationReport(total_samples=len(samples))
        
        # 1. Parse samples (if needed)
        parsed_samples = self._parse_samples(samples, report)
        
        # 2. Validate each sample
        for i, sample in enumerate(parsed_samples):
            sample_id = f"sample_{i}"
            result = ValidationResult(sample_id=sample_id)
            
            if sample is None:
                # JSON parsing failed
                result.is_valid = False
                result.json_valid = False
                result.errors.append("JSON parsing failed")
                report.json_errors += 1
                report.results.append(result)
                continue
            
            # 2a. Schema validation
            schema_valid, schema_errors = self._validate_schema(sample, schema)
            result.schema_valid = schema_valid
            result.errors.extend(schema_errors)
            if not schema_valid:
                report.schema_errors += 1
            
            # 2b. Fairness constraint validation
            if fairness_constraints:
                fairness_valid, fairness_errors = self._validate_fairness(
                    sample, fairness_constraints
                )
                result.fairness_valid = fairness_valid
                result.errors.extend(fairness_errors)
                if not fairness_valid:
                    report.fairness_errors += 1
            
            # 2c. Duplicate detection
            if existing_data is not None:
                duplicate_valid, duplicate_warnings = self._check_duplicate(
                    sample, existing_data, schema
                )
                result.duplicate_valid = duplicate_valid
                result.warnings.extend(duplicate_warnings)
                if not duplicate_valid:
                    report.duplicate_errors += 1
            
            # Overall validity
            result.is_valid = (
                result.json_valid and
                result.schema_valid and
                result.fairness_valid and
                result.duplicate_valid
            )
            
            if self.strict_mode and result.warnings:
                result.is_valid = False
            
            result.parsed_sample = sample
            report.results.append(result)
        
        # 3. Collect valid samples
        valid_samples = [
            r.parsed_sample for r in report.results 
            if r.is_valid and r.parsed_sample is not None
        ]
        
        if valid_samples:
            report.valid_samples_df = pd.DataFrame(valid_samples)
        
        # 4. Calculate statistics
        report.valid_samples = len(valid_samples)
        report.invalid_samples = report.total_samples - report.valid_samples
        report.validation_rate = report.valid_samples / report.total_samples if report.total_samples > 0 else 0.0
        
        # 5. Generate summary
        report.summary = self._generate_summary(report)
        
        return report
    
    def _parse_samples(
        self,
        samples: Union[List[Dict], List[str], pd.DataFrame],
        report: ValidationReport
    ) -> List[Optional[Dict]]:
        """Parse samples from various formats."""
        if isinstance(samples, pd.DataFrame):
            return samples.to_dict('records')
        
        if not samples:
            return []
        
        # Check first sample type
        if isinstance(samples[0], dict):
            return samples
        
        if isinstance(samples[0], str):
            # Parse JSON strings
            parsed = []
            for sample_str in samples:
                try:
                    sample = json.loads(sample_str)
                    parsed.append(sample)
                except json.JSONDecodeError as e:
                    parsed.append(None)
            return parsed
        
        return samples
    
    def _validate_schema(
        self,
        sample: Dict[str, Any],
        schema: Any
    ) -> Tuple[bool, List[str]]:
        """
        Validate sample against schema.
        
        Checks:
        - All required features present
        - Feature types match
        - Values within valid ranges/categories
        """
        errors = []
        
        # Check all features in schema
        for feature_name, feature_schema in schema.features.items():
            if feature_name not in sample:
                errors.append(f"Missing feature: {feature_name}")
                continue
            
            value = sample[feature_name]
            
            # Handle NaN/None values - treat as missing in non-strict mode
            try:
                is_nan = (
                    value is None or
                    (isinstance(value, float) and np.isnan(value)) or
                    (isinstance(value, (int, float)) and pd.isna(value))
                )
            except (TypeError, ValueError):
                is_nan = False
            
            if is_nan:
                if self.strict_mode:
                    errors.append(f"Missing value (NaN/None) for feature: {feature_name}")
                else:
                    # In non-strict mode, skip validation for NaN values (will be filled later)
                    continue
            
            # Check type and range
            try:
                from .schema_profiler import FeatureType
            except ImportError:
                from schema_profiler import FeatureType
            
            if feature_schema.type == FeatureType.BINARY:
                # For LLM-generated samples in non-strict mode, allow any numeric value
                # (will be mapped to nearest valid category)
                if not self.strict_mode:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{feature_name}: Invalid value {value}, must be numeric")
                else:
                    if value not in feature_schema.valid_categories:
                        errors.append(
                            f"{feature_name}: Invalid value {value}, "
                            f"expected one of {feature_schema.valid_categories}"
                        )
            
            elif feature_schema.type in [FeatureType.CATEGORICAL_NOMINAL, 
                                         FeatureType.CATEGORICAL_ORDINAL]:
                # For LLM-generated samples, allow any numeric value (will be mapped to nearest category)
                if not self.strict_mode:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{feature_name}: Invalid value {value}, must be numeric")
                else:
                    if value not in feature_schema.valid_categories:
                        errors.append(
                            f"{feature_name}: Invalid category {value}, "
                            f"expected one of {feature_schema.valid_categories}"
                        )
            
            elif feature_schema.type in [FeatureType.NUMERICAL_CONTINUOUS,
                                         FeatureType.NUMERICAL_DISCRETE]:
                try:
                    num_value = float(value)
                    
                    # Check range
                    # For LLM-generated samples, allow 20% margin outside range (will be clipped)
                    if not self.strict_mode:
                        if feature_schema.min_value is not None and feature_schema.max_value is not None:
                            range_size = feature_schema.max_value - feature_schema.min_value
                            margin = range_size * 0.2
                            if num_value < feature_schema.min_value - margin:
                                errors.append(
                                    f"{feature_name}: Value {num_value} too far below minimum {feature_schema.min_value}"
                                )
                            if num_value > feature_schema.max_value + margin:
                                errors.append(
                                    f"{feature_name}: Value {num_value} too far above maximum {feature_schema.max_value}"
                                )
                    else:
                        if feature_schema.min_value is not None and num_value < feature_schema.min_value:
                            errors.append(
                                f"{feature_name}: Value {num_value} below minimum {feature_schema.min_value}"
                            )
                        
                        if feature_schema.max_value is not None and num_value > feature_schema.max_value:
                            errors.append(
                                f"{feature_name}: Value {num_value} above maximum {feature_schema.max_value}"
                            )
                    
                    # Check if discrete should be integer
                    if feature_schema.type == FeatureType.NUMERICAL_DISCRETE:
                        if not float(num_value).is_integer():
                            errors.append(
                                f"{feature_name}: Expected integer, got {num_value}"
                            )
                
                except (ValueError, TypeError):
                    errors.append(
                        f"{feature_name}: Expected numerical value, got {value}"
                    )
        
        return len(errors) == 0, errors
    
    def _validate_fairness(
        self,
        sample: Dict[str, Any],
        fairness_constraints: List[Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate sample against fairness constraints.
        
        Checks:
        - Sensitive attributes have valid values
        - Sample respects target proportions (if applicable)
        """
        errors = []
        
        for constraint in fairness_constraints:
            # FairnessTarget uses 'attribute' not 'sensitive_attribute'
            sensitive_attr = getattr(constraint, 'attribute', getattr(constraint, 'sensitive_attribute', None))
            
            if not sensitive_attr:
                continue
                
            if sensitive_attr not in sample:
                errors.append(
                    f"Missing sensitive attribute: {sensitive_attr}"
                )
                continue
            
            value = sample[sensitive_attr]
            
            # Check if value is in target proportions (if available)
            target_proportions = getattr(constraint, 'target_proportions', None)
            if target_proportions:
                if value not in target_proportions:
                    errors.append(
                        f"{sensitive_attr}: Value {value} not in target proportions "
                        f"{list(target_proportions.keys())}"
                    )
        
        return len(errors) == 0, errors
    
    def _check_duplicate(
        self,
        sample: Dict[str, Any],
        existing_data: pd.DataFrame,
        schema: Any
    ) -> Tuple[bool, List[str]]:
        """
        Check if sample is duplicate using Gower distance.
        
        Gower distance handles mixed types (categorical + numerical).
        """
        warnings_list = []
        
        # Convert sample to DataFrame
        sample_df = pd.DataFrame([sample])
        
        # Ensure columns match
        common_cols = [col for col in sample_df.columns if col in existing_data.columns]
        if not common_cols:
            return True, []
        
        sample_df = sample_df[common_cols]
        existing_df = existing_data[common_cols]
        
        # Calculate Gower distance
        try:
            distances = self._gower_distance(sample_df, existing_df, schema)
            min_distance = np.min(distances)
            
            if min_distance < self.duplicate_threshold:
                warnings_list.append(
                    f"Potential duplicate detected (Gower distance: {min_distance:.4f})"
                )
                return False, warnings_list
        
        except Exception as e:
            warnings_list.append(f"Duplicate detection failed: {e}")
        
        return True, warnings_list
    
    def _gower_distance(
        self,
        sample_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        schema: Any
    ) -> np.ndarray:
        """
        Calculate Gower distance between sample and existing data.
        
        Gower distance = weighted average of:
        - Categorical: 0 if same, 1 if different
        - Numerical: |x1 - x2| / range
        """
        try:
            from .schema_profiler import FeatureType
        except ImportError:
            from schema_profiler import FeatureType
        
        distances = np.zeros(len(existing_df))
        n_features = 0
        
        for col in sample_df.columns:
            if col not in schema.features:
                continue
            
            feature_schema = schema.features[col]
            sample_val = sample_df[col].iloc[0]
            existing_vals = existing_df[col].values
            
            if feature_schema.type in [FeatureType.BINARY, 
                                       FeatureType.CATEGORICAL_NOMINAL,
                                       FeatureType.CATEGORICAL_ORDINAL]:
                # Categorical: 0 if same, 1 if different
                col_distances = (existing_vals != sample_val).astype(float)
            
            elif feature_schema.type in [FeatureType.NUMERICAL_CONTINUOUS,
                                         FeatureType.NUMERICAL_DISCRETE]:
                # Numerical: normalized absolute difference
                value_range = feature_schema.max_value - feature_schema.min_value
                if value_range > 0:
                    col_distances = np.abs(existing_vals - sample_val) / value_range
                else:
                    col_distances = np.zeros(len(existing_vals))
            
            else:
                continue
            
            distances += col_distances
            n_features += 1
        
        # Average across features
        if n_features > 0:
            distances /= n_features
        
        return distances
    
    def _generate_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        lines = []
        
        lines.append(f"Total: {report.total_samples}")
        lines.append(f"Valid: {report.valid_samples} ({report.validation_rate:.1%})")
        lines.append(f"Invalid: {report.invalid_samples}")
        
        if report.json_errors > 0:
            lines.append(f"JSON errors: {report.json_errors}")
        if report.schema_errors > 0:
            lines.append(f"Schema errors: {report.schema_errors}")
        if report.fairness_errors > 0:
            lines.append(f"Fairness errors: {report.fairness_errors}")
        if report.duplicate_errors > 0:
            lines.append(f"Duplicates: {report.duplicate_errors}")
        
        return " | ".join(lines)
    
    def print_report(self, report: ValidationReport, verbose: bool = True) -> None:
        """Print comprehensive validation report."""
        print(f"\n{'='*70}")
        print(f"VALIDATION REPORT")
        print(f"{'='*70}")
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total samples: {report.total_samples}")
        print(f"   Valid samples: {report.valid_samples} ({report.validation_rate:.1%})")
        print(f"   Invalid samples: {report.invalid_samples}")
        
        # Error breakdown
        if report.invalid_samples > 0:
            print(f"\n❌ ERROR BREAKDOWN:")
            if report.json_errors > 0:
                print(f"   JSON parsing errors: {report.json_errors}")
            if report.schema_errors > 0:
                print(f"   Schema validation errors: {report.schema_errors}")
            if report.fairness_errors > 0:
                print(f"   Fairness constraint errors: {report.fairness_errors}")
            if report.duplicate_errors > 0:
                print(f"   Duplicate detections: {report.duplicate_errors}")
        
        # Sample-level details (if verbose)
        if verbose and report.results:
            print(f"\n📋 SAMPLE-LEVEL RESULTS:")
            
            # Show first 5 invalid samples
            invalid_results = [r for r in report.results if not r.is_valid]
            if invalid_results:
                print(f"\n   Invalid Samples (showing first 5):")
                for result in invalid_results[:5]:
                    print(f"\n   {result.sample_id}:")
                    for error in result.errors:
                        print(f"      ❌ {error}")
                    for warning in result.warnings:
                        print(f"      ⚠️  {warning}")
                
                if len(invalid_results) > 5:
                    print(f"\n   ... and {len(invalid_results) - 5} more invalid samples")
            
            # Show first 3 valid samples
            valid_results = [r for r in report.results if r.is_valid]
            if valid_results:
                print(f"\n   Valid Samples (showing first 3):")
                for result in valid_results[:3]:
                    print(f"      ✅ {result.sample_id}")
        
        # Summary
        print(f"\n📝 SUMMARY:")
        print(f"   {report.summary}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test the Validator
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.schema_profiler import SchemaProfiler
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    
    print("="*70)
    print("Testing Enhanced Validator")
    print("="*70)
    
    validator = Validator(duplicate_threshold=0.05, strict_mode=False)
    schema_profiler = SchemaProfiler()
    auditor = FairnessAuditor(fairness_threshold=0.05)
    
    # Test on German Credit dataset
    dataset_name = 'german_credit'
    sensitive_cols = ['race', 'sex']
    
    print(f"\n\n{'='*70}")
    print(f"VALIDATING: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    split_data = load_split(dataset_name, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    # Get sensitive features
    available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
    if available_sensitive_cols:
        sensitive_features = X_train[available_sensitive_cols]
        audit_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
        fairness_targets = audit_report.fairness_targets
    else:
        fairness_targets = None
    
    # Profile schema
    schema = schema_profiler.profile(
        X_train,
        y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=fairness_targets,
        dataset_name=dataset_name
    )
    
    # Create test samples (mix of valid and invalid)
    minority_samples = X_train[y_train == 1].head(10)
    
    # Test 1: Valid samples (from actual data)
    print("\n\n" + "="*70)
    print("TEST 1: Validating real samples (should be 100% valid)")
    print("="*70)
    
    valid_samples = minority_samples.to_dict('records')
    report1 = validator.validate(
        valid_samples,
        schema,
        existing_data=X_train,
        fairness_constraints=schema.fairness_constraints
    )
    validator.print_report(report1, verbose=True)
    
    # Test 2: Invalid samples (modified to violate constraints)
    print("\n\n" + "="*70)
    print("TEST 2: Validating invalid samples (should have errors)")
    print("="*70)
    
    invalid_samples = minority_samples.to_dict('records')[:5]
    
    # Introduce errors
    if invalid_samples:
        # Missing feature
        del invalid_samples[0]['age']
        
        # Out of range
        if 'age' in invalid_samples[1]:
            invalid_samples[1]['age'] = 999.0
        
        # Invalid category
        if 'sex' in invalid_samples[2]:
            invalid_samples[2]['sex'] = 999
        
        # Invalid type
        if 'workclass' in invalid_samples[3]:
            invalid_samples[3]['workclass'] = "invalid_value"
    
    report2 = validator.validate(
        invalid_samples,
        schema,
        existing_data=X_train,
        fairness_constraints=schema.fairness_constraints
    )
    validator.print_report(report2, verbose=True)
    
    # Test 3: Duplicate detection
    print("\n\n" + "="*70)
    print("TEST 3: Duplicate detection (should detect duplicates)")
    print("="*70)
    
    # Use exact duplicates
    duplicate_samples = minority_samples.head(3).to_dict('records')
    
    report3 = validator.validate(
        duplicate_samples,
        schema,
        existing_data=X_train,
        fairness_constraints=schema.fairness_constraints
    )
    validator.print_report(report3, verbose=True)
    
    print("\n✅ Validator Test Complete")

