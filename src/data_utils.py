
"""
Data Utilities for Bank Term Deposit Prediction

This module provides comprehensive data loading, validation, and preprocessing
utilities for the bank term deposit prediction project. It handles both real
and synthetic datasets with robust error handling and validation.

Key Features:
- Safe CSV reading with multiple encoding/separator attempts
- Automatic target variable detection
- Binary target coercion with validation
- Dataset consistency validation
- Feature type inference
- Column alignment utilities

2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import warnings

# Candidate target column names for automatic detection
CANDIDATE_TARGET_COLUMNS = [
    "y", "target", "deposit", "subscribed", "subscription", 
    "term_deposit", "TermDeposit", "deposit_subscribed"
]

def read_csv_safe(file_path: str) -> pd.DataFrame:
    """
    Safely read CSV files with automatic separator and encoding detection.
    
    This function attempts multiple strategies to read CSV files, handling
    common issues with different separators (semicolon vs comma) and
    character encodings that are common in financial datasets.
    
    Args:
        file_path: Path to the CSV file to read
        
    Returns:
        pandas DataFrame containing the loaded data
        
    Raises:
        RuntimeError: If all reading attempts fail
        
    Note:
        Tries semicolon separator first (common in European datasets),
        then falls back to comma separator with various encodings.
    """
    reading_strategies = [
        # Strategy 1: Semicolon separator with UTF-8 (European datasets)
        {'sep': ';', 'encoding': 'utf-8'},
        # Strategy 2: Comma separator with UTF-8 (Standard CSV)
        {'sep': ',', 'encoding': 'utf-8'},
        # Strategy 3: Comma separator with Latin-1 (Legacy datasets)
        {'sep': ',', 'encoding': 'latin-1'},
        # Strategy 4: Semicolon separator with Latin-1 (Legacy European)
        {'sep': ';', 'encoding': 'latin-1'}
    ]
    
    for strategy in reading_strategies:
        try:
            dataframe = pd.read_csv(file_path, **strategy)
            # Validate that we got multiple columns (successful parsing)
            if dataframe.shape[1] > 1:
                print(f"Successfully read {file_path} using {strategy}")
                return dataframe
        except Exception as e:
            continue
    
    # If all strategies failed, raise an error
    raise RuntimeError(f"Failed to read CSV file {file_path} with all attempted strategies")

def infer_target_column(dataframe: pd.DataFrame, target_config: Dict[str, Any]) -> str:
    """
    Automatically infer the target column name from the dataset.
    
    This function attempts to identify the target variable column using
    either explicit configuration or automatic detection based on common
    naming patterns in financial datasets.
    
    Args:
        dataframe: Input DataFrame to analyze
        target_config: Configuration dictionary with target settings
        
    Returns:
        Name of the target column
        
    Raises:
        AssertionError: If configured target column doesn't exist
        
    Note:
        If target_config['name'] is 'auto', searches through common target
        column names. Falls back to last column if no match found.
    """
    target_column_name = target_config.get("name", "auto")
    
    # Use explicitly configured target column
    if target_column_name != "auto":
        assert target_column_name in dataframe.columns, \
            f"Configured target column '{target_column_name}' not found in dataset columns"
        return target_column_name
    
    # Automatic target column detection
    for candidate_name in CANDIDATE_TARGET_COLUMNS:
        if candidate_name in dataframe.columns:
            print(f"Auto-detected target column: '{candidate_name}'")
            return candidate_name
    
    # Fallback to last column (common ML convention)
    fallback_column = dataframe.columns[-1]
    print(f"No standard target column found, using last column: '{fallback_column}'")
    return fallback_column

def coerce_to_binary_target(target_series: pd.Series, positive_class_labels: List) -> pd.Series:
    """
    Convert target variable to binary format (0/1) with comprehensive validation.
    
    This function handles various target formats commonly found in financial
    datasets, including string labels ("yes"/"no") and numeric values (0/1).
    Provides robust error handling and validation.
    
    Args:
        target_series: Target variable as pandas Series
        positive_class_labels: List of values that should be mapped to 1 (positive class)
        
    Returns:
        Binary target series with values 0 and 1
        
    Note:
        - Handles missing values by filling with 0 (negative class)
        - Normalizes string values to lowercase for consistent matching
        - Validates final output contains only 0 and 1 values
    """
    # Handle missing values
    if target_series.isnull().any():
        missing_count = target_series.isnull().sum()
        warnings.warn(f"Found {missing_count} missing values in target, filling with 0 (negative class)")
        target_series = target_series.fillna(0)
    
    def normalize_value(value):
        """Normalize values for consistent comparison."""
        if isinstance(value, str):
            return value.strip().lower()
        return value
    
    # Normalize positive class labels for comparison
    normalized_positive_labels = set([normalize_value(label) for label in positive_class_labels])
    normalized_target = target_series.map(normalize_value)
    
    # Check if already in binary numeric format
    unique_values = set(pd.Series(normalized_target).dropna().unique())
    if unique_values.issubset({0, 1, 0.0, 1.0}):
        print("Target already in binary numeric format, converting to integer")
        return normalized_target.astype(int)
    
    # Convert string/mixed labels to binary
    binary_target = normalized_target.apply(
        lambda value: 1 if value in normalized_positive_labels else 0
    )
    
    # Validation of final result
    final_unique_values = set(binary_target.unique())
    if not final_unique_values.issubset({0, 1}):
        warnings.warn(f"Unexpected values after binary coercion: {final_unique_values}")
    
    # Report conversion statistics
    positive_count = (binary_target == 1).sum()
    negative_count = (binary_target == 0).sum()
    total_count = len(binary_target)
    
    print(f"Binary target conversion completed:")
    print(f"  Positive class (1): {positive_count:,} ({positive_count/total_count*100:.1f}%)")
    print(f"  Negative class (0): {negative_count:,} ({negative_count/total_count*100:.1f}%)")
    
    return binary_target.astype(int)

def split_dataset(dataframe: pd.DataFrame, target_column: str, test_size: float, 
                 random_state: int, use_stratification: bool = True) -> Tuple[Tuple[pd.DataFrame, pd.Series], 
                                                                            Tuple[pd.DataFrame, pd.Series]]:
    """
    Split dataset into training and testing sets with optional stratification.
    
    This function performs a train-test split while maintaining class balance
    through stratification, which is crucial for imbalanced datasets common
    in financial applications.
    
    Args:
        dataframe: Complete dataset to split
        target_column: Name of the target variable column
        test_size: Proportion of dataset to include in test split (0.0 to 1.0)
        random_state: Random seed for reproducible splits
        use_stratification: Whether to maintain class proportions in splits
        
    Returns:
        Tuple containing ((X_train, y_train), (X_test, y_test))
        
    Note:
        Stratification ensures both training and test sets have similar
        class distributions, preventing bias in model evaluation.
    """
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    feature_columns = dataframe.drop(columns=[target_column])
    target_values = dataframe[target_column]
    
    # Configure stratification
    stratification_target = target_values if use_stratification else None
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_columns, 
        target_values, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratification_target
    )
    
    # Report split statistics
    print(f"Dataset split completed:")
    print(f"  Training set: {len(X_train):,} samples ({(1-test_size)*100:.1f}%)")
    print(f"  Test set: {len(X_test):,} samples ({test_size*100:.1f}%)")
    
    if use_stratification:
        train_positive_rate = (y_train == 1).mean()
        test_positive_rate = (y_test == 1).mean()
        print(f"  Training positive class rate: {train_positive_rate:.3f}")
        print(f"  Test positive class rate: {test_positive_rate:.3f}")
    
    return (X_train, y_train), (X_test, y_test)

def infer_feature_types(dataframe: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    """
    Automatically infer numerical and categorical feature types from dataset.
    
    This function analyzes column data types to separate numerical and
    categorical features, which is essential for proper preprocessing
    pipeline configuration.
    
    Args:
        dataframe: Input DataFrame to analyze
        target_column: Name of target column to exclude from features
        
    Returns:
        Tuple containing (numerical_columns, categorical_columns)
        
    Note:
        - Numerical: int, float data types
        - Categorical: object, category data types
        - Target column is excluded from both lists
    """
    # Identify categorical columns (object or category dtype)
    categorical_columns = [
        column for column in dataframe.columns 
        if column != target_column and (
            dataframe[column].dtype == "object" or 
            dataframe[column].dtype.name == "category"
        )
    ]
    
    # Identify numerical columns (all others except target and categorical)
    numerical_columns = [
        column for column in dataframe.columns 
        if column != target_column and column not in categorical_columns
    ]
    
    # Report feature type analysis
    print(f"Feature type inference completed:")
    print(f"  Numerical features: {len(numerical_columns)} columns")
    print(f"  Categorical features: {len(categorical_columns)} columns")
    print(f"  Target column: '{target_column}' (excluded)")
    
    return numerical_columns, categorical_columns

def align_dataset_columns(training_features: pd.DataFrame, 
                         test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align column names and order between training and test datasets.
    
    This function ensures that training and test datasets have identical
    column structures, which is required for sklearn pipelines and prevents
    errors during model training and prediction.
    
    Args:
        training_features: Training dataset features
        test_features: Test dataset features
        
    Returns:
        Tuple of (aligned_training_features, aligned_test_features)
        
    Note:
        - Missing columns are filled with NaN values
        - Column order is standardized between datasets
        - Validates successful alignment before returning
    """
    # Identify missing columns in each dataset
    missing_in_test = [col for col in training_features.columns 
                      if col not in test_features.columns]
    missing_in_training = [col for col in test_features.columns 
                          if col not in training_features.columns]
    
    # Add missing columns to test set
    for column_name in missing_in_test:
        warnings.warn(f"Column '{column_name}' missing in test set, filling with NaN")
        test_features[column_name] = np.nan
        
    # Add missing columns to training set
    for column_name in missing_in_training:
        warnings.warn(f"Column '{column_name}' missing in training set, filling with NaN")
        training_features[column_name] = np.nan
        
    # Standardize column order (use test set order as reference)
    aligned_training_features = training_features[test_features.columns]
    aligned_test_features = test_features.copy()
    
    # Validate successful alignment
    training_columns = list(aligned_training_features.columns)
    test_columns = list(aligned_test_features.columns)
    
    assert training_columns == test_columns, \
        f"Column alignment failed: training={len(training_columns)}, test={len(test_columns)}"
    
    print(f"Column alignment completed: {len(training_columns)} columns aligned")
    
    return aligned_training_features, aligned_test_features

def validate_dataset_consistency(real_dataset: pd.DataFrame, synthetic_dataset: pd.DataFrame, 
                               target_column: str) -> Dict[str, Any]:
    """
    Comprehensive validation of consistency between real and synthetic datasets.
    
    This function performs detailed analysis to ensure synthetic data quality
    and compatibility with real data for transfer learning applications.
    
    Args:
        real_dataset: Real dataset DataFrame
        synthetic_dataset: Synthetic dataset DataFrame
        target_column: Name of the target variable column
        
    Returns:
        Dictionary containing validation results, warnings, and statistics
        
    Note:
        Analyzes feature overlap, schema compatibility, and target distributions
        to identify potential issues before model training.
    """
    validation_results = {
        'schema_compatibility': True,
        'validation_warnings': [],
        'feature_analysis': {},
        'target_distribution_analysis': {},
        'data_quality_metrics': {}
    }
    
    # Extract feature columns (exclude target and potential ID columns)
    real_feature_columns = set(real_dataset.columns) - {target_column}
    synthetic_feature_columns = set(synthetic_dataset.columns) - {target_column, 'id'}
    
    # Analyze feature overlap
    missing_in_synthetic = real_feature_columns - synthetic_feature_columns
    missing_in_real = synthetic_feature_columns - real_feature_columns
    common_feature_columns = real_feature_columns & synthetic_feature_columns
    
    # Calculate overlap statistics
    feature_overlap_ratio = (len(common_feature_columns) / len(real_feature_columns) 
                           if real_feature_columns else 0)
    
    validation_results['feature_analysis'] = {
        'common_features': sorted(list(common_feature_columns)),
        'missing_in_synthetic': sorted(list(missing_in_synthetic)),
        'missing_in_real': sorted(list(missing_in_real)),
        'feature_overlap_ratio': feature_overlap_ratio,
        'real_feature_count': len(real_feature_columns),
        'synthetic_feature_count': len(synthetic_feature_columns)
    }
    
    # Check for schema compatibility issues
    if missing_in_synthetic:
        warning_message = f"Features present in real but missing in synthetic data: {missing_in_synthetic}"
        validation_results['validation_warnings'].append(warning_message)
        validation_results['schema_compatibility'] = False
        
    if missing_in_real:
        info_message = f"Additional features in synthetic data: {missing_in_real}"
        validation_results['validation_warnings'].append(info_message)
    
    # Analyze target variable distributions
    real_target_distribution = real_dataset[target_column].value_counts(normalize=True).to_dict()
    synthetic_target_distribution = synthetic_dataset[target_column].value_counts(normalize=True).to_dict()
    
    # Calculate distribution similarity
    common_classes = set(real_target_distribution.keys()) & set(synthetic_target_distribution.keys())
    distribution_differences = {}
    
    for class_label in common_classes:
        real_proportion = real_target_distribution.get(class_label, 0)
        synthetic_proportion = synthetic_target_distribution.get(class_label, 0)
        distribution_differences[class_label] = abs(real_proportion - synthetic_proportion)
    
    validation_results['target_distribution_analysis'] = {
        'real_distribution': real_target_distribution,
        'synthetic_distribution': synthetic_target_distribution,
        'distribution_differences': distribution_differences,
        'max_distribution_difference': max(distribution_differences.values()) if distribution_differences else 0
    }
    
    # Calculate data quality metrics
    validation_results['data_quality_metrics'] = {
        'real_dataset_size': len(real_dataset),
        'synthetic_dataset_size': len(synthetic_dataset),
        'size_ratio': len(synthetic_dataset) / len(real_dataset) if len(real_dataset) > 0 else 0,
        'feature_overlap_score': feature_overlap_ratio,
        'target_distribution_similarity': 1 - max(distribution_differences.values()) if distribution_differences else 1
    }
    
    # Generate summary report
    print("Dataset Consistency Validation Results:")
    print(f"  Schema Compatibility: {'✅ PASS' if validation_results['schema_compatibility'] else '❌ ISSUES'}")
    print(f"  Feature Overlap: {feature_overlap_ratio:.1%} ({len(common_feature_columns)}/{len(real_feature_columns)})")
    print(f"  Target Distribution Similarity: {validation_results['data_quality_metrics']['target_distribution_similarity']:.3f}")
    
    if validation_results['validation_warnings']:
        print("  Warnings:")
        for warning in validation_results['validation_warnings']:
            print(f"    - {warning}")
    
    return validation_results
