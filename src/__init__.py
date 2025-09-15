"""
Bank Term Deposit Prediction - Source Package

This package contains the core modules for the bank term deposit prediction
machine learning pipeline, including data utilities, preprocessing, and
transfer learning components.

Modules:
    data_utils: Data loading, validation, and preprocessing utilities
    enhanced_preprocessing: Advanced feature engineering and preprocessing
    enhanced_transfer_learning: Transfer learning and adaptive methods

2025
"""

# Package version
__version__ = "1.0.0"

# Import key functions for easy access
from .data_utils import (
    read_csv_safe,
    infer_target_column,
    coerce_to_binary_target,
    split_dataset,
    infer_feature_types,
    align_dataset_columns,
    validate_dataset_consistency
)

from .enhanced_preprocessing import (
    create_enhanced_preprocessor,
    create_balanced_dataset,
    AdvancedFeatureEngineer,
    OutlierHandler,
    SmartTargetEncoder,
    FeedbackLearner
)

from .enhanced_transfer_learning import (
    AdaptiveTransferLearner,
    MultiStageTransferLearner,
    create_enhanced_transfer_learning_pipeline
)

# Define what gets imported with "from src import *"
__all__ = [
    # Data utilities
    'read_csv_safe',
    'infer_target_column', 
    'coerce_to_binary_target',
    'split_dataset',
    'infer_feature_types',
    'align_dataset_columns',
    'validate_dataset_consistency',
    
    # Enhanced preprocessing
    'create_enhanced_preprocessor',
    'create_balanced_dataset',
    'AdvancedFeatureEngineer',
    'OutlierHandler', 
    'SmartTargetEncoder',
    'FeedbackLearner',
    
    # Transfer learning
    'AdaptiveTransferLearner',
    'MultiStageTransferLearner',
    'create_enhanced_transfer_learning_pipeline'
]