"""
Enhanced Preprocessing Pipeline for Bank Term Deposit Prediction

This module provides advanced preprocessing capabilities including feature engineering,
outlier handling, target encoding, and feedback-based learning mechanisms. It extends
standard scikit-learn preprocessing with domain-specific enhancements for financial
machine learning applications.

Key Features:
- Advanced feature engineering with interaction and ratio features
- Robust outlier detection and handling
- Smart target encoding with regularization
- Feedback-based iterative learning
- Comprehensive data balancing techniques
- Modular pipeline construction

2025
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering transformer for financial datasets.
    
    This transformer creates sophisticated features including interactions,
    ratios, and aggregates that can improve model performance on financial
    prediction tasks. It's designed to work with both numerical and mixed
    data types commonly found in banking datasets.
    
    Parameters:
        create_interactions: Whether to create interaction features between numerical columns
        create_ratios: Whether to create ratio features between numerical columns  
        create_aggregates: Whether to create aggregate statistics across numerical columns
        polynomial_degree: Degree for polynomial feature expansion (currently unused)
        
    Attributes:
        feature_names_: List of original feature names from training data
    """
    
    def __init__(self, create_interactions: bool = True, create_ratios: bool = True, 
                 create_aggregates: bool = True, polynomial_degree: int = 2):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_aggregates = create_aggregates
        self.polynomial_degree = polynomial_degree
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """
        Fit the feature engineer to the training data.
        
        Args:
            X: Training features (DataFrame or array)
            y: Target values (ignored, for sklearn compatibility)
            
        Returns:
            self: Fitted transformer
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        """
        Transform the input data by creating advanced features.
        
        Args:
            X: Input features to transform
            
        Returns:
            DataFrame with original features plus engineered features
        """
        # Ensure DataFrame format
        if isinstance(X, pd.DataFrame):
            transformed_data = X.copy()
        else:
            column_names = (self.feature_names_ if self.feature_names_ 
                          else [f'feature_{i}' for i in range(X.shape[1])])
            transformed_data = pd.DataFrame(X, columns=column_names)
        
        # Identify numerical columns for feature engineering
        numerical_columns = transformed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create interaction features (multiplicative combinations)
        if self.create_interactions and len(numerical_columns) >= 2:
            print(f"Creating interaction features from {len(numerical_columns)} numerical columns")
            # Limit to top 5 columns to prevent feature explosion
            for i, first_column in enumerate(numerical_columns[:5]):
                for second_column in numerical_columns[i+1:6]:
                    interaction_name = f'{first_column}_x_{second_column}'
                    transformed_data[interaction_name] = (transformed_data[first_column] * 
                                                        transformed_data[second_column])
        
        # Create ratio features (division-based relationships)
        if self.create_ratios and len(numerical_columns) >= 2:
            print(f"Creating ratio features from {len(numerical_columns)} numerical columns")
            for i, numerator_column in enumerate(numerical_columns[:5]):
                for denominator_column in numerical_columns[i+1:6]:
                    # Prevent division by zero with small epsilon
                    safe_denominator = transformed_data[denominator_column].replace(0, 1e-8)
                    ratio_name = f'{numerator_column}_div_{denominator_column}'
                    transformed_data[ratio_name] = (transformed_data[numerator_column] / 
                                                  safe_denominator)
        
        # Create aggregate statistical features
        if self.create_aggregates and len(numerical_columns) > 1:
            print(f"Creating aggregate features from {len(numerical_columns)} numerical columns")
            numerical_data = transformed_data[numerical_columns]
            
            # Statistical aggregates across all numerical features
            transformed_data['numerical_sum'] = numerical_data.sum(axis=1)
            transformed_data['numerical_mean'] = numerical_data.mean(axis=1)
            transformed_data['numerical_std'] = numerical_data.std(axis=1).fillna(0)
            transformed_data['numerical_max'] = numerical_data.max(axis=1)
            transformed_data['numerical_min'] = numerical_data.min(axis=1)
            transformed_data['numerical_range'] = (transformed_data['numerical_max'] - 
                                                 transformed_data['numerical_min'])
        
        print(f"Feature engineering completed: {X.shape[1]} → {transformed_data.shape[1]} features")
        return transformed_data

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using Isolation Forest or statistical methods."""
    
    def __init__(self, method='isolation_forest', contamination=0.1, 
                 statistical_threshold=3.0):
        self.method = method
        self.contamination = contamination
        self.statistical_threshold = statistical_threshold
        self.outlier_detector_ = None
        self.numerical_cols_ = None
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
            X_num = X[self.numerical_cols_]
        else:
            X_num = X
            
        if self.method == 'isolation_forest':
            self.outlier_detector_ = IsolationForest(
                contamination=self.contamination, 
                random_state=42
            )
            self.outlier_detector_.fit(X_num)
        elif self.method == 'statistical':
            # Store statistical thresholds
            self.means_ = X_num.mean()
            self.stds_ = X_num.std()
            
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
            X_num = X_new[self.numerical_cols_]
        else:
            X_new = pd.DataFrame(X)
            X_num = X_new
            
        if self.method == 'isolation_forest':
            outlier_mask = self.outlier_detector_.predict(X_num) == -1
            # Cap outliers at 95th percentile
            for col in X_num.columns:
                upper_bound = X_num[col].quantile(0.95)
                lower_bound = X_num[col].quantile(0.05)
                X_new.loc[outlier_mask, col] = np.clip(
                    X_new.loc[outlier_mask, col], 
                    lower_bound, 
                    upper_bound
                )
        elif self.method == 'statistical':
            # Cap values beyond statistical threshold
            for col in X_num.columns:
                if hasattr(self.means_, 'loc'):
                    # If means_ is a pandas Series
                    upper_bound = self.means_.loc[col] + self.statistical_threshold * self.stds_.loc[col]
                    lower_bound = self.means_.loc[col] - self.statistical_threshold * self.stds_.loc[col]
                else:
                    # If means_ is a dictionary or other indexable
                    upper_bound = self.means_[col] + self.statistical_threshold * self.stds_[col]
                    lower_bound = self.means_[col] - self.statistical_threshold * self.stds_[col]
                X_new[col] = np.clip(X_new[col], lower_bound, upper_bound)
                
        return X_new

class SmartTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with regularization and cross-validation."""
    
    def __init__(self, smoothing=1.0, min_samples_leaf=1, noise_level=0.01):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.encoders_ = {}
        self.global_mean_ = None
        
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        else:
            categorical_cols = range(X.shape[1])
            X = pd.DataFrame(X, columns=[f'cat_{i}' for i in categorical_cols])
            
        self.global_mean_ = y.mean()
        
        for col in categorical_cols:
            # Calculate target mean for each category
            category_stats = pd.DataFrame({
                'target_mean': y.groupby(X[col]).mean(),
                'count': y.groupby(X[col]).count()
            }).fillna(self.global_mean_)
            
            # Apply smoothing
            smoothed_means = (
                category_stats['count'] * category_stats['target_mean'] + 
                self.smoothing * self.global_mean_
            ) / (category_stats['count'] + self.smoothing)
            
            self.encoders_[col] = smoothed_means.to_dict()
            
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        else:
            categorical_cols = range(X.shape[1])
            X_new = pd.DataFrame(X, columns=[f'cat_{i}' for i in categorical_cols])
            
        for col in categorical_cols:
            if col in self.encoders_:
                X_new[col] = X_new[col].map(self.encoders_[col]).fillna(self.global_mean_)
                # Add small amount of noise to prevent overfitting
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, len(X_new))
                    X_new[col] += noise
                    
        return X_new

def create_enhanced_preprocessor(numerical_cols: List[str], 
                               categorical_cols: List[str],
                               target_col: str = None,
                               use_advanced_features: bool = True,
                               use_outlier_handling: bool = True,
                               use_target_encoding: bool = True,
                               scaling_method: str = 'robust',
                               feature_selection_k: int = None) -> Pipeline:
    """
    Create an enhanced preprocessing pipeline.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        target_col: Target column name (for target encoding)
        use_advanced_features: Whether to create advanced features
        use_outlier_handling: Whether to handle outliers
        use_target_encoding: Whether to use target encoding for categoricals
        scaling_method: 'standard', 'robust', or 'power'
        feature_selection_k: Number of features to select (None for no selection)
    
    Returns:
        Enhanced preprocessing pipeline
    """
    
    # Numerical pipeline
    num_steps = []
    
    # Imputation
    num_steps.append(('imputer', KNNImputer(n_neighbors=5)))
    
    # Outlier handling
    if use_outlier_handling:
        num_steps.append(('outlier_handler', OutlierHandler(method='statistical')))
    
    # Scaling
    if scaling_method == 'standard':
        num_steps.append(('scaler', StandardScaler()))
    elif scaling_method == 'robust':
        num_steps.append(('scaler', RobustScaler()))
    elif scaling_method == 'power':
        num_steps.append(('scaler', PowerTransformer(method='yeo-johnson')))
    
    num_pipeline = Pipeline(num_steps)
    
    # Categorical pipeline
    cat_steps = []
    
    # Imputation
    cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
    
    # Encoding
    if use_target_encoding and target_col:
        cat_steps.append(('encoder', SmartTargetEncoder()))
    else:
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    
    cat_pipeline = Pipeline(cat_steps)
    
    # Combine pipelines
    preprocessor_steps = []
    
    # Column transformer
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='passthrough')
    
    preprocessor_steps.append(('column_transform', column_transformer))
    
    # Advanced feature engineering
    if use_advanced_features:
        preprocessor_steps.append(('feature_engineer', AdvancedFeatureEngineer()))
    
    # Feature selection
    if feature_selection_k:
        preprocessor_steps.append(('feature_selection', 
                                 SelectKBest(score_func=f_classif, k=feature_selection_k)))
    
    return Pipeline(preprocessor_steps)

class FeedbackLearner:
    """
    Implements feedback loop learning for iterative model improvement.
    """
    
    def __init__(self, base_model, max_iterations=5, improvement_threshold=0.01,
                 validation_split=0.2, random_state=42):
        self.base_model = base_model
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.validation_split = validation_split
        self.random_state = random_state
        self.iteration_history_ = []
        self.best_model_ = None
        self.best_score_ = -np.inf
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit model with feedback loop learning.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        
        # Initial train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, 
            random_state=self.random_state, stratify=y
        )
        
        current_model = self.base_model
        current_score = -np.inf
        
        for iteration in range(self.max_iterations):
            print(f"Feedback iteration {iteration + 1}/{self.max_iterations}")
            
            # Fit current model
            if sample_weight is not None:
                current_model.fit(X_train, y_train, 
                                sample_weight=sample_weight[:len(X_train)])
            else:
                current_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = current_model.predict(X_val)
            val_score = f1_score(y_val, y_val_pred)
            
            print(f"  Validation F1: {val_score:.4f}")
            
            # Check for improvement
            improvement = val_score - current_score
            
            if val_score > self.best_score_:
                self.best_score_ = val_score
                self.best_model_ = current_model
                print(f"  New best score: {val_score:.4f}")
            
            # Store iteration info
            self.iteration_history_.append({
                'iteration': iteration + 1,
                'validation_f1': val_score,
                'improvement': improvement
            })
            
            # Check stopping criteria
            if improvement < self.improvement_threshold:
                print(f"  Improvement below threshold ({self.improvement_threshold}). Stopping.")
                break
                
            # Prepare for next iteration with hard examples focus
            if iteration < self.max_iterations - 1:
                # Identify hard examples (misclassified)
                y_train_pred = current_model.predict(X_train)
                hard_examples = y_train != y_train_pred
                
                if hard_examples.sum() > 0:
                    # Create sample weights emphasizing hard examples
                    sample_weight_new = np.ones(len(X_train))
                    sample_weight_new[hard_examples] *= 2.0  # Double weight for hard examples
                    
                    # Update sample weights for next iteration
                    if sample_weight is None:
                        sample_weight = sample_weight_new
                    else:
                        sample_weight[:len(X_train)] = sample_weight_new
                        
            current_score = val_score
        
        # Final fit on full data with best configuration
        if self.best_model_ is not None:
            self.best_model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X):
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet!")
        return self.best_model_.predict(X)
    
    def predict_proba(self, X):
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet!")
        return self.best_model_.predict_proba(X)

def create_balanced_dataset(X, y, strategy='smote', random_state=42):
    """
    Create balanced dataset using various resampling techniques.
    
    Args:
        X: Features
        y: Target
        strategy: 'smote', 'adasyn', 'smote_tomek', 'undersample'
        random_state: Random state for reproducibility
    
    Returns:
        Balanced X, y
    """
    print(f"Original class distribution: {np.bincount(y)}")
    
    if strategy == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    
    return X_balanced, y_balanced