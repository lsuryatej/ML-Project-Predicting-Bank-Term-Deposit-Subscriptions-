"""
Enhanced Transfer Learning for Bank Term Deposit Prediction

This module implements sophisticated transfer learning techniques specifically
designed for financial machine learning applications. It provides adaptive
learning strategies that can effectively transfer knowledge from synthetic
data to real customer data.

Key Features:
- Adaptive transfer learning with feedback loops
- Multi-stage progressive learning
- Weighted ensemble methods
- Performance-based model selection
- Comprehensive validation and monitoring

2025
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

class AdaptiveTransferLearner(BaseEstimator, ClassifierMixin):
    """
    Advanced transfer learning with adaptive feedback loops.
    """
    
    def __init__(self, base_models: List, adaptation_strategy='weighted_ensemble',
                 feedback_iterations=3, validation_split=0.2, 
                 performance_threshold=0.02, random_state=42):
        """
        Initialize adaptive transfer learner.
        
        Args:
            base_models: List of (name, model) tuples
            adaptation_strategy: 'weighted_ensemble', 'best_model', 'stacking'
            feedback_iterations: Number of feedback iterations
            validation_split: Validation split ratio
            performance_threshold: Minimum improvement threshold
            random_state: Random state for reproducibility
        """
        self.base_models = base_models
        self.adaptation_strategy = adaptation_strategy
        self.feedback_iterations = feedback_iterations
        self.validation_split = validation_split
        self.performance_threshold = performance_threshold
        self.random_state = random_state
        
        # Internal state
        self.pretrained_models_ = {}
        self.adapted_models_ = {}
        self.model_weights_ = {}
        self.final_model_ = None
        self.adaptation_history_ = []
        self.feature_importance_ = None
        
    def pretrain(self, X_synthetic, y_synthetic):
        """
        Pretrain models on synthetic data.
        """
        print("Pretraining models on synthetic data...")
        
        for name, model in self.base_models:
            print(f"  Pretraining {name}...")
            model_copy = clone(model)
            model_copy.fit(X_synthetic, y_synthetic)
            self.pretrained_models_[name] = model_copy
            
            # Calculate feature importance if available
            if hasattr(model_copy, 'feature_importances_'):
                if self.feature_importance_ is None:
                    self.feature_importance_ = {}
                self.feature_importance_[name] = model_copy.feature_importances_
        
        print(f"Pretrained {len(self.pretrained_models_)} models")
        return self
    
    def adapt_with_feedback(self, X_real, y_real):
        """
        Adapt pretrained models using real data with feedback loops.
        """
        from sklearn.model_selection import train_test_split
        
        print("Adapting models with feedback loops...")
        
        # Split real data for adaptation and validation
        X_adapt, X_val, y_adapt, y_val = train_test_split(
            X_real, y_real, test_size=self.validation_split,
            random_state=self.random_state, stratify=y_real
        )
        
        # Initialize adapted models
        for name, pretrained_model in self.pretrained_models_.items():
            self.adapted_models_[name] = clone(pretrained_model)
            self.model_weights_[name] = 1.0  # Initial equal weights
        
        # Feedback loop iterations
        for iteration in range(self.feedback_iterations):
            print(f"\nFeedback iteration {iteration + 1}/{self.feedback_iterations}")
            
            iteration_scores = {}
            
            # Adapt each model
            for name, model in self.adapted_models_.items():
                # Fine-tune on real data
                model.fit(X_adapt, y_adapt)
                
                # Evaluate on validation set
                y_val_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred)
                iteration_scores[name] = val_f1
                
                print(f"  {name}: F1 = {val_f1:.4f}")
            
            # Update model weights based on performance
            self._update_model_weights(iteration_scores)
            
            # Check for convergence
            if iteration > 0:
                prev_scores = self.adaptation_history_[-1]['scores']
                improvements = {name: iteration_scores[name] - prev_scores[name] 
                              for name in iteration_scores.keys()}
                max_improvement = max(improvements.values())
                
                if max_improvement < self.performance_threshold:
                    print(f"Convergence reached (max improvement: {max_improvement:.4f})")
                    break
            
            # Store iteration history
            self.adaptation_history_.append({
                'iteration': iteration + 1,
                'scores': iteration_scores.copy(),
                'weights': self.model_weights_.copy()
            })
            
            # Adaptive sample weighting for next iteration
            if iteration < self.feedback_iterations - 1:
                X_adapt, y_adapt = self._adaptive_sample_weighting(
                    X_adapt, y_adapt, X_val, y_val
                )
        
        # Create final ensemble model
        self._create_final_model()
        
        return self
    
    def _update_model_weights(self, scores: Dict[str, float]):
        """Update model weights based on performance scores."""
        # Softmax weighting based on F1 scores
        score_values = np.array(list(scores.values()))
        exp_scores = np.exp(score_values - np.max(score_values))  # Numerical stability
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        for i, name in enumerate(scores.keys()):
            self.model_weights_[name] = softmax_weights[i]
    
    def _adaptive_sample_weighting(self, X_adapt, y_adapt, X_val, y_val):
        """
        Create adaptive sample weights based on ensemble predictions.
        """
        # Get ensemble predictions on adaptation set
        ensemble_pred = self._ensemble_predict(X_adapt)
        
        # Identify hard examples (misclassified by ensemble)
        hard_examples = (ensemble_pred != y_adapt)
        
        if hard_examples.sum() > 0:
            print(f"  Found {hard_examples.sum()} hard examples for next iteration")
            
            # Create weighted dataset emphasizing hard examples
            from sklearn.utils import resample
            
            # Oversample hard examples
            hard_indices = np.where(hard_examples)[0]
            easy_indices = np.where(~hard_examples)[0]
            
            # Sample more hard examples
            n_hard_samples = min(len(hard_indices) * 2, len(X_adapt) // 3)
            n_easy_samples = len(X_adapt) - n_hard_samples
            
            if n_hard_samples > 0 and n_easy_samples > 0:
                hard_sample_indices = resample(hard_indices, n_samples=n_hard_samples, 
                                             random_state=self.random_state)
                easy_sample_indices = resample(easy_indices, n_samples=n_easy_samples,
                                             random_state=self.random_state)
                
                combined_indices = np.concatenate([hard_sample_indices, easy_sample_indices])
                
                return X_adapt.iloc[combined_indices], y_adapt.iloc[combined_indices]
        
        return X_adapt, y_adapt
    
    def _ensemble_predict(self, X):
        """Make ensemble predictions using current model weights."""
        predictions = []
        weights = []
        
        for name, model in self.adapted_models_.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.model_weights_[name])
        
        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return (weighted_pred > 0.5).astype(int)
    
    def _create_final_model(self):
        """Create final ensemble model based on adaptation strategy."""
        if self.adaptation_strategy == 'weighted_ensemble':
            # Create weighted voting classifier
            estimators = [(name, model) for name, model in self.adapted_models_.items()]
            weights = [self.model_weights_[name] for name, _ in estimators]
            
            self.final_model_ = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )
            
        elif self.adaptation_strategy == 'best_model':
            # Select best performing model
            best_name = max(self.model_weights_.keys(), 
                          key=lambda x: self.model_weights_[x])
            self.final_model_ = self.adapted_models_[best_name]
            print(f"Selected best model: {best_name}")
            
        elif self.adaptation_strategy == 'stacking':
            # Create stacking ensemble
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            
            estimators = [(name, model) for name, model in self.adapted_models_.items()]
            self.final_model_ = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=self.random_state),
                cv=3
            )
    
    def fit(self, X, y):
        """Fit the final model on full real data."""
        if self.final_model_ is None:
            raise ValueError("Must call pretrain() and adapt_with_feedback() first!")
        
        self.final_model_.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the final model."""
        if self.final_model_ is None:
            raise ValueError("Model not fitted!")
        return self.final_model_.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.final_model_ is None:
            raise ValueError("Model not fitted!")
        return self.final_model_.predict_proba(X)
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation process."""
        if not self.adaptation_history_:
            return {}
        
        final_scores = self.adaptation_history_[-1]['scores']
        initial_scores = self.adaptation_history_[0]['scores']
        
        improvements = {name: final_scores[name] - initial_scores[name] 
                       for name in final_scores.keys()}
        
        return {
            'iterations': len(self.adaptation_history_),
            'final_scores': final_scores,
            'improvements': improvements,
            'best_model': max(final_scores.keys(), key=lambda x: final_scores[x]),
            'model_weights': self.model_weights_
        }

class MultiStageTransferLearner:
    """
    Multi-stage transfer learning with progressive adaptation.
    """
    
    def __init__(self, stages: List[Dict], random_state=42):
        """
        Initialize multi-stage transfer learner.
        
        Args:
            stages: List of stage configurations
                   Each stage: {'data_fraction': float, 'models': List, 'iterations': int}
            random_state: Random state for reproducibility
        """
        self.stages = stages
        self.random_state = random_state
        self.stage_results_ = []
        self.final_model_ = None
    
    def fit(self, X_synthetic, y_synthetic, X_real, y_real):
        """
        Fit using multi-stage progressive transfer learning.
        """
        from sklearn.model_selection import train_test_split
        
        print("Starting multi-stage transfer learning...")
        
        current_models = None
        
        for stage_idx, stage_config in enumerate(self.stages):
            print(f"\n=== Stage {stage_idx + 1}/{len(self.stages)} ===")
            
            data_fraction = stage_config['data_fraction']
            models = stage_config['models']
            iterations = stage_config.get('iterations', 3)
            
            # Sample real data for this stage
            if data_fraction < 1.0:
                X_stage, _, y_stage, _ = train_test_split(
                    X_real, y_real, train_size=data_fraction,
                    random_state=self.random_state, stratify=y_real
                )
            else:
                X_stage, y_stage = X_real, y_real
            
            print(f"Using {len(X_stage)} real samples ({data_fraction*100:.1f}%)")
            
            # Create adaptive transfer learner for this stage
            if current_models is None:
                # First stage: use provided models
                stage_models = models
            else:
                # Later stages: use models from previous stage
                stage_models = [(name, model) for name, model in current_models.items()]
            
            learner = AdaptiveTransferLearner(
                base_models=stage_models,
                feedback_iterations=iterations,
                random_state=self.random_state
            )
            
            # Pretrain (skip if not first stage)
            if stage_idx == 0:
                learner.pretrain(X_synthetic, y_synthetic)
            else:
                learner.pretrained_models_ = current_models
            
            # Adapt with feedback
            learner.adapt_with_feedback(X_stage, y_stage)
            
            # Fit final model
            learner.fit(X_stage, y_stage)
            
            # Store results
            stage_summary = learner.get_adaptation_summary()
            stage_summary['stage'] = stage_idx + 1
            stage_summary['data_fraction'] = data_fraction
            stage_summary['data_size'] = len(X_stage)
            self.stage_results_.append(stage_summary)
            
            # Update current models for next stage
            current_models = learner.adapted_models_
            self.final_model_ = learner.final_model_
        
        print("\nMulti-stage transfer learning completed!")
        return self
    
    def predict(self, X):
        """Make predictions using the final model."""
        if self.final_model_ is None:
            raise ValueError("Model not fitted!")
        return self.final_model_.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.final_model_ is None:
            raise ValueError("Model not fitted!")
        return self.final_model_.predict_proba(X)
    
    def get_stage_summary(self) -> pd.DataFrame:
        """Get summary of all stages."""
        if not self.stage_results_:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.stage_results_:
            row = {
                'stage': result['stage'],
                'data_fraction': result['data_fraction'],
                'data_size': result['data_size'],
                'iterations': result['iterations'],
                'best_model': result['best_model']
            }
            
            # Add final scores
            for model_name, score in result['final_scores'].items():
                row[f'{model_name}_f1'] = score
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

def create_enhanced_transfer_learning_pipeline(
    synthetic_data: Tuple[pd.DataFrame, pd.Series],
    real_data: Tuple[pd.DataFrame, pd.Series],
    base_models: List,
    strategy: str = 'adaptive',
    **kwargs
) -> Dict[str, Any]:
    """
    Create and run enhanced transfer learning pipeline.
    
    Args:
        synthetic_data: (X_synthetic, y_synthetic)
        real_data: (X_real, y_real)
        base_models: List of (name, model) tuples
        strategy: 'adaptive', 'multi_stage', or 'progressive'
        **kwargs: Additional parameters for the strategy
    
    Returns:
        Dictionary with results and trained model
    """
    X_synthetic, y_synthetic = synthetic_data
    X_real, y_real = real_data
    
    results = {
        'strategy': strategy,
        'synthetic_size': len(X_synthetic),
        'real_size': len(X_real),
        'models_used': [name for name, _ in base_models]
    }
    
    if strategy == 'adaptive':
        learner = AdaptiveTransferLearner(
            base_models=base_models,
            **kwargs
        )
        
        learner.pretrain(X_synthetic, y_synthetic)
        learner.adapt_with_feedback(X_real, y_real)
        learner.fit(X_real, y_real)
        
        results['adaptation_summary'] = learner.get_adaptation_summary()
        results['model'] = learner
        
    elif strategy == 'multi_stage':
        # Default multi-stage configuration
        default_stages = [
            {'data_fraction': 0.1, 'models': base_models, 'iterations': 2},
            {'data_fraction': 0.3, 'models': base_models, 'iterations': 3},
            {'data_fraction': 1.0, 'models': base_models, 'iterations': 3}
        ]
        
        stages = kwargs.get('stages', default_stages)
        
        learner = MultiStageTransferLearner(stages=stages)
        learner.fit(X_synthetic, y_synthetic, X_real, y_real)
        
        results['stage_summary'] = learner.get_stage_summary()
        results['model'] = learner
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return results