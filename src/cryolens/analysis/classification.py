"""
Classification performance evaluation for CryoLens embeddings.

This module provides tools for evaluating the discriminative power of
learned embeddings through classification tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, matthews_corrcoef
)
import logging

logger = logging.getLogger(__name__)

# Import classifiers with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install for classification evaluation.")


class ClassificationEvaluator:
    """
    Evaluator for classification performance on embeddings.
    
    This class provides methods for evaluating how well learned embeddings
    can discriminate between different protein structures.
    
    Attributes:
        classifier_type (str): Type of classifier to use
        n_folds (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        classifier_type: str = 'random_forest',
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize classification evaluator.
        
        Args:
            classifier_type: Type of classifier ('random_forest', 'logistic', 'svm')
            n_folds: Number of cross-validation folds
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Classification evaluation requires scikit-learn")
        
        self.classifier_type = classifier_type
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Initialize classifier
        self.classifier = self._create_classifier()
    
    def _create_classifier(self):
        """Create classifier based on type."""
        if self.classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.classifier_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                multi_class='multinomial'
            )
        elif self.classifier_type == 'svm':
            return SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        test_embeddings: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification performance.
        
        Args:
            embeddings: Training embeddings (n_samples x n_features)
            labels: Training labels
            test_embeddings: Optional test embeddings
            test_labels: Optional test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Encode labels if they're strings
        le = LabelEncoder()
        y_train = le.fit_transform(labels)
        
        results = {
            'classifier': self.classifier_type,
            'n_classes': len(le.classes_),
            'n_train_samples': len(embeddings),
            'n_features': embeddings.shape[1]
        }
        
        # Cross-validation on training set
        if len(np.unique(y_train)) > 1:
            cv = StratifiedKFold(n_splits=min(self.n_folds, min(np.bincount(y_train))))
            
            # Multiple metrics
            cv_scores = {}
            for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
                scores = cross_val_score(
                    self.classifier, embeddings, y_train,
                    cv=cv, scoring=metric, n_jobs=-1
                )
                cv_scores[f'cv_{metric}_mean'] = float(np.mean(scores))
                cv_scores[f'cv_{metric}_std'] = float(np.std(scores))
            
            results.update(cv_scores)
        
        # Train on full training set
        self.classifier.fit(embeddings, y_train)
        
        # Training set performance
        y_train_pred = self.classifier.predict(embeddings)
        results['train_accuracy'] = float(accuracy_score(y_train, y_train_pred))
        results['train_f1_macro'] = float(f1_score(y_train, y_train_pred, average='macro'))
        results['train_f1_weighted'] = float(f1_score(y_train, y_train_pred, average='weighted'))
        results['train_mcc'] = float(matthews_corrcoef(y_train, y_train_pred))
        
        # Test set performance if provided
        if test_embeddings is not None and test_labels is not None:
            y_test = le.transform(test_labels)
            y_test_pred = self.classifier.predict(test_embeddings)
            
            results['n_test_samples'] = len(test_embeddings)
            results['test_accuracy'] = float(accuracy_score(y_test, y_test_pred))
            results['test_f1_macro'] = float(f1_score(y_test, y_test_pred, average='macro'))
            results['test_f1_weighted'] = float(f1_score(y_test, y_test_pred, average='weighted'))
            results['test_precision_macro'] = float(precision_score(y_test, y_test_pred, average='macro'))
            results['test_recall_macro'] = float(recall_score(y_test, y_test_pred, average='macro'))
            results['test_mcc'] = float(matthews_corrcoef(y_test, y_test_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            results['test_confusion_matrix'] = cm.tolist()
            
            # Per-class metrics
            report = classification_report(
                y_test, y_test_pred,
                target_names=le.classes_,
                output_dict=True
            )
            results['test_per_class_metrics'] = report
        
        return results
    
    def compare_classifiers(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        classifier_types: List[str] = ['random_forest', 'logistic', 'svm']
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple classifiers.
        
        Args:
            embeddings: Embeddings to classify
            labels: True labels
            classifier_types: List of classifier types to compare
            
        Returns:
            Dictionary with results for each classifier
        """
        comparison_results = {}
        
        for clf_type in classifier_types:
            try:
                evaluator = ClassificationEvaluator(
                    classifier_type=clf_type,
                    n_folds=self.n_folds,
                    random_state=self.random_state
                )
                results = evaluator.evaluate(embeddings, labels)
                comparison_results[clf_type] = results
            except Exception as e:
                logger.warning(f"Error evaluating {clf_type}: {e}")
                comparison_results[clf_type] = {'error': str(e)}
        
        # Add summary statistics
        summary = {}
        metrics_to_compare = ['cv_accuracy_mean', 'cv_f1_macro_mean', 'train_mcc']
        
        for metric in metrics_to_compare:
            values = []
            for clf_type, results in comparison_results.items():
                if metric in results and 'error' not in results:
                    values.append(results[metric])
            
            if values:
                summary[f'best_{metric}'] = max(values)
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
        
        comparison_results['summary'] = summary
        
        return comparison_results
    
    def evaluate_progressive(
        self,
        embeddings_list: List[np.ndarray],
        labels_list: List[np.ndarray],
        particle_counts: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification with progressive particle counts.
        
        Args:
            embeddings_list: List of embedding sets
            labels_list: List of label sets
            particle_counts: Number of particles for each set
            
        Returns:
            Dictionary with progressive evaluation results
        """
        if particle_counts is None:
            particle_counts = list(range(1, len(embeddings_list) + 1))
        
        progressive_results = {
            'particle_counts': particle_counts,
            'accuracies': [],
            'f1_scores': [],
            'mcc_scores': []
        }
        
        for embeddings, labels in zip(embeddings_list, labels_list):
            if len(np.unique(labels)) < 2:
                # Skip if not enough classes
                progressive_results['accuracies'].append(np.nan)
                progressive_results['f1_scores'].append(np.nan)
                progressive_results['mcc_scores'].append(np.nan)
                continue
            
            try:
                results = self.evaluate(embeddings, labels)
                progressive_results['accuracies'].append(results.get('cv_accuracy_mean', np.nan))
                progressive_results['f1_scores'].append(results.get('cv_f1_macro_mean', np.nan))
                progressive_results['mcc_scores'].append(results.get('train_mcc', np.nan))
            except Exception as e:
                logger.warning(f"Error in progressive evaluation: {e}")
                progressive_results['accuracies'].append(np.nan)
                progressive_results['f1_scores'].append(np.nan)
                progressive_results['mcc_scores'].append(np.nan)
        
        # Calculate trends
        valid_indices = [i for i, v in enumerate(progressive_results['accuracies']) 
                        if not np.isnan(v)]
        
        if len(valid_indices) > 1:
            x = np.array([particle_counts[i] for i in valid_indices])
            y = np.array([progressive_results['accuracies'][i] for i in valid_indices])
            
            # Fit trend
            coeffs = np.polyfit(x, y, 1)
            progressive_results['accuracy_trend'] = {
                'slope': float(coeffs[0]),
                'improving': coeffs[0] > 0
            }
        
        return progressive_results


def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    classifier_type: str = 'random_forest'
) -> Dict[str, Any]:
    """
    Convenience function for embedding evaluation.
    
    Args:
        embeddings: Training embeddings
        labels: Training labels
        test_embeddings: Optional test embeddings
        test_labels: Optional test labels
        classifier_type: Type of classifier
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = ClassificationEvaluator(classifier_type=classifier_type)
    return evaluator.evaluate(embeddings, labels, test_embeddings, test_labels)


def cross_validate_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_types: List[str] = ['random_forest', 'logistic', 'svm'],
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Cross-validate multiple classifiers.
    
    Args:
        embeddings: Embeddings to classify
        labels: True labels
        classifier_types: Classifiers to compare
        n_folds: Number of CV folds
        
    Returns:
        Comparison results
    """
    evaluator = ClassificationEvaluator(n_folds=n_folds)
    return evaluator.compare_classifiers(embeddings, labels, classifier_types)
