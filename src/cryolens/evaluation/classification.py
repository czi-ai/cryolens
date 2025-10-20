"""
Statistical evaluation for classification tasks.

Provides cross-validation and significance testing for multi-class classification
without any hardcoded paths.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score
)
from scipy.stats import ttest_rel


def stratified_cross_validation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 10,
    random_seed: int = 171717,
    return_predictions: bool = False
) -> Dict:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        embeddings: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,) - can be strings or integers
        n_folds: Number of folds
        random_seed: Random seed for reproducibility
        return_predictions: If True, return predictions for per-class analysis
        
    Returns:
        Dictionary containing:
            - 'map_per_fold': MAP score for each fold
            - 'accuracy_per_fold': Accuracy for each fold
            - 'mean_map': Mean MAP across folds
            - 'std_map': Standard deviation of MAP
            - 'mean_accuracy': Mean accuracy
            - 'std_accuracy': Standard deviation of accuracy
            - 'predictions' (optional): List of (y_true, y_pred, y_scores) per fold
            
    Examples:
        >>> embeddings = np.random.randn(100, 40)
        >>> labels = np.repeat(['class_a', 'class_b', 'class_c'], [30, 35, 35])
        >>> results = stratified_cross_validation(embeddings, labels, n_folds=5)
        >>> print(f"MAP: {results['mean_map']:.3f} ± {results['std_map']:.3f}")
    """
    # Encode labels if they are strings
    if isinstance(labels[0], str):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        original_labels = labels
    else:
        labels_encoded = labels
        le = None
        original_labels = None
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    map_scores = []
    accuracy_scores = []
    predictions = [] if return_predictions else None
    
    for train_idx, test_idx in skf.split(embeddings, labels_encoded):
        # Split data
        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train = labels_encoded[train_idx]
        y_test = labels_encoded[test_idx]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=random_seed)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_scores = clf.predict_proba(X_test_scaled)
        
        # Store predictions if requested
        if return_predictions:
            # Store original string labels if available
            if le is not None:
                y_test_labels = [original_labels[i] for i in test_idx]
                y_pred_labels = le.inverse_transform(y_pred)
                predictions.append((y_test_labels, y_pred_labels, y_scores, clf.classes_))
            else:
                predictions.append((y_test, y_pred, y_scores, clf.classes_))
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # MAP for multi-class (one-vs-rest)
        map_scores_class = []
        unique_classes = np.unique(y_train)
        for i, class_label in enumerate(unique_classes):
            y_true_binary = (y_test == class_label).astype(int)
            
            # Find column index for this class in predictions
            class_idx = np.where(clf.classes_ == class_label)[0]
            if len(class_idx) > 0:
                y_score_binary = y_scores[:, class_idx[0]]
                
                # Only compute if there are positive samples
                if np.sum(y_true_binary) > 0:
                    map_score = average_precision_score(y_true_binary, y_score_binary)
                    map_scores_class.append(map_score)
        
        if map_scores_class:
            map_scores.append(np.mean(map_scores_class))
        else:
            map_scores.append(0.0)
        
        accuracy_scores.append(accuracy)
    
    result = {
        'map_per_fold': np.array(map_scores),
        'accuracy_per_fold': np.array(accuracy_scores),
        'mean_map': float(np.mean(map_scores)),
        'std_map': float(np.std(map_scores)),
        'mean_accuracy': float(np.mean(accuracy_scores)),
        'std_accuracy': float(np.std(accuracy_scores))
    }
    
    if return_predictions:
        result['predictions'] = predictions
        result['label_encoder'] = le
    
    return result


def compute_per_class_metrics_from_predictions(
    predictions: List[Tuple],
    class_names: List[str]
) -> Dict[str, Dict]:
    """
    Compute per-class metrics from saved CV predictions (fast - no retraining).
    
    This function reuses predictions from stratified_cross_validation() to compute
    per-class metrics without redundant training.
    
    Args:
        predictions: List of (y_true, y_pred, y_scores, classes) tuples from each fold
        class_names: List of class names to evaluate
        
    Returns:
        Dictionary mapping class names to their metrics
    """
    per_class_results = {}
    
    # Initialize storage for each class
    for class_name in class_names:
        per_class_results[class_name] = {
            'map_per_fold': [],
            'precision_per_fold': [],
            'recall_per_fold': []
        }
    
    # Process each fold's predictions
    for y_true, y_pred, y_scores, clf_classes in predictions:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # For each class, compute metrics
        for class_name in class_names:
            # Binary indicators for this class
            y_true_binary = (y_true == class_name).astype(int)
            y_pred_binary = (y_pred == class_name).astype(int)
            
            # Find column index for this class in probability scores
            if class_name in clf_classes:
                class_idx = np.where(clf_classes == class_name)[0][0]
                y_score_binary = y_scores[:, class_idx]
            else:
                # Class not present in this fold
                continue
            
            # MAP (only if class exists in test set)
            if np.sum(y_true_binary) > 0:
                map_score = average_precision_score(y_true_binary, y_score_binary)
                per_class_results[class_name]['map_per_fold'].append(map_score)
            
            # Precision and recall
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            per_class_results[class_name]['precision_per_fold'].append(precision)
            per_class_results[class_name]['recall_per_fold'].append(recall)
    
    # Aggregate results
    for class_name in class_names:
        for metric in ['map', 'precision', 'recall']:
            values = per_class_results[class_name][f'{metric}_per_fold']
            if values:
                per_class_results[class_name][f'mean_{metric}'] = float(np.mean(values))
                per_class_results[class_name][f'std_{metric}'] = float(np.std(values))
            else:
                per_class_results[class_name][f'mean_{metric}'] = 0.0
                per_class_results[class_name][f'std_{metric}'] = 0.0
    
    return per_class_results


def compute_per_class_metrics(
    embeddings: np.ndarray,
    labels: List[str],
    class_names: Optional[List[str]] = None,
    n_folds: int = 10,
    random_seed: int = 171717
) -> Dict[str, Dict]:
    """
    Compute per-class metrics with cross-validation.
    
    This is a convenience wrapper that runs CV and computes per-class metrics.
    For better performance when calling multiple times, use:
    1. stratified_cross_validation(..., return_predictions=True) once
    2. compute_per_class_metrics_from_predictions() multiple times
    
    Args:
        embeddings: Feature vectors
        labels: Class labels (strings)
        class_names: List of class names to evaluate (if None, uses all unique labels)
        n_folds: Number of CV folds
        random_seed: Random seed
        
    Returns:
        Dictionary mapping class names to their metrics
        
    Examples:
        >>> embeddings = np.random.randn(100, 40)
        >>> labels = ['class_a'] * 30 + ['class_b'] * 35 + ['class_c'] * 35
        >>> results = compute_per_class_metrics(embeddings, labels)
        >>> print(results['class_a']['mean_map'])
    """
    if class_names is None:
        class_names = sorted(set(labels))
    
    # Run CV with prediction tracking
    cv_results = stratified_cross_validation(
        embeddings, 
        labels, 
        n_folds=n_folds, 
        random_seed=random_seed,
        return_predictions=True
    )
    
    # Compute per-class metrics from saved predictions
    return compute_per_class_metrics_from_predictions(
        cv_results['predictions'],
        class_names
    )


def compute_statistical_significance(
    baseline_scores: np.ndarray,
    method_scores: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute statistical significance using paired t-test.
    
    Args:
        baseline_scores: Scores from baseline method (n_folds,)
        method_scores: Scores from test method (n_folds,)
        alpha: Significance level
        
    Returns:
        Dictionary containing:
            - 'p_value': Two-tailed p-value
            - 'mean_improvement': Mean difference
            - 'ci_lower': Lower bound of 95% CI
            - 'ci_upper': Upper bound of 95% CI
            - 'is_significant': Whether p < alpha
            
    Examples:
        >>> baseline = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
        >>> method = np.array([0.55, 0.57, 0.53, 0.56, 0.54])
        >>> sig = compute_statistical_significance(baseline, method)
        >>> print(f"p-value: {sig['p_value']:.4f}")
    """
    from scipy import stats
    
    t_stat, p_value = ttest_rel(method_scores, baseline_scores)
    
    # Compute improvement
    diff = method_scores - baseline_scores
    mean_improvement = np.mean(diff)
    
    # 95% confidence interval
    ci = stats.t.ppf(1 - alpha/2, len(diff) - 1) * np.std(diff, ddof=1) / np.sqrt(len(diff))
    
    return {
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'mean_improvement': float(mean_improvement),
        'ci_lower': float(mean_improvement - ci),
        'ci_upper': float(mean_improvement + ci),
        'is_significant': bool(p_value < alpha)
    }
