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
    random_seed: int = 171717
) -> Dict[str, np.ndarray]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        embeddings: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,) - can be strings or integers
        n_folds: Number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'map_per_fold': MAP score for each fold
            - 'accuracy_per_fold': Accuracy for each fold
            - 'mean_map': Mean MAP across folds
            - 'std_map': Standard deviation of MAP
            - 'mean_accuracy': Mean accuracy
            - 'std_accuracy': Standard deviation of accuracy
            
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
    else:
        labels_encoded = labels
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    map_scores = []
    accuracy_scores = []
    
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
    
    return {
        'map_per_fold': np.array(map_scores),
        'accuracy_per_fold': np.array(accuracy_scores),
        'mean_map': float(np.mean(map_scores)),
        'std_map': float(np.std(map_scores)),
        'mean_accuracy': float(np.mean(accuracy_scores)),
        'std_accuracy': float(np.std(accuracy_scores))
    }


def compute_per_class_metrics(
    embeddings: np.ndarray,
    labels: List[str],
    class_names: Optional[List[str]] = None,
    n_folds: int = 10,
    random_seed: int = 171717
) -> Dict[str, Dict]:
    """
    Compute per-class metrics with cross-validation.
    
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
    
    # Encode labels
    le = LabelEncoder()
    le.fit(class_names)
    labels_encoded = np.array([le.transform([l])[0] if l in class_names else -1 for l in labels])
    
    # Filter out labels not in class_names
    valid_mask = labels_encoded >= 0
    embeddings = embeddings[valid_mask]
    labels_encoded = labels_encoded[valid_mask]
    labels_filtered = [labels[i] for i in range(len(labels)) if valid_mask[i]]
    
    per_class_results = {}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Initialize storage for each class
    for class_name in class_names:
        per_class_results[class_name] = {
            'map_per_fold': [],
            'precision_per_fold': [],
            'recall_per_fold': []
        }
    
    for train_idx, test_idx in skf.split(embeddings, labels_encoded):
        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train = labels_encoded[train_idx]
        y_test = labels_encoded[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, random_state=random_seed)
        clf.fit(X_train_scaled, y_train)
        
        y_scores = clf.predict_proba(X_test_scaled)
        y_pred = clf.predict(X_test_scaled)
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            class_encoded = le.transform([class_name])[0]
            y_true_binary = (y_test == class_encoded).astype(int)
            y_pred_binary = (y_pred == class_encoded).astype(int)
            
            # Find column index for this class
            class_idx = np.where(clf.classes_ == class_encoded)[0]
            if len(class_idx) == 0:
                continue
            
            y_score_binary = y_scores[:, class_idx[0]]
            
            # MAP
            if np.sum(y_true_binary) > 0:  # Only if class exists in test set
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
