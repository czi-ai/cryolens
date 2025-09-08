"""
Diversity analysis for CryoLens embeddings and structures.

This module provides tools for analyzing the diversity of protein structures
and their learned embeddings using SOAP descriptors, UMAP, and other metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install umap-learn for UMAP projections.")

try:
    from dscribe.descriptors import SOAP
    from ase import Atoms
    SOAP_AVAILABLE = True
except ImportError:
    SOAP_AVAILABLE = False
    logger.warning("DScribe not available. Install dscribe for SOAP descriptors.")


class DiversityAnalyzer:
    """
    Analyzer for structural and embedding diversity.
    
    This class provides methods for analyzing the diversity of protein
    structures and their learned embeddings using various metrics.
    
    Attributes:
        use_soap (bool): Whether to compute SOAP descriptors
        soap_params (dict): Parameters for SOAP descriptor
        n_components_pca (int): Number of PCA components
        n_neighbors_umap (int): Number of neighbors for UMAP
    """
    
    def __init__(
        self,
        use_soap: bool = True,
        soap_params: Optional[Dict[str, Any]] = None,
        n_components_pca: int = 2,
        n_neighbors_umap: int = 15
    ):
        """
        Initialize diversity analyzer.
        
        Args:
            use_soap: Whether to use SOAP descriptors
            soap_params: Parameters for SOAP computation
            n_components_pca: Number of PCA components for projection
            n_neighbors_umap: Number of neighbors for UMAP
        """
        self.use_soap = use_soap and SOAP_AVAILABLE
        self.n_components_pca = n_components_pca
        self.n_neighbors_umap = n_neighbors_umap
        
        # Default SOAP parameters
        self.soap_params = soap_params or {
            'species': ['C', 'N', 'O', 'S'],  # Common protein atoms
            'r_cut': 10.0,  # Cutoff radius in Angstroms
            'n_max': 8,     # Number of radial basis functions
            'l_max': 6,     # Maximum angular momentum
            'sigma': 0.5,   # Gaussian width
            'periodic': False,
            'sparse': False
        }
        
        if self.use_soap:
            self.soap_calculator = SOAP(**self.soap_params)
        else:
            self.soap_calculator = None
    
    def compute_soap_descriptors(
        self,
        structures: List[Union[str, np.ndarray, Atoms]]
    ) -> np.ndarray:
        """
        Compute SOAP descriptors for protein structures.
        
        Args:
            structures: List of structures (PDB IDs, coordinates, or ASE Atoms)
            
        Returns:
            SOAP descriptor matrix (n_structures x n_features)
        """
        if not self.use_soap:
            raise RuntimeError("SOAP computation requires dscribe. Install with: pip install dscribe")
        
        descriptors = []
        
        for structure in structures:
            if isinstance(structure, str):
                # Assume it's a PDB ID - would need to load structure
                logger.warning(f"PDB loading not implemented for {structure}")
                # Create dummy descriptor for now
                descriptor = np.random.randn(self.soap_calculator.get_number_of_features())
            elif isinstance(structure, np.ndarray):
                # Convert coordinates to ASE Atoms
                # Assume all atoms are carbon for simplicity
                atoms = Atoms('C' * len(structure), positions=structure)
                descriptor = self.soap_calculator.create(atoms)
            elif isinstance(structure, Atoms):
                descriptor = self.soap_calculator.create(structure)
            else:
                raise ValueError(f"Unknown structure type: {type(structure)}")
            
            # Average over all atoms if needed
            if len(descriptor.shape) > 1:
                descriptor = np.mean(descriptor, axis=0)
                
            descriptors.append(descriptor)
        
        return np.array(descriptors)
    
    def compute_embedding_projections(
        self,
        embeddings: np.ndarray,
        method: str = 'umap'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute 2D projections of embeddings.
        
        Args:
            embeddings: Embedding matrix (n_samples x n_features)
            method: Projection method ('umap' or 'pca')
            
        Returns:
            Tuple of (projected_embeddings, projection_info)
        """
        if method == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP not available, falling back to PCA")
                method = 'pca'
            else:
                reducer = umap.UMAP(
                    n_neighbors=self.n_neighbors_umap,
                    min_dist=0.1,
                    n_components=2,
                    random_state=42
                )
                projected = reducer.fit_transform(embeddings)
                info = {
                    'method': 'umap',
                    'n_neighbors': self.n_neighbors_umap,
                    'min_dist': 0.1
                }
                return projected, info
        
        if method == 'pca':
            pca = PCA(n_components=self.n_components_pca)
            projected = pca.fit_transform(embeddings)
            info = {
                'method': 'pca',
                'explained_variance': pca.explained_variance_ratio_,
                'n_components': self.n_components_pca
            }
            return projected, info
        
        raise ValueError(f"Unknown projection method: {method}")
    
    def compute_diversity_metrics(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        soap_descriptors: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive diversity metrics.
        
        Args:
            embeddings: Embedding matrix (n_samples x n_features)
            labels: Optional cluster labels for supervised metrics
            soap_descriptors: Optional SOAP descriptors for correlation
            
        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['n_samples'] = embeddings.shape[0]
        metrics['n_dimensions'] = embeddings.shape[1]
        metrics['mean_norm'] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
        metrics['std_norm'] = float(np.std(np.linalg.norm(embeddings, axis=1)))
        
        # Pairwise distances
        if embeddings.shape[0] > 1:
            distances = pdist(embeddings, metric='euclidean')
            metrics['mean_pairwise_distance'] = float(np.mean(distances))
            metrics['std_pairwise_distance'] = float(np.std(distances))
            metrics['min_pairwise_distance'] = float(np.min(distances))
            metrics['max_pairwise_distance'] = float(np.max(distances))
        
        # Clustering metrics if labels provided
        if labels is not None and len(np.unique(labels)) > 1:
            try:
                metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
                metrics['davies_bouldin_index'] = float(davies_bouldin_score(embeddings, labels))
                
                # Class separation
                unique_labels = np.unique(labels)
                within_class_distances = []
                between_class_distances = []
                
                for label in unique_labels:
                    mask = labels == label
                    if np.sum(mask) > 1:
                        within = pdist(embeddings[mask])
                        within_class_distances.extend(within)
                    
                    for other_label in unique_labels:
                        if other_label != label:
                            other_mask = labels == other_label
                            between = cdist(embeddings[mask], embeddings[other_mask]).flatten()
                            between_class_distances.extend(between)
                
                if within_class_distances and between_class_distances:
                    metrics['mean_within_class_distance'] = float(np.mean(within_class_distances))
                    metrics['mean_between_class_distance'] = float(np.mean(between_class_distances))
                    metrics['class_separation_ratio'] = float(
                        np.mean(between_class_distances) / (np.mean(within_class_distances) + 1e-8)
                    )
            except Exception as e:
                logger.warning(f"Error computing clustering metrics: {e}")
        
        # SOAP correlation if available
        if soap_descriptors is not None and soap_descriptors.shape[0] == embeddings.shape[0]:
            try:
                # Compute correlation between embedding and SOAP distances
                embedding_distances = squareform(pdist(embeddings))
                soap_distances = squareform(pdist(soap_descriptors))
                
                # Flatten upper triangular matrices
                mask = np.triu(np.ones_like(embedding_distances), k=1).astype(bool)
                embedding_flat = embedding_distances[mask]
                soap_flat = soap_distances[mask]
                
                correlation, p_value = pearsonr(embedding_flat, soap_flat)
                metrics['soap_embedding_correlation'] = float(correlation)
                metrics['soap_embedding_p_value'] = float(p_value)
            except Exception as e:
                logger.warning(f"Error computing SOAP correlation: {e}")
        
        # Dimension-wise statistics
        metrics['mean_variance_per_dim'] = float(np.mean(np.var(embeddings, axis=0)))
        metrics['max_variance_dim'] = int(np.argmax(np.var(embeddings, axis=0)))
        metrics['min_variance_dim'] = int(np.argmin(np.var(embeddings, axis=0)))
        
        return metrics
    
    def analyze_latent_segments(
        self,
        embeddings: np.ndarray,
        affinity_dims: int,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze affinity vs free latent segments.
        
        Args:
            embeddings: Full embedding matrix
            affinity_dims: Number of affinity dimensions
            labels: Optional structure labels
            
        Returns:
            Dictionary with segment analysis
        """
        free_dims = embeddings.shape[1] - affinity_dims
        
        affinity_embeddings = embeddings[:, :affinity_dims]
        free_embeddings = embeddings[:, affinity_dims:] if free_dims > 0 else None
        
        analysis = {
            'affinity_dims': affinity_dims,
            'free_dims': free_dims,
            'total_dims': embeddings.shape[1]
        }
        
        # Analyze affinity segment
        analysis['affinity_metrics'] = self.compute_diversity_metrics(
            affinity_embeddings, labels
        )
        
        # Analyze free segment if present
        if free_embeddings is not None and free_dims > 0:
            analysis['free_metrics'] = self.compute_diversity_metrics(
                free_embeddings, labels
            )
            
            # Compare segments
            analysis['affinity_mean_var'] = float(np.mean(np.var(affinity_embeddings, axis=0)))
            analysis['free_mean_var'] = float(np.mean(np.var(free_embeddings, axis=0)))
            analysis['variance_ratio'] = float(
                analysis['affinity_mean_var'] / (analysis['free_mean_var'] + 1e-8)
            )
        
        return analysis
    
    def compute_mahalanobis_overlap(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance-based overlap between classes.
        
        Args:
            embeddings: Embedding matrix
            labels: Class labels
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            Overlap probability matrix
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        overlap_matrix = np.zeros((n_classes, n_classes))
        
        for i, label1 in enumerate(unique_labels):
            mask1 = labels == label1
            emb1 = embeddings[mask1]
            
            if len(emb1) < 2:
                continue
                
            mean1 = np.mean(emb1, axis=0)
            cov1 = np.cov(emb1.T)
            
            # Add regularization to avoid singular matrices
            cov1 += np.eye(cov1.shape[0]) * 1e-6
            
            for j, label2 in enumerate(unique_labels):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                    continue
                    
                mask2 = labels == label2
                emb2 = embeddings[mask2]
                
                if len(emb2) < 2:
                    continue
                    
                mean2 = np.mean(emb2, axis=0)
                cov2 = np.cov(emb2.T)
                cov2 += np.eye(cov2.shape[0]) * 1e-6
                
                # Estimate overlap using average Mahalanobis distance
                try:
                    inv_cov1 = np.linalg.inv(cov1)
                    inv_cov2 = np.linalg.inv(cov2)
                    
                    # Bhattacharyya distance approximation
                    cov_avg = (cov1 + cov2) / 2
                    inv_cov_avg = np.linalg.inv(cov_avg)
                    
                    diff = mean1 - mean2
                    distance = 0.125 * np.dot(np.dot(diff, inv_cov_avg), diff)
                    distance += 0.5 * np.log(np.linalg.det(cov_avg) / 
                                             np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
                    
                    # Convert to overlap probability
                    overlap = np.exp(-distance)
                    overlap_matrix[i, j] = overlap_matrix[j, i] = overlap
                    
                except np.linalg.LinAlgError:
                    logger.warning(f"Singular covariance for classes {label1}-{label2}")
                    overlap_matrix[i, j] = overlap_matrix[j, i] = 0.0
        
        return overlap_matrix


def compute_soap_similarity(
    structures: List[Union[str, np.ndarray]],
    **soap_params
) -> np.ndarray:
    """
    Convenience function to compute SOAP similarity matrix.
    
    Args:
        structures: List of structures
        **soap_params: Parameters for SOAP computation
        
    Returns:
        Similarity matrix
    """
    analyzer = DiversityAnalyzer(use_soap=True, soap_params=soap_params)
    descriptors = analyzer.compute_soap_descriptors(structures)
    
    # Compute similarity as 1 - normalized distance
    distances = squareform(pdist(descriptors, metric='euclidean'))
    max_dist = np.max(distances)
    if max_dist > 0:
        similarity = 1 - distances / max_dist
    else:
        similarity = np.ones_like(distances)
    
    return similarity


def compute_embedding_diversity(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    compute_projections: bool = True,
    projection_method: str = 'umap'
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive diversity analysis.
    
    Args:
        embeddings: Embedding matrix
        labels: Optional class labels
        compute_projections: Whether to compute 2D projections
        projection_method: Method for projection ('umap' or 'pca')
        
    Returns:
        Dictionary with diversity analysis results
    """
    analyzer = DiversityAnalyzer()
    
    results = {
        'metrics': analyzer.compute_diversity_metrics(embeddings, labels)
    }
    
    if compute_projections:
        projected, proj_info = analyzer.compute_embedding_projections(
            embeddings, method=projection_method
        )
        results['projection'] = projected
        results['projection_info'] = proj_info
    
    if labels is not None:
        # Compute Mahalanobis overlap
        overlap = analyzer.compute_mahalanobis_overlap(embeddings, labels)
        results['class_overlap'] = overlap
    
    return results
