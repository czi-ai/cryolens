"""
Background noise generator for CryoLens training.

This module provides generators for creating background noise samples that match
the distribution characteristics of real cryoET data, including ice thickness variations,
noise patterns, and missing wedge artifacts.
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class BackgroundGenerator:
    """Base class for background noise generation."""
    
    def __init__(self, box_size: int = 48, seed: Optional[int] = None):
        """Initialize background generator.
        
        Parameters
        ----------
        box_size : int
            Size of the volume box (default: 48)
        seed : int or None
            Random seed for reproducibility
        """
        self.box_size = box_size
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self, **kwargs) -> np.ndarray:
        """Generate a background noise volume.
        
        Returns
        -------
        np.ndarray
            Background noise volume of shape (box_size, box_size, box_size)
        """
        raise NotImplementedError("Subclasses must implement generate()")


class GaussianNoiseGenerator(BackgroundGenerator):
    """Simple Gaussian noise generator."""
    
    def __init__(self, box_size: int = 48, mean: float = 0.0, std: float = 1.0, 
                 seed: Optional[int] = None):
        """Initialize Gaussian noise generator.
        
        Parameters
        ----------
        box_size : int
            Size of the volume box
        mean : float
            Mean of the Gaussian distribution
        std : float
            Standard deviation of the Gaussian distribution
        seed : int or None
            Random seed
        """
        super().__init__(box_size, seed)
        self.mean = mean
        self.std = std
    
    def generate(self, **kwargs) -> np.ndarray:
        """Generate Gaussian noise volume."""
        return np.random.normal(self.mean, self.std, 
                              (self.box_size, self.box_size, self.box_size)).astype(np.float32)


class CryoETBackgroundGenerator(BackgroundGenerator):
    """
    Realistic cryoET background generator that simulates:
    - Ice contamination and thickness variations
    - Shot noise and detector noise
    - CTF-like effects
    - Missing wedge artifacts
    """
    
    def __init__(self, 
                 box_size: int = 48,
                 ice_mean: float = 0.0,
                 ice_std: float = 0.2,
                 shot_noise_lambda: float = 0.5,
                 detector_noise_std: float = 0.05,
                 ctf_defocus_range: Tuple[float, float] = (1.0, 3.0),
                 missing_wedge_angle: float = 60.0,
                 smooth_sigma: float = 1.0,
                 seed: Optional[int] = None):
        """Initialize cryoET background generator.
        
        Parameters
        ----------
        box_size : int
            Size of the volume box
        ice_mean : float
            Mean intensity of ice background
        ice_std : float
            Standard deviation of ice variations
        shot_noise_lambda : float
            Lambda parameter for Poisson shot noise
        detector_noise_std : float
            Standard deviation of detector noise
        ctf_defocus_range : tuple
            Range of defocus values for CTF simulation (in micrometers)
        missing_wedge_angle : float
            Angle of the missing wedge in degrees
        smooth_sigma : float
            Gaussian smoothing sigma for ice variations
        seed : int or None
            Random seed
        """
        super().__init__(box_size, seed)
        self.ice_mean = ice_mean
        self.ice_std = ice_std
        self.shot_noise_lambda = shot_noise_lambda
        self.detector_noise_std = detector_noise_std
        self.ctf_defocus_range = ctf_defocus_range
        self.missing_wedge_angle = missing_wedge_angle
        self.smooth_sigma = smooth_sigma
        
        # Precompute missing wedge mask in Fourier space
        self._compute_wedge_mask()
    
    def _compute_wedge_mask(self):
        """Compute the missing wedge mask in Fourier space."""
        # Create frequency coordinates
        freqs = np.fft.fftfreq(self.box_size)
        kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
        
        # Calculate angles relative to z-axis
        angles = np.abs(np.arctan2(np.sqrt(kx**2 + ky**2), np.abs(kz)))
        
        # Create wedge mask
        wedge_rad = np.radians(self.missing_wedge_angle / 2)
        self.wedge_mask = angles > wedge_rad
    
    def generate(self, source_volume: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate realistic cryoET background noise.
        
        Parameters
        ----------
        source_volume : np.ndarray or None
            Optional source volume to extract statistics from
            
        Returns
        -------
        np.ndarray
            Background noise volume
        """
        # 1. Generate ice background with spatial variations
        ice = self._generate_ice_background()
        
        # 2. Add shot noise (Poisson noise)
        if self.shot_noise_lambda > 0:
            shot_noise = np.random.poisson(self.shot_noise_lambda, 
                                         (self.box_size, self.box_size, self.box_size))
            shot_noise = (shot_noise - self.shot_noise_lambda) / np.sqrt(self.shot_noise_lambda)
            ice += shot_noise * 0.1  # Scale down shot noise
        
        # 3. Add detector noise (Gaussian)
        if self.detector_noise_std > 0:
            detector_noise = np.random.normal(0, self.detector_noise_std,
                                           (self.box_size, self.box_size, self.box_size))
            ice += detector_noise
        
        # 4. Apply CTF-like modulation
        ice = self._apply_ctf_modulation(ice)
        
        # 5. Apply missing wedge in Fourier space
        ice = self._apply_missing_wedge(ice)
        
        # 6. Match statistics if source volume provided
        if source_volume is not None:
            ice = self._match_statistics(ice, source_volume)
        
        return ice.astype(np.float32)
    
    def _generate_ice_background(self) -> np.ndarray:
        """Generate spatially-varying ice background."""
        # Start with random noise
        ice = np.random.normal(self.ice_mean, self.ice_std,
                             (self.box_size, self.box_size, self.box_size))
        
        # Add low-frequency ice thickness variations
        thickness_variation = np.random.normal(0, 1,
                                             (self.box_size // 4, self.box_size // 4, self.box_size // 4))
        thickness_variation = ndimage.zoom(thickness_variation, 4, order=1)
        thickness_variation = thickness_variation[:self.box_size, :self.box_size, :self.box_size]
        
        # Smooth the variations
        if self.smooth_sigma > 0:
            thickness_variation = gaussian_filter(thickness_variation, self.smooth_sigma)
        
        # Combine base ice with thickness variations
        ice += thickness_variation * self.ice_std * 0.5
        
        return ice
    
    def _apply_ctf_modulation(self, volume: np.ndarray) -> np.ndarray:
        """Apply simplified CTF-like modulation in Fourier space."""
        # Random defocus value
        defocus = np.random.uniform(*self.ctf_defocus_range)
        
        # FFT of volume
        fft_vol = np.fft.fftn(volume)
        
        # Create radial frequency grid
        freqs = np.fft.fftfreq(self.box_size)
        kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
        k_rad = np.sqrt(kx**2 + ky**2 + kz**2)
        
        # Simplified CTF (sin function of radial frequency and defocus)
        # This is a very simplified model for demonstration
        ctf = np.sin(2 * np.pi * defocus * k_rad)
        
        # Apply CTF
        fft_vol *= (0.7 + 0.3 * ctf)  # Partial CTF effect
        
        # Inverse FFT
        return np.real(np.fft.ifftn(fft_vol))
    
    def _apply_missing_wedge(self, volume: np.ndarray) -> np.ndarray:
        """Apply missing wedge artifact in Fourier space."""
        # FFT
        fft_vol = np.fft.fftn(volume)
        
        # Apply wedge mask
        fft_vol *= self.wedge_mask
        
        # Inverse FFT
        return np.real(np.fft.ifftn(fft_vol))
    
    def _match_statistics(self, background: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Match the statistics of background to source volume."""
        # Normalize background
        bg_mean = np.mean(background)
        bg_std = np.std(background)
        if bg_std > 0:
            background = (background - bg_mean) / bg_std
        
        # Match to source statistics
        src_mean = np.mean(source)
        src_std = np.std(source)
        
        return background * src_std + src_mean


class OnlineBackgroundGenerator:
    """
    Online background generator that creates backgrounds on-the-fly during training.
    Can optionally cache generated backgrounds for efficiency.
    """
    
    def __init__(self, 
                 generator_type: str = "cryoet",
                 generator_params: Optional[Dict[str, Any]] = None,
                 cache_size: int = 0,
                 seed: Optional[int] = None):
        """Initialize online background generator.
        
        Parameters
        ----------
        generator_type : str
            Type of generator to use ("gaussian" or "cryoet")
        generator_params : dict or None
            Parameters for the generator
        cache_size : int
            Number of backgrounds to cache (0 = no caching)
        seed : int or None
            Random seed
        """
        self.generator_type = generator_type
        self.cache_size = cache_size
        self.cache = []
        
        # Default parameters
        default_params = {
            "box_size": 48,
            "seed": seed
        }
        
        if generator_params:
            default_params.update(generator_params)
        
        # Create the appropriate generator
        if generator_type == "gaussian":
            self.generator = GaussianNoiseGenerator(**default_params)
        elif generator_type == "cryoet":
            # Add cryoET-specific defaults
            cryoet_defaults = {
                "ice_mean": 0.0,
                "ice_std": 0.15,
                "shot_noise_lambda": 0.3,
                "detector_noise_std": 0.05,
                "ctf_defocus_range": (1.0, 3.0),
                "missing_wedge_angle": 60.0,
                "smooth_sigma": 1.5
            }
            cryoet_defaults.update(default_params)
            self.generator = CryoETBackgroundGenerator(**cryoet_defaults)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
    
    def get_background(self, source_volume: Optional[np.ndarray] = None) -> np.ndarray:
        """Get a background sample, either from cache or newly generated.
        
        Parameters
        ----------
        source_volume : np.ndarray or None
            Optional source volume for statistics matching
            
        Returns
        -------
        np.ndarray
            Background noise volume
        """
        # Use cache if available and not using source matching
        if self.cache_size > 0 and source_volume is None and len(self.cache) > 0:
            # Randomly select from cache
            idx = np.random.randint(0, len(self.cache))
            return self.cache[idx].copy()
        
        # Generate new background
        if hasattr(self.generator, 'generate'):
            if isinstance(self.generator, CryoETBackgroundGenerator):
                background = self.generator.generate(source_volume=source_volume)
            else:
                background = self.generator.generate()
        else:
            # Fallback for simple generators
            background = self.generator.generate()
        
        # Add to cache if caching is enabled and not using source matching
        if self.cache_size > 0 and source_volume is None:
            if len(self.cache) < self.cache_size:
                self.cache.append(background.copy())
            else:
                # Replace random element in cache
                idx = np.random.randint(0, self.cache_size)
                self.cache[idx] = background.copy()
        
        return background
    
    def clear_cache(self):
        """Clear the background cache."""
        self.cache = []


def create_background_generator(config: Dict[str, Any]) -> OnlineBackgroundGenerator:
    """Factory function to create a background generator from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - type: "gaussian" or "cryoet"
        - params: dict of generator-specific parameters
        - cache_size: int, number of backgrounds to cache
        - seed: random seed
        
    Returns
    -------
    OnlineBackgroundGenerator
        Configured background generator
    """
    return OnlineBackgroundGenerator(
        generator_type=config.get("type", "cryoet"),
        generator_params=config.get("params", {}),
        cache_size=config.get("cache_size", 100),
        seed=config.get("seed", None)
    )
