#!/usr/bin/env python3
"""
Minimal CryoLens FastAPI Server using refactored modules

This server provides REST API endpoints for CryoLens VAE functionality including:
- Volume reconstruction
- Feature extraction (embeddings, pose, global weight)
- Gaussian splat extraction
- Segmented decoder outputs
- Optional final convolution control

Usage:
    python minimal_cryolens_server.py --checkpoint /path/to/checkpoint.pt --port 8023

Based on the original copick_server_fastapi.py but simplified to use only
the refactored CryoLens modules without Copick dependencies.
"""

import os
import sys
import json
import torch
import click
import uvicorn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cryolens.server")

# Import refactored CryoLens modules
from cryolens.utils.checkpoint_loading import load_vae_model
from cryolens.utils.normalization import normalize_volume, denormalize_volume, get_volume_statistics
from cryolens.inference import InferencePipeline, create_inference_pipeline
from cryolens.splats import extract_gaussian_splats


# Pydantic models for request/response
class VolumeReconstructionRequest(BaseModel):
    """Request model for volume reconstruction."""
    input_volume: List[List[List[float]]]  # 3D array as nested lists
    use_segmented_decoder: bool = True
    use_final_convolution: bool = True
    return_splat_data: bool = False
    return_embeddings: bool = False

class VolumeFeaturesRequest(BaseModel):
    """Request model for feature extraction."""
    input_volume: List[List[List[float]]]  # 3D array as nested lists
    return_pose: bool = True
    return_global_weight: bool = True
    return_variance: bool = False

class GaussianSplatRequest(BaseModel):
    """Request model for Gaussian splat extraction."""
    input_volume: List[List[List[float]]]  # 3D array as nested lists
    return_centroids: bool = True
    return_sigmas: bool = True
    return_weights: bool = True
    return_colors: bool = False  # For future RGB support

class BatchProcessingRequest(BaseModel):
    """Request model for batch processing multiple volumes."""
    input_volumes: List[List[List[List[float]]]]  # List of 3D arrays
    operation: str = "reconstruction"  # "reconstruction", "features", or "splats"
    batch_size: int = 4


class CryoLensServer:
    """Minimal CryoLens FastAPI server using refactored modules."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the server with a model checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the VAE checkpoint file
        device : str, optional
            Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        self.model = None
        self.config = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the VAE model and create inference pipeline."""
        try:
            logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
            
            # Load model using refactored checkpoint loading
            self.model, self.config = load_vae_model(
                self.checkpoint_path,
                device=self.device,
                load_config=True,
                strict_loading=False
            )
            
            logger.info(f"Model loaded successfully with config: {self.config}")
            
            # Create inference pipeline for convenience
            self.pipeline = create_inference_pipeline(
                self.checkpoint_path,
                device=self.device
            )
            
            logger.info("Inference pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def create_app(self, cors_origins: Optional[List[str]] = None) -> FastAPI:
        """
        Create and configure the FastAPI application.
        
        Parameters
        ----------
        cors_origins : List[str], optional
            List of allowed CORS origins
            
        Returns
        -------
        FastAPI
            Configured FastAPI application
        """
        app = FastAPI(
            title="CryoLens VAE Server",
            description="Minimal server for CryoLens VAE processing using refactored modules",
            version="1.0.0"
        )
        
        # Add CORS middleware if needed
        if cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Register endpoints
        self._register_endpoints(app)
        
        return app
    
    def _register_endpoints(self, app: FastAPI):
        """Register all API endpoints."""
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "device": str(self.device),
                "checkpoint": self.checkpoint_path
            }
        
        @app.get("/model_info")
        async def model_info():
            """Get information about the loaded model."""
            if self.config is None:
                raise HTTPException(status_code=500, detail="Model not loaded")
            
            return {
                "config": self.config,
                "model_type": type(self.model).__name__ if self.model else None,
                "device": str(self.device)
            }
        
        @app.post("/reconstruct")
        async def reconstruct_volume(request: VolumeReconstructionRequest):
            """
            Reconstruct a volume using the VAE.
            
            Supports:
            - Regular or segmented decoder output
            - Optional final convolution
            - Optional splat data and embeddings
            """
            try:
                # Convert input to numpy array
                input_volume = np.array(request.input_volume, dtype=np.float32)
                logger.info(f"Processing volume with shape: {input_volume.shape}")
                
                # Process through pipeline
                result = self.pipeline.process_volume(
                    input_volume,
                    return_embeddings=request.return_embeddings,
                    return_reconstruction=True,
                    return_splats=request.return_splat_data
                )
                
                # Prepare response
                response = {
                    "input_shape": list(input_volume.shape),
                    "reconstruction": result['reconstruction'].tolist()
                }
                
                if request.return_embeddings and 'embeddings' in result:
                    response['embeddings'] = result['embeddings'].tolist()
                
                if request.return_splat_data and 'splats' in result:
                    response['splats'] = {
                        'centroids': result['splats']['centroids'].tolist(),
                        'sigmas': result['splats']['sigmas'].tolist(),
                        'weights': result['splats']['weights'].tolist()
                    }
                
                # Handle segmented decoder outputs if available
                if request.use_segmented_decoder and 'segmented_outputs' in result:
                    response['segmented_outputs'] = {
                        'affinity': result['segmented_outputs']['affinity'].tolist(),
                        'free': result['segmented_outputs']['free'].tolist()
                    }
                
                return response
                
            except Exception as e:
                logger.error(f"Error in reconstruct_volume: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/extract_features")
        async def extract_features(request: VolumeFeaturesRequest):
            """
            Extract features from a volume.
            
            Returns embeddings (mu), pose, global weight, and optionally variance.
            """
            try:
                # Convert input to numpy array
                input_volume = np.array(request.input_volume, dtype=np.float32)
                logger.info(f"Extracting features from volume with shape: {input_volume.shape}")
                
                # Normalize volume
                normalization_method = self.config.get('normalization', 'z-score')
                volume_normalized, norm_stats = normalize_volume(
                    input_volume,
                    method=normalization_method,
                    return_stats=True
                )
                
                # Process through encoder
                with torch.no_grad():
                    volume_tensor = torch.tensor(volume_normalized).float()
                    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    mu, log_var, pose, global_weight = self.model.encode(volume_tensor)
                    
                    # Convert to numpy
                    mu_np = mu.cpu().numpy()[0]
                    log_var_np = log_var.cpu().numpy()[0] if request.return_variance else None
                    pose_np = pose.cpu().numpy()[0] if pose is not None and request.return_pose else None
                    global_weight_np = global_weight.cpu().numpy()[0] if global_weight is not None and request.return_global_weight else None
                
                response = {
                    "embeddings": mu_np.tolist(),
                    "latent_dims": int(mu_np.shape[0])
                }
                
                if request.return_variance and log_var_np is not None:
                    response["log_variance"] = log_var_np.tolist()
                
                if request.return_pose and pose_np is not None:
                    response["pose"] = pose_np.tolist()
                
                if request.return_global_weight and global_weight_np is not None:
                    response["global_weight"] = global_weight_np.tolist()
                
                return response
                
            except Exception as e:
                logger.error(f"Error in extract_features: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/extract_gaussian_splats")
        async def extract_gaussian_splats(request: GaussianSplatRequest):
            """
            Extract Gaussian splat parameters from a volume.
            
            Returns centroids, sigmas, and weights for the Gaussian splats.
            """
            try:
                # Convert input to numpy array
                input_volume = np.array(request.input_volume, dtype=np.float32)
                logger.info(f"Extracting Gaussian splats from volume with shape: {input_volume.shape}")
                
                # Extract splats using the refactored function
                centroids, sigmas, weights, embeddings = extract_gaussian_splats(
                    self.model,
                    np.expand_dims(input_volume, axis=0),  # Add batch dimension
                    self.config,
                    device=self.device,
                    batch_size=1
                )
                
                # Remove batch dimension
                centroids = centroids[0]
                sigmas = sigmas[0]
                weights = weights[0]
                
                response = {}
                
                if request.return_centroids:
                    response["centroids"] = centroids.tolist()
                
                if request.return_sigmas:
                    response["sigmas"] = sigmas.tolist()
                
                if request.return_weights:
                    response["weights"] = weights.tolist()
                
                # Add summary statistics
                response["num_splats"] = int(weights.shape[0])
                response["significant_splats"] = int(np.sum(weights > 0.01))
                
                return response
                
            except Exception as e:
                logger.error(f"Error in extract_gaussian_splats: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/batch_process")
        async def batch_process(request: BatchProcessingRequest):
            """
            Process multiple volumes in batch.
            
            Supports reconstruction, feature extraction, or splat extraction.
            """
            try:
                # Convert input volumes to numpy array
                input_volumes = np.array(request.input_volumes, dtype=np.float32)
                num_volumes = len(input_volumes)
                logger.info(f"Batch processing {num_volumes} volumes with operation: {request.operation}")
                
                if request.operation == "reconstruction":
                    results = self.pipeline.process_batch(
                        input_volumes,
                        batch_size=request.batch_size,
                        return_embeddings=False,
                        return_reconstruction=True
                    )
                    
                    return {
                        "num_processed": num_volumes,
                        "reconstructions": results['reconstruction'].tolist()
                    }
                
                elif request.operation == "features":
                    results = self.pipeline.process_batch(
                        input_volumes,
                        batch_size=request.batch_size,
                        return_embeddings=True,
                        return_reconstruction=False
                    )
                    
                    return {
                        "num_processed": num_volumes,
                        "embeddings": results['embeddings'].tolist()
                    }
                
                elif request.operation == "splats":
                    centroids, sigmas, weights, embeddings = extract_gaussian_splats(
                        self.model,
                        input_volumes,
                        self.config,
                        device=self.device,
                        batch_size=request.batch_size
                    )
                    
                    return {
                        "num_processed": num_volumes,
                        "centroids": centroids.tolist(),
                        "sigmas": sigmas.tolist(),
                        "weights": weights.tolist()
                    }
                
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown operation: {request.operation}"
                    )
                
            except Exception as e:
                logger.error(f"Error in batch_process: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/statistics")
        async def get_statistics(
            volume_data: str = Query(..., description="Base64 encoded volume data")
        ):
            """
            Get statistics for a volume.
            
            Returns mean, std, min, max, and percentiles.
            """
            try:
                # For this example, we'll accept JSON array instead of base64
                # In production, you might want to use base64 for efficiency
                import json
                volume_list = json.loads(volume_data)
                volume = np.array(volume_list, dtype=np.float32)
                
                stats = get_volume_statistics(volume)
                
                return {
                    "shape": list(volume.shape),
                    "statistics": stats
                }
                
            except Exception as e:
                logger.error(f"Error in get_statistics: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))


@click.command()
@click.option(
    '--checkpoint',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='Path to VAE checkpoint file'
)
@click.option(
    '--host',
    default='0.0.0.0',
    help='Host to bind the server to'
)
@click.option(
    '--port',
    '-p',
    default=8023,
    type=int,
    help='Port to bind the server to'
)
@click.option(
    '--cors',
    default='*',
    help='CORS origins (comma-separated or * for all)'
)
@click.option(
    '--device',
    type=click.Choice(['cuda', 'cpu', 'auto']),
    default='auto',
    help='Device to use for inference'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Enable auto-reload for development'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level'
)
def main(checkpoint, host, port, cors, device, reload, log_level):
    """
    Start the CryoLens FastAPI server.
    
    Example:
        python minimal_cryolens_server.py -c model.pt -p 8023
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Determine device
    if device == 'auto':
        device = None  # Let the server auto-detect
    
    # Parse CORS origins
    if cors == '*':
        cors_origins = ['*']
    else:
        cors_origins = [origin.strip() for origin in cors.split(',')]
    
    logger.info(f"Starting CryoLens server")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"CORS origins: {cors_origins}")
    logger.info(f"Device: {device or 'auto-detect'}")
    
    # Create server
    try:
        server = CryoLensServer(checkpoint, device=device)
        app = server.create_app(cors_origins=cors_origins)
        
        # Run with uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level.lower()
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
