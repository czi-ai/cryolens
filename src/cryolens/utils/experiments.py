import mlflow
from pathlib import Path
import json
import time
import typer
from typing import List, Dict, Any
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_robust_session(tracking_uri: str):
    """Create a requests session with retry logic and timeouts"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        connect=3,
        read=3,
        status=3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=None
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Test connection
    logger.info("Testing connection to MLflow server...")
    try:
        test_response = session.get(tracking_uri, timeout=5)
        test_response.raise_for_status()
        logger.info("Successfully connected to MLflow server")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow server: {str(e)}")
        raise
        
    return session

def make_request(session, method, url, **kwargs):
    """Make a request with timeout and error handling"""
    try:
        kwargs['timeout'] = kwargs.get('timeout', 30)
        response = session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out: {url}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {url}, Error: {str(e)}")
        raise

def get_experiment_runs(session, base_url, experiment_id, tracking_uri):
    """Get experiment runs with fallback to different API versions"""
    try:
        # Try API 2.0 first
        response = make_request(
            session,
            'POST',
            f"{base_url}/experiments/search-runs",
            json={
                "experiment_ids": [experiment_id],
                "filter_string": "",
                "max_results": 1000
            }
        )
        return response.json().get("runs", [])
    except requests.exceptions.HTTPError as e:
        if getattr(e.response, 'status_code', None) == 404:
            # Fall back to API 1.0
            try:
                response = make_request(
                    session,
                    'GET',
                    f"{base_url.replace('api/2.0', 'api/1.0')}/experiments/{experiment_id}/runs",
                    params={"max_results": 1000}
                )
                return response.json().get("runs", [])
            except:
                # If both fail, try using MLflow client directly
                logger.warning("Falling back to MLflow client for run retrieval")
                import mlflow
                client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                return [run.to_dictionary() for run in client.search_runs([experiment_id])]
        raise

def create_experiments(
    configs: List[Dict[str, Any]],
    experiment_name: str,
    script_path: str,
    script_args: Dict[str, str],
    mlflow_config_path: str = "/mnt/czi-sci-ai/imaging-models/systems/mlflow/mlflow_server_metadata.json"
):
    """Create MLflow experiments for MLC training."""
    # Load MLflow configuration
    with open(mlflow_config_path, 'r') as f:
        mlflow_config = json.load(f)['mlflow_server']
    
    # Setup MLflow base URL
    tracking_uri = mlflow_config['tracking_uri']
    if not tracking_uri.endswith('/'):
        tracking_uri += '/'
    base_url = tracking_uri + 'api/2.0/mlflow'
    logger.info(f"Base url: {base_url}")
    
    session = create_robust_session(tracking_uri)
    
    # First get or create experiment
    logger.info(f"Attempting to get experiment by name: {experiment_name}")
    try:
        response = make_request(
            session,
            'GET',
            f"{base_url}/experiments/get-by-name",
            params={"experiment_name": experiment_name}
        )
        experiment_data = response.json()
        experiment_exists = True
        experiment_id = experiment_data["experiment"]["experiment_id"]
        lifecycle_stage = experiment_data["experiment"]["lifecycle_stage"]
        logger.info(f"Found experiment ID: {experiment_id}, stage: {lifecycle_stage}")
    except requests.exceptions.HTTPError as e:
        if getattr(e.response, 'status_code', None) == 404:
            logger.info("Experiment not found, will create new one")
            experiment_exists = False
            experiment_id = None
            lifecycle_stage = None
        else:
            raise

    if experiment_exists:
        # Get and delete existing runs using the new function
        runs = get_experiment_runs(session, base_url, experiment_id, tracking_uri)
        
        for run in runs:
            run_id = run["info"]["run_id"]
            logger.info(f"Deleting run: {run_id}")
            make_request(
                session,
                'POST',
                f"{base_url}/runs/delete",
                json={"run_id": run_id}
            )
            
        if lifecycle_stage == 'deleted':
            logger.info(f"Restoring deleted experiment: {experiment_name}")
            make_request(
                session,
                'POST',
                f"{base_url}/experiments/restore",
                json={"experiment_id": experiment_id}
            )
            
        # Set experiment tags
        for tag in [
            {"key": "script_path", "value": script_path},
            {"key": "script_args", "value": json.dumps(script_args)}
        ]:
            make_request(
                session,
                'POST',
                f"{base_url}/experiments/set-experiment-tag",
                json={
                    "experiment_id": experiment_id,
                    "key": tag["key"],
                    "value": tag["value"]
                }
            )
    else:
        # Create new experiment
        logger.info(f"Creating new experiment: {experiment_name}")
        response = make_request(
            session,
            'POST',
            f"{base_url}/experiments/create",
            json={
                "name": experiment_name,
                "artifact_location": str(Path(mlflow_config['artifacts_dir']) / experiment_name),
                "tags": [
                    {"key": "script_path", "value": script_path},
                    {"key": "script_args", "value": json.dumps(script_args)}
                ]
            }
        )
        experiment_id = response.json()["experiment_id"]

    logger.info(f"Using experiment with ID: {experiment_id}")

    # Create runs
    for i, config in enumerate(configs, 1):
        logger.info(f"Creating run {i}/{len(configs)}")
        
        # Create a unique experiment_id string that includes the experiment name and index
        unique_experiment_id = f"{experiment_name}_{i}"
        
        response = make_request(
            session,
            'POST',
            f"{base_url}/runs/create",
            json={
                "experiment_id": experiment_id,
                "tags": [
                    {"key": "status", "value": "pending"},
                    {"key": "last_update", "value": str(time.time())},
                    {"key": "worker_id", "value": ""},
                    {"key": "unique_experiment_id", "value": unique_experiment_id}
                ]
            }
        )
        run_id = response.json()["run"]["info"]["run_id"]
        
        # Log unique_experiment_id as a parameter for use in the training script
        make_request(
            session,
            'POST',
            f"{base_url}/runs/log-parameter",
            json={
                "run_id": run_id,
                "key": "unique_experiment_id",
                "value": unique_experiment_id
            }
        )
        
        # Log parameters
        for key, value in config.items():
            logger.info(f"Logging parameter: {key}={value}")
            make_request(
                session,
                'POST',
                f"{base_url}/runs/log-parameter",
                json={
                    "run_id": run_id,
                    "key": key,
                    "value": str(value)
                }
            )

    logger.info(f"Created {len(configs)} experiment runs")
    logger.info(f"MLflow UI available at: {mlflow_config['tracking_uri']}")