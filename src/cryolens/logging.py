import logging
from pathlib import Path
import os
from .utils.version import get_git_commit_hash, get_version_info, log_version_info

# Function to set up logging to both console and file
def setup_logging(experiment_dir, rank=0):
    """Configure logging to output to both console and file in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        rank: Process rank in distributed setup (default: 0)
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory within experiment directory
    logs_dir = Path(experiment_dir) / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure root logger with basic settings
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up formatting
    formatter = logging.Formatter('%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s')
    
    # Add rank to log context
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - separate log file for each rank
    log_file = logs_dir / f"training_rank_{rank}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Master log that combines all outputs (rank 0 only creates this file)
    if rank == 0:
        master_log_file = logs_dir / "master.log"
        master_handler = logging.FileHandler(master_log_file)
        master_handler.setFormatter(formatter)
        logger.addHandler(master_handler)
    
    # Log version information (only for rank 0 to avoid duplicate logs)
    if rank == 0:
        log_version_info(logger)
        
        # Get version info for file storage
        version_info = get_version_info()
        git_hash = version_info['git_commit_hash']
        
        # Store git hash in a file for future reference
        git_hash_file = logs_dir / "git_commit_hash.txt"
        try:
            with open(git_hash_file, 'w') as f:
                f.write(git_hash)
                if version_info['has_uncommitted_changes']:
                    f.write("\nWARNING: Repository had uncommitted changes!")
            logger.info(f"Git commit hash saved to: {git_hash_file}")
        except Exception as e:
            logger.warning(f"Failed to save git commit hash to file: {str(e)}")
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Class to capture stdout/stderr and redirect to logger
class LogRedirector:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ""
    
    def write(self, message):
        # Only log non-empty messages
        if message and message.strip():
            self.buffer += message
            if '\n' in message:
                self.flush()
            
    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer.rstrip())
            self.buffer = ""