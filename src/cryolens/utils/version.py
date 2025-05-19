"""
Version utilities for CryoLens.
"""

import os
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_git_commit_hash():
    """Get the current git commit hash of the repository.
    
    Returns:
        str: Git commit hash or 'unknown' if git is not available or not in a git repository
    """
    try:
        # Try to get the git commit hash using subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode == 0:
            # Return the commit hash (stripping any whitespace)
            git_hash = result.stdout.strip()
            return git_hash
        else:
            # If git command failed, try environment variable (useful in CI/CD environments)
            git_hash = os.environ.get('GIT_COMMIT_HASH')
            if git_hash:
                return git_hash
            else:
                return 'unknown'
    except (FileNotFoundError, subprocess.SubprocessError):
        # Git not available
        return 'unknown'

def get_git_status():
    """Check if the git repository has uncommitted changes.
    
    Returns:
        bool: True if there are uncommitted changes, False otherwise
    """
    try:
        # Check if there are any uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False
        )
        
        if result.returncode == 0:
            # If there's any output, there are uncommitted changes
            return bool(result.stdout.strip())
        else:
            return False
    except (FileNotFoundError, subprocess.SubprocessError):
        # Git not available
        return False

def get_version_info():
    """Get comprehensive version information including git hash and status.
    
    Returns:
        dict: Dictionary containing version information
    """
    git_hash = get_git_commit_hash()
    has_uncommitted_changes = get_git_status()
    
    return {
        'git_commit_hash': git_hash,
        'has_uncommitted_changes': has_uncommitted_changes,
        'version_string': f"{git_hash}{' (with uncommitted changes)' if has_uncommitted_changes else ''}"
    }

def log_version_info(logger=None):
    """Log version information.
    
    Args:
        logger: Logger to use (default: module logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    version_info = get_version_info()
    logger.info(f"Git commit hash: {version_info['git_commit_hash']}")
    
    if version_info['has_uncommitted_changes']:
        logger.warning("Repository has uncommitted changes!")
