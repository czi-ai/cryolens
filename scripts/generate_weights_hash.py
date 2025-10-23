#!/usr/bin/env python
"""Generate SHA256 hash for model weights.

This helper script generates SHA256 hashes for CryoLens model weight files
using pooch. The hash can then be added to the WEIGHTS_REGISTRY in
src/cryolens/utils/checkpoint_loading.py for integrity verification.

Usage:
    python scripts/generate_weights_hash.py <path_to_weights_file>

Example:
    # Download weights first
    wget https://czi-cryolens.s3-us-west-2.amazonaws.com/weights/cryolens_v001.pt
    
    # Generate hash
    python scripts/generate_weights_hash.py cryolens_v001.pt
    
    # Output will be:
    # File: cryolens_v001.pt
    # Hash: sha256:abc123...
    #
    # Add to WEIGHTS_REGISTRY:
    # 'hash': 'sha256:abc123...',
"""

import sys
import pooch
from pathlib import Path


def generate_hash(file_path: str):
    """Generate and print hash for a file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Generating hash for: {file_path}")
    print(f"File size: {path.stat().st_size / (1024**3):.2f} GB")
    print("This may take a moment for large files...\n")
    
    hash_value = pooch.file_hash(file_path)
    
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Hash: {hash_value}")
    print("=" * 60)
    print("\nAdd to WEIGHTS_REGISTRY in src/cryolens/utils/checkpoint_loading.py:")
    print(f"'hash': '{hash_value}',")
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nError: Please provide a file path")
        print("Usage: python generate_weights_hash.py <path_to_weights>")
        sys.exit(1)
    
    generate_hash(sys.argv[1])


if __name__ == '__main__':
    main()
