#!/usr/bin/env python
"""
UV-compatible wrapper for visualizing similarity pairs.
This script can be run with: uv run visualize_pairs.py [arguments]
"""

import sys
import os

# Add the src directory to Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main visualization script
from visualize_similarity_pairs import main

if __name__ == "__main__":
    main()
