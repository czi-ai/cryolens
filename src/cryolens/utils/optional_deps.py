"""
Utilities for handling optional dependencies with MPL licenses.

This module provides helper functions and classes for gracefully handling
optional dependencies that have been removed from the core requirements due
to licensing concerns (MPL).
"""

from typing import Iterator, Iterable, TypeVar, Optional
import sys

T = TypeVar('T')


class _MockTqdm:
    """Mock tqdm class that provides no-op progress bar functionality."""
    
    def __init__(self, iterable: Optional[Iterable[T]] = None, *args, **kwargs):
        self.iterable = iterable
        self.desc = kwargs.get('desc', '')
        self.total = kwargs.get('total', None)
        self.n = 0
        
    def __iter__(self) -> Iterator[T]:
        """Iterate without progress display."""
        if self.iterable is None:
            return iter([])
        for item in self.iterable:
            self.n += 1
            yield item
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n: int = 1):
        """No-op update method."""
        self.n += n
    
    def set_description(self, desc: str):
        """No-op set description method."""
        self.desc = desc


def get_tqdm():
    """
    Get tqdm if available, otherwise return a no-op mock.
    
    Returns
    -------
    tqdm or _MockTqdm
        The tqdm class if available, otherwise a mock that provides no progress display.
        
    Examples
    --------
    >>> tqdm = get_tqdm()
    >>> for item in tqdm(range(100), desc="Processing"):
    ...     process(item)
    """
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        print("Note: tqdm not installed. Install it with: pip install tqdm", file=sys.stderr)
        print("Progress bars will be disabled.", file=sys.stderr)
        return _MockTqdm


def check_gemmi():
    """
    Check if gemmi is available and provide installation instructions if not.
    
    Returns
    -------
    module or None
        The gemmi module if available, None otherwise.
        
    Examples
    --------
    >>> gemmi = check_gemmi()
    >>> if gemmi is None:
    ...     raise ImportError("gemmi is required for PDB processing")
    """
    try:
        import gemmi
        return gemmi
    except ImportError:
        return None


def require_gemmi():
    """
    Require gemmi to be installed, raising an informative error if not.
    
    Returns
    -------
    module
        The gemmi module.
        
    Raises
    ------
    ImportError
        If gemmi is not installed, with instructions on how to install it.
        
    Examples
    --------
    >>> gemmi = require_gemmi()
    >>> structure = gemmi.read_structure("file.pdb")
    """
    gemmi = check_gemmi()
    if gemmi is None:
        raise ImportError(
            "gemmi is required for PDB file processing but is not installed.\n"
            "gemmi is licensed under MPL-2.0 and must be installed separately.\n\n"
            "Install it with:\n"
            "  pip install gemmi>=0.7.0\n\n"
            "Or install cryolens with PDB processing support:\n"
            "  pip install cryolens[pdb-processing]"
        )
    return gemmi
