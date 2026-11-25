"""PyHFO headless EEG analysis engine."""

import os
from importlib.metadata import PackageNotFoundError, version

from .paths import pyhfo_home

# These libraries select their cache on first import. Respect a host application's
# explicit settings and keep generated files out of a writable site-packages.
_cache = pyhfo_home() / "cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache / "numba"))

try:
    __version__ = version("pyhfo")
except PackageNotFoundError:  # Uninstalled source checkout or frozen application.
    from ._version import VERSION as __version__
