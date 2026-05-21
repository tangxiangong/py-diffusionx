from importlib.metadata import PackageNotFoundError, version

from .types import DType
from . import random
from . import simulation
from . import distribution

try:
    __version__ = version("diffusionx")
except PackageNotFoundError:  # running from source without an installed dist
    __version__ = "0.2.1"

__all__ = [
    "__version__",
    "random",
    "distribution",
    "DType",
    "simulation",
]
