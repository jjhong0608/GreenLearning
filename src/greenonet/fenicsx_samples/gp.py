"""Backward-compatible imports for the shared complex source GP core."""

from greenonet.complex_sources.gp import (
    GaussianProcessSourceSampler,
    squared_exponential_kernel,
    stable_symmetric_factor,
)

__all__ = [
    "GaussianProcessSourceSampler",
    "squared_exponential_kernel",
    "stable_symmetric_factor",
]
