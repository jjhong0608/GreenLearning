"""Deterministic source generation and providers for complex geometry."""

from greenonet.complex_sources.generator import (
    ComplexSourceGenerationConfig,
    ComplexSourceGenerator,
    SourceOnlySampleWriter,
)
from greenonet.complex_sources.geometry import (
    GeometryGridLoader,
    RawComplexGeometryGrid,
)
from greenonet.complex_sources.gp import (
    GaussianProcessSourceSampler,
    squared_exponential_kernel,
    stable_symmetric_factor,
)
from greenonet.complex_sources.providers import (
    ComplexSourceProvider,
    ComplexSourceSample,
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
    NpzComplexSourceProvider,
    generate_fixed_rhs,
)
from greenonet.complex_sources.seeding import SPLIT_IDS, derive_indexed_seed

__all__ = [
    "SPLIT_IDS",
    "ComplexSourceGenerationConfig",
    "ComplexSourceGenerator",
    "ComplexSourceProvider",
    "ComplexSourceSample",
    "GaussianProcessSourceSampler",
    "GeometryGridLoader",
    "IndexedGpComplexSourceProvider",
    "IndexedGpParameters",
    "NpzComplexSourceProvider",
    "RawComplexGeometryGrid",
    "SourceOnlySampleWriter",
    "derive_indexed_seed",
    "generate_fixed_rhs",
    "squared_exponential_kernel",
    "stable_symmetric_factor",
]
