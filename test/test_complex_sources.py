from __future__ import annotations

import numpy as np
import pytest

from greenonet.complex_sources import (
    GaussianProcessSourceSampler,
    GeometryGridLoader,
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
    NpzComplexSourceProvider,
    derive_indexed_seed,
    generate_fixed_rhs,
)
from test.complex_fixtures import write_geometry_npz


def test_indexed_source_is_fixed_by_split_and_index(tmp_path):
    geometry = GeometryGridLoader().load(write_geometry_npz(tmp_path / "geometry.npz"))
    parameters = IndexedGpParameters(
        seed=17,
        lengthscale=0.2,
        amplitude=1.5,
        mean=-0.25,
    )
    provider = IndexedGpComplexSourceProvider(
        geometry,
        split="train",
        sample_count=3,
        parameters=parameters,
    )

    first = provider[1]
    provider[0]
    repeated = provider[1]
    validation = IndexedGpComplexSourceProvider(
        geometry,
        split="valid",
        sample_count=3,
        parameters=parameters,
    )[1]

    np.testing.assert_array_equal(first.rhs, repeated.rhs)
    assert not np.array_equal(first.rhs, validation.rhs)
    assert first.sample_index == 1
    assert first.sample_name == "sample_000001"
    assert first.sol is None
    assert first.flux is None


def test_generate_fixed_rhs_masks_outside_domain(tmp_path):
    geometry = GeometryGridLoader().load(write_geometry_npz(tmp_path / "geometry.npz"))
    sampler = GaussianProcessSourceSampler(
        geometry.grid_x,
        geometry.grid_y,
        lengthscale=0.3,
        amplitude=0.8,
        mean=0.1,
        seed=5,
    )

    rhs = generate_fixed_rhs(
        geometry,
        sampler,
        base_seed=5,
        split="train",
        sample_index=2,
    )

    mask = np.zeros(geometry.full_grid_shape, dtype=bool)
    mask[geometry.valid_grid_y_index, geometry.valid_grid_x_index] = True
    assert rhs.shape == geometry.full_grid_shape
    assert rhs.dtype == np.float64
    assert np.isfinite(rhs).all()
    np.testing.assert_array_equal(rhs[~mask], np.zeros(np.count_nonzero(~mask)))


def test_npz_and_indexed_providers_have_bitwise_rhs_parity(tmp_path):
    geometry = GeometryGridLoader().load(write_geometry_npz(tmp_path / "geometry.npz"))
    parameters = IndexedGpParameters(
        seed=9,
        lengthscale=0.25,
        amplitude=1.2,
        mean=0.4,
    )
    indexed = IndexedGpComplexSourceProvider(
        geometry,
        split="train",
        sample_count=2,
        parameters=parameters,
    )
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    np.savez(source_dir / "sample_000000.npz", rhs=indexed[0].rhs)
    np.savez(source_dir / "sample_000001.npz", rhs=indexed[1].rhs)
    stored = NpzComplexSourceProvider(
        source_dir,
        reference_diagnostics=False,
    )

    np.testing.assert_array_equal(stored[0].rhs, indexed[0].rhs)
    np.testing.assert_array_equal(stored[1].rhs, indexed[1].rhs)


def test_npz_provider_reference_contract_is_explicit(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    rhs = np.ones((5, 5), dtype=np.float64)
    np.savez(source_dir / "sample_000000.npz", rhs=rhs)

    source_only = NpzComplexSourceProvider(
        source_dir,
        reference_diagnostics=False,
    )[0]
    assert source_only.sol is None
    assert source_only.flux is None

    with pytest.raises(KeyError, match="sol"):
        NpzComplexSourceProvider(
            source_dir,
            reference_diagnostics=True,
        )[0]


def test_indexed_seed_contract_matches_seed_sequence():
    expected = int(
        np.random.SeedSequence([11, 1, 47]).generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )

    assert derive_indexed_seed(11, "valid", 47) == expected


def test_indexed_provider_rejects_unknown_split_at_construction(tmp_path):
    geometry = GeometryGridLoader().load(write_geometry_npz(tmp_path / "geometry.npz"))

    with pytest.raises(ValueError, match="Unknown split"):
        IndexedGpComplexSourceProvider(
            geometry,
            split="invalid",  # type: ignore[arg-type]
            sample_count=1,
            parameters=IndexedGpParameters(),
        )
