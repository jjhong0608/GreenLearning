from __future__ import annotations

import pytest
import torch

from greenonet.config import GreenQuadratureConfig, TrainingConfig
from greenonet.green_quadrature import (
    interpolate_source_on_unit_grid,
    reconstruct_split_gauss_legendre,
    split_gauss_legendre_nodes,
)


def test_green_quadrature_config_parses_nested_training_dict() -> None:
    config = TrainingConfig(
        green_quadrature={
            "enabled": True,
            "rule": "split_gauss_legendre",
            "order": 3,
            "source_sampling_factor": 4,
            "source_interpolation": "cubic",
        }
    )

    assert config.green_quadrature.enabled is True
    assert config.green_quadrature.order == 3
    assert config.green_quadrature.source_sampling_factor == 4
    assert config.green_quadrature.source_interpolation == "cubic"


@pytest.mark.parametrize(
    "raw",
    [
        {"order": 0},
        {"source_sampling_factor": 0},
        {"source_interpolation": "nearest"},
        {"rule": "trapezoid"},
        {"unknown": True},
    ],
)
def test_green_quadrature_config_rejects_invalid_values(raw) -> None:
    with pytest.raises((TypeError, ValueError)):
        GreenQuadratureConfig.from_raw(raw)


def test_split_gauss_legendre_nodes_have_unit_weight_sum() -> None:
    t_grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    nodes, weights = split_gauss_legendre_nodes(t_grid, order=4)

    assert nodes.shape == (3, 8)
    assert weights.shape == (3, 8)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(3, dtype=torch.float64),
    )
    assert torch.all((nodes >= 0.0) & (nodes <= 1.0))


def test_interpolate_source_on_unit_grid_is_linear() -> None:
    grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    source = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25, 0.75]], dtype=torch.float64)

    values = interpolate_source_on_unit_grid(
        unit_grid=grid,
        source=source,
        query_points=query,
    )

    torch.testing.assert_close(
        values,
        torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float64),
    )


def test_interpolate_source_on_unit_grid_supports_natural_cubic() -> None:
    grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    source = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25, 0.75]], dtype=torch.float64)

    values = interpolate_source_on_unit_grid(
        unit_grid=grid,
        source=source,
        query_points=query,
        method="cubic",
    )

    torch.testing.assert_close(
        values,
        torch.tensor([[[[0.6875, 0.6875]]]], dtype=torch.float64),
    )


def test_interpolate_source_on_unit_grid_cubic_handles_batched_sources() -> None:
    grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    source = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, -2.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    query = torch.tensor([[0.25, 0.75], [0.0, 1.0]], dtype=torch.float64)

    values = interpolate_source_on_unit_grid(
        unit_grid=grid,
        source=source,
        query_points=query,
        method="cubic",
    )

    assert values.shape == (2, 2, 2, 2)
    torch.testing.assert_close(
        values[0, 1, 0],
        torch.tensor([1.375, 1.375], dtype=torch.float64),
    )
    torch.testing.assert_close(
        values[1, 0, 0],
        torch.tensor([-0.6875, -0.6875], dtype=torch.float64),
    )


def test_interpolate_source_on_unit_grid_cubic_two_points_matches_linear() -> None:
    grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    source = torch.tensor([[[0.0, 2.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25, 0.75]], dtype=torch.float64)

    cubic = interpolate_source_on_unit_grid(
        unit_grid=grid,
        source=source,
        query_points=query,
        method="cubic",
    )
    linear = interpolate_source_on_unit_grid(
        unit_grid=grid,
        source=source,
        query_points=query,
        method="linear",
    )

    torch.testing.assert_close(cubic, linear)


def test_interpolate_source_on_unit_grid_rejects_non_increasing_grid() -> None:
    grid = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    source = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25]], dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_source_on_unit_grid(
            unit_grid=grid,
            source=source,
            query_points=query,
            method="cubic",
        )


def test_reconstruct_split_gauss_legendre_constant_integrand() -> None:
    target_grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    source_grid = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    source = torch.ones((2, 3, 9), dtype=torch.float64)
    kernel_nodes = torch.ones((3, 5, 6), dtype=torch.float64)

    reconstruction = reconstruct_split_gauss_legendre(
        kernel_nodes=kernel_nodes,
        source=source,
        source_grid=source_grid,
        target_grid=target_grid,
        order=3,
    )

    torch.testing.assert_close(
        reconstruction,
        torch.ones((2, 3, 5), dtype=torch.float64),
    )


def test_reconstruct_split_gauss_legendre_accepts_cubic_source_interpolation() -> None:
    target_grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    source_grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    source = torch.ones((2, 3, 5), dtype=torch.float64)
    kernel_nodes = torch.ones((3, 5, 4), dtype=torch.float64)

    reconstruction = reconstruct_split_gauss_legendre(
        kernel_nodes=kernel_nodes,
        source=source,
        source_grid=source_grid,
        target_grid=target_grid,
        order=2,
        source_interpolation="cubic",
    )

    torch.testing.assert_close(
        reconstruction,
        torch.ones((2, 3, 5), dtype=torch.float64),
    )
