import torch
import pytest

from greenonet.trainer import Trainer


def _make_trunk_grid(m_points: int) -> torch.Tensor:
    return torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, m_points),
            torch.linspace(0.0, 1.0, m_points),
            indexing="ij",
        ),
        dim=-1,
    )


def test_green_loss_zero_when_reconstructed() -> None:
    m = 3
    trunk_grid = _make_trunk_grid(m)
    prediction = torch.ones((2, 1, m, m), dtype=torch.float64)
    source = torch.ones((1, 2, 1, m), dtype=torch.float64)
    solution = torch.ones((1, 2, 1, m), dtype=torch.float64)
    loss, rel = Trainer._green_reconstruction_loss(
        prediction=prediction,
        source=source,
        solution=solution,
        trunk_grid=trunk_grid,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert rel.item() == pytest.approx(0.0, abs=1e-12)


def test_green_loss_positive_when_mismatch() -> None:
    m = 3
    trunk_grid = _make_trunk_grid(m)
    prediction = torch.ones((2, 1, m, m), dtype=torch.float64)
    source = torch.ones((1, 2, 1, m), dtype=torch.float64)
    solution = torch.zeros((1, 2, 1, m), dtype=torch.float64)
    loss, rel = Trainer._green_reconstruction_loss(
        prediction=prediction,
        source=source,
        solution=solution,
        trunk_grid=trunk_grid,
    )
    assert loss.item() > 0.0
    assert rel.item() > 0.0
