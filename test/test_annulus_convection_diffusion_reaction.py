from pathlib import Path

import torch

from greenonet.coefficients import load_coefficient_functions


ANNULUS_COEFFICIENT_PATH = Path("coefficients/Annulus_Convection_Diffusion_Reaction.py")
BASE_COEFFICIENT_PATH = Path("coefficients/Convection_Diffusion_Reaction.py")
INNER_RADIUS = 0.2
OUTER_RADIUS = 0.5


def test_annulus_diffusion_and_reaction_match_base_coefficients() -> None:
    annulus = load_coefficient_functions(ANNULUS_COEFFICIENT_PATH)
    base = load_coefficient_functions(BASE_COEFFICIENT_PATH)
    x = torch.tensor([-0.31, -0.11, 0.17, 0.36], dtype=torch.float64)
    y = torch.tensor([0.08, -0.29, 0.27, -0.12], dtype=torch.float64)

    torch.testing.assert_close(annulus.a_fun(x, y), base.a_fun(x, y))
    torch.testing.assert_close(annulus.apx_fun(x, y), base.apx_fun(x, y))
    torch.testing.assert_close(annulus.apy_fun(x, y), base.apy_fun(x, y))
    torch.testing.assert_close(annulus.c_fun(x, y), base.c_fun(x, y))


def test_annulus_convection_vanishes_on_both_boundaries() -> None:
    coefficients = load_coefficient_functions(ANNULUS_COEFFICIENT_PATH)
    angles = torch.linspace(0.0, 2.0 * torch.pi, 17, dtype=torch.float64)

    for radius in (INNER_RADIUS, OUTER_RADIUS):
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        torch.testing.assert_close(
            coefficients.bx_fun(x, y),
            torch.zeros_like(x),
            rtol=0.0,
            atol=1e-15,
        )
        torch.testing.assert_close(
            coefficients.by_fun(x, y),
            torch.zeros_like(y),
            rtol=0.0,
            atol=1e-15,
        )


def test_annulus_convection_is_tangential_and_counter_clockwise() -> None:
    coefficients = load_coefficient_functions(ANNULUS_COEFFICIENT_PATH)
    angles = torch.linspace(0.0, 2.0 * torch.pi, 16, dtype=torch.float64)[:-1]
    radius = torch.tensor(0.35, dtype=torch.float64)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)

    bx = coefficients.bx_fun(x, y)
    by = coefficients.by_fun(x, y)
    radial_component = x * bx + y * by
    counter_clockwise_component = -y * bx + x * by

    torch.testing.assert_close(
        radial_component,
        torch.zeros_like(radial_component),
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.all(counter_clockwise_component > 0.0)


def test_annulus_convection_is_divergence_free() -> None:
    coefficients = load_coefficient_functions(ANNULUS_COEFFICIENT_PATH)
    x = torch.tensor([-0.31, -0.12, 0.18, 0.36], dtype=torch.float64)
    y = torch.tensor([0.13, -0.32, 0.29, -0.11], dtype=torch.float64)
    x.requires_grad_(True)
    y.requires_grad_(True)

    dbx_dx = torch.autograd.grad(
        coefficients.bx_fun(x, y).sum(),
        x,
        create_graph=True,
    )[0]
    dby_dy = torch.autograd.grad(coefficients.by_fun(x, y).sum(), y)[0]

    torch.testing.assert_close(
        dbx_dx + dby_dy,
        torch.zeros_like(x),
        rtol=0.0,
        atol=1e-14,
    )


def test_annulus_convection_is_finite_at_origin() -> None:
    coefficients = load_coefficient_functions(ANNULUS_COEFFICIENT_PATH)
    origin = torch.zeros(1, dtype=torch.float64)

    assert torch.isfinite(coefficients.bx_fun(origin, origin)).all()
    assert torch.isfinite(coefficients.by_fun(origin, origin)).all()
