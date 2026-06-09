from pathlib import Path

import pytest
import torch

from greenonet.coefficients import (
    default_coefficient_functions,
    load_coefficient_functions,
)


def test_default_coefficient_functions_match_previous_train_cli_formulas() -> None:
    coeffs = default_coefficient_functions()
    x = torch.tensor([0.125, 0.25, 0.5], dtype=torch.float64)
    y = torch.tensor([0.25, 0.375, 0.5], dtype=torch.float64)

    expected_a = 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
    expected_apx = torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
    expected_apy = (
        2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(4 * torch.pi * y)
    )

    torch.testing.assert_close(coeffs.a_fun(x, y), expected_a)
    torch.testing.assert_close(coeffs.apx_fun(x, y), expected_apx)
    torch.testing.assert_close(coeffs.apy_fun(x, y), expected_apy)
    torch.testing.assert_close(coeffs.bx_fun(x, y), torch.zeros_like(x))
    torch.testing.assert_close(coeffs.by_fun(x, y), torch.zeros_like(x))
    torch.testing.assert_close(coeffs.c_fun(x, y), torch.zeros_like(x))


def test_load_coefficient_functions_uses_default_when_path_is_none() -> None:
    coeffs = load_coefficient_functions(None)
    default_coeffs = default_coefficient_functions()
    x = torch.tensor([0.25], dtype=torch.float64)
    y = torch.tensor([0.5], dtype=torch.float64)

    torch.testing.assert_close(coeffs.a_fun(x, y), default_coeffs.a_fun(x, y))


def test_load_coefficient_functions_imports_custom_file(tmp_path: Path) -> None:
    coeff_path = tmp_path / "custom_coefficients.py"
    coeff_path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return x + y + 1.0",
                "def apx_fun(x, y): return torch.ones_like(x) * 2.0",
                "def apy_fun(x, y): return torch.ones_like(y) * 3.0",
                "def bx_fun(x, y): return torch.ones_like(x) * 4.0",
                "def by_fun(x, y): return torch.ones_like(y) * 6.0",
                "def c_fun(x, y): return torch.ones_like(y) * 5.0",
            ]
        )
    )
    x = torch.tensor([0.25, 0.5], dtype=torch.float64)
    y = torch.tensor([0.5, 0.75], dtype=torch.float64)

    coeffs = load_coefficient_functions(coeff_path)

    torch.testing.assert_close(coeffs.a_fun(x, y), x + y + 1.0)
    torch.testing.assert_close(coeffs.apx_fun(x, y), torch.ones_like(x) * 2.0)
    torch.testing.assert_close(coeffs.apy_fun(x, y), torch.ones_like(y) * 3.0)
    torch.testing.assert_close(coeffs.bx_fun(x, y), torch.ones_like(x) * 4.0)
    torch.testing.assert_close(coeffs.by_fun(x, y), torch.ones_like(y) * 6.0)
    torch.testing.assert_close(coeffs.c_fun(x, y), torch.ones_like(y) * 5.0)


def test_load_coefficient_functions_supports_legacy_b_fun(tmp_path: Path) -> None:
    coeff_path = tmp_path / "legacy_coefficients.py"
    coeff_path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return x + y + 1.0",
                "def apx_fun(x, y): return torch.ones_like(x) * 2.0",
                "def apy_fun(x, y): return torch.ones_like(y) * 3.0",
                "def b_fun(x, y): return x - y",
                "def c_fun(x, y): return torch.ones_like(y) * 5.0",
            ]
        )
    )
    x = torch.tensor([0.25, 0.5], dtype=torch.float64)
    y = torch.tensor([0.5, 0.75], dtype=torch.float64)

    coeffs = load_coefficient_functions(coeff_path)

    torch.testing.assert_close(coeffs.bx_fun(x, y), x - y)
    torch.testing.assert_close(coeffs.by_fun(x, y), x - y)


def test_sinusoidal_coefficient_example_matches_default() -> None:
    example_path = Path("configs/sinusoidal_coefficients.py")
    coeffs = load_coefficient_functions(example_path)
    default_coeffs = default_coefficient_functions()
    x = torch.tensor([0.125, 0.25, 0.5], dtype=torch.float64)
    y = torch.tensor([0.25, 0.375, 0.5], dtype=torch.float64)

    torch.testing.assert_close(coeffs.a_fun(x, y), default_coeffs.a_fun(x, y))
    torch.testing.assert_close(coeffs.apx_fun(x, y), default_coeffs.apx_fun(x, y))
    torch.testing.assert_close(coeffs.apy_fun(x, y), default_coeffs.apy_fun(x, y))
    torch.testing.assert_close(coeffs.bx_fun(x, y), default_coeffs.bx_fun(x, y))
    torch.testing.assert_close(coeffs.by_fun(x, y), default_coeffs.by_fun(x, y))
    torch.testing.assert_close(coeffs.c_fun(x, y), default_coeffs.c_fun(x, y))


def test_divergence_free_convection_diffusion_example() -> None:
    coeffs = load_coefficient_functions(
        Path("coefficients/Divergence_Free_Convection_Diffusion.py")
    )
    x = torch.tensor([0.125, 0.25, 0.5], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.25, 0.375, 0.5], dtype=torch.float64, requires_grad=True)
    amplitude = 2.0

    expected_a = 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
    expected_apx = torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
    expected_apy = (
        torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)
    )
    expected_bx = (
        amplitude * torch.pi * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    )
    expected_by = (
        -amplitude * torch.pi * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    )

    torch.testing.assert_close(coeffs.a_fun(x, y), expected_a)
    torch.testing.assert_close(coeffs.apx_fun(x, y), expected_apx)
    torch.testing.assert_close(coeffs.apy_fun(x, y), expected_apy)
    torch.testing.assert_close(coeffs.bx_fun(x, y), expected_bx)
    torch.testing.assert_close(coeffs.by_fun(x, y), expected_by)
    torch.testing.assert_close(coeffs.c_fun(x, y), torch.zeros_like(x))

    dbx_dx = torch.autograd.grad(coeffs.bx_fun(x, y).sum(), x, create_graph=True)[0]
    dby_dy = torch.autograd.grad(coeffs.by_fun(x, y).sum(), y)[0]
    torch.testing.assert_close(dbx_dx + dby_dy, torch.zeros_like(x))


def test_load_coefficient_functions_rejects_missing_callable(tmp_path: Path) -> None:
    coeff_path = tmp_path / "missing_coefficients.py"
    coeff_path.write_text(
        "\n".join(
            [
                "def a_fun(x, y): return x",
                "def apx_fun(x, y): return x",
                "def apy_fun(x, y): return y",
                "def b_fun(x, y): return x",
            ]
        )
    )

    with pytest.raises(ValueError, match="c_fun"):
        load_coefficient_functions(coeff_path)


def test_load_coefficient_functions_rejects_non_callable(tmp_path: Path) -> None:
    coeff_path = tmp_path / "non_callable_coefficients.py"
    coeff_path.write_text(
        "\n".join(
            [
                "def a_fun(x, y): return x",
                "def apx_fun(x, y): return x",
                "def apy_fun(x, y): return y",
                "b_fun = 3.0",
                "def c_fun(x, y): return y",
            ]
        )
    )

    with pytest.raises(TypeError, match="b_fun"):
        load_coefficient_functions(coeff_path)


def test_load_coefficient_functions_rejects_one_sided_directional_b(
    tmp_path: Path,
) -> None:
    coeff_path = tmp_path / "one_sided_coefficients.py"
    coeff_path.write_text(
        "\n".join(
            [
                "def a_fun(x, y): return x",
                "def apx_fun(x, y): return x",
                "def apy_fun(x, y): return y",
                "def bx_fun(x, y): return x",
                "def c_fun(x, y): return y",
            ]
        )
    )

    with pytest.raises(ValueError, match="bx_fun.*by_fun"):
        load_coefficient_functions(coeff_path)


def test_load_coefficient_functions_rejects_mixed_directional_and_legacy_b(
    tmp_path: Path,
) -> None:
    coeff_path = tmp_path / "mixed_coefficients.py"
    coeff_path.write_text(
        "\n".join(
            [
                "def a_fun(x, y): return x",
                "def apx_fun(x, y): return x",
                "def apy_fun(x, y): return y",
                "def bx_fun(x, y): return x",
                "def by_fun(x, y): return y",
                "def b_fun(x, y): return x + y",
                "def c_fun(x, y): return y",
            ]
        )
    )

    with pytest.raises(ValueError, match="legacy 'b_fun'"):
        load_coefficient_functions(coeff_path)
