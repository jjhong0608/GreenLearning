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
    torch.testing.assert_close(coeffs.b_fun(x, y), torch.zeros_like(x))
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
                "def b_fun(x, y): return torch.ones_like(x) * 4.0",
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
    torch.testing.assert_close(coeffs.b_fun(x, y), torch.ones_like(x) * 4.0)
    torch.testing.assert_close(coeffs.c_fun(x, y), torch.ones_like(y) * 5.0)


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
