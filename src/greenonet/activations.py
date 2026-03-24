from __future__ import annotations

import torch
from torch import Tensor, nn


class FastPolyval(nn.Module):  # type: ignore[misc]
    """Polynomial evaluation with Horner's method; coefficients are learnable parameters."""

    def __init__(self, coeffs: Tensor) -> None:
        super().__init__()
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, x: Tensor) -> Tensor:
        result = self.coeffs[0].expand_as(x)
        for coeff in self.coeffs[1:]:
            result = result * x + coeff
        return result


class FastPolyvalWithoutConstant(nn.Module):  # type: ignore[misc]
    """Horner's method without constant term; multiplies by x once more at the end."""

    def __init__(self, coeffs: Tensor) -> None:
        super().__init__()
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, x: Tensor) -> Tensor:
        coeffs = self.coeffs[:-1]
        result = coeffs[0].expand_as(x)
        for coeff in coeffs[1:]:
            result = result * x + coeff
        return result * x


class RationalActivation(nn.Module):  # type: ignore[misc]
    """Rational activation used in the original GreenONet codebase."""

    def __init__(self) -> None:
        super().__init__()
        numerator_weights: Tensor = torch.tensor(
            [1.1915, 1.5957, 0.5, 0.0218], dtype=torch.float64
        )
        denominator_weights: Tensor = torch.tensor([2.383, 0.0], dtype=torch.float64)
        self.register_buffer(
            "numerator_weights",
            numerator_weights,
            persistent=False,
        )
        self.register_buffer(
            "denominator_weights",
            denominator_weights,
            persistent=False,
        )
        self.numerator = FastPolyval(numerator_weights)
        self.denominator = FastPolyvalWithoutConstant(denominator_weights)

    def forward(self, x: Tensor) -> Tensor:
        return torch.div(self.numerator(x), 1 + self.denominator(x).abs())
