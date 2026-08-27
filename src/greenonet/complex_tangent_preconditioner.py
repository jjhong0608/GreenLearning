from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from greenonet.complex_axial_response_operator import TangentColumnGramTerms

TangentPreconditionerVariant = Literal[
    "separable",
    "exact_diagonal",
    "absolute_cross_axis",
    "normalized_quadratic_cross_axis",
]

TANGENT_PRECONDITIONER_VARIANTS: tuple[TangentPreconditionerVariant, ...] = (
    "separable",
    "exact_diagonal",
    "absolute_cross_axis",
    "normalized_quadratic_cross_axis",
)


@dataclass(frozen=True)
class TangentPreconditionerTerms:
    """All static diagonal tangent-preconditioner terms for one operator."""

    variant: TangentPreconditionerVariant
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    rho: torch.Tensor
    q: torch.Tensor
    separable_base: torch.Tensor
    exact_base: torch.Tensor
    absolute_base: torch.Tensor
    quadratic_base: torch.Tensor
    selected_base: torch.Tensor
    gain_scale: torch.Tensor
    q_epsilon: torch.Tensor
    damping: torch.Tensor
    separable_denominator: torch.Tensor
    exact_denominator: torch.Tensor
    absolute_denominator: torch.Tensor
    quadratic_denominator: torch.Tensor
    denominator: torch.Tensor
    cauchy_violation: torch.Tensor
    cauchy_violation_max: torch.Tensor
    exact_roundoff_clamp_mask: torch.Tensor
    exact_roundoff_clamp_count: int

    def base_for(self, variant: TangentPreconditionerVariant) -> torch.Tensor:
        return {
            "separable": self.separable_base,
            "exact_diagonal": self.exact_base,
            "absolute_cross_axis": self.absolute_base,
            "normalized_quadratic_cross_axis": self.quadratic_base,
        }[variant]

    def denominator_for(self, variant: TangentPreconditionerVariant) -> torch.Tensor:
        return {
            "separable": self.separable_denominator,
            "exact_diagonal": self.exact_denominator,
            "absolute_cross_axis": self.absolute_denominator,
            "normalized_quadratic_cross_axis": self.quadratic_denominator,
        }[variant]


def build_tangent_preconditioner_terms(
    *,
    gram: TangentColumnGramTerms,
    variant: TangentPreconditionerVariant,
    relative_lambda: float,
    denominator_relative_eps: float,
    cross_axis_relative_eps: float,
) -> TangentPreconditionerTerms:
    """Build the four matrix-free diagonal candidates and select one."""

    if variant not in TANGENT_PRECONDITIONER_VARIANTS:
        raise ValueError(f"Unsupported tangent preconditioner variant: {variant}.")
    a, b, c = gram.a, gram.b, gram.c
    _validate_gram_terms(gram)
    for name, value, positive in (
        ("relative_lambda", relative_lambda, False),
        ("denominator_relative_eps", denominator_relative_eps, True),
        ("cross_axis_relative_eps", cross_axis_relative_eps, True),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric.")
        scalar = float(value)
        if (
            not torch.isfinite(torch.tensor(scalar))
            or scalar < 0.0
            or (positive and scalar == 0.0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be finite and {qualifier}.")

    separable_base = a + b
    gain_scale = separable_base.mean()
    if not torch.isfinite(gain_scale) or bool((gain_scale <= 0.0).item()):
        raise ValueError(
            "Tangent preconditioner gain scale must be finite and positive."
        )

    dtype_eps = torch.finfo(a.dtype).eps
    scale = torch.maximum(separable_base, gain_scale.expand_as(separable_base))
    cauchy_tolerance = 128.0 * dtype_eps * scale
    ab_root = torch.sqrt(torch.clamp_min(a * b, 0.0))
    cauchy_violation = torch.abs(c) - ab_root
    if torch.any(cauchy_violation > cauchy_tolerance):
        maximum = float(cauchy_violation.max().item())
        raise ValueError(
            "Cross-axis tangent Gram term violates Cauchy-Schwarz beyond "
            f"roundoff tolerance: max_violation={maximum:.6e}."
        )

    exact_raw = separable_base + 2.0 * c
    invalid_exact = exact_raw < -cauchy_tolerance
    if torch.any(invalid_exact):
        minimum = float(exact_raw.min().item())
        raise ValueError(
            "Exact tangent diagonal is negative beyond roundoff tolerance: "
            f"min={minimum:.6e}."
        )
    exact_roundoff_clamp_mask = exact_raw < 0.0
    exact_base = torch.where(
        exact_roundoff_clamp_mask,
        torch.zeros_like(exact_raw),
        exact_raw,
    )

    q_epsilon = float(cross_axis_relative_eps) * gain_scale
    q = c.square() / (separable_base + q_epsilon)
    rho_denominator = torch.maximum(ab_root, q_epsilon.expand_as(ab_root))
    rho = c / rho_denominator
    absolute_base = separable_base + 2.0 * torch.abs(c)
    quadratic_base = separable_base + 4.0 * q
    damping = (float(relative_lambda) + float(denominator_relative_eps)) * gain_scale

    bases = {
        "separable": separable_base,
        "exact_diagonal": exact_base,
        "absolute_cross_axis": absolute_base,
        "normalized_quadratic_cross_axis": quadratic_base,
    }
    denominators = {name: base + damping for name, base in bases.items()}
    for name, denominator in denominators.items():
        if not torch.all(torch.isfinite(denominator)) or torch.any(denominator <= 0.0):
            raise ValueError(
                f"Tangent preconditioner denominator {name} must be finite and positive."
            )

    selected_base = bases[variant]
    denominator = denominators[variant]
    return TangentPreconditionerTerms(
        variant=variant,
        a=a.detach(),
        b=b.detach(),
        c=c.detach(),
        rho=rho.detach(),
        q=q.detach(),
        separable_base=separable_base.detach(),
        exact_base=exact_base.detach(),
        absolute_base=absolute_base.detach(),
        quadratic_base=quadratic_base.detach(),
        selected_base=selected_base.detach(),
        gain_scale=gain_scale.detach(),
        q_epsilon=q_epsilon.detach(),
        damping=damping.detach(),
        separable_denominator=denominators["separable"].detach(),
        exact_denominator=denominators["exact_diagonal"].detach(),
        absolute_denominator=denominators["absolute_cross_axis"].detach(),
        quadratic_denominator=denominators["normalized_quadratic_cross_axis"].detach(),
        denominator=denominator.detach(),
        cauchy_violation=cauchy_violation.detach(),
        cauchy_violation_max=torch.clamp_min(cauchy_violation, 0.0).max().detach(),
        exact_roundoff_clamp_mask=exact_roundoff_clamp_mask.detach(),
        exact_roundoff_clamp_count=int(exact_roundoff_clamp_mask.sum().item()),
    )


def _validate_gram_terms(gram: TangentColumnGramTerms) -> None:
    a, b, c = gram.a, gram.b, gram.c
    if a.shape != b.shape or a.shape != c.shape or a.dim() != 1:
        raise ValueError("Tangent column Gram terms must be same-shape vectors.")
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise ValueError("Tangent column Gram terms must share a dtype.")
    if a.device != b.device or a.device != c.device:
        raise ValueError("Tangent column Gram terms must share a device.")
    if not a.dtype.is_floating_point:
        raise ValueError("Tangent column Gram terms must use floating dtype.")
    if not torch.all(torch.isfinite(a)) or not torch.all(torch.isfinite(b)):
        raise ValueError("Tangent self Gram terms must be finite.")
    if torch.any(a < 0.0) or torch.any(b < 0.0):
        raise ValueError("Tangent self Gram terms must be non-negative.")
    if not torch.all(torch.isfinite(c)):
        raise ValueError("Tangent cross-axis Gram terms must be finite.")
