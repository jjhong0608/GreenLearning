from __future__ import annotations

import torch

from greenonet.green_interval import (
    evaluate_green_pairs,
    transform_interval_coefficients,
    transform_interval_source,
)


class EchoGreen(torch.nn.Module):
    def forward_pairs(
        self,
        trunk_coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> torch.Tensor:
        del a_vals, ap_vals, b_vals, c_vals
        return trunk_coords[..., 0].unsqueeze(0) + 2.0 * trunk_coords[..., 1].unsqueeze(
            0
        )


def test_unit_interval_coefficient_scaling():
    length = torch.tensor([2.0, 0.5], dtype=torch.float64)
    coeffs = transform_interval_coefficients(
        a_phys=torch.ones((2, 3), dtype=torch.float64),
        ap_phys=torch.full((2, 3), 2.0, dtype=torch.float64),
        b_phys=torch.full((2, 3), 3.0, dtype=torch.float64),
        c_phys=torch.full((2, 3), 4.0, dtype=torch.float64),
        length=length,
    )
    source = transform_interval_source(
        torch.full((2, 3), 5.0, dtype=torch.float64),
        length,
    )

    torch.testing.assert_close(coeffs.a_unit[0], torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(
        coeffs.ap_unit[0], torch.full((3,), 4.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        coeffs.b_unit[0], torch.full((3,), 6.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        coeffs.c_unit[0], torch.full((3,), 16.0, dtype=torch.float64)
    )
    torch.testing.assert_close(source[0], torch.full((3,), 20.0, dtype=torch.float64))


def test_green_pair_evaluation_does_not_apply_length_factor():
    t = torch.tensor([0.25, 0.75], dtype=torch.float64)
    eta = torch.tensor([0.5], dtype=torch.float64)

    kernel = evaluate_green_pairs(
        EchoGreen(),
        a_unit=torch.ones(4, dtype=torch.float64),
        ap_unit=torch.zeros(4, dtype=torch.float64),
        b_unit=torch.zeros(4, dtype=torch.float64),
        c_unit=torch.zeros(4, dtype=torch.float64),
        t_eval=t,
        eta_eval=eta,
    )

    torch.testing.assert_close(
        kernel,
        torch.tensor([[1.25], [1.75]], dtype=torch.float64),
    )
