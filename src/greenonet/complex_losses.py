from __future__ import annotations

from typing import cast

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata


def physical_edge_energy_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> torch.Tensor:
    if u_phi_valid.shape != u_psi_valid.shape:
        raise ValueError("u_phi_valid and u_psi_valid must have matching shapes.")
    if u_phi_valid.dim() != 2:
        raise ValueError("u_phi_valid and u_psi_valid must have shape (B, P).")
    if a_valid.dim() == 1:
        a_valid = a_valid.unsqueeze(0).expand_as(u_phi_valid)
    if a_valid.shape != u_phi_valid.shape:
        raise ValueError("a_valid must have shape (P,) or (B, P).")

    residual = u_phi_valid - u_psi_valid
    loss = residual.new_zeros(())
    loss = loss + _edge_energy(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.x_edges.to(residual.device),
        spacing=geometry.hx.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    loss = loss + _edge_energy(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.y_edges.to(residual.device),
        spacing=geometry.hy.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    return loss


def _edge_energy(
    *,
    residual: torch.Tensor,
    a_valid: torch.Tensor,
    edges: torch.Tensor,
    spacing: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    if edges.numel() == 0:
        return residual.new_zeros(())
    left = edges[:, 0]
    right = edges[:, 1]
    derivative = (residual[:, right] - residual[:, left]) / spacing
    a_face = 0.5 * (a_valid[:, left] + a_valid[:, right])
    per_batch = (a_face * derivative.pow(2)).sum(dim=-1) * area
    return per_batch.mean()


def relative_l2_valid(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes.")
    numerator = torch.linalg.vector_norm(pred - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(eps)
    return cast(torch.Tensor, (numerator / denominator).mean())
