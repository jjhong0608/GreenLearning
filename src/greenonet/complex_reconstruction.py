from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.green_interval import evaluate_green_pairs


@dataclass(frozen=True)
class ComplexReconstructionResult:
    u_phi_valid: torch.Tensor
    u_psi_valid: torch.Tensor
    projected_response: torch.Tensor

    @property
    def u_mean_valid(self) -> torch.Tensor:
        return 0.5 * (self.u_phi_valid + self.u_psi_valid)


def reconstruct_from_projected_response(
    *,
    green_model: torch.nn.Module,
    geometry: ComplexGeometryMetadata,
    projected_response: torch.Tensor,
    x_green_branch: torch.Tensor,
    y_green_branch: torch.Tensor,
) -> ComplexReconstructionResult:
    """Reconstruct directly from projected unit-interval response sources."""

    if projected_response.dim() != 3 or projected_response.shape[1] != 2:
        raise ValueError("projected_response must have shape (B, 2, P).")
    if projected_response.shape[-1] != geometry.num_points:
        raise ValueError("projected_response point count does not match geometry.")
    bsz, _axis, point_count = projected_response.shape
    u_phi = torch.zeros(
        (bsz, point_count),
        dtype=projected_response.dtype,
        device=projected_response.device,
    )
    u_psi = torch.zeros_like(u_phi)
    _reconstruct_axis(
        green_model=green_model,
        ptr=geometry.x_recon_ptr,
        t=geometry.x_recon_t,
        weight=geometry.x_recon_weight,
        valid_index=geometry.x_recon_valid_index,
        source_valid=projected_response[:, 0],
        green_branch=x_green_branch,
        output_valid=u_phi,
    )
    _reconstruct_axis(
        green_model=green_model,
        ptr=geometry.y_recon_ptr,
        t=geometry.y_recon_t,
        weight=geometry.y_recon_weight,
        valid_index=geometry.y_recon_valid_index,
        source_valid=projected_response[:, 1],
        green_branch=y_green_branch,
        output_valid=u_psi,
    )
    return ComplexReconstructionResult(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        projected_response=projected_response,
    )


def _reconstruct_axis(
    *,
    green_model: torch.nn.Module,
    ptr: torch.Tensor,
    t: torch.Tensor,
    weight: torch.Tensor,
    valid_index: torch.Tensor,
    source_valid: torch.Tensor,
    green_branch: torch.Tensor,
    output_valid: torch.Tensor,
) -> None:
    if green_branch.dim() != 4 or green_branch.shape[2] != 4:
        raise ValueError("green_branch must have shape (B, S, 4, M).")
    device = source_valid.device
    ptr = ptr.to(device)
    t = t.to(device)
    weight = weight.to(device)
    valid_index = valid_index.to(device)
    for segment_index in range(int(ptr.numel()) - 1):
        start = int(ptr[segment_index].item())
        end = int(ptr[segment_index + 1].item())
        node_t = t[start:end]
        node_weight = weight[start:end]
        node_valid_index = valid_index[start:end]
        node_source = torch.zeros(
            (source_valid.shape[0], end - start),
            dtype=source_valid.dtype,
            device=device,
        )
        interior = node_valid_index >= 0
        if torch.any(interior):
            node_source[:, interior] = source_valid[:, node_valid_index[interior]]
        with torch.no_grad():
            branch = green_branch[0, segment_index]
            kernel = evaluate_green_pairs(
                green_model,
                a_unit=branch[0],
                ap_unit=branch[1],
                b_unit=branch[2],
                c_unit=branch[3],
                t_eval=node_t,
                eta_eval=node_t,
            ).to(device=device, dtype=source_valid.dtype)
        node_solution = torch.matmul(node_source * node_weight.unsqueeze(0), kernel.T)
        if torch.any(interior):
            output_valid[:, node_valid_index[interior]] = node_solution[:, interior]
