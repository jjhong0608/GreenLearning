from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_tangent_context_io import (
    TangentContextIdentity,
    TangentResponseContextStore,
)
from greenonet.complex_tangent_preconditioner import (
    TANGENT_PRECONDITIONER_VARIANTS,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig


def _operator(
    device: torch.device | str = "cpu",
) -> FrozenBidirectionalResponseOperator:
    dtype = torch.float64
    return FrozenBidirectionalResponseOperator(
        x=FrozenAxialResponseOperator(
            axis="x",
            point_count=4,
            blocks=(
                AxialResponseBlock(
                    torch.tensor([0, 1], dtype=torch.long, device=device),
                    torch.tensor([[1.0, 0.2], [0.1, 0.8]], dtype=dtype, device=device),
                ),
                AxialResponseBlock(
                    torch.tensor([2, 3], dtype=torch.long, device=device),
                    torch.tensor(
                        [[0.7, -0.1], [0.25, 1.1]], dtype=dtype, device=device
                    ),
                ),
            ),
        ),
        y=FrozenAxialResponseOperator(
            axis="y",
            point_count=4,
            blocks=(
                AxialResponseBlock(
                    torch.tensor([0, 2], dtype=torch.long, device=device),
                    torch.tensor([[0.9, -0.2], [0.3, 1.2]], dtype=dtype, device=device),
                ),
                AxialResponseBlock(
                    torch.tensor([1, 3], dtype=torch.long, device=device),
                    torch.tensor(
                        [[1.1, 0.1], [-0.15, 0.75]], dtype=dtype, device=device
                    ),
                ),
            ),
        ),
    )


def _config(
    *,
    variant: str = "separable",
    subspace_dimension: int = 1,
) -> SymmetricTangentGreenResponseProjectionConfig:
    return SymmetricTangentGreenResponseProjectionConfig(
        subspace_dimension=subspace_dimension,
        max_subspace_dimension=max(8, subspace_dimension),
        eta=0.1,
        eta_strategy="closed_loop_exact_line_search",
        relative_lambda=0.01,
        denominator_relative_eps=1.0e-12,
        preconditioner_variant=variant,  # type: ignore[arg-type]
        cross_axis_relative_eps=1.0e-12,
    )


def _context(
    *,
    variant: str = "separable",
    subspace_dimension: int = 1,
    device: torch.device | str = "cpu",
) -> SymmetricTangentGreenResponseContext:
    return SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=_operator(device),
        point_mass=torch.tensor(0.125, dtype=torch.float64, device=device),
        config=_config(
            variant=variant,
            subspace_dimension=subspace_dimension,
        ),
    )


def _identity() -> TangentContextIdentity:
    return TangentContextIdentity(
        geometry_semantic_sha256="sha256:geometry",
        green_state_dict_sha256="sha256:green",
        x_green_branch_sha256="sha256:x-branch",
        y_green_branch_sha256="sha256:y-branch",
        reconstruction_contract_id=(
            TangentResponseContextStore.RECONSTRUCTION_CONTRACT_ID
        ),
        floating_dtype="float64",
        point_count=4,
        point_mass_hex=float(0.125).hex(),
    )


def _load_manifest_and_tensors(
    path: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        manifest = json.loads((handle.metadata() or {})["manifest_json"])
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return manifest, tensors


def test_schema_v2_round_trip_is_shared_by_all_variants_and_k(tmp_path: Path) -> None:
    path = tmp_path / "tangent_response_context.safetensors"
    built = _context()
    manifest = TangentResponseContextStore.save(
        path=path,
        context=built,
        identity=_identity(),
    )

    assert manifest["schema_version"] == 2
    assert path.is_file()
    assert path.with_suffix(".json").is_file()

    source = torch.tensor(
        [[0.2, -0.4, 0.8, -0.1], [-0.5, 0.3, 0.1, 0.9]],
        dtype=torch.float64,
    )
    mismatch = torch.tensor(
        [[0.4, -0.2, 0.7, -0.5], [0.1, 0.8, -0.3, 0.2]],
        dtype=torch.float64,
    )
    for variant in TANGENT_PRECONDITIONER_VARIANTS:
        for dimension in (1, 2, 3, 4, 9):
            config = _config(variant=variant, subspace_dimension=dimension)
            loaded = TangentResponseContextStore.load(
                path=path,
                identity=_identity(),
                config=config,
                device=torch.device("cpu"),
            )
            loaded_context = (
                SymmetricTangentGreenResponseContext.from_preconditioner_terms(
                    response_operator=loaded.response_operator,
                    point_mass=loaded.point_mass,
                    terms=loaded.terms,
                    config=config,
                )
            )
            reference = _context(
                variant=variant,
                subspace_dimension=dimension,
            )
            assert torch.equal(loaded_context.denominator, reference.denominator)
            assert torch.equal(
                loaded_context.response_operator.x.forward(source),
                reference.response_operator.x.forward(source),
            )
            assert torch.equal(
                loaded_context.response_operator.y.adjoint(source),
                reference.response_operator.y.adjoint(source),
            )
            gradient = reference.tangent_gradient(mismatch)
            loaded_gradient = loaded_context.tangent_gradient(mismatch)
            assert torch.equal(loaded_gradient, gradient)
            reference_step = reference.tangent_step(
                mismatch=mismatch,
                gradient=gradient,
            )
            loaded_step = loaded_context.tangent_step(
                mismatch=mismatch,
                gradient=loaded_gradient,
            )
            assert torch.equal(loaded_step.delta, reference_step.delta)
            if dimension == 1:
                assert loaded_step.residual_gradient_post is None
                assert reference_step.residual_gradient_post is None
            else:
                assert loaded_step.residual_gradient_post is not None
                assert reference_step.residual_gradient_post is not None
                assert torch.equal(
                    loaded_step.residual_gradient_post,
                    reference_step.residual_gradient_post,
                )


def test_sidecar_rejects_identity_and_static_config_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "context.safetensors"
    TangentResponseContextStore.save(
        path=path,
        context=_context(),
        identity=_identity(),
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        TangentResponseContextStore.load(
            path=path,
            identity=replace(
                _identity(),
                green_state_dict_sha256="sha256:different-green",
            ),
            config=_config(),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="relative_lambda mismatch"):
        TangentResponseContextStore.load(
            path=path,
            identity=_identity(),
            config=replace(_config(), relative_lambda=0.02),
            device=torch.device("cpu"),
        )


def test_sidecar_rejects_corrupt_tensor_payload(tmp_path: Path) -> None:
    path = tmp_path / "context.safetensors"
    TangentResponseContextStore.save(
        path=path,
        context=_context(),
        identity=_identity(),
    )
    manifest, tensors = _load_manifest_and_tensors(path)
    tensors["gamma_x_squared"] = tensors["gamma_x_squared"].clone()
    tensors["gamma_x_squared"][0] += 1.0
    save_file(
        tensors,
        str(path),
        metadata={"manifest_json": json.dumps(manifest)},
    )

    with pytest.raises(ValueError, match="payload digest mismatch"):
        TangentResponseContextStore.load(
            path=path,
            identity=_identity(),
            config=_config(),
            device=torch.device("cpu"),
        )


def test_sidecar_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TangentResponseContextStore.load(
            path=tmp_path / "missing.safetensors",
            identity=_identity(),
            config=_config(),
            device=torch.device("cpu"),
        )


def test_static_payload_accepts_roundoff_but_rejects_material_drift() -> None:
    tensors = TangentResponseContextStore._context_tensors(_context())
    gain_scale = tensors["gain_scale"]
    tensors["gain_scale"] = torch.nextafter(
        gain_scale,
        torch.full_like(gain_scale, torch.inf),
    )

    TangentResponseContextStore._validate_static_payload(tensors)

    tensors["gain_scale"] = gain_scale * (1.0 + 1.0e-8)
    with pytest.raises(
        ValueError,
        match=r"gain scale invariant failed: max_abs_error=.*rtol=.*atol=",
    ):
        TangentResponseContextStore._validate_static_payload(tensors)


def test_static_payload_rejects_material_denominator_drift() -> None:
    tensors = TangentResponseContextStore._context_tensors(_context())
    tensors["separable_denominator"] = tensors["separable_denominator"].clone()
    tensors["separable_denominator"][0] += tensors["gain_scale"].item() * 1.0e-8

    with pytest.raises(ValueError, match="separable_denominator invariant failed"):
        TangentResponseContextStore._validate_static_payload(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
@pytest.mark.parametrize("subspace_dimension", [1, 9])
def test_cuda_context_save_load_preserves_response_and_balance(
    tmp_path: Path,
    subspace_dimension: int,
) -> None:
    device = torch.device("cuda")
    path = tmp_path / f"context_k{subspace_dimension}.safetensors"
    built = _context(device=device)
    manifest = TangentResponseContextStore.save(
        path=path,
        context=built,
        identity=_identity(),
    )
    config = _config(subspace_dimension=subspace_dimension)
    loaded = TangentResponseContextStore.load(
        path=path,
        identity=_identity(),
        config=config,
        device=device,
    )
    restored = SymmetricTangentGreenResponseContext.from_preconditioner_terms(
        response_operator=loaded.response_operator,
        point_mass=loaded.point_mass,
        terms=loaded.terms,
        config=config,
    )

    assert loaded.manifest["context_id"] == manifest["context_id"]
    torch.testing.assert_close(restored.denominator, built.denominator)
    rhs = torch.tensor([[0.7, -0.4, 0.2, 0.9]], dtype=torch.float64, device=device)
    proposal_difference = torch.tensor(
        [[0.1, 0.3, -0.5, 0.2]], dtype=torch.float64, device=device
    )
    phi = 0.5 * (rhs + proposal_difference)
    psi = 0.5 * (rhs - proposal_difference)
    mismatch = restored.response_operator.x.forward(
        phi
    ) - restored.response_operator.y.forward(psi)
    gradient = restored.tangent_gradient(mismatch)
    step = restored.tangent_step(mismatch=mismatch, gradient=gradient)
    corrected_phi = phi + step.delta
    corrected_psi = psi - step.delta

    torch.testing.assert_close(
        corrected_phi + corrected_psi,
        rhs,
        rtol=0.0,
        atol=32.0 * torch.finfo(torch.float64).eps,
    )
    torch.testing.assert_close(
        restored.response_operator.x.forward(rhs),
        built.response_operator.x.forward(rhs),
    )
    torch.testing.assert_close(
        restored.response_operator.y.adjoint(rhs),
        built.response_operator.y.adjoint(rhs),
    )
