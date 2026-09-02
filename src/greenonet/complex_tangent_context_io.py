from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, TYPE_CHECKING, cast

import torch

from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_tangent_preconditioner import TangentPreconditionerTerms
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig
from greenonet.config import TangentContextCheckpointConfig

if TYPE_CHECKING:
    from greenonet.complex_tangent_projection import (
        SymmetricTangentGreenResponseContext,
    )


@dataclass(frozen=True)
class TangentContextIdentity:
    geometry_semantic_sha256: str
    green_state_dict_sha256: str
    x_green_branch_sha256: str
    y_green_branch_sha256: str
    reconstruction_contract_id: str
    floating_dtype: str
    point_count: int
    point_mass_hex: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "geometry_semantic_sha256": self.geometry_semantic_sha256,
            "green_state_dict_sha256": self.green_state_dict_sha256,
            "x_green_branch_sha256": self.x_green_branch_sha256,
            "y_green_branch_sha256": self.y_green_branch_sha256,
            "reconstruction_contract_id": self.reconstruction_contract_id,
            "floating_dtype": self.floating_dtype,
            "point_count": self.point_count,
            "point_mass_hex": self.point_mass_hex,
        }


@dataclass(frozen=True)
class LoadedTangentContext:
    response_operator: FrozenBidirectionalResponseOperator
    point_mass: torch.Tensor
    terms: TangentPreconditionerTerms
    manifest: dict[str, Any]


def resolve_tangent_context_path(
    *,
    checkpoint: TangentContextCheckpointConfig,
    cli_override: Path | None,
    default_path: Path,
) -> Path | None:
    """Resolve CLI, config, and deterministic-default context paths in order."""

    if not checkpoint.enabled:
        if cli_override is not None:
            raise ValueError(
                "--tangent-context requires "
                "coupling_training.tangent_context_checkpoint.enabled=true."
            )
        return None
    if cli_override is not None:
        return cli_override
    if checkpoint.path is not None:
        return checkpoint.path
    return default_path


class TangentResponseContextStore:
    """Strict schema-v2 safetensors storage for a frozen tangent context."""

    FORMAT_NAME = "greenonet_symmetric_tangent_context"
    SCHEMA_VERSION = 2
    FORMULA_SUITE_ID = "tangent_diagonal_preconditioner_suite_v2"
    RECONSTRUCTION_CONTRACT_ID = (
        "segment_green_kernel_nonuniform_unit_quadrature_physical_source_l2_v1"
    )
    _FLOATING_INVARIANT_EPS_FACTOR = 512.0

    _PRECONDITIONER_KEYS = (
        "gamma_x_squared",
        "gamma_y_squared",
        "cross_axis_inner_product",
        "normalized_correlation",
        "normalized_quadratic_cross_axis",
        "separable_preconditioner_base",
        "exact_preconditioner_base",
        "absolute_preconditioner_base",
        "quadratic_preconditioner_base",
        "separable_denominator",
        "exact_denominator",
        "absolute_denominator",
        "quadratic_denominator",
        "gain_scale",
        "q_epsilon",
        "damping",
        "point_mass",
        "cauchy_violation",
        "cauchy_violation_max",
        "exact_roundoff_clamp_mask",
        "exact_roundoff_clamp_count",
    )

    @classmethod
    def identity(
        cls,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
        point_mass: torch.Tensor,
    ) -> TangentContextIdentity:
        x_canonical = cls._canonical_branch(x_green_branch, "x_green_branch")
        y_canonical = cls._canonical_branch(y_green_branch, "y_green_branch")
        geometry_items = sorted(
            (
                name,
                value,
            )
            for name, value in geometry.__dict__.items()
            if isinstance(value, torch.Tensor)
        )
        state_items = sorted(green_model.state_dict().items())
        mass_value = float(point_mass.detach().cpu().reshape(-1)[0].item())
        return TangentContextIdentity(
            geometry_semantic_sha256=cls._hash_named_tensors(geometry_items),
            green_state_dict_sha256=cls._hash_named_tensors(state_items),
            x_green_branch_sha256=cls.tensor_sha256("x_green_branch", x_canonical),
            y_green_branch_sha256=cls.tensor_sha256("y_green_branch", y_canonical),
            reconstruction_contract_id=cls.RECONSTRUCTION_CONTRACT_ID,
            floating_dtype=cls._dtype_name(x_canonical.dtype),
            point_count=geometry.num_points,
            point_mass_hex=mass_value.hex(),
        )

    @classmethod
    def save(
        cls,
        *,
        path: Path,
        context: SymmetricTangentGreenResponseContext,
        identity: TangentContextIdentity,
    ) -> dict[str, Any]:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        cls._validate_identity_against_context(identity, context)
        tensors = cls._context_tensors(context)
        cls._validate_static_payload(tensors)
        payload_sha256 = cls._hash_named_tensors(sorted(tensors.items()))
        manifest = cls._manifest(
            context=context,
            identity=identity,
            tensors=tensors,
            payload_sha256=payload_sha256,
        )
        metadata = {
            "manifest_json": json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        manifest_path = path.with_suffix(".json")
        manifest_temporary = manifest_path.with_name(
            f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            save_file(tensors, str(temporary), metadata=metadata)
            cls._verify_written_sidecar(
                path=temporary,
                identity=identity,
                config=SymmetricTangentGreenResponseProjectionConfig(
                    relative_lambda=context.relative_lambda,
                    denominator_relative_eps=context.denominator_relative_eps,
                    preconditioner_variant=context.preconditioner_variant,
                    cross_axis_relative_eps=context.cross_axis_relative_eps,
                ),
            )
            manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(temporary, path)
            os.replace(manifest_temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
            manifest_temporary.unlink(missing_ok=True)
        return manifest

    @classmethod
    def load(
        cls,
        *,
        path: Path,
        identity: TangentContextIdentity,
        config: SymmetricTangentGreenResponseProjectionConfig,
        device: torch.device,
    ) -> LoadedTangentContext:
        from safetensors import safe_open

        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(  # type: ignore[no-untyped-call]
            str(path), framework="pt", device="cpu"
        ) as handle:
            metadata = handle.metadata() or {}
            manifest_raw = metadata.get("manifest_json")
            if manifest_raw is None:
                raise ValueError(
                    "Tangent context sidecar has no manifest_json metadata."
                )
            try:
                manifest = json.loads(manifest_raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Tangent context manifest_json is not valid JSON."
                ) from error
            if not isinstance(manifest, dict):
                raise ValueError("Tangent context manifest must be a JSON object.")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        cls._validate_manifest(
            manifest=manifest,
            tensors=tensors,
            identity=identity,
            config=config,
        )
        x = cls._unpack_axis("x", tensors, device)
        y = cls._unpack_axis("y", tensors, device)
        terms = cls._terms_from_tensors(tensors, config, device)
        point_mass = tensors["point_mass"].to(device=device).reshape(())
        return LoadedTangentContext(
            response_operator=FrozenBidirectionalResponseOperator(x=x, y=y),
            point_mass=point_mass,
            terms=terms,
            manifest=manifest,
        )

    @classmethod
    def validate_runtime_branches(
        cls,
        *,
        identity: TangentContextIdentity,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> None:
        x = cls._canonical_branch(x_green_branch, "x_green_branch")
        y = cls._canonical_branch(y_green_branch, "y_green_branch")
        if cls.tensor_sha256("x_green_branch", x) != identity.x_green_branch_sha256:
            raise ValueError("x Green branch changed after tangent context creation.")
        if cls.tensor_sha256("y_green_branch", y) != identity.y_green_branch_sha256:
            raise ValueError("y Green branch changed after tangent context creation.")

    @classmethod
    def tensor_sha256(cls, name: str, tensor: torch.Tensor) -> str:
        return cls._hash_named_tensors([(name, tensor)])

    @staticmethod
    def _canonical_branch(branch: torch.Tensor, name: str) -> torch.Tensor:
        if branch.dim() != 4 or branch.shape[0] < 1:
            raise ValueError(f"{name} must have shape (B, S, 4, M).")
        canonical = branch[:1]
        if branch.shape[0] > 1 and not torch.equal(
            branch,
            canonical.expand_as(branch),
        ):
            raise ValueError(
                "Tangent context persistence requires one fixed coefficient/Green "
                f"operator per run; {name} varies across samples."
            )
        return canonical

    @classmethod
    def _context_tensors(
        cls,
        context: SymmetricTangentGreenResponseContext,
    ) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        tensors.update(cls._pack_axis("x", context.response_operator.x))
        tensors.update(cls._pack_axis("y", context.response_operator.y))
        values: dict[str, torch.Tensor] = {
            "gamma_x_squared": context.gamma_x_squared,
            "gamma_y_squared": context.gamma_y_squared,
            "cross_axis_inner_product": context.cross_axis_inner_product,
            "normalized_correlation": context.normalized_correlation,
            "normalized_quadratic_cross_axis": (
                context.normalized_quadratic_cross_axis
            ),
            "separable_preconditioner_base": (context.separable_preconditioner_base),
            "exact_preconditioner_base": context.exact_preconditioner_base,
            "absolute_preconditioner_base": context.absolute_preconditioner_base,
            "quadratic_preconditioner_base": context.quadratic_preconditioner_base,
            "separable_denominator": context.separable_denominator,
            "exact_denominator": context.exact_denominator,
            "absolute_denominator": context.absolute_denominator,
            "quadratic_denominator": context.quadratic_denominator,
            "gain_scale": context.gain_scale.reshape(1),
            "q_epsilon": context.q_epsilon.reshape(1),
            "damping": context.damping.reshape(1),
            "point_mass": context.point_mass.reshape(1),
            "cauchy_violation": context.cauchy_violation,
            "cauchy_violation_max": context.cauchy_violation_max.reshape(1),
            "exact_roundoff_clamp_mask": (
                context.exact_roundoff_clamp_mask.to(torch.int8)
            ),
            "exact_roundoff_clamp_count": torch.tensor(
                [context.exact_roundoff_clamp_count], dtype=torch.int64
            ),
        }
        tensors.update(
            {key: value.detach().cpu().contiguous() for key, value in values.items()}
        )
        return tensors

    @staticmethod
    def _pack_axis(
        prefix: str,
        operator: FrozenAxialResponseOperator,
    ) -> dict[str, torch.Tensor]:
        sizes = torch.tensor(
            [block.valid_indices.numel() for block in operator.blocks],
            dtype=torch.int64,
        )
        index_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), sizes.cumsum(dim=0)))
        matrix_sizes = sizes.square()
        matrix_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64), matrix_sizes.cumsum(dim=0))
        )
        return {
            f"{prefix}_block_sizes": sizes,
            f"{prefix}_index_ptr": index_ptr,
            f"{prefix}_valid_indices": torch.cat(
                [block.valid_indices.detach().cpu() for block in operator.blocks]
            ),
            f"{prefix}_matrix_ptr": matrix_ptr,
            f"{prefix}_matrix_values": torch.cat(
                [
                    block.matrix.detach().cpu().contiguous().reshape(-1)
                    for block in operator.blocks
                ]
            ),
        }

    @classmethod
    def _unpack_axis(
        cls,
        prefix: str,
        tensors: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> FrozenAxialResponseOperator:
        sizes = tensors[f"{prefix}_block_sizes"]
        index_ptr = tensors[f"{prefix}_index_ptr"]
        indices = tensors[f"{prefix}_valid_indices"]
        matrix_ptr = tensors[f"{prefix}_matrix_ptr"]
        values = tensors[f"{prefix}_matrix_values"]
        if sizes.dim() != 1 or sizes.numel() < 1 or torch.any(sizes <= 0):
            raise ValueError(f"Malformed {prefix} tangent context block sizes.")
        expected_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64), sizes.cumsum(dim=0))
        )
        expected_matrix_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64), sizes.square().cumsum(dim=0))
        )
        if not torch.equal(index_ptr, expected_ptr) or not torch.equal(
            matrix_ptr, expected_matrix_ptr
        ):
            raise ValueError(f"Malformed {prefix} tangent context block pointers.")
        if (
            int(index_ptr[-1].item()) != indices.numel()
            or int(matrix_ptr[-1].item()) != values.numel()
        ):
            raise ValueError(f"Malformed {prefix} tangent context packed lengths.")
        blocks: list[AxialResponseBlock] = []
        for block_index, size_tensor in enumerate(sizes):
            size = int(size_tensor.item())
            i0, i1 = int(index_ptr[block_index]), int(index_ptr[block_index + 1])
            m0, m1 = int(matrix_ptr[block_index]), int(matrix_ptr[block_index + 1])
            blocks.append(
                AxialResponseBlock(
                    valid_indices=indices[i0:i1].to(device=device, dtype=torch.long),
                    matrix=values[m0:m1].reshape(size, size).to(device=device),
                )
            )
        point_count = int(indices.numel())
        if prefix not in {"x", "y"}:
            raise ValueError(f"Unsupported tangent response axis: {prefix}.")
        return FrozenAxialResponseOperator(
            axis=cast(Literal["x", "y"], prefix),
            point_count=point_count,
            blocks=tuple(blocks),
        )

    @classmethod
    def _terms_from_tensors(
        cls,
        tensors: Mapping[str, torch.Tensor],
        config: SymmetricTangentGreenResponseProjectionConfig,
        device: torch.device,
    ) -> TangentPreconditionerTerms:
        def get(key: str) -> torch.Tensor:
            return tensors[key].to(device=device)

        bases = {
            "separable": get("separable_preconditioner_base"),
            "exact_diagonal": get("exact_preconditioner_base"),
            "absolute_cross_axis": get("absolute_preconditioner_base"),
            "normalized_quadratic_cross_axis": get("quadratic_preconditioner_base"),
        }
        denominators = {
            "separable": get("separable_denominator"),
            "exact_diagonal": get("exact_denominator"),
            "absolute_cross_axis": get("absolute_denominator"),
            "normalized_quadratic_cross_axis": get("quadratic_denominator"),
        }
        variant = config.preconditioner_variant
        return TangentPreconditionerTerms(
            variant=variant,
            a=get("gamma_x_squared"),
            b=get("gamma_y_squared"),
            c=get("cross_axis_inner_product"),
            rho=get("normalized_correlation"),
            q=get("normalized_quadratic_cross_axis"),
            separable_base=bases["separable"],
            exact_base=bases["exact_diagonal"],
            absolute_base=bases["absolute_cross_axis"],
            quadratic_base=bases["normalized_quadratic_cross_axis"],
            selected_base=bases[variant],
            gain_scale=get("gain_scale").reshape(()),
            q_epsilon=get("q_epsilon").reshape(()),
            damping=get("damping").reshape(()),
            separable_denominator=denominators["separable"],
            exact_denominator=denominators["exact_diagonal"],
            absolute_denominator=denominators["absolute_cross_axis"],
            quadratic_denominator=denominators["normalized_quadratic_cross_axis"],
            denominator=denominators[variant],
            cauchy_violation=get("cauchy_violation"),
            cauchy_violation_max=get("cauchy_violation_max").reshape(()),
            exact_roundoff_clamp_mask=get("exact_roundoff_clamp_mask").bool(),
            exact_roundoff_clamp_count=int(
                tensors["exact_roundoff_clamp_count"].item()
            ),
        )

    @classmethod
    def _manifest(
        cls,
        *,
        context: SymmetricTangentGreenResponseContext,
        identity: TangentContextIdentity,
        tensors: Mapping[str, torch.Tensor],
        payload_sha256: str,
    ) -> dict[str, Any]:
        static_config = {
            "relative_lambda": context.relative_lambda,
            "denominator_relative_eps": context.denominator_relative_eps,
            "cross_axis_relative_eps": context.cross_axis_relative_eps,
        }
        identity_payload = {
            "identity": identity.as_dict(),
            "formula_suite_id": cls.FORMULA_SUITE_ID,
            **static_config,
        }
        context_id = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    {
                        **identity_payload,
                        "tensor_payload_sha256": payload_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        return {
            "format_name": cls.FORMAT_NAME,
            "schema_version": cls.SCHEMA_VERSION,
            "formula_suite_id": cls.FORMULA_SUITE_ID,
            "context_id": context_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "point_count": context.num_points,
            "x_block_count": len(context.response_operator.x.blocks),
            "y_block_count": len(context.response_operator.y.blocks),
            "floating_dtype": cls._dtype_name(context.denominator.dtype),
            "stored_device": "cpu",
            "operator_formula": "H_s=K_s*W_s*L_s^2",
            "gradient_formula": "g=(H_x+H_y)^T*M_Omega*m",
            "created_with_preconditioner_variant": context.preconditioner_variant,
            "identity": identity.as_dict(),
            "tensor_payload_sha256": payload_sha256,
            **static_config,
        }

    @classmethod
    def _validate_manifest(
        cls,
        *,
        manifest: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
        identity: TangentContextIdentity,
        config: SymmetricTangentGreenResponseProjectionConfig,
    ) -> None:
        if manifest.get("format_name") != cls.FORMAT_NAME:
            raise ValueError("Unsupported tangent context format_name.")
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported tangent context schema_version.")
        if manifest.get("formula_suite_id") != cls.FORMULA_SUITE_ID:
            raise ValueError("Unsupported tangent context formula suite.")
        if manifest.get("identity") != identity.as_dict():
            raise ValueError("Tangent context compatibility identity mismatch.")
        if manifest.get("floating_dtype") != identity.floating_dtype:
            raise ValueError("Tangent context floating dtype identity mismatch.")
        if manifest.get("point_count") != identity.point_count:
            raise ValueError("Tangent context point count identity mismatch.")
        for key, expected in (
            ("relative_lambda", config.relative_lambda),
            ("denominator_relative_eps", config.denominator_relative_eps),
            ("cross_axis_relative_eps", config.cross_axis_relative_eps),
        ):
            if manifest.get(key) != float(expected):
                raise ValueError(f"Tangent context {key} mismatch.")
        required = set(cls._PRECONDITIONER_KEYS)
        for prefix in ("x", "y"):
            required.update(
                {
                    f"{prefix}_block_sizes",
                    f"{prefix}_index_ptr",
                    f"{prefix}_valid_indices",
                    f"{prefix}_matrix_ptr",
                    f"{prefix}_matrix_values",
                }
            )
        missing = sorted(required - set(tensors))
        unknown = sorted(set(tensors) - required)
        if missing or unknown:
            raise ValueError(
                "Tangent context tensor schema mismatch: "
                f"missing={missing}, unknown={unknown}."
            )
        actual_payload = cls._hash_named_tensors(sorted(tensors.items()))
        if manifest.get("tensor_payload_sha256") != actual_payload:
            raise ValueError("Tangent context tensor payload digest mismatch.")
        expected_context_id = cls._context_id(
            identity=identity,
            payload_sha256=actual_payload,
            relative_lambda=float(config.relative_lambda),
            denominator_relative_eps=float(config.denominator_relative_eps),
            cross_axis_relative_eps=float(config.cross_axis_relative_eps),
        )
        if manifest.get("context_id") != expected_context_id:
            raise ValueError("Tangent context ID mismatch.")
        floating_dtype = manifest.get("floating_dtype")
        for key, tensor in tensors.items():
            if tensor.dtype.is_floating_point and cls._dtype_name(tensor.dtype) != (
                floating_dtype
            ):
                raise ValueError(f"Tangent context tensor {key} has wrong dtype.")
        for prefix in ("x", "y"):
            for suffix in (
                "block_sizes",
                "index_ptr",
                "valid_indices",
                "matrix_ptr",
            ):
                key = f"{prefix}_{suffix}"
                if tensors[key].dtype != torch.int64:
                    raise ValueError(f"Tangent context tensor {key} must use int64.")
        cls._validate_static_payload(tensors)

    @classmethod
    def _validate_identity_against_context(
        cls,
        identity: TangentContextIdentity,
        context: SymmetricTangentGreenResponseContext,
    ) -> None:
        if identity.point_count != context.num_points:
            raise ValueError("Tangent context identity point count mismatch.")
        if identity.floating_dtype != cls._dtype_name(context.denominator.dtype):
            raise ValueError("Tangent context identity floating dtype mismatch.")
        if identity.reconstruction_contract_id != cls.RECONSTRUCTION_CONTRACT_ID:
            raise ValueError("Tangent context reconstruction contract mismatch.")
        point_mass_hex = float(context.point_mass.detach().cpu().item()).hex()
        if identity.point_mass_hex != point_mass_hex:
            raise ValueError("Tangent context identity point mass mismatch.")

    @classmethod
    def _validate_static_payload(
        cls,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        floating = [
            tensor for tensor in tensors.values() if tensor.dtype.is_floating_point
        ]
        if any(not torch.all(torch.isfinite(tensor)) for tensor in floating):
            raise ValueError("Tangent context payload contains non-finite values.")
        a = tensors["gamma_x_squared"]
        b = tensors["gamma_y_squared"]
        c = tensors["cross_axis_inner_product"]
        separable = a + b
        gain_scale = tensors["gain_scale"].reshape(())
        damping = tensors["damping"].reshape(())
        q_epsilon = tensors["q_epsilon"].reshape(())
        expected = {
            "separable_preconditioner_base": separable,
            "exact_preconditioner_base": torch.clamp_min(separable + 2.0 * c, 0.0),
            "absolute_preconditioner_base": separable + 2.0 * c.abs(),
            "quadratic_preconditioner_base": (
                separable + 4.0 * c.square() / (separable + q_epsilon)
            ),
        }
        expected_q = c.square() / (separable + q_epsilon)
        cls._validate_floating_invariant(
            name="normalized_quadratic_cross_axis",
            actual=tensors["normalized_quadratic_cross_axis"],
            expected=expected_q,
            reference_scale=gain_scale,
        )
        cls._validate_floating_invariant(
            name="gain scale",
            actual=gain_scale,
            expected=separable.mean(),
            reference_scale=gain_scale,
        )
        for key, expected_base in expected.items():
            cls._validate_floating_invariant(
                name=key,
                actual=tensors[key],
                expected=expected_base,
                reference_scale=gain_scale,
            )
        for base_key, denominator_key in (
            ("separable_preconditioner_base", "separable_denominator"),
            ("exact_preconditioner_base", "exact_denominator"),
            ("absolute_preconditioner_base", "absolute_denominator"),
            ("quadratic_preconditioner_base", "quadratic_denominator"),
        ):
            cls._validate_floating_invariant(
                name=denominator_key,
                actual=tensors[denominator_key],
                expected=tensors[base_key] + damping,
                reference_scale=gain_scale,
            )

    @classmethod
    def _validate_floating_invariant(
        cls,
        *,
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        reference_scale: torch.Tensor,
    ) -> None:
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                f"Tangent context {name} invariant shape/dtype mismatch: "
                f"actual_shape={tuple(actual.shape)}, "
                f"expected_shape={tuple(expected.shape)}, "
                f"actual_dtype={actual.dtype}, expected_dtype={expected.dtype}."
            )
        epsilon = torch.finfo(expected.dtype).eps
        rtol = cls._FLOATING_INVARIANT_EPS_FACTOR * epsilon
        scale = max(
            float(reference_scale.detach().abs().max().item()),
            torch.finfo(expected.dtype).tiny,
        )
        atol = rtol * scale
        if torch.allclose(actual, expected, rtol=rtol, atol=atol):
            return

        absolute_error = (actual - expected).abs()
        relative_floor = torch.as_tensor(
            scale * epsilon,
            dtype=expected.dtype,
            device=expected.device,
        )
        relative_error = absolute_error / expected.abs().clamp_min(relative_floor)
        raise ValueError(
            f"Tangent context {name} invariant failed: "
            f"max_abs_error={float(absolute_error.max().item()):.6e}, "
            f"max_rel_error={float(relative_error.max().item()):.6e}, "
            f"rtol={rtol:.6e}, atol={atol:.6e}."
        )

    @classmethod
    def _context_id(
        cls,
        *,
        identity: TangentContextIdentity,
        payload_sha256: str,
        relative_lambda: float,
        denominator_relative_eps: float,
        cross_axis_relative_eps: float,
    ) -> str:
        payload = {
            "identity": identity.as_dict(),
            "formula_suite_id": cls.FORMULA_SUITE_ID,
            "relative_lambda": relative_lambda,
            "denominator_relative_eps": denominator_relative_eps,
            "cross_axis_relative_eps": cross_axis_relative_eps,
            "tensor_payload_sha256": payload_sha256,
        }
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )

    @classmethod
    def _verify_written_sidecar(
        cls,
        *,
        path: Path,
        identity: TangentContextIdentity,
        config: SymmetricTangentGreenResponseProjectionConfig,
    ) -> None:
        from safetensors import safe_open

        with safe_open(  # type: ignore[no-untyped-call]
            str(path), framework="pt", device="cpu"
        ) as handle:
            metadata = handle.metadata() or {}
            manifest_raw = metadata.get("manifest_json")
            if manifest_raw is None:
                raise ValueError("Written tangent context has no manifest_json.")
            manifest = json.loads(manifest_raw)
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        cls._validate_manifest(
            manifest=manifest,
            tensors=tensors,
            identity=identity,
            config=config,
        )

    @classmethod
    def _hash_named_tensors(
        cls,
        items: list[tuple[str, torch.Tensor]],
    ) -> str:
        digest = hashlib.sha256()
        for name, tensor in items:
            value = tensor.detach().cpu().contiguous()
            header = json.dumps(
                {
                    "name": name,
                    "dtype": cls._dtype_name(value.dtype),
                    "shape": list(value.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).replace("torch.", "")
