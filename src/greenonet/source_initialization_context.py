"""Explicit, verified runtime rebuilding for cross-device branch identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from greenonet.complex_coupling_data import ComplexCouplingBatch
from greenonet.complex_tangent_context_io import (
    TangentContextIdentity,
    TangentResponseContextStore,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContextCache,
)
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig


def verified_runtime_cache(
    path: Path,
    model: torch.nn.Module,
    batch: ComplexCouplingBatch,
    config: SymmetricTangentGreenResponseProjectionConfig,
) -> tuple[SymmetricTangentGreenResponseContextCache | None, dict[str, Any]]:
    actual = TangentResponseContextStore.identity(
        green_model=model,
        geometry=batch.geometry,
        x_green_branch=batch.x_green_branch,
        y_green_branch=batch.y_green_branch,
        point_mass=batch.geometry.hx * batch.geometry.hy,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Paper audit requires the archived context: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
        manifest = json.loads((handle.metadata() or {})["manifest_json"])
    stored = TangentContextIdentity(**manifest["identity"])
    differences = {
        key: dict(runtime=value, archived=stored.as_dict()[key])
        for key, value in actual.as_dict().items()
        if value != stored.as_dict()[key]
    }
    info: dict[str, Any] = dict(
        runtime_identity=actual.as_dict(),
        archived_identity=stored.as_dict(),
        identity_differences=differences,
        rebuilt=False,
    )
    if not differences:
        return None, info
    if set(differences) - {"x_green_branch_sha256", "y_green_branch_sha256"}:
        raise ValueError(
            f"Non-branch context identity mismatch for {path}: {differences}"
        )
    # Validate the archived payload under its own identity. This does NOT assert
    # that its branch inputs equal the runtime inputs; compare operators below.
    archived = TangentResponseContextStore.load(
        path=path,
        identity=stored,
        config=config,
        device=batch.rhs_valid.device,
    )
    cache = SymmetricTangentGreenResponseContextCache(config)
    fresh = cache.get_or_build(
        green_model=model,
        geometry=batch.geometry,
        x_green_branch=batch.x_green_branch,
        y_green_branch=batch.y_green_branch,
    )
    largest = 0.0
    for axis in ("x", "y"):
        old_blocks = getattr(archived.response_operator, axis).blocks
        new_blocks = getattr(fresh.response_operator, axis).blocks
        for old, new in zip(old_blocks, new_blocks, strict=True):
            torch.testing.assert_close(
                old.valid_indices, new.valid_indices, rtol=0, atol=0
            )
            torch.testing.assert_close(old.matrix, new.matrix, rtol=1e-10, atol=1e-14)
            largest = max(largest, float((old.matrix - new.matrix).abs().max()))
    torch.testing.assert_close(
        fresh.gamma_x_squared, archived.terms.a, rtol=1e-10, atol=1e-14
    )
    torch.testing.assert_close(
        fresh.gamma_y_squared, archived.terms.b, rtol=1e-10, atol=1e-14
    )
    torch.testing.assert_close(
        fresh.denominator,
        archived.terms.denominator_for(config.preconditioner_variant),
        rtol=1e-10,
        atol=1e-14,
    )
    info.update(
        rebuilt=True,
        archived_payload_validated=True,
        operator_max_abs_difference=largest,
        operator_rtol=1e-10,
        operator_atol=1e-14,
        original_sidecar_modified=False,
        reason="Branch identity differs; runtime operator rebuilt and compared, not identity check relaxed.",
    )
    return cache, info
