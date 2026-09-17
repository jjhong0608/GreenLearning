"""Check the Poisson weak-defect contribution from source quadrature mismatch."""

import argparse
from pathlib import Path

import numpy as np
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_frozen_tangent_csv import _json
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_weak_closure import (
    assemble_directional_weak_residuals,
    build_directional_weak_context,
)


def main(root: Path) -> None:
    torch.set_num_threads(4)
    geometry = load_complex_geometry(Path("data/geometry/unit_square_h_1_128.npz"))
    context = build_directional_weak_context(
        geometry,
        load_coefficient_functions(
            Path("numerical_examples/unit_square/coefficients.py")
        ),
    )
    maximum = 0.0
    relative = []
    for bi in range(4):
        name = f"unit_square_seed0_batch{bi:03d}.npz"
        with np.load(
            Path("docs/analysis/reference_green_reoptimization/sources") / name
        ) as data:
            pair = torch.from_numpy(data["reference_optimized_k64"])
            ids = data["sample_ids"].copy()
        load = assemble_directional_weak_residuals(
            u_valid=torch.zeros_like(pair[:, 0]),
            projected_physical=pair,
            context=context,
        )
        with np.load(root / "raw" / name) as data:
            np.testing.assert_array_equal(ids, data["sample_ids"])
            for channel, key, negative_mass in (
                (0, "phi_x", load.x),
                (1, "psi_y", load.y),
            ):
                actual = torch.from_numpy(data[f"RR_k64_{key}"])
                predicted = context.point_area * pair[:, channel] + negative_mass
                difference = actual - predicted
                maximum = max(maximum, float(difference.abs().max()))
                relative.extend((difference.norm(dim=1) / actual.norm(dim=1)).tolist())
    if max(relative) > 1e-7:
        raise RuntimeError("Load-identity check did not reproduce the defect")
    _json(
        root / "poisson_load_identity.json",
        dict(
            run_id="unit_square_seed0",
            k=64,
            samples=100,
            directions=2,
            formula="r_axis = (hx*hy I - M_axis) source_axis",
            max_abs_difference=maximum,
            max_relative_difference=max(relative),
            mean_relative_difference=float(np.mean(relative)),
            interpretation="Reference own-axis defect reproduced by nodal quadrature versus consistent P1 load mismatch",
        ),
    )
    print(maximum, max(relative))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=Path("docs/analysis/weak_green_residual_audit")
    )
    main(parser.parse_args().outdir)
