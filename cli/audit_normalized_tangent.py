"""Isolated normalized-response prototype; never a production projection mode."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import torch
from rich.logging import RichHandler

from cli.audit_uniform_tangent import UniformTangentAudit
from greenonet.complex_tangent_projection import KrylovSubspaceStepResult
from greenonet.complex_tangent_subspace_audit import TangentSubspaceAuditRequest


def normalized_step(
    *,
    context,
    mismatch,
    gradient,
    max_dimension,
    relative_eps,
    monotonicity_relative_tol=1e-10,
    direction_scale=1.0,
):
    """Two-pass paired MGS; relative_eps is a squared independence threshold.

    A source max-norm prescale prevents arbitrary direction magnitudes from
    underflowing response norms. Both rescalings preserve the source/response
    relation. Safe inactive denominators are one, including during backward.
    """
    if not 0 < relative_eps < 1 or max_dimension < 1:
        raise ValueError("Invalid independence threshold or dimension")
    mass = context.point_mass
    residual, grad = mismatch, gradient
    delta = torch.zeros_like(gradient)
    directions, pairs, responses, coefficients, activities = [], [], [], [], []
    deltas, residuals, costs = [], [], []
    initial_cost = mass * mismatch.square().sum(-1)
    for _ in range(max_dimension):
        source = direction_scale * grad / context.denominator
        source_scale = source.abs().amax(-1)
        valid = source_scale > 0
        safe = torch.where(valid, source_scale, torch.ones_like(source_scale))
        source = source / safe[:, None]
        pair = context.response_operator.forward_pair(torch.stack((source, source), 1))
        response = pair.sum(1)
        energy = mass * response.square().sum(-1)
        valid = valid & (energy > torch.finfo(energy.dtype).tiny)
        norm = torch.sqrt(torch.where(valid, energy, torch.ones_like(energy)))
        source, pair, response = (
            source / norm[:, None],
            pair / norm[:, None, None],
            response / norm[:, None],
        )
        before_energy = mass * response.square().sum(-1)
        for _pass in range(2):
            for old_source, old_pair, old_response in zip(
                directions, pairs, responses, strict=True
            ):
                cross = mass * (response * old_response).sum(-1)
                source = source - cross[:, None] * old_source
                pair = pair - cross[:, None, None] * old_pair
                response = response - cross[:, None] * old_response
        after_energy = mass * response.square().sum(-1)
        active = valid & (after_energy > relative_eps * before_energy)
        norm = torch.sqrt(
            torch.where(active, after_energy, torch.ones_like(after_energy))
        )
        source = torch.where(active[:, None], source / norm[:, None], 0.0)
        pair = torch.where(active[:, None, None], pair / norm[:, None, None], 0.0)
        response = torch.where(active[:, None], response / norm[:, None], 0.0)
        numerator = mass * (residual * response).sum(-1)
        response_energy = mass * response.square().sum(-1)
        denominator = torch.where(active, response_energy * (1 + relative_eps), 1.0)
        coefficient = numerator / denominator
        delta = delta - coefficient[:, None] * source
        residual = residual - coefficient[:, None] * response
        cost = mass * residual.square().sum(-1)
        previous = costs[-1] if costs else initial_cost
        if torch.any(
            cost
            > previous
            + monotonicity_relative_tol * initial_cost
            + torch.finfo(cost.dtype).tiny
        ):
            raise RuntimeError("Normalized response cost increased")
        grad = context.tangent_gradient(residual)
        for value in (source, pair, response, coefficient, delta, residual, grad):
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite normalized correction")
        directions.append(source)
        pairs.append(pair)
        responses.append(response)
        coefficients.append(coefficient)
        activities.append(active)
        deltas.append(delta)
        residuals.append(residual)
        costs.append(cost)
    response_stack = torch.stack(responses)
    gram = mass * torch.einsum("kbp,lbp->bkl", response_stack, response_stack)
    off = gram.abs().masked_fill(
        torch.eye(max_dimension, dtype=torch.bool, device=gram.device)[None], 0
    )
    return KrylovSubspaceStepResult(
        directions=torch.stack(directions),
        directional_responses=torch.stack(pairs),
        response_directions=response_stack,
        coefficients=torch.stack(coefficients),
        direction_active=torch.stack(activities),
        deltas=torch.stack(deltas),
        mismatches=torch.stack(residuals),
        costs=torch.stack(costs),
        residual_gradient_post=grad,
        response_gram=gram,
        response_orthogonality_max=torch.stack(
            [off[:, :k, :k].amax((1, 2)) for k in range(1, max_dimension + 1)]
        ),
        line_search_numerator_0=mass * (mismatch * responses[0]).sum(-1),
        line_search_denominator_0=torch.where(
            activities[0], gram[:, 0, 0] * (1 + relative_eps), 1.0
        ),
    )


class NormalizedAudit(UniformTangentAudit):
    include_identity = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prototype_checks = []

    def _evaluate_prepared_batch(self, batch, prepared, *, context):
        # Process-local replacement of the audit wrapper, not production core.
        with patch(
            "greenonet.complex_tangent_subspace_audit.matrix_free_krylov_subspace_audit",
            normalized_step,
        ):
            result = super()._evaluate_prepared_batch(batch, prepared, context=context)
        evaluation, subspace = result
        pair = context.response_operator.forward_pair(
            torch.stack((subspace.deltas[-1], subspace.deltas[-1]), 1)
        )
        torch.testing.assert_close(
            prepared.mismatch + pair.sum(1),
            subspace.mismatches[-1],
            rtol=1e-9,
            atol=1e-12,
        )
        relation_error = float(
            (prepared.mismatch + pair.sum(1) - subspace.mismatches[-1]).abs().max()
        )
        with torch.enable_grad():
            source = prepared.symmetric_physical.detach().clone().requires_grad_(True)
            solution = context.response_operator.forward_pair(source)
            mismatch = solution[:, 0] - solution[:, 1]
            check = normalized_step(
                context=context,
                mismatch=mismatch,
                gradient=context.tangent_gradient(mismatch),
                max_dimension=2,
                relative_eps=self.request.subspace_relative_eps,
            )
            objective = check.costs[-1].sum() + check.deltas[-1].square().mean()
            derivative = torch.autograd.grad(objective, source)[0]
            if not torch.isfinite(derivative).all():
                raise RuntimeError("Nonfinite frozen-proposal backward")
        self.prototype_checks.append(
            {
                "sample_ids": batch.sample_indices.tolist(),
                "denominator_min": float(context.denominator.min()),
                "denominator_max": float(context.denominator.max()),
                "source_response_relation_max_abs": relation_error,
                "backward_max_abs": float(derivative.abs().max()),
                "response_orthogonality_max": float(
                    subspace.response_orthogonality_max.max()
                ),
                "all_backward_finite": True,
            }
        )
        return evaluation, subspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "coupling-checkpoint", "green-checkpoint", "outdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("normalized_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        RichHandler(show_path=False),
        logging.FileHandler(args.outdir / "audit.log"),
    ):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    import hashlib
    import json

    audit = NormalizedAudit(
        TangentSubspaceAuditRequest(
            config=args.config,
            coupling_checkpoint=args.coupling_checkpoint,
            green_checkpoint=args.green_checkpoint,
            outdir=args.outdir,
            device="cpu",
            batch_size=5,
            max_subspace_dimension=4,
            subspace_relative_eps=1e-12,
        ),
        logger=logger,
    )
    result = audit.run()
    result["algorithm"] = "audit-only paired response normalization + two-pass MGS"
    result["independence_squared_ratio_threshold"] = 1e-12
    result["prototype_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["exclude_methods"] = ["k1_production"]
    result["coefficient_basis"] = (
        "normalized response basis; not legacy raw-direction eta"
    )
    result["prototype_checks"] = audit.prototype_checks
    (args.outdir / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
