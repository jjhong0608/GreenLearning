"""Compare frozen training K4/K5 and target-K checkpoints at a common K."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.logging import RichHandler

from audit_equal_split_initialization import InitializationAudit, base, digest


class LearnedAudit(InitializationAudit):
    initializations = ("learned",)


def validate_native(
    actual: pd.DataFrame, reference: pd.DataFrame, k: int, target_k: int = 10
) -> dict:
    selected = actual[actual.K == k].set_index("sample_id").sort_index()
    saved = reference.set_index("sample_id").sort_index()
    assert selected.index.equals(saved.index) and selected.index.is_unique
    differences = {}
    for metric, original in (
        ("rel_sol", "rel_sol"),
        ("rel_equal", "rel_sol_equal_mean"),
        ("energy", "loss_energy_optimized"),
        ("response_cost", f"tangent_response_cost_k{k}"),
    ):
        np.testing.assert_allclose(
            selected[metric], saved[original], rtol=1e-7, atol=1e-12
        )
        differences[metric] = float(abs(selected[metric] - saved[original]).max())
    for _, group in actual.groupby("sample_id"):
        assert group.sort_values("K").K.tolist() == list(range(target_k + 1))
        costs = group.sort_values("K").response_cost.to_numpy()
        assert np.all(np.diff(costs) <= 1e-10 * costs[0] + 1e-24)
    return differences


def summarize(out: Path, target_k: int = 10) -> None:
    rows = pd.read_csv(out / "per_sample.csv")
    assert not rows.duplicated(["seed", "training_k", "K", "sample_id"]).any()
    assert len(rows) == 4 * 3 * (target_k + 1) * 100
    metrics = [
        "rel_sol",
        "rel_equal",
        "rel_u_phi",
        "rel_u_psi",
        "energy",
        "response_cost",
        "correction_l2",
    ]
    records = []
    for (seed, train_k, eval_k), group in rows.groupby(["seed", "training_k", "K"]):
        assert len(group) == 100 and set(group.sample_id) == set(range(100))
        record = dict(seed=seed, training_k=train_k, evaluation_k=eval_k)
        for metric in metrics:
            values = group[metric]
            assert np.isfinite(values).all()
            record.update(
                {
                    f"{metric}_mean": values.mean(),
                    f"{metric}_median": values.median(),
                    f"{metric}_p95": values.quantile(0.95),
                    f"{metric}_max": values.max(),
                }
            )
        records.append(record)
    per_seed = pd.DataFrame(records)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    per_seed.groupby(["training_k", "evaluation_k"]).agg(
        {
            column: ["mean", "std"]
            for column in per_seed
            if column not in ("seed", "training_k", "evaluation_k")
        }
    ).to_csv(out / "summary.csv")
    final = rows[rows.K == target_k]
    pairs = []
    for seed, group in final.groupby("seed"):
        reference = group[group.training_k == target_k].set_index("sample_id")
        for train_k in (4, 5):
            candidate = group[group.training_k == train_k].set_index("sample_id")
            difference = candidate.rel_sol - reference.rel_sol
            pairs.append(
                dict(
                    seed=seed,
                    training_k=train_k,
                    evaluation_k=target_k,
                    mean_difference=difference.mean(),
                    better_sample_count=int((difference < 0).sum()),
                    worse_sample_count=int((difference > 0).sum()),
                )
            )
    pd.DataFrame(pairs).to_csv(out / "paired_comparison.csv", index=False)
    configurations = []
    for seed in range(4):
        paired = []
        for k in (4, 5, target_k):
            path = next(
                Path("checkpoints/numerical_examples/pentagram").glob(
                    f"*/seed{seed}/pentagram_k{k}_seed{seed}/config_used.json"
                )
            )
            config = json.loads(path.read_text())
            configurations.append(config)
            normalized = json.loads(json.dumps(config))
            tangent = normalized["coupling_model"]["balance_projection"][
                "symmetric_tangent_green_response"
            ]
            tangent.pop("subspace_dimension")
            tangent.pop("max_subspace_dimension")
            normalized.pop("tangent_subspace_dimension_provenance")
            paired.append(normalized)
        assert paired[0] == paired[1] == paired[2]
    dataset = configurations[0]["dataset"]
    source_paths = [
        Path(dataset["geometry_path"]),
        Path(dataset["coefficient_functions_path"]),
    ]
    source_paths += sorted(Path(dataset["test_path"]).glob("*.npz"))
    assert len(source_paths) == 102
    checks = json.loads((out / "validation.json").read_text())
    assert len(checks) == 12
    for item in checks:
        assert item["protected_sha256"] == {
            p: digest(Path(p)) for p in item["protected_sha256"]
        }
    provenance = dict(
        device="cuda:1",
        dtype="float64",
        torch_version=torch.__version__,
        training=False,
        row_count=len(rows),
        checkpoint_count=len(checks),
        samples_per_checkpoint=100,
        paired_config_difference="K fields only",
        evaluation=f"Same balanced network proposal, unrestarted production subspace K0..{target_k}",
        input_sha256={str(p): digest(p) for p in source_paths},
        code_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("cli/audit_equal_split_initialization.py"),
                Path("src/greenonet/complex_tangent_subspace_audit.py"),
                Path("src/greenonet/complex_tangent_projection.py"),
            ]
        },
        native_max_abs={
            metric: max(item["native_max_abs"][metric] for item in checks)
            for metric in checks[0]["native_max_abs"]
        },
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-k", type=int, choices=(9, 10), default=10)
    args = parser.parse_args()
    root = Path("checkpoints/numerical_examples/pentagram")
    out = Path("docs/analysis/pentagram_training_k_transfer")
    if args.target_k != 10:
        out = out.with_name(f"{out.name}_k{args.target_k}")
    out.mkdir(parents=True, exist_ok=args.resume)
    logger = logging.getLogger("TrainingKTransfer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (RichHandler(), logging.FileHandler(out / "run.log")):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    torch.set_num_threads(4)
    checks = json.loads((out / "validation.json").read_text()) if args.resume else []
    collected = [pd.read_csv(out / "per_sample.csv")] if args.resume else []
    completed = {(item["seed"], item["training_k"]) for item in checks}
    for item in checks:
        assert item["protected_sha256"] == {
            path: digest(Path(path)) for path in item["protected_sha256"]
        }
    for seed in range(4):
        for k in (4, 5, args.target_k):
            if (seed, k) in completed:
                continue
            run = next(root.glob(f"*/seed{seed}/pentagram_k{k}_seed{seed}"))
            target = out / f"train_k{k}_seed{seed}"
            target.mkdir(exist_ok=args.resume)
            original = run / "config_used.json"
            config = json.loads(original.read_text())
            checkpoint = run / "complex_coupling_model_best_energy.safetensors"
            context = run / "tangent_response_context.safetensors"
            green = Path(config["pipeline"]["green_pretrained_path"])
            artifact = run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
            protected = [original, checkpoint, context, green, artifact]
            hashes = {str(p): digest(p) for p in protected}
            config["coupling_training"]["compile"] = {"enabled": False}
            config["coupling_training"]["tangent_context_checkpoint"].update(
                enabled=True,
                load_policy="required" if seed in (0, 2) else "never",
                path=str(context.resolve())
                if seed in (0, 2)
                else str(target / "unused_context.safetensors"),
                save_after_build=False,
            )
            effective = target / "evaluation_config.json"
            effective.write_text(json.dumps(config, indent=2))
            request = base.TangentSubspaceAuditRequest(
                config=effective,
                coupling_checkpoint=checkpoint,
                green_checkpoint=green,
                tangent_context=context if seed in (0, 2) else None,
                outdir=target,
                device="cuda:1",
                batch_size=5,
                max_subspace_dimension=args.target_k,
            )
            audit = LearnedAudit(request, logger=logger)
            audit.execute(timing_repeats=1)
            rows = pd.read_csv(target / "per_sample.csv")
            differences = validate_native(rows, pd.read_csv(artifact), k, args.target_k)
            assert hashes == {str(p): digest(p) for p in protected}
            checks.append(
                dict(
                    seed=seed,
                    training_k=k,
                    native_max_abs=differences,
                    protected_sha256=hashes,
                    normalization=audit.tangent_context.direction_normalization,
                    context_source="saved" if seed in (0, 2) else "rebuilt_on_gpu",
                )
            )
            rows["training_k"] = k
            rows["seed"] = seed
            collected.append(rows)
            pd.concat(collected).to_csv(out / "per_sample.csv", index=False)
            (out / "validation.json").write_text(json.dumps(checks, indent=2))
            logger.info("Validated training K%d seed%d", k, seed)
            del audit
            gc.collect()
            torch.cuda.empty_cache()
    summarize(out, args.target_k)


if __name__ == "__main__":
    main()
