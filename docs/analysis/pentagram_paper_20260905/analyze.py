"""Read frozen paper runs and supplement their directional test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_tangent_geometry_selection import (
    AxialSegmentTopologyAnalyzer,
    global_reach_fraction,
    pointwise_reach_fraction,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "checkpoints/numerical_examples/pentagram"
KS = (0, 1, 2, 3, 4, 5, 9, 10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def log_timing(path: Path) -> dict:
    """Rich dates are inherited by continuation lines; no filesystem mtimes."""
    events = {}
    current = None
    first = None
    last = None
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(
            r"\[(\d{4})년 (\d{2})월 (\d{2})일 (\d{2})시 (\d{2})분 (\d{2})초\]",
            line,
        )
        if match:
            current = datetime(*map(int, match.groups()))
        else:
            match = re.search(r"\[(\d{2}/\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})\]", line)
            if match:
                current = datetime.strptime(
                    " ".join(match.groups()), "%m/%d/%y %H:%M:%S"
                )
        if current is not None:
            first = first or current
            last = current
        match = re.search(r"epoch (\d+) train", line)
        if match and current is not None:
            events[int(match.group(1))] = current
    values = [
        (events[e] - events[e - 1]).total_seconds()
        for e in range(11, 101)
        if e in events and e - 1 in events
    ]
    return {
        "timestamped_train_epochs": len(events),
        "steady_epoch_seconds": float(np.median(values)) if values else None,
        "trainer_wall_hours": (events[100] - first).total_seconds() / 3600
        if 100 in events and first
        else None,
        "log_span_hours_including_export": (last - first).total_seconds() / 3600
        if first and last
        else None,
        "log_first_timestamp": str(first),
        "log_last_timestamp": str(last),
    }


class PaperComparison:
    def __init__(self) -> None:
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        (OUT / "figures").mkdir(exist_ok=True)
        self.logger = logging.getLogger("PentagramPaperComparison")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        for handler in (
            RichHandler(rich_tracebacks=True, omit_repeated_times=False),
            logging.FileHandler(OUT / "analysis.log"),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.runs = [
            (hw, seed, k, RUN_ROOT / hw / f"seed{seed}/pentagram_k{k}_seed{seed}")
            for hw, seeds in (("nvidia_a40", (0, 2)), ("mac_studio", (1, 3)))
            for seed in seeds
            for k in KS
        ]

    def collect(self) -> None:
        records, samples, curves, stages, manifests, identities = [], [], [], [], [], []
        configs = {}
        for hw, seed, k, run in self.runs:
            artifact = run / "artifacts_best_energy"
            summary = read_json(artifact / "summary.json")
            cfg = read_json(run / "config_used.json")
            configs[f"s{seed}_k{k}"] = flatten(cfg)
            df = pd.read_csv(artifact / "metrics/per_sample_metrics.csv").sort_values(
                "sample_id"
            )
            train = pd.read_csv(run / "complex_training_metrics.csv")
            val = train.loc[train.split == "val"]
            tr = train.loc[train.split == "train"]
            assert len(df) == 100 and df.sample_id.tolist() == list(range(100))
            assert len(val) == len(tr) == 100
            assert val.global_step.tolist() == list(range(24, 2401, 24))
            assert np.allclose(df.loss, df.loss_energy_bulk, rtol=1e-14, atol=0)
            for key in ("loss", "rel_sol", "rel_flux"):
                np.testing.assert_allclose(
                    df[key].mean(),
                    summary["aggregate_metrics"][key + "_mean"],
                    rtol=1e-12,
                )
            assert summary["coupling_checkpoint"].endswith(
                "complex_coupling_model_best_energy.safetensors"
            )
            assert summary["reference_targets_used_for_training"] is False
            best = val.loc[val.loss_energy_optimized.idxmin()]
            log = (run / "training.log").read_text()
            rec = {
                "hardware": hw,
                "seed": seed,
                "K": k,
                "run_path": str(run.relative_to(ROOT)),
                "best_epoch": int(best.epoch),
                "best_step": int(best.global_step),
                "best_val_energy": best.loss_energy_optimized,
                "final_val_energy": val.iloc[-1].loss_energy_optimized,
                "final_train_energy": tr.iloc[-1].loss_energy_optimized,
                "val_end_over_best": val.iloc[-1].loss / best.loss,
                "val_last10_over_previous10": val.tail(10).loss.mean()
                / val.iloc[-20:-10].loss.mean(),
                "train_last10_over_previous10": tr.tail(10).loss.mean()
                / tr.iloc[-20:-10].loss.mean(),
                "optimizer_ms": tr.optimizer_step_time_mean_ms.mean(),
                "peak_device_memory_mib": tr.optimizer_peak_memory_mib.max(),
                "success_marker": (run / "_SUCCESS").exists(),
                "parameter_count": int(
                    re.search(r"trainable_parameter_count=(\d+)", log)[1]
                ),
                "cpu_runtime_log": next(
                    (
                        line
                        for line in log.splitlines()
                        if line.startswith("runtime_cpu")
                    ),
                    "not_recorded",
                ),
                "artifact_original_outdir": summary["outdir"],
                **log_timing(run / "popen_stdout_stderr.log"),
            }
            for name in df.select_dtypes(include="number"):
                if name != "sample_id":
                    rec[name + "_mean"] = df[name].mean()
                    rec[name + "_max"] = df[name].max()
            for name in ("rel_sol", "rel_flux", "rel_sol_equal_mean"):
                rec[name + "_p95"] = df[name].quantile(0.95)
                rec[name + "_median"] = df[name].median()
            final = pd.read_csv(run / "metrics/test_per_sample_metrics.csv")
            rec["final_checkpoint_rel_sol_mean"] = final.rel_sol.mean()
            rec["weak_relative_improvement"] = (
                1 - df.rel_sol.mean() / df.rel_sol_equal_mean.mean()
            )
            rec["weak_sample_win_fraction"] = (
                df.rel_sol < df.rel_sol_equal_mean
            ).mean()
            sidecar = run / "tangent_response_context.json"
            if sidecar.exists():
                identity = read_json(sidecar)["identity"]
                identities.append({"seed": seed, "K": k, **identity})
            if k == 1:
                assert (
                    cfg["coupling_model"]["balance_projection"][
                        "symmetric_tangent_green_response"
                    ]["eta_cap_enabled"]
                    is False
                )
            for step in range(1, k + 1):
                key = f"tangent_response_cost_k{step}"
                if key not in df:
                    continue
                previous = (
                    df[f"tangent_response_cost_k{step - 1}"] if step > 1 else None
                )
                stages.append(
                    {
                        "seed": seed,
                        "trained_K": k,
                        "stage": step,
                        "cost_mean": df[key].mean(),
                        "ratio_of_mean_costs": df[key].mean() / previous.mean()
                        if previous is not None
                        else None,
                        "mean_sample_ratio": (df[key] / previous).mean()
                        if previous is not None
                        else None,
                        "active_fraction": df[
                            f"tangent_direction_{step - 1}_active"
                        ].mean(),
                    }
                )
                if previous is not None:
                    assert np.all(df[key] <= previous * (1 + 1e-10) + 1e-25)
            records.append(rec)
            samples.append(df.assign(seed=seed, K=k, hardware=hw))
            curves.append(train.assign(seed=seed, K=k, hardware=hw))
            for file in (
                run / "config_used.json",
                run / "complex_training_metrics.csv",
                run / "training.log",
                run / "popen_stdout_stderr.log",
                run / "metrics/test_per_sample_metrics.csv",
                artifact / "summary.json",
                artifact / "metrics/per_sample_metrics.csv",
                run / "complex_coupling_model_best_energy.safetensors",
                run / "tangent_response_context.json",
            ):
                if not file.exists():
                    assert file.name == "tangent_response_context.json" and k == 0
                    continue
                manifests.append(
                    {
                        "path": str(file.relative_to(ROOT)),
                        "sha256": sha256(file),
                        "bytes": file.stat().st_size,
                    }
                )
        self.records = pd.DataFrame(records).sort_values(["K", "seed"])
        self.samples = pd.concat(samples, ignore_index=True)
        self.curves = pd.concat(curves, ignore_index=True)
        self.stages = pd.DataFrame(stages)
        self.records.to_csv(OUT / "tables/run_metrics.csv", index=False)
        self.samples.to_csv(OUT / "tables/best_energy_test_samples.csv", index=False)
        self.curves.to_csv(OUT / "tables/training_curves.csv", index=False)
        self.stages.to_csv(OUT / "tables/tangent_stages.csv", index=False)
        pd.DataFrame(identities).to_csv(
            OUT / "tables/context_identities.csv", index=False
        )
        differences = {}
        for key in sorted(set().union(*(set(c) for c in configs.values()))):
            values = {run: c.get(key) for run, c in configs.items()}
            if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
                differences[key] = values
        (OUT / "config_differences.json").write_text(json.dumps(differences, indent=2))
        (OUT / "input_manifest.json").write_text(json.dumps(manifests, indent=2))
        self.logger.info("Validated 32 complete runs, 3200 best-energy sample rows")

    def geometry(self) -> None:
        geometry_path = ROOT / "data/geometry/pentagram_r05_h00078125.npz"
        topology = AxialSegmentTopologyAnalyzer.from_npz(geometry_path).analyze()
        rows = []
        for k in range(1, 11):
            reach = pointwise_reach_fraction(topology, k)
            rows.append(
                {
                    "K": k,
                    "global": global_reach_fraction(topology, k),
                    "q05": float(np.quantile(reach, 0.05)),
                    "minimum": float(reach.min()),
                    "full_point_fraction": float((reach == 1).mean()),
                }
            )
        self.reach = pd.DataFrame(rows)
        self.reach.to_csv(OUT / "tables/geometry_reach.csv", index=False)
        assert (
            topology.a_graph_diameter == 8
            and self.reach.loc[self.reach.K == 9, "global"].item() == 1
        )
        self.logger.info(
            "Geometry verified: P=%d, A-graph diameter=%d",
            topology.num_points,
            topology.a_graph_diameter,
        )

    def reference_audit(self) -> None:
        geometry = np.load(ROOT / "data/geometry/pentagram_r05_h00078125.npz")
        iy, ix = geometry["valid_grid_y_index"], geometry["valid_grid_x_index"]
        rows, manifest = [], []
        for sid, path in enumerate(
            sorted(
                (ROOT / "data/complex_samples/pentagram_r05_h00078125/test").glob(
                    "*.npz"
                )
            )
        ):
            with np.load(path) as sample:
                f, phi, psi = (sample[key][iy, ix] for key in ("rhs", "phi", "psi"))
            residual = f - phi - psi
            pair_norm = np.sqrt(np.sum(phi**2 + psi**2))
            rows.append(
                {
                    "sample_id": sid,
                    "file_stem": path.stem,
                    "target_balance_relative_rhs": np.linalg.norm(residual)
                    / np.linalg.norm(f),
                    "balance_constrained_rel_flux_lower_bound": np.linalg.norm(residual)
                    / (np.sqrt(2) * pair_norm),
                }
            )
            manifest.append(
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            )
        assert len(rows) == 100
        pd.DataFrame(rows).to_csv(
            OUT / "tables/reference_target_audit.csv", index=False
        )
        for path in (
            ROOT / "data/geometry/pentagram_r05_h00078125.npz",
            ROOT
            / "data/complex_samples/pentagram_r05_h00078125/generation_summary.json",
            ROOT / "coefficients/CDR_pentagram.py",
            ROOT / "checkpoints/pentagram/green/model.safetensors",
        ):
            manifest.append(
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            )
        (OUT / "reference_manifest.json").write_text(json.dumps(manifest, indent=2))

    def replay(self, limit: int | None) -> None:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        dataset = None
        green = None
        for index, (hw, seed, k, run) in enumerate(self.runs):
            if limit is not None and index >= limit:
                break
            outdir = OUT / "replay" / f"s{seed}_k{k}"
            outdir.mkdir(parents=True, exist_ok=True)
            destination = outdir / "metrics.csv"
            if destination.exists():
                continue
            configs = load_coupling_artifact_configs(run / "config_used.json")
            request = CouplingArtifactRequest(
                config=run / "config_used.json",
                coupling_checkpoint=run
                / "complex_coupling_model_best_energy.safetensors",
                green_checkpoint=ROOT / "checkpoints/pentagram/green/model.safetensors",
                outdir=outdir,
                device="cpu",
            )
            loader = ComplexCouplingArtifactExporter(request, logger=self.logger)
            model = loader._load_complex_model(configs, torch.device("cpu"))
            if green is None:
                green = loader._load_green_model(configs, torch.device("cpu"))
            if dataset is None:
                geometry = load_complex_geometry(
                    configs.dataset.geometry_path, dtype=configs.dataset.dtype
                )
                dataset = ComplexCouplingDataset(
                    configs.dataset.test_path,
                    geometry,
                    load_coefficient_functions(
                        configs.dataset.coefficient_functions_path
                    ),
                    branch_input_dim=configs.coupling_model.branch_input_dim,
                    dtype=configs.dataset.dtype,
                    coefficient_terms=configs.coupling_model.coefficient_terms,
                    integration_rule=configs.coupling_training.integration_rule,
                )
            # Rebuild on the analysis device, never write inside a frozen run.
            training = replace(
                configs.coupling_training,
                tangent_context_checkpoint=replace(
                    configs.coupling_training.tangent_context_checkpoint, enabled=False
                ),
            )
            evaluator = ComplexCouplingEvaluator(
                model=model,
                green_model=green,
                config=training,
                device=torch.device("cpu"),
                work_dir=outdir,
            )
            rows, inference_seconds = [], 0.0
            original = pd.read_csv(
                run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
            ).set_index("sample_id")
            start = time.perf_counter()
            with torch.no_grad():
                for batch in DataLoader(
                    dataset, batch_size=10, collate_fn=complex_coupling_collate_fn
                ):
                    t0 = time.perf_counter()
                    pred = evaluator.predict_batch(batch)
                    inference_seconds += time.perf_counter() - t0
                    sol = batch.sol_valid.numpy()
                    norm = np.linalg.norm(sol, axis=1)
                    fields = {
                        "u_phi": pred.reconstruction.u_phi_valid.numpy(),
                        "u_psi": pred.reconstruction.u_psi_valid.numpy(),
                        "sol": pred.cross_axis_reconstruction.u_pred_valid.numpy(),
                    }
                    errors = {
                        f"rel_{name}": np.linalg.norm(v - sol, axis=1) / norm
                        for name, v in fields.items()
                    }
                    physical = pred.projection.projected_physical.numpy()
                    target = batch.flux_valid.numpy()
                    for offset, sid in enumerate(batch.sample_indices.tolist()):
                        row = {"hardware": hw, "seed": seed, "K": k, "sample_id": sid}
                        row.update(
                            {key: value[offset] for key, value in errors.items()}
                        )
                        row["rel_phi"] = np.linalg.norm(
                            physical[offset, 0] - target[offset, 0]
                        ) / np.linalg.norm(target[offset, 0])
                        row["rel_psi"] = np.linalg.norm(
                            physical[offset, 1] - target[offset, 1]
                        ) / np.linalg.norm(target[offset, 1])
                        row["balance_max_abs"] = np.max(
                            np.abs(
                                physical[offset].sum(axis=0)
                                - batch.rhs_valid[offset].numpy()
                            )
                        )
                        row["artifact_rel_sol_abs_difference"] = abs(
                            row["rel_sol"] - original.loc[sid, "rel_sol"]
                        )
                        # Tight enough to detect a wrong checkpoint or projection, allows platform reductions.
                        np.testing.assert_allclose(
                            row["rel_sol"],
                            original.loc[sid, "rel_sol"],
                            rtol=2e-7,
                            atol=2e-10,
                        )
                        rows.append(row)
                    if 15 in batch.sample_indices.tolist():
                        offset = batch.sample_indices.tolist().index(15)
                        np.savez_compressed(
                            outdir / "sample15.npz",
                            coords=batch.geometry.coords_valid.numpy(),
                            sol=sol[offset],
                            rhs=batch.rhs_valid[offset].numpy(),
                            **{
                                name: val[offset]
                                for name, val in fields.items()
                                if name != "sol"
                            },
                            u_pred=fields["sol"][offset],
                        )
            pd.DataFrame(rows).to_csv(destination, index=False)
            (outdir / "provenance.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": sha256(request.coupling_checkpoint),
                        "device": "cpu",
                        "threads": 4,
                        "batch_size": 10,
                        "seconds_including_setup": time.perf_counter() - start,
                        "prediction_seconds_including_first_context_build": inference_seconds,
                        "no_training": True,
                    },
                    indent=2,
                )
            )
            self.logger.info(
                "Replayed seed=%d K=%d: %.2fs, max metric drift %.3g",
                seed,
                k,
                time.perf_counter() - start,
                max(row["artifact_rel_sol_abs_difference"] for row in rows),
            )

    def aggregate(self) -> None:
        replay_files = list((OUT / "replay").glob("*/metrics.csv"))
        if len(replay_files) == 32:
            for _, seed, k, run in self.runs:
                provenance = read_json(OUT / f"replay/s{seed}_k{k}/provenance.json")
                assert provenance["checkpoint_sha256"] == sha256(
                    run / "complex_coupling_model_best_energy.safetensors"
                )
            replay = pd.concat(
                [pd.read_csv(p) for p in replay_files], ignore_index=True
            )
            assert len(replay) == 3200
            assert not replay.duplicated(["seed", "K", "sample_id"]).any()
            assert np.isfinite(replay.select_dtypes(include="number")).all().all()
            assert replay.balance_max_abs.max() < 1e-12
            verified = replay.merge(
                self.samples[["seed", "K", "sample_id", "rel_sol"]],
                on=["seed", "K", "sample_id"],
                suffixes=("_replay", "_artifact"),
                validate="one_to_one",
            )
            np.testing.assert_allclose(
                verified.rel_sol_replay,
                verified.rel_sol_artifact,
                rtol=2e-7,
                atol=2e-10,
            )
            k1 = self.samples[self.samples.K == 1]
            assert (k1.tangent_eta_capped == 0).all()
            np.testing.assert_array_equal(k1.tangent_eta_star, k1.tangent_eta_applied)
            (OUT / "verification.json").write_text(
                json.dumps(
                    {
                        "complete_runs": 32,
                        "test_samples_per_run": 100,
                        "optimizer_calls_per_run": 2400,
                        "validation_events_per_run": 100,
                        "same_parameter_count": int(
                            self.records.parameter_count.iloc[0]
                        ),
                        "replayed_sample_rows": len(replay),
                        "rel_sol_max_abs_replay_difference": float(
                            replay.artifact_rel_sol_abs_difference.max()
                        ),
                        "projection_max_abs_balance_residual": float(
                            replay.balance_max_abs.max()
                        ),
                        "k1_eta_uncapped_verified": True,
                        "nested_cost_nonincrease_verified": True,
                        "all_recorded_subspace_directions_active": bool(
                            (self.stages.active_fraction == 1).all()
                        ),
                        "geometry_A_graph_diameter": 8,
                        "first_full_reach_K": 9,
                        "training_executed": False,
                    },
                    indent=2,
                )
            )
            replay.to_csv(OUT / "tables/directional_test_samples.csv", index=False)
            grouped = (
                replay.groupby(["K", "seed"])[
                    ["rel_u_phi", "rel_u_psi", "rel_phi", "rel_psi"]
                ]
                .mean()
                .add_suffix("_mean")
                .reset_index()
            )
            self.records = self.records.merge(
                grouped, on=["K", "seed"], validate="one_to_one"
            )
            self.records.to_csv(OUT / "tables/run_metrics.csv", index=False)
            self.directional_figures(replay)
        metrics = [
            "rel_sol_mean",
            "rel_sol_equal_mean_mean",
            "rel_flux_mean",
            "rel_sol_p95",
            "rel_sol_max",
            "loss_mean",
            "loss_energy_boundary_mean",
            "best_epoch",
            "val_end_over_best",
            "steady_epoch_seconds",
            "trainer_wall_hours",
            "weak_relative_improvement",
        ]
        metrics += [
            key
            for key in (
                "rel_u_phi_mean",
                "rel_u_psi_mean",
                "rel_phi_mean",
                "rel_psi_mean",
            )
            if key in self.records
        ]
        agg = self.records.groupby("K")[metrics].agg(["mean", "std"])
        agg.columns = ["_".join(col) for col in agg.columns]
        agg.reset_index().to_csv(OUT / "tables/aggregate_by_k.csv", index=False)
        paired = []
        indexed = self.records.set_index(["K", "seed"])
        for low, high in zip(KS[:-1], KS[1:], strict=True):
            for seed in range(4):
                a, b = indexed.loc[low, seed], indexed.loc[high, seed]
                paired.append(
                    {
                        "from_K": low,
                        "to_K": high,
                        "seed": seed,
                        "rel_sol_reduction": 1 - b.rel_sol_mean / a.rel_sol_mean,
                        "rel_flux_reduction": 1 - b.rel_flux_mean / a.rel_flux_mean,
                        "energy_reduction": 1 - b.loss_mean / a.loss_mean,
                    }
                )
        pd.DataFrame(paired).to_csv(OUT / "tables/paired_effects.csv", index=False)
        self.records.groupby(["hardware", "K"])[
            [
                "steady_epoch_seconds",
                "trainer_wall_hours",
                "optimizer_ms",
                "peak_device_memory_mib",
                "rel_sol_mean",
            ]
        ].mean().to_csv(OUT / "tables/hardware_costs.csv")
        colors = ["#2364aa", "#b33d50", "#008575", "#8359a3"]
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Solution error: paired training seeds",
                "Directional-source pair error",
            ),
        )
        for seed, color in enumerate(colors):
            d = self.records[self.records.seed == seed]
            for col, key in ((1, "rel_sol_mean"), (2, "rel_flux_mean")):
                fig.add_trace(
                    go.Scatter(
                        x=d.K,
                        y=100 * d[key],
                        mode="lines+markers",
                        name=f"seed {seed}",
                        marker_color=color,
                        showlegend=col == 1,
                    ),
                    row=1,
                    col=col,
                )
        fig.update_xaxes(title_text="Tangent dimension K", tickvals=KS)
        fig.update_yaxes(title_text="Mean test relative error (%)")
        self.save_figure(fig, "accuracy_by_seed")
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Global and lower-5% reach", "Worst point reach"),
        )
        for key, name in (
            ("global", "Global"),
            ("q05", "Lower 5%"),
            ("minimum", "Minimum"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=self.reach.K,
                    y=100 * self.reach[key],
                    mode="lines+markers",
                    name=name,
                ),
                row=1,
                col=2 if key == "minimum" else 1,
            )
        fig.update_xaxes(title_text="K", dtick=1)
        fig.update_yaxes(title_text="Structural reach (%)")
        self.save_figure(fig, "geometry_reach")
        fig = make_subplots(rows=2, cols=4, subplot_titles=[f"K={k}" for k in KS])
        for i, k in enumerate(KS):
            for seed, color in enumerate(colors):
                d = self.curves[
                    (self.curves.K == k)
                    & (self.curves.seed == seed)
                    & (self.curves.split == "val")
                ]
                fig.add_trace(
                    go.Scatter(
                        x=d.global_step,
                        y=d.loss,
                        mode="lines",
                        name=f"seed {seed}",
                        marker_color=color,
                        showlegend=i == 0,
                    ),
                    row=i // 4 + 1,
                    col=i % 4 + 1,
                )
        fig.update_yaxes(type="log", title_text="Validation bulk energy")
        fig.update_xaxes(title_text="Optimizer calls")
        self.save_figure(fig, "validation_energy", height=720)
        fig = go.Figure()
        for hw, symbol in (("nvidia_a40", "circle"), ("mac_studio", "diamond")):
            for seed, color in enumerate(colors):
                d = self.records[
                    (self.records.hardware == hw) & (self.records.seed == seed)
                ]
                if d.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=d.steady_epoch_seconds / 24,
                        y=d.rel_sol_mean * 100,
                        mode="lines+markers+text",
                        text=[f"K={k}" for k in d.K],
                        textposition="top center",
                        name=f"{hw}, seed {seed}",
                        marker={"color": color, "symbol": symbol},
                    )
                )
        fig.update_xaxes(
            title_text="Steady epoch time / 24 (seconds; includes validation)"
        )
        fig.update_yaxes(title_text="Mean test solution error (%)")
        self.save_figure(fig, "time_accuracy")
        fig = go.Figure()
        for k in KS[2:]:
            d = (
                self.stages[self.stages.trained_K == k]
                .groupby("stage")
                .cost_mean.agg(["mean", "std"])
            )
            fig.add_trace(
                go.Scatter(
                    x=d.index,
                    y=d["mean"],
                    error_y={"array": d["std"]},
                    mode="lines+markers",
                    name=f"trained K={k}",
                )
            )
        fig.update_xaxes(title_text="Internal correction stage j", dtick=1)
        fig.update_yaxes(
            title_text="Mean J_j (squared physical L2 mismatch)", type="log"
        )
        self.save_figure(fig, "internal_response_cost")
        self.logger.info("Aggregate tables and five Plotly figures generated")
        self.write_appendix()

    def write_appendix(self) -> None:
        sections = [
            "# Pentagram Paper Experiment: Numerical Appendix\n",
            "Generated from all 32 best-energy runs. Errors are percentages. "
            "The standard deviation is across four seed-level means (ddof=1), not across 400 independent tests.\n",
        ]

        def table(headers: list[str], rows: list[list[str]]) -> str:
            return (
                "\n".join(
                    [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join(["---"] * len(headers)) + " |",
                    ]
                    + ["| " + " | ".join(row) + " |" for row in rows]
                )
                + "\n"
            )

        metrics = [
            "rel_sol_mean",
            "rel_sol_equal_mean_mean",
            "rel_u_phi_mean",
            "rel_u_psi_mean",
            "rel_flux_mean",
        ]
        metrics = [key for key in metrics if key in self.records]
        rows = []
        for k, group in self.records.groupby("K"):
            rows.append(
                [str(k)]
                + [
                    f"{group[key].mean() * 100:.4f} +/- {group[key].std() * 100:.4f}"
                    for key in metrics
                ]
            )
        sections += ["## Four-Seed Summary\n", table(["K", *metrics], rows)]
        rows = [
            [
                str(int(row.K)),
                str(int(row.seed)),
                row.hardware,
                str(int(row.best_epoch)),
                *[f"{row[key] * 100:.4f}" for key in metrics],
                f"{row.loss_mean:.7e}",
            ]
            for _, row in self.records.iterrows()
        ]
        sections += [
            "## All 32 Runs\n",
            table(
                ["K", "Seed", "Hardware", "Best epoch", *metrics, "Test bulk energy"],
                rows,
            ),
        ]
        rows = [
            [
                str(int(r.K)),
                *[f"{100 * r[key]:.8f}" for key in ("global", "q05", "minimum")],
            ]
            for _, r in self.reach.iterrows()
        ]
        sections += [
            "## Geometry-Only Reach (%)\n",
            table(["K", "Global", "Lower 5%", "Minimum"], rows),
        ]
        rows = [
            [
                str(int(r.K)),
                str(int(r.seed)),
                f"{r.steady_epoch_seconds:.2f}",
                f"{r.trainer_wall_hours:.3f}",
                f"{r.val_end_over_best:.5f}",
                f"{r.final_checkpoint_rel_sol_mean * 100:.4f}",
            ]
            for _, r in self.records.iterrows()
        ]
        sections += [
            "## Timing and Final-Checkpoint Audit\n",
            "Timing is from each individual run log, not the stale parent queue logs. "
            "Steady epochs use the median of timestamp intervals 10->11 through 99->100; "
            "these intervals include validation. Trainer span excludes post-training test/export.\n",
            table(
                [
                    "K",
                    "Seed",
                    "Steady epoch seconds",
                    "Trainer hours",
                    "Final/best validation energy",
                    "Final-model test error (%)",
                ],
                rows,
            ),
        ]
        rows = []
        for (k, step), group in self.stages.groupby(["trained_K", "stage"]):
            ratio = group.ratio_of_mean_costs.mean()
            rows.append(
                [
                    str(k),
                    str(step),
                    f"{group.cost_mean.mean():.8e}",
                    "-" if np.isnan(ratio) else f"{ratio:.6f}",
                    f"{group.active_fraction.mean():.4f}",
                ]
            )
        sections += [
            "## Internal Tangent Stages\n",
            "Each row is a four-seed average. Ratios average seed-level ratios of 100-sample mean costs. "
            "Different trained K use different networks and initial proposals; do not treat columns from different networks as a single trajectory.\n",
            table(
                [
                    "Trained K",
                    "Stage j",
                    "Mean J_j",
                    "Mean seed J_j/J_(j-1)",
                    "Active fraction",
                ],
                rows,
            ),
        ]
        (OUT / "numerical_appendix.md").write_text("\n".join(sections))

    def directional_figures(self, replay: pd.DataFrame) -> None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Directional solutions and final prediction",
                "Test error tails",
            ),
        )
        for key, name in (
            ("rel_u_phi_mean", "u_phi"),
            ("rel_u_psi_mean", "u_psi"),
            ("rel_sol_mean", "weak blend"),
        ):
            d = self.records.groupby("K")[key].agg(["mean", "std"])
            fig.add_trace(
                go.Scatter(
                    x=d.index,
                    y=100 * d["mean"],
                    error_y={"array": 100 * d["std"]},
                    mode="lines+markers",
                    name=name,
                ),
                row=1,
                col=1,
            )
        for key, name in (
            ("rel_sol_mean", "Mean"),
            ("rel_sol_p95", "95th percentile"),
            ("rel_sol_max", "Worst sample"),
        ):
            d = self.records.groupby("K")[key].mean()
            fig.add_trace(
                go.Scatter(x=d.index, y=100 * d, mode="lines+markers", name=name),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="K", tickvals=KS)
        fig.update_yaxes(title_text="Relative solution error (%)")
        self.save_figure(fig, "directional_solutions_and_tails")
        # Fixed seed and sample selected before viewing this comparison; no per-K cherry-picking.
        selected = (0, 4, 5, 9, 10)
        arrays = [np.load(OUT / f"replay/s0_k{k}/sample15.npz") for k in selected]
        vmax = max(
            np.max(np.abs(a[key] - a["sol"]))
            for a in arrays
            for key in ("u_phi", "u_psi", "u_pred")
        )
        fig = make_subplots(
            rows=3,
            cols=5,
            subplot_titles=[
                f"{label}, K={k}"
                for label in ("u_phi", "u_psi", "u_pred")
                for k in selected
            ],
            horizontal_spacing=0.018,
            vertical_spacing=0.10,
        )
        for col, arr in enumerate(arrays, 1):
            coords = arr["coords"]
            for row, key in enumerate(("u_phi", "u_psi", "u_pred"), 1):
                error = np.abs(arr[key] - arr["sol"])
                fig.add_trace(
                    go.Scattergl(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode="markers",
                        showlegend=False,
                        marker={"size": 2.5, "color": error, "coloraxis": "coloraxis"},
                        customdata=error,
                        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>abs error=%{customdata:.4g}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                axis = (row - 1) * 5 + col
                fig.update_yaxes(
                    scaleanchor=f"x{axis}" if axis > 1 else "x",
                    scaleratio=1,
                    showticklabels=col == 1,
                    tickfont={"size": 11},
                    row=row,
                    col=col,
                )
                fig.update_xaxes(
                    showticklabels=row == 3,
                    tickvals=[-0.4, 0, 0.4],
                    tickfont={"size": 11},
                    row=row,
                    col=col,
                )
        fig.update_layout(
            coloraxis={
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": vmax,
                "colorbar": {
                    "title": {"text": "Absolute error", "side": "right"},
                    "thickness": 14,
                },
            },
            title="Fixed seed 0, test sample 15: identical color range across all fields and K",
        )
        self.save_figure(fig, "sample15_error_fields", height=850)

    @staticmethod
    def save_figure(fig: go.Figure, name: str, height: int = 540) -> None:
        fig.update_layout(
            template="plotly_white",
            width=1100,
            height=height,
            font={"size": 15},
            margin={"l": 75, "r": 40, "t": 60, "b": 65},
        )
        fig.write_html(OUT / f"figures/{name}.html", include_plotlyjs="directory")
        fig.write_image(OUT / f"figures/{name}.png", scale=1.4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    analysis = PaperComparison()
    analysis.collect()
    analysis.geometry()
    analysis.reference_audit()
    if args.replay:
        analysis.replay(args.limit)
    analysis.aggregate()
