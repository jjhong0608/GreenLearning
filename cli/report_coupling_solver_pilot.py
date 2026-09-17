"""Validate saved pilot metrics and build a compact numerical table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


class PilotReport:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self) -> None:
        hashes = json.loads((self.root / "input_hashes.json").read_text())
        executed_cli = Path("cli/audit_coupling_solver_pilot.py").resolve()
        archive = self.root / "executed_audit.py"
        for name, expected in hashes.items():
            path = Path(name)
            if path == executed_cli and archive.exists():
                path = archive
            if _sha256(path) != expected:
                raise ValueError(f"Recorded input changed: {path}")
        _json(
            self.root / "current_code_provenance.json",
            dict(
                evaluated_cli_hash=hashes[str(executed_cli)],
                current_cli_hash=_sha256(executed_cli),
                executed_snapshot=str(archive) if archive.exists() else None,
                numerical_inputs_unchanged=True,
                post_run_cleanup="np.asarray return typing and explicit NPZ keywords only; evaluated snapshot preserved",
            ),
        )
        rows = read_rows(self.root / "per_sample.csv")
        assert len(rows) == 30
        structures = read_rows(self.root / "structure.csv")
        assert {r["example"] for r in structures} == {
            "unit_square",
            "disk",
            "annulus",
            "pentagram",
        }
        baseline = {
            (r["example"], r["sample_id"]): r
            for r in read_rows(
                Path(
                    "docs/analysis/paper_source_initialization_audit_v2/learned_baseline_per_sample.csv"
                )
            )
            if r["seed"] == "0"
        }
        summary = []
        checks = []
        for example in ("unit_square", "disk"):
            direct = json.loads((self.root / f"{example}_direct.json").read_text())
            assert direct["qr_full_rank_screen"] and direct["lu_status"] == 0
            stop_records = {}
            for method in ("gmres", "lsmr"):
                stop_records[method] = [
                    json.loads(
                        (self.root / f"{example}_{method}_sample{i}.json").read_text()
                    )
                    for i in range(3)
                ]
            for method in ("learned", "lu", "qr", "gmres", "lsmr"):
                selected = [
                    r
                    for r in rows
                    if r["example"] == example and r["condition"] == method
                ]
                assert len(selected) == 3
                errors = np.array([float(r["rel_sol"]) for r in selected])
                summary.append(
                    dict(
                        example=example,
                        method=method,
                        samples=3,
                        converged_samples=sum(
                            s["status"] in ((0,) if method == "gmres" else (1, 2, 4, 5))
                            for s in stop_records[method]
                        )
                        if method in stop_records
                        else None,
                        mean_percent=float(errors.mean() * 100),
                        p95_percent=float(np.quantile(errors, 0.95) * 100),
                        max_percent=float(errors.max() * 100),
                        maximum_balance=max(
                            float(r["balance_max_abs"]) for r in selected
                        ),
                        max_equation_relative_residual=max(
                            float(r["equation_relative_residual"]) for r in selected
                        )
                        if method != "learned"
                        else None,
                        max_delta_relative_to_lu=max(
                            float(r["delta_relative_to_lu"]) for r in selected
                        )
                        if method != "learned"
                        else None,
                    )
                )
                for r in selected:
                    assert float(r["balance_max_abs"]) < 1e-11
                    if method == "learned":
                        np.testing.assert_allclose(
                            float(r["rel_sol"]),
                            float(baseline[example, r["sample_id"]]["rel_sol"]),
                            rtol=1e-8,
                            atol=1e-12,
                        )
                    elif method in ("lu", "qr"):
                        assert float(r["delta_relative_to_lu"]) < 1e-6
                        assert float(r["equation_relative_residual"]) < 1e-8
            checks.append(
                dict(
                    example=example,
                    native_metric_reproduced=True,
                    direct_lu_qr_agreement=True,
                    iterative_status={
                        name: [s["status"] for s in records]
                        for name, records in stop_records.items()
                    },
                    iterative_agreement={
                        name: all(
                            float(r["delta_relative_to_lu"]) < 1e-6
                            and float(r["equation_relative_residual"]) < 1e-8
                            for r in rows
                            if r["example"] == example and r["condition"] == name
                        )
                        for name in stop_records
                    },
                    nonconvergence_preserved_not_silently_accepted=True,
                )
            )
        _write_csv(self.root / "summary.csv", summary)
        lines = [
            "# Pilot numerical tables",
            "",
            "All errors in percent. Three samples only; not full-test tail estimates.",
            "",
            "| Example | Method | Mean | P95 | Max | Relative residual | Relative d difference to LU |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for entry in summary:
            lines.append(
                f"| {entry['example']} | {entry['method']} | {entry['mean_percent']:.8f} | {entry['p95_percent']:.8f} | {entry['max_percent']:.8f} | {entry['max_equation_relative_residual']} | {entry['max_delta_relative_to_lu']} |"
            )
        (self.root / "tables.md").write_text("\n".join(lines) + "\n")
        _json(self.root / "report_verification.json", checks)
        _json(
            self.root / "report_input_hashes.json",
            {
                str(p): _sha256(p)
                for p in [
                    self.root / "per_sample.csv",
                    self.root / "structure.csv",
                    Path(
                        "docs/analysis/paper_source_initialization_audit_v2/learned_baseline_per_sample.csv"
                    ),
                    Path(
                        "docs/analysis/learned_k_extension_square_disk_v2/per_sample.csv"
                    ),
                ]
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    PilotReport(parser.parse_args().outdir).run()
