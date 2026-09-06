"""Sequential frozen best-energy checkpoint audit; never trains or plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from greenonet.complex_frozen_tangent_csv import (
    FrozenTangentCsvRequest,
    run_frozen_tangent_csv,
)


class AuditFrozenTangentCsvCLI:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
        self.parser.add_argument("--outdir", type=Path, required=True)
        self.parser.add_argument(
            "--device", required=True, help="cpu or explicit cuda:N; no fallback"
        )
        for name, default in (
            ("baseline-k", 10),
            ("max-k", 64),
            ("batch-size", 10),
            ("num-threads", 4),
            ("warmup-repeats", 3),
            ("timing-repeats", 5),
        ):
            self.parser.add_argument(f"--{name}", type=int, default=default)
        for name in ("green-checkpoint", "geometry", "test-path", "coefficients"):
            self.parser.add_argument(f"--{name}", type=Path)
        self.parser.add_argument(
            "--benchmark",
            action="store_true",
            help="Independently time every K; excludes setup and reference metrics",
        )
        self.parser.add_argument("--overwrite", action="store_true")

    def run(self) -> None:
        args = vars(self.parser.parse_args())
        args["run_dirs"] = tuple(args["run_dirs"])
        request = FrozenTangentCsvRequest(**args)
        run_frozen_tangent_csv(request)


if __name__ == "__main__":
    AuditFrozenTangentCsvCLI().run()
