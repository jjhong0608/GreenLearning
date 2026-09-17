"""Evaluate frozen Annulus best-energy reconstructions without training."""

import argparse
from pathlib import Path

from greenonet.complex_reconstruction_audit import (
    ReconstructionAudit,
    ReconstructionAuditRequest,
)


class AnnulusAuditCLI:
    def run(self) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
        parser.add_argument("--outdir", type=Path, required=True)
        parser.add_argument("--device", choices=["cpu", "cuda:1"], required=True)
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--num-threads", type=int, default=4)
        args = vars(parser.parse_args())
        args["run_dirs"] = tuple(args["run_dirs"])
        ReconstructionAudit(ReconstructionAuditRequest(**args)).run()


if __name__ == "__main__":
    AnnulusAuditCLI().run()
