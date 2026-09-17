"""Sequential CPU audit of twelve frozen Example 1 best-energy checkpoints."""

import argparse
from pathlib import Path

from greenonet.unit_square_trunk_audit import TrunkAuditRequest, UnitSquareTrunkAudit


class TrunkAuditCLI:
    def run(self) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--run-root", type=Path, required=True)
        parser.add_argument("--outdir", type=Path, required=True)
        parser.add_argument("--device", choices=["cpu"], default="cpu")
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--num-threads", type=int, default=4)
        UnitSquareTrunkAudit(TrunkAuditRequest(**vars(parser.parse_args()))).run()


if __name__ == "__main__":
    TrunkAuditCLI().run()
