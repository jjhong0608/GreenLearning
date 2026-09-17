"""Run the four-example, read-only paper source initialization audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("preflight", "reference", "initialization", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    args = parser.parse_args()
    SourceInitializationAudit(SourceInitializationRequest(**vars(args))).run()


if __name__ == "__main__":
    main()
