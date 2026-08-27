from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_coupling_artifacts import export_complex_coupling_artifacts
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    export_coupling_artifacts,
)


def build_coupling_artifact_logger(outdir: Path) -> logging.Logger:
    """Create the shared console/file logger used by both export entrypoints."""

    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ExportCouplingArtifacts")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(funcName)s - %(message)s")
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        omit_repeated_times=False,
    )
    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        outdir / "export_coupling_artifacts.log",
        mode="w",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    return logger


def export_coupling_artifact_request(
    request: CouplingArtifactRequest,
    *,
    logger: logging.Logger,
) -> dict[str, object]:
    """Dispatch one artifact request through the configured geometry path."""

    with request.config.open() as fp:
        raw = json.load(fp)
    dataset_raw = raw.get("dataset", {})
    if isinstance(dataset_raw, dict) and dataset_raw.get("geometry_mode") == "complex":
        summary = export_complex_coupling_artifacts(request, logger=logger)
    else:
        summary = export_coupling_artifacts(request, logger=logger)
    logger.info(
        "Completed CouplingNet artifact export (selected_samples=%s)",
        summary["selected_samples"],
    )
    return summary
