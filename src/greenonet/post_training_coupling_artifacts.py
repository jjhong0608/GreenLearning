from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from greenonet.complex_tangent_context_io import resolve_tangent_context_path
from greenonet.config import (
    CouplingArtifactsConfig,
    CouplingTrainingConfig,
    DatasetConfig,
    PipelineConfig,
    TangentContextCheckpointConfig,
)
from greenonet.coupling_artifact_runtime import (
    build_coupling_artifact_logger,
    export_coupling_artifact_request,
)
from greenonet.coupling_artifacts import CouplingArtifactRequest


@dataclass(frozen=True)
class PostTrainingCouplingArtifactPaths:
    """Resolved fixed paths for one automatic best-energy export."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    coefficients: Path | None
    tangent_context: Path | None


class FullReferenceTestDatasetValidator:
    """Validate every test NPZ before a long artifact-enabled training run."""

    @classmethod
    def validate(cls, test_path: Path | str) -> tuple[Path, ...]:
        directory = Path(test_path)
        files = tuple(sorted(directory.glob("*.npz")))
        if not files:
            raise FileNotFoundError(f"No npz files found in {directory}")
        for path in files:
            cls._validate_file(path)
        return files

    @staticmethod
    def _validate_file(path: Path) -> None:
        with np.load(path, allow_pickle=False) as raw:
            keys = set(raw.files)
        missing = sorted({"rhs", "sol"} - keys)
        if missing:
            raise KeyError(
                f"{path} is missing full-reference keys: {', '.join(missing)}"
            )
        if not ({"phi", "psi"}.issubset(keys) or {"uxx", "uyy"}.issubset(keys)):
            raise KeyError(
                f"{path} must contain directional targets as phi/psi or uxx/uyy."
            )


class PostTrainingCouplingArtifactRunner:
    """Run the existing complex exporter after training from fixed run outputs."""

    BEST_ENERGY_CHECKPOINT = "complex_coupling_model_best_energy.safetensors"
    OUTPUT_DIRECTORY = "artifacts_best_energy"
    CONFIG_USED = "config_used.json"
    GREEN_CHECKPOINT = "model.safetensors"
    TANGENT_CONTEXT = "tangent_response_context.safetensors"

    def __init__(
        self,
        *,
        config: CouplingArtifactsConfig,
        dataset: DatasetConfig,
        coupling_training: CouplingTrainingConfig,
        pipeline: PipelineConfig,
        work_dir: Path | str,
        tangent_context_override: Path | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError(
                "PostTrainingCouplingArtifactRunner requires "
                "coupling_artifacts.enabled=true."
            )
        self.config = config
        self.dataset = dataset
        self.coupling_training = coupling_training
        self.pipeline = pipeline
        self.work_dir = Path(work_dir).resolve()
        self.tangent_context_override = tangent_context_override

    def resolve_paths(self) -> PostTrainingCouplingArtifactPaths:
        config_path = (self.work_dir / self.CONFIG_USED).resolve()
        coupling_checkpoint = (self.work_dir / self.BEST_ENERGY_CHECKPOINT).resolve()
        outdir = (coupling_checkpoint.parent / self.OUTPUT_DIRECTORY).resolve()
        if self.pipeline.run_green:
            green_checkpoint = (self.work_dir / self.GREEN_CHECKPOINT).resolve()
        else:
            if self.pipeline.green_pretrained_path is None:
                raise ValueError(
                    "pipeline.green_pretrained_path is required for post-training "
                    "artifact export when pipeline.run_green=false."
                )
            green_checkpoint = Path(self.pipeline.green_pretrained_path).resolve()

        tangent_checkpoint = TangentContextCheckpointConfig.from_raw(
            self.coupling_training.tangent_context_checkpoint
        )
        tangent_context = resolve_tangent_context_path(
            checkpoint=tangent_checkpoint,
            cli_override=self.tangent_context_override,
            default_path=(self.work_dir / self.TANGENT_CONTEXT),
        )
        if tangent_context is not None:
            tangent_context = tangent_context.resolve()

        coefficients = self.dataset.coefficient_functions_path
        return PostTrainingCouplingArtifactPaths(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            coefficients=None if coefficients is None else coefficients.resolve(),
            tangent_context=tangent_context,
        )

    def build_request(self) -> CouplingArtifactRequest:
        paths = self.resolve_paths()
        for label, path in (
            ("materialized config", paths.config),
            ("best-energy CouplingNet checkpoint", paths.coupling_checkpoint),
            ("GreenNet checkpoint", paths.green_checkpoint),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        self._require_empty_output_directory(paths.outdir)
        return CouplingArtifactRequest(
            config=paths.config,
            coupling_checkpoint=paths.coupling_checkpoint,
            green_checkpoint=paths.green_checkpoint,
            outdir=paths.outdir,
            coefficients=paths.coefficients,
            device=self.config.device,
            theme=self.config.theme,
            selected_samples=self.config.selected_samples,
            plot_workers=self.config.plot_workers,
            save_generated_data=self.config.save_generated_data,
            coefficient_vector_max_points=(self.config.coefficient_vector_max_points),
            show_domain_boundary=self.config.show_domain_boundary,
            visualization_mesh=(
                None
                if self.config.visualization_mesh is None
                else self.config.visualization_mesh.resolve()
            ),
            directional_color_quantile=self.config.directional_color_quantile,
            tangent_context=paths.tangent_context,
            generation_trigger="post_training",
            checkpoint_selector="best_energy",
        )

    def run(self) -> dict[str, object]:
        request = self.build_request()
        logger = build_coupling_artifact_logger(request.outdir)
        logger.info(
            "Starting post-training best-energy artifact export "
            "config=%s coupling_checkpoint=%s green_checkpoint=%s outdir=%s",
            request.config,
            request.coupling_checkpoint,
            request.green_checkpoint,
            request.outdir,
        )
        return export_coupling_artifact_request(request, logger=logger)

    @staticmethod
    def _require_empty_output_directory(outdir: Path) -> None:
        if not outdir.exists():
            return
        if not outdir.is_dir():
            raise NotADirectoryError(outdir)
        if any(outdir.iterdir()):
            raise FileExistsError(
                f"Post-training artifact output directory is not empty: {outdir}"
            )
