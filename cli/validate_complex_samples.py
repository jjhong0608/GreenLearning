from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rich.logging import RichHandler

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_geometry import load_complex_geometry
from greenonet.fenicsx_samples.geometry import (
    GeometryGridLoader,
    RawComplexGeometryGrid,
)


@dataclass(frozen=True)
class ComplexSampleValidationConfig:
    """Configuration for validating complex Coupling sample directories."""

    geometry: Path
    sample_root: Path
    splits: tuple[str, ...]
    coefficients: Path | None
    branch_input_dim: int = 4
    max_balance_residual: float = 1.0e-2
    outside_tol: float = 1.0e-12

    def __post_init__(self) -> None:
        if not self.splits:
            raise ValueError("--splits must contain at least one split name.")
        if self.branch_input_dim <= 0:
            raise ValueError("--branch-input-dim must be positive.")
        if self.max_balance_residual < 0.0:
            raise ValueError("--max-balance-residual must be non-negative.")
        if self.outside_tol < 0.0:
            raise ValueError("--outside-tol must be non-negative.")


@dataclass(frozen=True)
class SplitValidationSummary:
    split: str
    sample_count: int
    balance_residual_mean: float
    balance_residual_max: float
    outside_abs_max: float


class FullGridValidationMixin:
    """Full-grid array validation helpers shared across sample files."""

    REQUIRED_KEYS: tuple[str, ...] = ("rhs", "sol", "phi", "psi")

    @staticmethod
    def valid_mask(geometry: RawComplexGeometryGrid) -> np.ndarray:
        mask = np.zeros(geometry.full_grid_shape, dtype=bool)
        mask[geometry.valid_grid_y_index, geometry.valid_grid_x_index] = True
        return mask

    def load_arrays(
        self,
        path: Path,
        *,
        full_grid_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        with np.load(path) as raw:
            missing = sorted(set(self.REQUIRED_KEYS) - set(raw.files))
            if missing:
                raise KeyError(f"{path} is missing keys: {', '.join(missing)}")
            arrays = {
                key: np.asarray(raw[key], dtype=np.float64)
                for key in self.REQUIRED_KEYS
            }
        for key, value in arrays.items():
            if value.shape != full_grid_shape:
                raise ValueError(
                    f"{path}:{key} has shape {value.shape}; expected {full_grid_shape}."
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{path}:{key} contains non-finite values.")
        return arrays

    @staticmethod
    def outside_abs_max(
        arrays: dict[str, np.ndarray], outside_mask: np.ndarray
    ) -> float:
        if not np.any(outside_mask):
            return 0.0
        maxima = [
            float(np.max(np.abs(value[outside_mask])))
            for value in arrays.values()
            if value.size
        ]
        return max(maxima, default=0.0)

    @staticmethod
    def balance_residual(
        arrays: dict[str, np.ndarray], valid_mask: np.ndarray
    ) -> float:
        rhs = arrays["rhs"][valid_mask]
        residual = arrays["phi"][valid_mask] + arrays["psi"][valid_mask] - rhs
        denominator = max(float(np.linalg.norm(rhs)), 1.0e-12)
        return float(np.linalg.norm(residual) / denominator)


class ComplexSampleValidator(FullGridValidationMixin):
    """Validate sample NPZ files against the complex Coupling dataset contract."""

    def __init__(
        self,
        config: ComplexSampleValidationConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def validate(self) -> dict[str, object]:
        geometry_grid = GeometryGridLoader().load(self.config.geometry)
        valid_mask = self.valid_mask(geometry_grid)
        outside_mask = ~valid_mask
        split_summaries = [
            self._validate_split(geometry_grid, valid_mask, outside_mask, split)
            for split in self.config.splits
        ]
        probe = self._probe_dataset_load()
        payload = {
            "geometry": str(self.config.geometry),
            "sample_root": str(self.config.sample_root),
            "splits": [asdict(summary) for summary in split_summaries],
            "max_balance_residual": max(
                summary.balance_residual_max for summary in split_summaries
            ),
            "max_outside_abs": max(
                summary.outside_abs_max for summary in split_summaries
            ),
            "dataset_probe": probe,
            "thresholds": {
                "max_balance_residual": self.config.max_balance_residual,
                "outside_tol": self.config.outside_tol,
            },
        }
        self._write_summary(payload)
        if payload["max_balance_residual"] > self.config.max_balance_residual:
            raise ValueError(
                "Balance residual threshold exceeded: "
                f"{payload['max_balance_residual']:.6e} > "
                f"{self.config.max_balance_residual:.6e}."
            )
        self.logger.info(
            "validated %d split(s); max balance residual %.6e",
            len(split_summaries),
            payload["max_balance_residual"],
        )
        return payload

    def _validate_split(
        self,
        geometry: RawComplexGeometryGrid,
        valid_mask: np.ndarray,
        outside_mask: np.ndarray,
        split: str,
    ) -> SplitValidationSummary:
        split_dir = self.config.sample_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Sample split directory does not exist: {split_dir}"
            )
        files = sorted(split_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz samples found in {split_dir}")
        residuals: list[float] = []
        outside_maxima: list[float] = []
        for path in files:
            arrays = self.load_arrays(path, full_grid_shape=geometry.full_grid_shape)
            outside_max = self.outside_abs_max(arrays, outside_mask)
            if outside_max > self.config.outside_tol:
                raise ValueError(
                    f"{path} has nonzero outside-domain values; max abs "
                    f"{outside_max:.6e} > {self.config.outside_tol:.6e}."
                )
            residuals.append(self.balance_residual(arrays, valid_mask))
            outside_maxima.append(outside_max)
        summary = SplitValidationSummary(
            split=split,
            sample_count=len(files),
            balance_residual_mean=float(np.mean(residuals)),
            balance_residual_max=max(residuals),
            outside_abs_max=max(outside_maxima, default=0.0),
        )
        self.logger.info(
            "%s: %d samples, balance max %.6e",
            split,
            summary.sample_count,
            summary.balance_residual_max,
        )
        return summary

    def _probe_dataset_load(self) -> dict[str, object]:
        probe_split = (
            "train" if "train" in self.config.splits else self.config.splits[0]
        )
        dataset = ComplexCouplingDataset(
            self.config.sample_root / probe_split,
            load_complex_geometry(self.config.geometry),
            load_coefficient_functions(self.config.coefficients),
            branch_input_dim=self.config.branch_input_dim,
            dtype=torch.float64,
        )
        item = dataset[0]
        return {
            "split": probe_split,
            "sample_count": len(dataset),
            "first_file_stem": item.file_stem,
            "num_valid_points": int(item.rhs_valid.numel()),
            "has_flux": bool(item.has_flux),
        }

    def _write_summary(self, payload: dict[str, object]) -> None:
        self.config.sample_root.mkdir(parents=True, exist_ok=True)
        path = self.config.sample_root / "validation_summary.json"
        path.write_text(json.dumps(payload, indent=2))
        self.logger.info("wrote validation summary to %s", path)


class ValidateComplexSamplesCLI:
    """CLI for validating complex Coupling sample NPZ directories."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Validate complex Coupling sample NPZ files."
        )
        parser.add_argument("--geometry", type=Path, required=True)
        parser.add_argument("--sample-root", type=Path, required=True)
        parser.add_argument(
            "--splits",
            nargs="+",
            default=("train", "valid", "test"),
        )
        parser.add_argument("--coefficients", type=Path, default=None)
        parser.add_argument("--branch-input-dim", type=int, default=4)
        parser.add_argument("--max-balance-residual", type=float, default=1.0e-2)
        parser.add_argument("--outside-tol", type=float, default=1.0e-12)
        self.parser = parser

    def parse_config(
        self,
        argv: Sequence[str] | None = None,
    ) -> ComplexSampleValidationConfig:
        args = self.parser.parse_args(argv)
        return ComplexSampleValidationConfig(
            geometry=args.geometry,
            sample_root=args.sample_root,
            splits=tuple(str(split) for split in args.splits),
            coefficients=args.coefficients,
            branch_input_dim=int(args.branch_input_dim),
            max_balance_residual=float(args.max_balance_residual),
            outside_tol=float(args.outside_tol),
        )

    @staticmethod
    def _build_logger(sample_root: Path) -> logging.Logger:
        sample_root.mkdir(parents=True, exist_ok=True)
        build_logger = logging.getLogger("ValidateComplexSamples")
        build_logger.handlers.clear()
        build_logger.propagate = False
        build_logger.setLevel(logging.INFO)
        logging.root.handlers.clear()

        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            sample_root / "validate_complex_samples.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        build_logger.addHandler(rich_handler)
        build_logger.addHandler(file_handler)
        return build_logger

    def run(self, argv: Sequence[str] | None = None) -> dict[str, object]:
        config = self.parse_config(argv)
        build_logger = self._build_logger(config.sample_root)
        return ComplexSampleValidator(config, logger=build_logger).validate()


def main() -> None:
    ValidateComplexSamplesCLI().run()


if __name__ == "__main__":
    main()
