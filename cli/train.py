from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import torch

from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CompileConfig,
    CouplingBranchFusionConfig,
    CouplingBestRelSolCheckpointConfig,
    CouplingCoefficientTermsConfig,
    CouplingGeometryBranchConfig,
    CouplingLossesConfig,
    CouplingLossTermConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingTrainingConfig,
    CouplingTrunkPositionalEncodingConfig,
    DatasetConfig,
    ComplexCanonicalEnergyConfig,
    GreenResponseFeatureConfig,
    IndexedGpSourceConfig,
    ModelConfig,
    PipelineConfig,
    SourceStencilLiftConfig,
    TerminalConfig,
    TrainingConfig,
    reject_retired_coupling_training_options,
    validate_active_training_seeds,
    validate_complex_coupling_source_config,
)
from greenonet.compile_utils import maybe_compile_model, model_state_dict_for_save
from greenonet.coefficients import (
    CoefficientFunction,
    CoefficientFunctions,
    load_coefficient_functions,
)
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_coupling_trainer import ComplexCouplingTrainer
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_tangent_geometry_selection import (
    GeometryTangentDimensionResolver,
    materialized_tangent_config,
)
from greenonet.complex_sources import (
    GeometryGridLoader,
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
)
from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule
from greenonet.coupling_model import CouplingNet
from greenonet.coupling_optimizer import ComplexCouplingOptimizerFactory
from greenonet.coupling_trainer import CouplingTrainer
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.green_lr_scheduler import GreenLearningRateSchedule
from greenonet.green_optimizer import GreenOptimizerFactory
from greenonet.model import GreenONetModel
from greenonet.runner import run_complex_green_o_net, run_green_o_net
from greenonet.runtime import apply_runtime_cpu_settings, write_runtime_cpu_summary
from greenonet.reproducibility import TrainingSeedContext


class TrainCLI:
    """Command-line entrypoint for model training."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Train GreenONet and CouplingNet.")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/default.json",
            help="Path to JSON configuration file.",
        )
        parser.add_argument(
            "--work-dir",
            type=str,
            default="checkpoints/run",
            help="Directory to store logs and artifacts.",
        )
        parser.add_argument(
            "--tangent-context",
            type=Path,
            default=None,
            help=(
                "Optional tangent response context sidecar override. Requires "
                "tangent_context_checkpoint.enabled=true."
            ),
        )
        self.parser = parser

    def _build_configs(
        self, config_path: Path
    ) -> tuple[
        DatasetConfig,
        ModelConfig,
        TrainingConfig,
        CouplingModelConfig,
        CouplingTrainingConfig,
        PipelineConfig,
        TerminalConfig,
    ]:
        with config_path.open() as fp:
            raw = json.load(fp)
        terminal_cfg = self._build_terminal_config(raw.get("terminal"))
        dataset_cfg = DatasetConfig.from_raw(raw["dataset"])

        # model_kwargs = dict(raw["model"])
        # model_dtype = model_kwargs.pop("dtype", "float64")
        # model_kwargs["dtype"] = getattr(torch, model_dtype)
        # model_cfg = ModelConfig(**model_kwargs)
        # training_cfg = TrainingConfig(**raw["training"])
        model_kwargs = dict(raw.get("model", {}))
        model_dtype = model_kwargs.pop("dtype", "float64")
        model_kwargs["dtype"] = getattr(torch, model_dtype)
        model_cfg = ModelConfig(**model_kwargs)
        training_cfg = self._build_training_config(raw.get("training", {}))

        coupling_model_kwargs = dict(raw.get("coupling_model", {}))
        if "coupler" in coupling_model_kwargs:
            raise TypeError("coupling_model.coupler has been removed.")
        source_lift_raw = coupling_model_kwargs.pop("source_stencil_lift", None)
        source_lift_cfg = self._build_source_stencil_lift_config(
            source_lift_raw,
            "coupling_model",
        )
        coefficient_terms_raw = coupling_model_kwargs.pop("coefficient_terms", None)
        coefficient_terms_cfg = self._build_coefficient_terms_config(
            coefficient_terms_raw,
            "coupling_model",
        )
        branch_fusion_raw = coupling_model_kwargs.pop("branch_fusion", None)
        branch_fusion_cfg = self._build_branch_fusion_config(
            branch_fusion_raw,
            "coupling_model",
        )
        geometry_branch_raw = coupling_model_kwargs.pop("geometry_branch", None)
        geometry_branch_cfg = self._build_geometry_branch_config(
            geometry_branch_raw,
            "coupling_model",
        )
        green_response_raw = coupling_model_kwargs.pop("green_response_feature", None)
        green_response_cfg = self._build_green_response_feature_config(
            green_response_raw,
            "coupling_model",
        )
        positional_raw = coupling_model_kwargs.pop("trunk_positional_encoding", None)
        positional_cfg = self._build_trunk_positional_encoding_config(
            positional_raw,
            "coupling_model",
        )
        axis_1d_trunk_raw = coupling_model_kwargs.pop("axis_1d_trunk", None)
        axis_1d_trunk_cfg = self._build_axis_1d_trunk_config(
            axis_1d_trunk_raw,
            "coupling_model",
        )
        balance_projection_raw = coupling_model_kwargs.pop("balance_projection", None)
        balance_projection_cfg = self._build_balance_projection_config(
            balance_projection_raw,
            "coupling_model",
        )
        cm_dtype = coupling_model_kwargs.pop("dtype", "float64")
        coupling_model_kwargs["dtype"] = getattr(torch, cm_dtype)
        coupling_model_cfg = CouplingModelConfig(
            balance_projection=balance_projection_cfg,
            source_stencil_lift=source_lift_cfg,
            coefficient_terms=coefficient_terms_cfg,
            branch_fusion=branch_fusion_cfg,
            geometry_branch=geometry_branch_cfg,
            green_response_feature=green_response_cfg,
            trunk_positional_encoding=positional_cfg,
            axis_1d_trunk=axis_1d_trunk_cfg,
            **coupling_model_kwargs,
        )

        coupling_training_cfg = self._build_coupling_training_config(
            raw.get("coupling_training", {})
        )
        pipeline_cfg = PipelineConfig(**raw.get("pipeline", {}))
        return (
            dataset_cfg,
            model_cfg,
            training_cfg,
            coupling_model_cfg,
            coupling_training_cfg,
            pipeline_cfg,
            terminal_cfg,
        )

    @staticmethod
    def _build_terminal_config(raw_terminal: object | None) -> TerminalConfig:
        if raw_terminal is None:
            return TerminalConfig()
        if not isinstance(raw_terminal, dict):
            raise TypeError("terminal must be an object.")
        return TerminalConfig(**dict(raw_terminal))

    @staticmethod
    def _active_device_types(
        training_cfg: TrainingConfig,
        coupling_training_cfg: CouplingTrainingConfig,
        pipeline_cfg: PipelineConfig,
    ) -> set[str]:
        active_device_types: set[str] = set()
        if pipeline_cfg.run_green:
            active_device_types.add(torch.device(training_cfg.device).type)
        if pipeline_cfg.run_coupling:
            active_device_types.add(torch.device(coupling_training_cfg.device).type)
        return active_device_types

    @staticmethod
    def _write_config_used(
        *,
        config_path: Path,
        work_dir: Path,
        dataset_cfg: DatasetConfig,
        training_cfg: TrainingConfig | None = None,
        coupling_training_cfg: CouplingTrainingConfig,
        coupling_model_cfg: CouplingModelConfig,
        pipeline_cfg: PipelineConfig,
        tangent_dimension_provenance: dict[str, object] | None = None,
    ) -> None:
        destination = work_dir / "config_used.json"
        materialize_green = pipeline_cfg.run_green
        materialize_coupling = pipeline_cfg.run_coupling
        materialize_complex_coupling = (
            dataset_cfg.geometry_mode == "complex" and materialize_coupling
        )
        if not materialize_green and not materialize_coupling:
            shutil.copy2(config_path, destination)
            return
        with config_path.open() as fp:
            payload = json.load(fp)
        if materialize_green:
            if training_cfg is None:
                raise ValueError(
                    "training_cfg is required when pipeline.run_green is true."
                )
            training = payload.setdefault("training", {})
            if not isinstance(training, dict):
                raise TypeError("training must be an object.")
            green_factory = GreenOptimizerFactory(training_cfg)
            training["optimizer"] = green_factory.resolved_config()
            training["seed"] = training_cfg.seed
            training["deterministic_algorithms"] = training_cfg.deterministic_algorithms
            payload["green_optimizer_provenance"] = green_factory.provenance().as_dict()
            payload["green_learning_rate_schedule"] = (
                GreenLearningRateSchedule.configured_config(training_cfg)
            )
            if training_cfg.seed is not None:
                payload["green_training_seed_provenance"] = TrainingSeedContext(
                    stage="green",
                    base_seed=training_cfg.seed,
                    deterministic_algorithms=training_cfg.deterministic_algorithms,
                    device=training_cfg.device,
                ).as_dict()

        if materialize_coupling:
            coupling_training = payload.setdefault("coupling_training", {})
            if not isinstance(coupling_training, dict):
                raise TypeError("coupling_training must be an object.")
            coupling_training["seed"] = coupling_training_cfg.seed
            coupling_training["deterministic_algorithms"] = (
                coupling_training_cfg.deterministic_algorithms
            )
            payload["coupling_learning_rate_schedule"] = (
                CouplingLearningRateSchedule.configured_config(coupling_training_cfg)
            )
            if coupling_training_cfg.seed is not None:
                payload["coupling_training_seed_provenance"] = TrainingSeedContext(
                    stage="coupling",
                    base_seed=coupling_training_cfg.seed,
                    deterministic_algorithms=(
                        coupling_training_cfg.deterministic_algorithms
                    ),
                    device=coupling_training_cfg.device,
                ).as_dict()

        if materialize_complex_coupling:
            coupling_training = payload["coupling_training"]
            assert isinstance(coupling_training, dict)
            factory = ComplexCouplingOptimizerFactory(coupling_training_cfg)
            coupling_training["optimizer"] = factory.resolved_config()
            canonical_energy = ComplexCanonicalEnergyConfig.from_raw(
                coupling_training_cfg.canonical_energy
            )
            coupling_training["canonical_energy"] = asdict(canonical_energy)
            payload["optimizer_provenance"] = factory.provenance().as_dict()
            dataset = payload.setdefault("dataset", {})
            if not isinstance(dataset, dict):
                raise TypeError("dataset must be an object.")
            dataset["coupling_source"] = asdict(dataset_cfg.coupling_source)
            dataset["reference_diagnostics"] = asdict(dataset_cfg.reference_diagnostics)
            payload["complex_source_provenance"] = {
                "fixed_across_epochs": True,
                "backend": dataset_cfg.coupling_source.mode,
                "sample_identity": "base_seed_split_id_sample_index",
                "test_reference_backend": "npz",
                "training_seed_independent": True,
                "indexed_gp_base_seed": (
                    dataset_cfg.coupling_source.indexed_gp.seed
                    if dataset_cfg.coupling_source.mode == "indexed_gp"
                    else None
                ),
            }
            if tangent_dimension_provenance is not None:
                coupling_model = payload.setdefault("coupling_model", {})
                if not isinstance(coupling_model, dict):
                    raise TypeError("coupling_model must be an object.")
                projection = coupling_model.setdefault("balance_projection", {})
                if not isinstance(projection, dict):
                    raise TypeError(
                        "coupling_model.balance_projection must be an object."
                    )
                projection["symmetric_tangent_green_response"] = (
                    materialized_tangent_config(coupling_model_cfg)
                )
                payload["tangent_subspace_dimension_provenance"] = (
                    tangent_dimension_provenance
                )
        destination.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def _should_apply_cpu_runtime(
        cls,
        training_cfg: TrainingConfig,
        coupling_training_cfg: CouplingTrainingConfig,
        pipeline_cfg: PipelineConfig,
    ) -> bool:
        return "cpu" in cls._active_device_types(
            training_cfg,
            coupling_training_cfg,
            pipeline_cfg,
        )

    @staticmethod
    def _build_compile_config(
        raw_compile: object | None, section_name: str
    ) -> CompileConfig:
        if raw_compile is None:
            return CompileConfig()
        if isinstance(raw_compile, dict):
            return CompileConfig(**raw_compile)
        raise TypeError(f"{section_name}.compile must be an object.")

    @staticmethod
    def _build_source_stencil_lift_config(
        raw_source_lift: object | None,
        section_name: str,
    ) -> SourceStencilLiftConfig:
        if raw_source_lift is None:
            return SourceStencilLiftConfig()
        if not isinstance(raw_source_lift, dict):
            raise TypeError(f"{section_name}.source_stencil_lift must be an object.")
        return SourceStencilLiftConfig(**dict(raw_source_lift))

    @staticmethod
    def _build_green_response_feature_config(
        raw_green_response: object | None,
        section_name: str,
    ) -> GreenResponseFeatureConfig:
        if raw_green_response is None:
            return GreenResponseFeatureConfig()
        if not isinstance(raw_green_response, dict):
            raise TypeError(f"{section_name}.green_response_feature must be an object.")
        return GreenResponseFeatureConfig(**dict(raw_green_response))

    @staticmethod
    def _build_coefficient_terms_config(
        raw_coefficient_terms: object | None,
        section_name: str,
    ) -> CouplingCoefficientTermsConfig:
        if raw_coefficient_terms is None:
            return CouplingCoefficientTermsConfig()
        if not isinstance(raw_coefficient_terms, dict):
            raise TypeError(f"{section_name}.coefficient_terms must be an object.")
        return CouplingCoefficientTermsConfig(**dict(raw_coefficient_terms))

    @staticmethod
    def _build_branch_fusion_config(
        raw_branch_fusion: object | None,
        section_name: str,
    ) -> CouplingBranchFusionConfig:
        try:
            return CouplingBranchFusionConfig.from_raw(raw_branch_fusion)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{section_name}.{exc}") from exc

    @staticmethod
    def _build_geometry_branch_config(
        raw_geometry_branch: object | None,
        section_name: str,
    ) -> CouplingGeometryBranchConfig:
        try:
            return CouplingGeometryBranchConfig.from_raw(raw_geometry_branch)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{section_name}.{exc}") from exc

    @staticmethod
    def _build_trunk_positional_encoding_config(
        raw_positional: object | None,
        section_name: str,
    ) -> CouplingTrunkPositionalEncodingConfig:
        if raw_positional is None:
            return CouplingTrunkPositionalEncodingConfig()
        if not isinstance(raw_positional, dict):
            raise TypeError(
                f"{section_name}.trunk_positional_encoding must be an object."
            )
        return CouplingTrunkPositionalEncodingConfig(**dict(raw_positional))

    @staticmethod
    def _build_axis_1d_trunk_config(
        raw_axis_1d_trunk: object | None,
        section_name: str,
    ) -> Axis1DTrunkConfig:
        try:
            return Axis1DTrunkConfig.from_raw(raw_axis_1d_trunk)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{section_name}.{exc}") from exc

    @staticmethod
    def _build_balance_projection_config(
        raw_balance_projection: object | None,
        section_name: str,
    ) -> BalanceProjectionConfig:
        try:
            return BalanceProjectionConfig.from_raw(raw_balance_projection)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{section_name}.{exc}") from exc

    @classmethod
    def _build_training_config(
        cls,
        raw_training: dict[str, object],
    ) -> TrainingConfig:
        training_kwargs = dict(raw_training)
        compile_raw = training_kwargs.pop("compile", None)
        compile_cfg = cls._build_compile_config(compile_raw, "training")
        config = TrainingConfig(compile=compile_cfg, **training_kwargs)
        GreenOptimizerFactory(config)
        GreenLearningRateSchedule.validate_config(config)
        return config

    @staticmethod
    def _build_coupling_training_config(
        raw_training: dict[str, object],
    ) -> CouplingTrainingConfig:
        coupling_training_kwargs = dict(raw_training)
        deprecated_loss_keys = {
            "lambda_consistency",
            "flux_consistency_enabled",
            "lambda_flux_consistency",
            "energy_consistency_enabled",
            "lambda_energy_consistency",
        }
        found_deprecated = sorted(
            key for key in deprecated_loss_keys if key in coupling_training_kwargs
        )
        if found_deprecated:
            raise TypeError(
                "deprecated flat coupling loss config is not supported; use "
                "coupling_training.losses.* instead "
                f"({', '.join(found_deprecated)})."
            )
        if "hybrid_detach" in coupling_training_kwargs:
            raise TypeError("coupling_training.hybrid_detach has been removed.")
        if "stage2" in coupling_training_kwargs:
            raise TypeError("coupling_training.stage2 has been removed.")
        reject_retired_coupling_training_options(coupling_training_kwargs)
        losses_raw = coupling_training_kwargs.pop("losses", None)
        compile_raw = coupling_training_kwargs.pop("compile", None)
        periodic_raw = coupling_training_kwargs.pop("periodic_checkpoint", None)
        best_rel_sol_raw = coupling_training_kwargs.pop("best_rel_sol_checkpoint", None)
        losses_cfg = TrainCLI._build_coupling_losses_config(
            losses_raw, "coupling_training"
        )
        compile_cfg = TrainCLI._build_compile_config(compile_raw, "coupling_training")
        if periodic_raw is None:
            periodic_cfg = CouplingPeriodicCheckpointConfig()
        elif isinstance(periodic_raw, dict):
            periodic_cfg = CouplingPeriodicCheckpointConfig(**periodic_raw)
        else:
            raise TypeError("coupling_training.periodic_checkpoint must be an object.")
        if best_rel_sol_raw is None:
            best_rel_sol_cfg = CouplingBestRelSolCheckpointConfig()
        elif isinstance(best_rel_sol_raw, dict):
            best_rel_sol_cfg = CouplingBestRelSolCheckpointConfig(**best_rel_sol_raw)
        else:
            raise TypeError(
                "coupling_training.best_rel_sol_checkpoint must be an object."
            )
        config = CouplingTrainingConfig(
            losses=losses_cfg,
            compile=compile_cfg,
            periodic_checkpoint=periodic_cfg,
            best_rel_sol_checkpoint=best_rel_sol_cfg,
            **coupling_training_kwargs,
        )
        CouplingLearningRateSchedule.validate_config(config)
        return config

    @staticmethod
    def _build_coupling_losses_config(
        raw_losses: object | None, section_name: str
    ) -> CouplingLossesConfig:
        if raw_losses is None:
            return CouplingLossesConfig()
        if not isinstance(raw_losses, dict):
            raise TypeError(f"{section_name}.losses must be an object.")

        loss_kwargs = dict(raw_losses)
        defaults = CouplingLossesConfig()
        parsed: dict[str, CouplingLossTermConfig] = {}
        for key in (
            "l2_consistency",
            "energy_consistency",
            "cross_consistency",
            "balance_loss",
            "symmetric_boundary_loss",
        ):
            raw_term = loss_kwargs.pop(key, None)
            if raw_term is None:
                parsed[key] = getattr(defaults, key)
            elif isinstance(raw_term, dict):
                parsed[key] = CouplingLossTermConfig(**raw_term)
            else:
                raise TypeError(f"{section_name}.losses.{key} must be an object.")
        if loss_kwargs:
            unknown = ", ".join(sorted(loss_kwargs))
            raise TypeError(f"{section_name}.losses has unknown keys: {unknown}.")
        return CouplingLossesConfig(**parsed)

    @staticmethod
    def _coeff_from_coords(
        coords: torch.Tensor, fn: CoefficientFunction
    ) -> torch.Tensor:
        x = coords[..., 0]
        y = coords[..., 1]
        return fn(x, y)

    @staticmethod
    def _convection_from_coords(
        coords: torch.Tensor,
        bx_fun: CoefficientFunction,
        by_fun: CoefficientFunction,
    ) -> torch.Tensor:
        if coords.shape[0] != 2:
            raise ValueError("Expected coords first dimension to contain x/y axes.")
        bx = bx_fun(coords[0, ..., 0], coords[0, ..., 1])
        by = by_fun(coords[1, ..., 0], coords[1, ..., 1])
        return torch.stack((bx, by), dim=0)

    @staticmethod
    def _compute_green_kernel(
        model: GreenONetModel,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor | None = None,
        b_vals: torch.Tensor | None = None,
        c_vals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Green kernel (2,n,m,m) using provided coefficients per axial line."""
        m_points = coords.shape[2]
        device = next(model.parameters()).device
        trunk_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, m_points, device=device),
                torch.linspace(0.0, 1.0, m_points, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        if a_vals.dim() == 3:
            a_vals = a_vals.unsqueeze(0)
        if ap_vals is None:
            ap_vals = torch.zeros_like(a_vals)
        if b_vals is None:
            b_vals = torch.zeros_like(a_vals)
        if c_vals is None:
            c_vals = torch.zeros_like(a_vals)
        if ap_vals.dim() == 3:
            ap_vals = ap_vals.unsqueeze(0)
        if b_vals.dim() == 3:
            b_vals = b_vals.unsqueeze(0)
        if c_vals.dim() == 3:
            c_vals = c_vals.unsqueeze(0)
        with torch.no_grad():
            kernel = model(
                trunk_grid=trunk_grid,
                a_vals=a_vals.to(device),
                ap_vals=ap_vals.to(device),
                b_vals=b_vals.to(device),
                c_vals=c_vals.to(device),
            )  # (2,n,m,m)
        # from greenonet.greens import ExactGreenFunction
        # kernel_class = ExactGreenFunction(torch.linspace(0, 1, m_points, device=device), a=a_vals.to(device))
        # kernel0 = kernel_class().squeeze(0)
        # kernel0 = kernel_class.poisson().unsqueeze(0).unsqueeze(0).expand_as(kernel)
        return kernel.cpu()

    def _run_complex_coupling(
        self,
        *,
        dataset_cfg: DatasetConfig,
        coupling_model_cfg: CouplingModelConfig,
        coupling_training_cfg: CouplingTrainingConfig,
        pipeline_cfg: PipelineConfig,
        terminal_cfg: TerminalConfig,
        coeffs: CoefficientFunctions,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        work_dir: Path,
        tangent_context_path: Path | None,
    ) -> None:
        if coupling_training_cfg.seed is None:
            raise ValueError(
                "coupling_training.seed is required for complex CouplingNet training."
            )
        seed_context = TrainingSeedContext(
            stage="coupling",
            base_seed=coupling_training_cfg.seed,
            deterministic_algorithms=coupling_training_cfg.deterministic_algorithms,
            device=coupling_training_cfg.device,
        )
        seed_context.configure_process()
        if dataset_cfg.geometry_path is None:
            raise ValueError("dataset.geometry_path is required for complex mode.")
        validate_complex_coupling_source_config(
            dataset_cfg,
            coupling_training_cfg,
        )
        train_dataset = self._build_complex_source_dataset(
            split="train",
            dataset_cfg=dataset_cfg,
            coupling_model_cfg=coupling_model_cfg,
            coupling_training_cfg=coupling_training_cfg,
            geometry=geometry,
            coeffs=coeffs,
        )
        if train_dataset is None:
            raise RuntimeError("Complex training source unexpectedly resolved to none.")
        validation_dataset = self._build_complex_source_dataset(
            split="valid",
            dataset_cfg=dataset_cfg,
            coupling_model_cfg=coupling_model_cfg,
            coupling_training_cfg=coupling_training_cfg,
            geometry=geometry,
            coeffs=coeffs,
        )
        test_dataset = None
        if dataset_cfg.test_path is not None:
            test_dataset = ComplexCouplingDataset(
                dataset_cfg.test_path,
                geometry,
                coeffs,
                branch_input_dim=coupling_model_cfg.branch_input_dim,
                dtype=dataset_cfg.dtype,
                coefficient_terms=coupling_model_cfg.coefficient_terms,
                integration_rule=coupling_training_cfg.integration_rule,
                reference_diagnostics=True,
            )

        seed_context.apply("model")
        coupling_model = ComplexCouplingNet(coupling_model_cfg)
        if pipeline_cfg.coupling_pretrained_path is not None:
            load_state_dict_auto(coupling_model, pipeline_cfg.coupling_pretrained_path)
        seed_context.apply("runtime")
        trainer = ComplexCouplingTrainer(
            model=coupling_model,
            config=coupling_training_cfg,
            work_dir=work_dir,
            green_model=green_model,
            terminal_width=terminal_cfg.width,
            seed_context=seed_context,
            tangent_context_path=tangent_context_path,
        )
        trainer.train(train_dataset, validation_dataset)
        if test_dataset is not None:
            evaluator = ComplexCouplingEvaluator(
                model=coupling_model,
                green_model=green_model,
                config=coupling_training_cfg,
                device=torch.device(coupling_training_cfg.device),
                work_dir=work_dir,
                terminal_width=terminal_cfg.width,
                tangent_context_path=tangent_context_path,
                tangent_context_default_path=(
                    work_dir / "tangent_response_context.safetensors"
                ),
            )
            evaluator.evaluate(
                test_dataset,
                dataset_name="test",
                batch_size=coupling_training_cfg.batch_size,
            )

    @staticmethod
    def _build_complex_source_dataset(
        *,
        split: Literal["train", "valid"],
        dataset_cfg: DatasetConfig,
        coupling_model_cfg: CouplingModelConfig,
        coupling_training_cfg: CouplingTrainingConfig,
        geometry: ComplexGeometryMetadata,
        coeffs: CoefficientFunctions,
    ) -> ComplexCouplingDataset | None:
        source = dataset_cfg.coupling_source
        diagnostics = dataset_cfg.reference_diagnostics
        if source.mode == "npz":
            data_dir = (
                dataset_cfg.training_path
                if split == "train"
                else dataset_cfg.validation_path
            )
            if data_dir is None:
                if split == "train":
                    raise ValueError(
                        "dataset.training_path is required for NPZ training."
                    )
                return None
            return ComplexCouplingDataset(
                data_dir,
                geometry,
                coeffs,
                branch_input_dim=coupling_model_cfg.branch_input_dim,
                dtype=dataset_cfg.dtype,
                coefficient_terms=coupling_model_cfg.coefficient_terms,
                integration_rule=coupling_training_cfg.integration_rule,
                reference_diagnostics=(
                    diagnostics.training if split == "train" else diagnostics.validation
                ),
            )

        if dataset_cfg.geometry_path is None:
            raise ValueError("dataset.geometry_path is required for indexed GP.")
        indexed = cast(IndexedGpSourceConfig, source.indexed_gp)
        count = indexed.num_train if split == "train" else indexed.num_valid
        if count == 0:
            return None
        raw_geometry = GeometryGridLoader().load(dataset_cfg.geometry_path)
        provider = IndexedGpComplexSourceProvider(
            raw_geometry,
            split=split,
            sample_count=count,
            parameters=IndexedGpParameters(
                seed=indexed.seed,
                lengthscale=indexed.lengthscale,
                amplitude=indexed.amplitude,
                mean=indexed.mean,
            ),
        )
        return ComplexCouplingDataset(
            None,
            geometry,
            coeffs,
            branch_input_dim=coupling_model_cfg.branch_input_dim,
            dtype=dataset_cfg.dtype,
            coefficient_terms=coupling_model_cfg.coefficient_terms,
            integration_rule=coupling_training_cfg.integration_rule,
            reference_diagnostics=False,
            source_provider=provider,
        )

    def run(self) -> None:
        args = self.parser.parse_args()
        config_path = Path(args.config)
        (
            dataset_cfg,
            model_cfg,
            training_cfg,
            coupling_model_cfg,
            coupling_training_cfg,
            pipeline_cfg,
            terminal_cfg,
        ) = self._build_configs(config_path)

        validate_active_training_seeds(
            training=training_cfg,
            coupling_training=coupling_training_cfg,
            pipeline=pipeline_cfg,
        )

        if pipeline_cfg.run_coupling:
            validate_complex_coupling_source_config(
                dataset_cfg,
                coupling_training_cfg,
            )
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        complex_geometry: ComplexGeometryMetadata | None = None
        tangent_dimension_provenance: dict[str, object] | None = None
        if pipeline_cfg.run_coupling and dataset_cfg.geometry_mode == "complex":
            if dataset_cfg.geometry_path is None:
                raise ValueError("dataset.geometry_path is required for complex mode.")
            complex_geometry = load_complex_geometry(
                dataset_cfg.geometry_path,
                dtype=dataset_cfg.dtype,
            )
            resolution = GeometryTangentDimensionResolver.resolve(
                model_config=coupling_model_cfg,
                geometry=complex_geometry,
                geometry_path=dataset_cfg.geometry_path,
            )
            coupling_model_cfg = resolution.model_config
            tangent_dimension_provenance = resolution.provenance
        self._write_config_used(
            config_path=config_path,
            work_dir=work_dir,
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            coupling_training_cfg=coupling_training_cfg,
            coupling_model_cfg=coupling_model_cfg,
            pipeline_cfg=pipeline_cfg,
            tangent_dimension_provenance=tangent_dimension_provenance,
        )
        if self._should_apply_cpu_runtime(
            training_cfg,
            coupling_training_cfg,
            pipeline_cfg,
        ):
            runtime_state = apply_runtime_cpu_settings()
            write_runtime_cpu_summary(work_dir, runtime_state)

        coeffs = load_coefficient_functions(dataset_cfg.coefficient_functions_path)

        green_model = GreenONetModel(model_cfg)
        green_kernel: torch.Tensor | None = None

        if pipeline_cfg.run_green:
            if dataset_cfg.geometry_mode == "complex":
                if dataset_cfg.geometry_path is None:
                    raise ValueError(
                        "dataset.geometry_path is required for complex GreenNet training."
                    )
                green_trainer = run_complex_green_o_net(
                    coeffs=coeffs,
                    geometry_path=dataset_cfg.geometry_path,
                    activation=model_cfg.activation,
                    work_dir=work_dir,
                    ndata=dataset_cfg.samples_per_line,
                    validation_ndata=dataset_cfg.validation_samples_per_line,
                    seed=cast(int, training_cfg.seed),
                    scale_length=dataset_cfg.scale_length,
                    validation_scale_length=dataset_cfg.validation_scale_length,
                    deterministic=dataset_cfg.deterministic,
                    sampler_mode=dataset_cfg.sampler_mode,
                    validation_sampler_mode=dataset_cfg.validation_sampler_mode,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    terminal_width=terminal_cfg.width,
                )
            else:
                green_trainer = run_green_o_net(
                    a_fun=coeffs.a_fun,
                    apx_fun=coeffs.apx_fun,
                    apy_fun=coeffs.apy_fun,
                    bx_fun=coeffs.bx_fun,
                    by_fun=coeffs.by_fun,
                    c_fun=coeffs.c_fun,
                    activation=model_cfg.activation,
                    work_dir=work_dir,
                    ndata=dataset_cfg.samples_per_line,
                    validation_ndata=dataset_cfg.validation_samples_per_line,
                    seed=cast(int, training_cfg.seed),
                    scale_length=dataset_cfg.scale_length,
                    validation_scale_length=dataset_cfg.validation_scale_length,
                    use_operator_learning=dataset_cfg.use_operator_learning,
                    deterministic=dataset_cfg.deterministic,
                    sampler_mode=dataset_cfg.sampler_mode,
                    validation_sampler_mode=dataset_cfg.validation_sampler_mode,
                    n_points_per_line=model_cfg.branch_input_dim,
                    step_size=dataset_cfg.step_size,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    terminal_width=terminal_cfg.width,
                )
            green_model.load_state_dict(model_state_dict_for_save(green_trainer.model))
        else:
            if pipeline_cfg.green_pretrained_path is None:
                raise ValueError(
                    "green_pretrained_path must be set when not training GreenONet"
                )
            try:
                green_model, model_cfg = load_model_with_config(
                    pipeline_cfg.green_pretrained_path
                )
            except Exception:
                load_state_dict_auto(green_model, pipeline_cfg.green_pretrained_path)

        if pipeline_cfg.run_coupling:
            if dataset_cfg.geometry_mode == "complex":
                if complex_geometry is None:
                    raise RuntimeError(
                        "Complex geometry was not resolved before training."
                    )
                self._run_complex_coupling(
                    dataset_cfg=dataset_cfg,
                    coupling_model_cfg=coupling_model_cfg,
                    coupling_training_cfg=coupling_training_cfg,
                    pipeline_cfg=pipeline_cfg,
                    terminal_cfg=terminal_cfg,
                    coeffs=coeffs,
                    green_model=green_model,
                    geometry=complex_geometry,
                    work_dir=work_dir,
                    tangent_context_path=args.tangent_context,
                )
                return
            if coupling_training_cfg.seed is None:
                raise ValueError(
                    "coupling_training.seed is required for CouplingNet training."
                )
            coupling_seed_context = TrainingSeedContext(
                stage="coupling",
                base_seed=coupling_training_cfg.seed,
                deterministic_algorithms=(
                    coupling_training_cfg.deterministic_algorithms
                ),
                device=coupling_training_cfg.device,
            )
            coupling_seed_context.configure_process()
            train_dir = dataset_cfg.training_path or Path("2D_data_variable")
            val_dir = dataset_cfg.validation_path
            coupling_train_dataset = CouplingDataset(
                data_dir=train_dir,
                step_size=dataset_cfg.step_size,
                n_points_per_line=dataset_cfg.n_points_per_line,
                dtype=dataset_cfg.dtype,
                integration_rule=coupling_training_cfg.integration_rule,
                a_fun=coeffs.a_fun,
                bx_fun=coeffs.bx_fun,
                by_fun=coeffs.by_fun,
                c_fun=coeffs.c_fun,
                ap_fun_x=coeffs.apx_fun,
                ap_fun_y=coeffs.apy_fun,
            )
            coupling_val_dataset = None
            if val_dir is not None:
                coupling_val_dataset = CouplingDataset(
                    data_dir=val_dir,
                    step_size=dataset_cfg.step_size,
                    n_points_per_line=dataset_cfg.n_points_per_line,
                    dtype=dataset_cfg.dtype,
                    integration_rule=coupling_training_cfg.integration_rule,
                    a_fun=coeffs.a_fun,
                    bx_fun=coeffs.bx_fun,
                    by_fun=coeffs.by_fun,
                    c_fun=coeffs.c_fun,
                    ap_fun_x=coeffs.apx_fun,
                    ap_fun_y=coeffs.apy_fun,
                )
            test_dir = dataset_cfg.test_path
            coupling_test_dataset = None
            if test_dir is not None:
                coupling_test_dataset = CouplingDataset(
                    data_dir=test_dir,
                    step_size=dataset_cfg.step_size,
                    n_points_per_line=dataset_cfg.n_points_per_line,
                    dtype=dataset_cfg.dtype,
                    integration_rule=coupling_training_cfg.integration_rule,
                    a_fun=coeffs.a_fun,
                    bx_fun=coeffs.bx_fun,
                    by_fun=coeffs.by_fun,
                    c_fun=coeffs.c_fun,
                    ap_fun_x=coeffs.apx_fun,
                    ap_fun_y=coeffs.apy_fun,
                )
            if green_kernel is None:
                sample = coupling_train_dataset[0]
                sample_coords = sample[0]
                sample_kappa = sample[6]
                sample_ap = sample[9]
                device = torch.device(coupling_training_cfg.device)
                sample_b = self._convection_from_coords(
                    sample_coords,
                    bx_fun=coeffs.bx_fun,
                    by_fun=coeffs.by_fun,
                )
                sample_c = self._coeff_from_coords(sample_coords, coeffs.c_fun)
                green_model = maybe_compile_model(
                    green_model.to(device),
                    training_cfg.compile,
                    model_name="GreenONetModel",
                )
                green_kernel = self._compute_green_kernel(
                    green_model,
                    sample_coords.to(device),
                    a_vals=sample_kappa.to(device),
                    ap_vals=sample_ap.to(device),
                    b_vals=sample_b.to(device),
                    c_vals=sample_c.to(device),
                )
            coupling_seed_context.apply("model")
            coupling_model = CouplingNet(coupling_model_cfg)
            if pipeline_cfg.coupling_pretrained_path is not None:
                try:
                    coupling_model, coupling_model_cfg = load_model_with_config(
                        pipeline_cfg.coupling_pretrained_path
                    )
                except Exception:
                    load_state_dict_auto(
                        coupling_model, pipeline_cfg.coupling_pretrained_path
                    )
            coupling_seed_context.apply("runtime")
            coupling_trainer = CouplingTrainer(
                model=coupling_model,
                config=coupling_training_cfg,
                work_dir=work_dir,
                green_kernel=green_kernel,
                model_cfg=coupling_model_cfg,
                terminal_width=terminal_cfg.width,
                seed_context=coupling_seed_context,
            )
            coupling_trainer.train(coupling_train_dataset, coupling_val_dataset)
            if coupling_test_dataset is not None:
                evaluator = CouplingEvaluator(
                    model=coupling_model,
                    green_kernel=green_kernel,
                    device=torch.device(coupling_training_cfg.device),
                    work_dir=work_dir,
                    integration_rule=coupling_training_cfg.integration_rule,
                    terminal_width=terminal_cfg.width,
                )
                evaluator.evaluate(
                    coupling_test_dataset,
                    dataset_name="test",
                    batch_size=coupling_training_cfg.batch_size,
                )


if __name__ == "__main__":
    TrainCLI().run()
