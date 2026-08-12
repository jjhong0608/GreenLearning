from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import torch
from torch import Tensor

from greenonet.axial import AxialLines, make_square_axial_lines
from greenonet.backward_sampler import BackwardSampler
from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_data import (
    ComplexGreenDataset,
    generate_complex_green_data,
)
from greenonet.complex_green_trainer import ComplexGreenTrainer
from greenonet.config import ModelConfig, TrainingConfig
from greenonet.data import AxialDataset
from greenonet.logging_mixin import LoggingMixin
from greenonet.model import GreenONetModel
from greenonet.reproducibility import TrainingSeedContext
from greenonet.sampler import ForwardSampler
from greenonet.trainer import Trainer


class GreenONetRunner(LoggingMixin):
    """Cleaned runner inspired by the original GreenONet script."""

    def __init__(
        self,
        work_dir: Path | str,
        terminal_width: int | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.terminal_width = terminal_width
        super().__init__(
            logger_name="GreenONetRunner",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )

    def _build_axial_lines(
        self,
        use_operator_learning: bool,
        step_size: float | None = None,
        n_points_per_line: int | None = None,
    ) -> AxialLines:
        if step_size is None:
            step_size = 0.25 if use_operator_learning else 0.5
        return make_square_axial_lines(
            step_size=step_size, n_points_per_line=n_points_per_line
        )

    def run(
        self,
        a_fun: Callable[[Tensor, Tensor], Tensor],
        apx_fun: Callable[[Tensor, Tensor], Tensor],
        apy_fun: Callable[[Tensor, Tensor], Tensor],
        activation: Literal["tanh", "relu", "gelu", "rational"],
        ndata: int,
        validation_ndata: int,
        seed: int,
        scale_length: float | tuple[float, float],
        validation_scale_length: float | tuple[float, float] | None,
        use_operator_learning: bool,
        deterministic: bool,
        b_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
        c_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
        b_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
        bx_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
        by_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
        c_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
        step_size: float | None = None,
        n_points_per_line: int | None = None,
        sampler_mode: Literal["forward", "backward"] = "forward",
        validation_sampler_mode: Literal["forward", "backward"] | None = None,
        model_cfg: ModelConfig | None = None,
        training_cfg: TrainingConfig | None = None,
    ) -> Trainer:
        cfg_training = training_cfg or TrainingConfig(
            learning_rate=5e-3,
            epochs=max(5, ndata * 2),
            batch_size=32,
            log_interval=1,
            seed=seed,
        )
        base_seed = seed if cfg_training.seed is None else cfg_training.seed
        if cfg_training.seed is not None and cfg_training.seed != seed:
            raise ValueError(
                "run_green_o_net seed must match training.seed when both are set."
            )
        seed_context = TrainingSeedContext(
            stage="green",
            base_seed=base_seed,
            deterministic_algorithms=cfg_training.deterministic_algorithms,
            device=cfg_training.device,
        )
        seed_context.configure_process()

        axial_lines = self._build_axial_lines(
            use_operator_learning, step_size, n_points_per_line
        )

        def resolve_sampler_cls(
            mode: Literal["forward", "backward"],
        ) -> type[ForwardSampler] | type[BackwardSampler]:
            if mode == "forward":
                return ForwardSampler
            if mode == "backward":
                return BackwardSampler
            raise ValueError(f"Unsupported sampler_mode: {mode}")

        def make_sampler(
            mode: Literal["forward", "backward"],
            sample_count: int,
            sampler_scale_length: float | tuple[float, float],
        ) -> ForwardSampler | BackwardSampler:
            sampler_cls = resolve_sampler_cls(mode)
            return sampler_cls(
                axial_lines=axial_lines,
                data_size_per_each_line=sample_count,
                scale_length=sampler_scale_length,
                deterministic=deterministic,
                integration_rule=cfg_training.integration_rule,
            )

        seed_context.apply("data_train")
        sampler = make_sampler(sampler_mode, ndata, scale_length)
        self.logger.info(
            "Sampling data: mode=%s, ndata=%s, scale_length=%s, lines=%s",
            sampler_mode,
            ndata,
            scale_length,
            len(axial_lines.xaxial_lines) + len(axial_lines.yaxial_lines),
        )

        def zeros(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        c_fun = c_fun or zeros
        c_fun_y = c_fun_y or c_fun
        bx_fun, by_fun = ForwardSampler._resolve_convection_functions(
            bx_fun=bx_fun,
            by_fun=by_fun,
            b_fun=b_fun,
            b_fun_y=b_fun_y,
        )

        data = sampler.generate_dataset(
            a_fun=a_fun,
            ap_fun=apx_fun,
            bx_fun=bx_fun,
            by_fun=by_fun,
            c_fun=c_fun,
            a_fun_y=a_fun,
            ap_fun_y=apy_fun,
            c_fun_y=c_fun_y,
        )
        dataset = AxialDataset(data)
        validation_dataset: AxialDataset | None = None
        if cfg_training.compute_validation_rel_sol:
            if validation_ndata <= 0:
                raise ValueError(
                    "validation_ndata must be > 0 when compute_validation_rel_sol=True."
                )
            val_scale_length = (
                validation_scale_length
                if validation_scale_length is not None
                else scale_length
            )
            val_sampler_mode = (
                validation_sampler_mode
                if validation_sampler_mode is not None
                else sampler_mode
            )
            seed_context.apply("data_valid")
            validation_sampler = make_sampler(
                val_sampler_mode, validation_ndata, val_scale_length
            )
            self.logger.info(
                "Sampling validation data: mode=%s, ndata=%s, scale_length=%s",
                val_sampler_mode,
                validation_ndata,
                val_scale_length,
            )
            validation_data = validation_sampler.generate_dataset(
                a_fun=a_fun,
                ap_fun=apx_fun,
                bx_fun=bx_fun,
                by_fun=by_fun,
                c_fun=c_fun,
                a_fun_y=a_fun,
                ap_fun_y=apy_fun,
                c_fun_y=c_fun_y,
            )
            validation_dataset = AxialDataset(validation_data)
        m_points = dataset.coords.shape[2]

        cfg_model = model_cfg or ModelConfig(
            input_dim=2,
            hidden_dim=64,
            depth=4,
            activation=activation,
            use_bias=True,
            dropout=0.0,
            branch_input_dim=m_points,
        )
        if cfg_model.branch_input_dim != m_points or cfg_model.activation != activation:
            cfg_model = ModelConfig(
                **{
                    **cfg_model.__dict__,
                    "branch_input_dim": m_points,
                    "activation": activation,
                }
            )
        seed_context.apply("model")
        model = GreenONetModel(cfg_model)
        seed_context.apply("runtime")
        trainer = Trainer(
            model=model,
            config=cfg_training,
            work_dir=self.work_dir,
            model_cfg=cfg_model,
            terminal_width=self.terminal_width,
            seed_context=seed_context,
        )
        trainer.train(dataset, validation_dataset)
        return trainer


def run_green_o_net(
    a_fun: Callable[[Tensor, Tensor], Tensor],
    apx_fun: Callable[[Tensor, Tensor], Tensor],
    apy_fun: Callable[[Tensor, Tensor], Tensor],
    activation: Literal["tanh", "relu", "gelu", "rational"],
    work_dir: Path,
    ndata: int,
    seed: int,
    scale_length: float | tuple[float, float],
    use_operator_learning: bool,
    deterministic: bool,
    validation_ndata: int = 0,
    validation_scale_length: float | tuple[float, float] | None = None,
    validation_sampler_mode: Literal["forward", "backward"] | None = None,
    b_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
    c_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
    b_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
    bx_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
    by_fun: Callable[[Tensor, Tensor], Tensor] | None = None,
    c_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
    step_size: float | None = None,
    n_points_per_line: int | None = None,
    sampler_mode: Literal["forward", "backward"] = "forward",
    model_cfg: ModelConfig | None = None,
    training_cfg: TrainingConfig | None = None,
    terminal_width: int | None = None,
) -> Trainer:
    runner = GreenONetRunner(work_dir=work_dir, terminal_width=terminal_width)
    return runner.run(
        a_fun=a_fun,
        apx_fun=apx_fun,
        apy_fun=apy_fun,
        b_fun=b_fun,
        c_fun=c_fun,
        b_fun_y=b_fun_y,
        bx_fun=bx_fun,
        by_fun=by_fun,
        c_fun_y=c_fun_y,
        activation=activation,
        ndata=ndata,
        validation_ndata=validation_ndata,
        seed=seed,
        scale_length=scale_length,
        validation_scale_length=validation_scale_length,
        use_operator_learning=use_operator_learning,
        deterministic=deterministic,
        validation_sampler_mode=validation_sampler_mode,
        step_size=step_size,
        n_points_per_line=n_points_per_line,
        sampler_mode=sampler_mode,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
    )


def run_complex_green_o_net(
    *,
    coeffs: CoefficientFunctions,
    geometry_path: Path,
    activation: Literal["tanh", "relu", "gelu", "rational"],
    work_dir: Path,
    ndata: int,
    seed: int,
    scale_length: float | tuple[float, float],
    deterministic: bool,
    validation_ndata: int = 0,
    validation_scale_length: float | tuple[float, float] | None = None,
    validation_sampler_mode: Literal["forward", "backward"] | None = None,
    sampler_mode: Literal["forward", "backward"] = "forward",
    model_cfg: ModelConfig | None = None,
    training_cfg: TrainingConfig | None = None,
    terminal_width: int | None = None,
) -> ComplexGreenTrainer:
    cfg_training = training_cfg or TrainingConfig(
        learning_rate=5e-3,
        epochs=max(5, ndata * 2),
        batch_size=32,
        log_interval=1,
        seed=seed,
    )
    base_seed = seed if cfg_training.seed is None else cfg_training.seed
    if cfg_training.seed is not None and cfg_training.seed != seed:
        raise ValueError(
            "run_complex_green_o_net seed must match training.seed when both are set."
        )
    seed_context = TrainingSeedContext(
        stage="green",
        base_seed=base_seed,
        deterministic_algorithms=cfg_training.deterministic_algorithms,
        device=cfg_training.device,
    )
    seed_context.configure_process()
    cfg_model = model_cfg or ModelConfig(
        input_dim=2,
        hidden_dim=64,
        depth=4,
        activation=activation,
        use_bias=True,
        dropout=0.0,
        branch_input_dim=16,
    )
    if cfg_model.activation != activation:
        cfg_model = ModelConfig(
            **{
                **cfg_model.__dict__,
                "activation": activation,
            }
        )
    if cfg_model.branch_input_dim < 2:
        raise ValueError("model.branch_input_dim must be at least 2.")
    if ndata <= 0:
        raise ValueError("dataset.samples_per_line must be positive in complex mode.")

    work_dir.mkdir(parents=True, exist_ok=True)
    runner_logger = LoggingMixin(
        logger_name="ComplexGreenONetRunner",
        work_dir=work_dir,
        terminal_width=terminal_width,
    ).logger
    geometry = load_complex_geometry(geometry_path, dtype=cfg_model.dtype)
    runner_logger.info(
        "Sampling complex Green data: geometry=%s, intervals=%s, samples_per_interval=%s",
        geometry_path,
        geometry.num_x_segments + geometry.num_y_segments,
        ndata,
    )
    seed_context.apply("data_train")
    train_data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=cfg_model.branch_input_dim,
        samples_per_interval=ndata,
        sampler_mode=sampler_mode,
        scale_length=scale_length,
        deterministic=deterministic,
        integration_rule=cfg_training.integration_rule,
        source_sampling_factor=(
            cfg_training.green_quadrature.source_sampling_factor
            if cfg_training.green_quadrature.enabled
            else 1
        ),
        dtype=cfg_model.dtype,
    )
    train_dataset = ComplexGreenDataset(train_data)

    validation_dataset: ComplexGreenDataset | None = None
    if cfg_training.compute_validation_rel_sol:
        if validation_ndata <= 0:
            raise ValueError(
                "validation_samples_per_line must be > 0 when compute_validation_rel_sol=True."
            )
        val_scale_length = (
            validation_scale_length
            if validation_scale_length is not None
            else scale_length
        )
        val_sampler_mode = (
            validation_sampler_mode
            if validation_sampler_mode is not None
            else sampler_mode
        )
        runner_logger.info(
            "Sampling complex validation Green data: mode=%s, samples_per_interval=%s",
            val_sampler_mode,
            validation_ndata,
        )
        seed_context.apply("data_valid")
        validation_data = generate_complex_green_data(
            geometry,
            coeffs,
            branch_input_dim=cfg_model.branch_input_dim,
            samples_per_interval=validation_ndata,
            sampler_mode=val_sampler_mode,
            scale_length=val_scale_length,
            deterministic=deterministic,
            integration_rule=cfg_training.integration_rule,
            source_sampling_factor=(
                cfg_training.green_quadrature.source_sampling_factor
                if cfg_training.green_quadrature.enabled
                else 1
            ),
            dtype=cfg_model.dtype,
        )
        validation_dataset = ComplexGreenDataset(validation_data)

    seed_context.apply("model")
    model = GreenONetModel(cfg_model)
    seed_context.apply("runtime")
    trainer = ComplexGreenTrainer(
        model=model,
        config=cfg_training,
        work_dir=work_dir,
        model_cfg=cfg_model,
        coeffs=coeffs,
        terminal_width=terminal_width,
        seed_context=seed_context,
    )
    trainer.train(train_dataset, validation_dataset)
    return trainer
