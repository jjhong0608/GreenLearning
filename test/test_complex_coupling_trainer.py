from __future__ import annotations

import csv
from dataclasses import replace

import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_coupling_trainer import (
    ComplexCouplingTrainer,
    complex_metric_keys_are_safe,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CompileConfig,
    ComplexPreProjectionFusionConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingBestEnergyCheckpointConfig,
    CouplingBestPhysicsCheckpointConfig,
    CouplingBestRelSolCheckpointConfig,
    CouplingLossTermConfig,
    CouplingLossesConfig,
    CouplingModelConfig,
    CouplingTrainingConfig,
    TransverseTrunkConfig,
)
from test.complex_fixtures import (
    ConstantGreen,
    write_coefficients,
    write_geometry_npz,
    write_sample_npz,
)


def test_complex_trainer_one_step_has_no_cross_metrics_or_logs(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            pre_projection_fusion=ComplexPreProjectionFusionConfig(
                enabled=True,
                nonlinear_hidden_dim=8,
                nonlinear_depth=1,
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                num_frequencies=1,
                max_frequency=1.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    training = CouplingTrainingConfig(
        epochs=1,
        batch_size=1,
        log_interval=1,
        learning_rate=1e-3,
        device="cpu",
        compile=CompileConfig(enabled=False),
        relative_split_consistency=ComplexRelativeSplitConsistencyConfig(enabled=True),
        weak_operator_closure=ComplexWeakOperatorClosureConfig(enabled=True),
        losses=CouplingLossesConfig(
            cross_consistency=CouplingLossTermConfig(enabled=True, weight=99.0),
            balance_loss=CouplingLossTermConfig(enabled=True, weight=99.0),
        ),
    )
    trainer = ComplexCouplingTrainer(
        model=model,
        config=training,
        work_dir=tmp_path / "physical_symmetric",
        green_model=ConstantGreen(1.0),
        terminal_width=120,
    )

    trainer.train(dataset)

    assert (
        tmp_path / "physical_symmetric" / "complex_coupling_model.safetensors"
    ).exists()
    assert (tmp_path / "physical_symmetric" / "complex_training_metrics.csv").exists()
    assert complex_metric_keys_are_safe(trainer.metric_rows[0].keys())
    assert "cross" not in (tmp_path / "physical_symmetric" / "training.log").read_text()
    assert "loss_energy_consistency" in trainer.metric_rows[0]
    assert "loss_energy_bulk" in trainer.metric_rows[0]
    assert "loss_energy_boundary" in trainer.metric_rows[0]
    assert "loss_energy_boundary_x" in trainer.metric_rows[0]
    assert "loss_energy_boundary_y" in trainer.metric_rows[0]
    assert "loss_energy_length_balanced" not in trainer.metric_rows[0]
    assert "loss_energy_regular" not in trainer.metric_rows[0]
    assert "loss_energy_transition" not in trainer.metric_rows[0]
    assert "transition_edge_fraction" not in trainer.metric_rows[0]
    assert "loss_split_relative" in trainer.metric_rows[0]
    assert "loss_split_mass_relative" in trainer.metric_rows[0]
    assert "loss_weak_operator_closure" in trainer.metric_rows[0]
    assert "loss_weak_operator_x" in trainer.metric_rows[0]
    assert "loss_weak_operator_y" in trainer.metric_rows[0]
    assert "pre_projection_fusion_gate" in trainer.metric_rows[0]
    assert "loss_trace_gluing" not in trainer.metric_rows[0]
    assert "loss_trace_carrier_transition" not in trainer.metric_rows[0]
    row = trainer.metric_rows[0]
    assert float(row["learning_rate"]) == pytest.approx(1.0e-3)
    assert float(row["loss_split_relative"]) == pytest.approx(
        float(row["loss_split_energy_relative"])
        + float(row["loss_split_mass_relative"])
    )
    assert float(row["loss_weak_operator_closure"]) == pytest.approx(
        0.5 * (float(row["loss_weak_operator_x"]) + float(row["loss_weak_operator_y"]))
    )
    assert float(row["loss"]) == pytest.approx(
        float(row["loss_split_relative"]) + float(row["loss_weak_operator_closure"])
    )
    training_log = (tmp_path / "physical_symmetric" / "training.log").read_text()
    assert "kind=fixed" in training_log
    assert "learning_rate=1.000000e-03" in training_log
    assert "pre-projection fusion enabled=True" in training_log
    assert "pre_projection_fusion_gate=" in training_log


def test_complex_trainer_applies_and_records_warmup_cosine_schedule(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    work_dir = tmp_path / "scheduled"
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=3,
            batch_size=1,
            log_interval=1,
            learning_rate=6.0e-3,
            use_lr_schedule=True,
            warmup_epochs=2,
            min_lr=1.0e-3,
            device="cpu",
            compile=CompileConfig(enabled=False),
        ),
        work_dir=work_dir,
        green_model=ConstantGreen(1.0),
    )

    trainer.train(dataset, dataset)

    expected = [3.0e-3, 6.0e-3, 1.0e-3]
    train_rows = [row for row in trainer.metric_rows if row["split"] == "train"]
    val_rows = [row for row in trainer.metric_rows if row["split"] == "val"]
    assert [float(row["learning_rate"]) for row in train_rows] == pytest.approx(
        expected
    )
    assert [float(row["learning_rate"]) for row in val_rows] == pytest.approx(expected)

    with (work_dir / "complex_training_metrics.csv").open(newline="") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert "learning_rate" in csv_rows[0]
    assert [float(row["learning_rate"]) for row in csv_rows[::2]] == pytest.approx(
        expected
    )
    training_log = (work_dir / "training.log").read_text()
    assert "kind=linear_warmup_cosine_decay" in training_log
    assert "configured_warmup_epochs=2" in training_log
    assert "effective_warmup_epochs=2" in training_log
    assert "learning_rate=3.000000e-03" in training_log
    assert "learning_rate=1.000000e-03" in training_log


def test_complex_trainer_selects_best_checkpoint_by_reference_free_energy(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            device="cpu",
            compile=CompileConfig(enabled=False),
            best_energy_checkpoint=CouplingBestEnergyCheckpointConfig(enabled=True),
            best_physics_checkpoint=CouplingBestPhysicsCheckpointConfig(enabled=True),
            relative_split_consistency=ComplexRelativeSplitConsistencyConfig(
                enabled=True
            ),
            weak_operator_closure=ComplexWeakOperatorClosureConfig(enabled=True),
        ),
        work_dir=tmp_path / "best_energy",
        green_model=ConstantGreen(1.0),
    )

    trainer.train(dataset, dataset)

    assert (
        tmp_path / "best_energy" / "complex_coupling_model_best_energy.safetensors"
    ).is_file()
    assert (
        tmp_path / "best_energy" / "complex_coupling_model_best_physics.safetensors"
    ).is_file()
    assert not (
        tmp_path / "best_energy" / "complex_coupling_model_best_rel_sol.safetensors"
    ).exists()


def test_complex_training_loss_graph_excludes_reference_targets(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            device="cpu",
            compile=CompileConfig(enabled=False),
            relative_split_consistency=ComplexRelativeSplitConsistencyConfig(
                enabled=True
            ),
            weak_operator_closure=ComplexWeakOperatorClosureConfig(enabled=True),
        ),
        work_dir=tmp_path / "reference_free",
        green_model=ConstantGreen(1.0),
    )
    batch = complex_coupling_collate_fn([dataset[0]])
    batch = replace(
        batch,
        sol_valid=batch.sol_valid.clone().requires_grad_(),
        flux_valid=batch.flux_valid.clone().requires_grad_(),
    )

    result = trainer._forward_batch(batch)
    result.loss.backward()

    assert batch.sol_valid.grad is None
    assert batch.flux_valid.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_complex_trainer_rejects_reference_based_checkpoint_selection(tmp_path):
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="reference sol must not select"):
        ComplexCouplingTrainer(
            model=model,
            config=CouplingTrainingConfig(
                epochs=1,
                best_rel_sol_checkpoint=CouplingBestRelSolCheckpointConfig(
                    enabled=True
                ),
            ),
            work_dir=tmp_path / "rejected",
            green_model=ConstantGreen(1.0),
        )
