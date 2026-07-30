from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_coupling_trainer import (
    ComplexCouplingTrainer,
    complex_metric_keys_are_safe,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_sources import (
    GeometryGridLoader,
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
)
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
from greenonet.optimizers import SOAP


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
                hidden_dim=8,
                depth=1,
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
    work_dir = tmp_path / "single_residual_mlp"
    trainer = ComplexCouplingTrainer(
        model=model,
        config=training,
        work_dir=work_dir,
        green_model=ConstantGreen(1.0),
        terminal_width=120,
    )

    trainer.train(dataset)

    assert (work_dir / "complex_coupling_model.safetensors").exists()
    assert (work_dir / "complex_training_metrics.csv").exists()
    assert complex_metric_keys_are_safe(trainer.metric_rows[0].keys())
    assert "cross" not in (work_dir / "training.log").read_text()
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
    assert "pre_projection_fusion_gate" not in trainer.metric_rows[0]
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
    training_log = (work_dir / "training.log").read_text()
    assert "kind=fixed" in training_log
    assert "learning_rate=1.000000e-03" in training_log
    assert "pre-projection fusion enabled=True" in training_log
    assert "architecture=single_nonlinear_residual_mlp" in training_log
    assert "input_dim=2" in training_log
    assert "hidden_dim=8" in training_log
    assert "depth=1" in training_log
    assert "identity_skip=true" in training_log
    assert "final_initialization=zeros" in training_log
    assert "explicit_geometry_features=false" in training_log
    assert "pre_projection_fusion_gate" not in training_log


def test_complex_trainer_omits_reference_metrics_for_rhs_only_data(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir, include_solution=False, include_flux=False)
    dataset = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        reference_diagnostics=False,
    )
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
    work_dir = tmp_path / "source_only"
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            log_interval=1,
            device="cpu",
            compile=CompileConfig(enabled=False),
            best_energy_checkpoint=CouplingBestEnergyCheckpointConfig(enabled=True),
        ),
        work_dir=work_dir,
        green_model=ConstantGreen(1.0),
    )

    trainer.train(dataset, dataset)

    assert (work_dir / "complex_coupling_model_best_energy.safetensors").is_file()
    assert all(
        "rel_sol" not in row and "rel_flux" not in row for row in trainer.metric_rows
    )
    csv_header = (work_dir / "complex_training_metrics.csv").read_text().splitlines()[0]
    assert "rel_sol" not in csv_header
    assert "rel_flux" not in csv_header
    log = (work_dir / "training.log").read_text()
    assert "rel_sol=" not in log
    assert "rel_flux=" not in log


def test_complex_evaluator_omits_reference_metrics_for_rhs_only_data(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir, include_solution=False, include_flux=False)
    dataset = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        reference_diagnostics=False,
    )
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
    work_dir = tmp_path / "evaluation"
    evaluator = ComplexCouplingEvaluator(
        model=model,
        green_model=ConstantGreen(1.0),
        config=CouplingTrainingConfig(
            batch_size=1,
            device="cpu",
            compile=CompileConfig(enabled=False),
        ),
        device=torch.device("cpu"),
        work_dir=work_dir,
    )

    summary = evaluator.evaluate(
        dataset,
        dataset_name="source_only",
        batch_size=1,
    )

    assert "rel_sol" not in summary
    assert "rel_flux" not in summary
    summary_payload = json.loads(
        (work_dir / "metrics" / "source_only_metrics.json").read_text()
    )
    assert "rel_sol" not in summary_payload
    assert "rel_flux" not in summary_payload
    csv_header = (
        (work_dir / "metrics" / "source_only_per_sample_metrics.csv")
        .read_text()
        .splitlines()[0]
    )
    assert "rel_sol" not in csv_header
    assert "rel_flux" not in csv_header


def test_complex_trainer_accepts_fixed_indexed_gp_provider(tmp_path):
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    geometry = load_complex_geometry(geometry_path)
    raw_geometry = GeometryGridLoader().load(geometry_path)
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    provider = IndexedGpComplexSourceProvider(
        raw_geometry,
        split="train",
        sample_count=2,
        parameters=IndexedGpParameters(seed=3),
    )
    dataset = ComplexCouplingDataset(
        None,
        geometry,
        coeffs,
        branch_input_dim=4,
        reference_diagnostics=False,
        source_provider=provider,
    )
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
            log_interval=1,
            device="cpu",
            compile=CompileConfig(enabled=False),
        ),
        work_dir=tmp_path / "indexed_gp",
        green_model=ConstantGreen(1.0),
    )
    before = dataset[1].rhs_valid.clone()

    trainer.train(dataset)

    torch.testing.assert_close(dataset[1].rhs_valid, before, rtol=0.0, atol=0.0)
    assert all("rel_sol" not in row for row in trainer.metric_rows)


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


def test_complex_trainer_soap_smoke_records_provenance_and_telemetry(
    tmp_path,
    monkeypatch,
):
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
    work_dir = tmp_path / "soap"
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=2,
            batch_size=1,
            log_interval=1,
            learning_rate=2.0e-3,
            weight_decay=0.01,
            use_lr_schedule=True,
            warmup_epochs=1,
            min_lr=5.0e-4,
            device="cpu",
            compile=CompileConfig(enabled=False),
            optimizer={
                "name": "soap",
                "betas": [0.95, 0.95],
                "profile_step_time": True,
                "soap": {
                    "precondition_frequency": 1,
                    "max_precondition_dim": 16,
                },
            },
        ),
        work_dir=work_dir,
        green_model=ConstantGreen(1.0),
    )
    call_order: list[str] = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = SOAP.step

    def tracked_clip(*args, **kwargs):
        call_order.append("clip")
        return original_clip(*args, **kwargs)

    def tracked_step(optimizer, closure=None):
        call_order.append("step")
        return original_step(optimizer, closure)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", tracked_clip)
    monkeypatch.setattr(SOAP, "step", tracked_step)

    trainer.train(dataset, dataset)

    provenance = json.loads((work_dir / "optimizer_provenance.json").read_text())
    assert provenance["name"] == "soap"
    assert provenance["betas"] == [0.95, 0.95]
    assert provenance["soap"]["precondition_frequency"] == 1
    assert provenance["checkpoint_policy"] == "model_only_no_optimizer_resume"
    train_rows = [row for row in trainer.metric_rows if row["split"] == "train"]
    val_rows = [row for row in trainer.metric_rows if row["split"] == "val"]
    assert [float(row["learning_rate"]) for row in train_rows] == pytest.approx(
        [2.0e-3, 5.0e-4]
    )
    assert all(float(row["optimizer_step_count"]) == 1.0 for row in train_rows)
    assert float(train_rows[0]["optimizer_basis_refresh_count"]) == 0.0
    assert float(train_rows[1]["optimizer_basis_refresh_count"]) == 1.0
    assert all(float(row["optimizer_step_time_mean_ms"]) > 0.0 for row in train_rows)
    assert all("optimizer_step_time_mean_ms" not in row for row in val_rows)
    assert (work_dir / "complex_coupling_model.safetensors").is_file()
    with (work_dir / "complex_training_metrics.csv").open(newline="") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert "optimizer_step_time_mean_ms" in csv_rows[0]
    training_log = (work_dir / "training.log").read_text()
    assert "optimizer name=soap" in training_log
    assert "frequency_unit=optimizer_step" in training_log
    assert "first_step_initializes_preconditioner=true" in training_log
    assert call_order == ["clip", "step", "clip", "step"]


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
