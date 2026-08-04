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
from greenonet.complex_losses import relative_l2_valid
from greenonet.complex_sources import (
    GeometryGridLoader,
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
)
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CompileConfig,
    ComplexCanonicalEnergyConfig,
    ComplexCrossAxisReconstructionConfig,
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


@pytest.mark.parametrize("fusion_mode", ["residual", "absolute"])
def test_complex_trainer_one_step_has_no_cross_metrics_or_logs(
    tmp_path,
    fusion_mode,
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
            pre_projection_fusion=ComplexPreProjectionFusionConfig(
                enabled=True,
                mode=fusion_mode,
                hidden_dim=8,
                depth=1,
                final_layer_init_scale=0.0,
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
    work_dir = tmp_path / f"single_fusion_mlp_{fusion_mode}"
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
    assert "loss_energy_optimized" in trainer.metric_rows[0]
    assert trainer.metric_rows[0]["boundary_weight"] == pytest.approx(1.0)
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
    assert "architecture=single_nonlinear_fusion_mlp" in training_log
    assert f"mode={fusion_mode}" in training_log
    assert "input_dim=2" in training_log
    assert "hidden_dim=8" in training_log
    assert "depth=1" in training_log
    assert f"identity_skip={str(fusion_mode == 'residual').lower()}" in training_log
    assert "final_initialization=scaled_torch_linear_default" in training_log
    assert "final_layer_init_scale=0" in training_log
    assert "explicit_geometry_features=false" in training_log
    assert "pre_projection_fusion_gate" not in training_log


def test_column_diagonal_green_response_context_is_cached_in_train_and_eval(
    tmp_path,
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
            balance_projection=BalanceProjectionConfig(
                mode="column_diagonal_green_response",
                column_diagonal_green_response={"gain_exponent": 0.25},
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
    )
    work_dir = tmp_path / "column_train"
    trainer = ComplexCouplingTrainer(
        model=model,
        config=training,
        work_dir=work_dir,
        green_model=ConstantGreen(1.0),
    )

    trainer.train(dataset, dataset)

    assert trainer.column_diagonal_green_response_context_build_count == 1
    context = trainer.column_diagonal_green_response_context
    assert context is not None
    assert not context.gamma_x_squared.requires_grad
    assert context.gain_exponent == 0.25
    training_log = (work_dir / "training.log").read_text()
    assert "column-diagonal Green-response context" in training_log
    assert "gain_exponent=0.250000" in training_log
    assert "row_norm_used=false" in training_log

    evaluator = ComplexCouplingEvaluator(
        model=model,
        green_model=ConstantGreen(1.0),
        config=training,
        device=torch.device("cpu"),
        work_dir=tmp_path / "column_eval",
    )
    batch = complex_coupling_collate_fn([dataset[0]])
    first = evaluator.predict_batch(batch)
    second = evaluator.predict_batch(batch)

    assert evaluator.column_diagonal_green_response_context_build_count == 1
    evaluator_context = evaluator.column_diagonal_green_response_context
    assert evaluator_context is not None
    assert evaluator_context.gain_exponent == 0.25
    torch.testing.assert_close(
        evaluator_context.correction_weight_phi,
        context.correction_weight_phi,
    )
    evaluation_log = (tmp_path / "column_eval" / "training.log").read_text()
    assert "gain_exponent=0.250000" in evaluation_log
    torch.testing.assert_close(
        first.projection.projected_physical,
        second.projection.projected_physical,
    )
    torch.testing.assert_close(
        first.projection.physical_balance_residual,
        torch.zeros_like(first.projection.physical_balance_residual),
        atol=1e-12,
        rtol=1e-12,
    )


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
            cross_axis_reconstruction=ComplexCrossAxisReconstructionConfig(
                enabled=True
            ),
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
    assert trainer.cross_axis_reconstructor.context_build_count == 0


def test_cross_axis_reconstruction_changes_only_detached_solution_diagnostic(
    tmp_path,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)

    def build_model(enabled: bool) -> ComplexCouplingNet:
        return ComplexCouplingNet(
            CouplingModelConfig(
                branch_input_dim=4,
                hidden_dim=4,
                depth=1,
                dtype=torch.float64,
                balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
                cross_axis_reconstruction=(
                    ComplexCrossAxisReconstructionConfig(enabled=enabled)
                ),
                axis_1d_trunk=Axis1DTrunkConfig(
                    enabled=True,
                    transverse_trunk=TransverseTrunkConfig(
                        enabled=True,
                        length_context=True,
                    ),
                ),
            )
        )

    disabled_model = build_model(False)
    enabled_model = build_model(True)
    enabled_model.load_state_dict(disabled_model.state_dict())
    training = CouplingTrainingConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        compile=CompileConfig(enabled=False),
    )
    disabled_trainer = ComplexCouplingTrainer(
        model=disabled_model,
        config=training,
        work_dir=tmp_path / "disabled",
        green_model=ConstantGreen(1.0),
    )
    enabled_trainer = ComplexCouplingTrainer(
        model=enabled_model,
        config=training,
        work_dir=tmp_path / "enabled",
        green_model=ConstantGreen(1.0),
    )
    batch = complex_coupling_collate_fn([dataset[0]])

    disabled = disabled_trainer._forward_batch(batch)
    enabled = enabled_trainer._forward_batch(batch)

    torch.testing.assert_close(enabled.loss, disabled.loss)
    torch.testing.assert_close(
        enabled.projection.projected_physical,
        disabled.projection.projected_physical,
    )
    torch.testing.assert_close(
        enabled.reconstruction.u_phi_valid,
        disabled.reconstruction.u_phi_valid,
    )
    torch.testing.assert_close(
        enabled.reconstruction.u_psi_valid,
        disabled.reconstruction.u_psi_valid,
    )
    assert enabled.cross_axis_reconstruction is not None
    assert enabled.cross_axis_reconstruction.reliability is not None
    expected_rel_sol = relative_l2_valid(
        enabled.cross_axis_reconstruction.u_pred_valid,
        batch.sol_valid,
    )
    torch.testing.assert_close(enabled.metrics["rel_sol"], expected_rel_sol)
    assert enabled.metrics["rel_sol"].requires_grad is False
    assert enabled_trainer.cross_axis_reconstructor.context_build_count == 1


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


def test_complex_evaluator_reports_official_and_equal_mean_solution_metrics(
    tmp_path,
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
            cross_axis_reconstruction=ComplexCrossAxisReconstructionConfig(
                enabled=True
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    evaluator = ComplexCouplingEvaluator(
        model=model,
        green_model=ConstantGreen(1.0),
        config=CouplingTrainingConfig(batch_size=1, device="cpu"),
        device=torch.device("cpu"),
        work_dir=tmp_path / "evaluation_reliability",
    )
    batch = complex_coupling_collate_fn([dataset[0]])

    prediction = evaluator.predict_batch(batch)
    row = evaluator._sample_metric_row(prediction, 0)
    evaluator.predict_batch(batch)

    reconstruction = prediction.cross_axis_reconstruction
    expected_official = relative_l2_valid(
        reconstruction.u_pred_valid,
        batch.sol_valid,
    )
    expected_equal = relative_l2_valid(
        reconstruction.u_equal_mean_valid,
        batch.sol_valid,
    )
    assert float(row["rel_sol"]) == pytest.approx(float(expected_official.item()))
    assert float(row["rel_sol_equal_mean"]) == pytest.approx(
        float(expected_equal.item())
    )
    assert "weak_weight_phi_mean" in row
    assert "weak_weight_phi_min" in row
    assert "weak_weight_phi_max" in row
    assert "weak_support_fraction" in row
    assert evaluator.cross_axis_reconstructor.context_build_count == 1


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


def test_complex_trainer_selects_best_energy_checkpoint_by_optimized_energy(
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
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=2,
            batch_size=1,
            device="cpu",
            compile=CompileConfig(enabled=False),
            canonical_energy=ComplexCanonicalEnergyConfig(boundary_weight=0.0),
            best_energy_checkpoint=CouplingBestEnergyCheckpointConfig(enabled=True),
        ),
        work_dir=tmp_path / "best_optimized_energy",
        green_model=ConstantGreen(1.0),
    )
    validation_metrics = iter(
        (
            {
                "loss": 2.0,
                "loss_energy_optimized": 2.0,
                "loss_energy_consistency": 1.0,
            },
            {
                "loss": 1.0,
                "loss_energy_optimized": 1.0,
                "loss_energy_consistency": 2.0,
            },
        )
    )
    saved: list[str] = []
    monkeypatch.setattr(
        trainer,
        "_run_epoch",
        lambda *_args, **_kwargs: {
            "loss": 1.0,
            "loss_energy_optimized": 1.0,
            "loss_energy_consistency": 1.0,
        },
    )
    monkeypatch.setattr(
        trainer,
        "_evaluate_loader",
        lambda *_args, **_kwargs: next(validation_metrics),
    )
    monkeypatch.setattr(trainer, "_save_checkpoint", saved.append)

    trainer.train(dataset, dataset)

    assert saved.count("complex_coupling_model_best_energy.safetensors") == 2


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


def test_complex_tangent_projection_smoke_reuses_context_and_preserves_autograd(
    tmp_path,
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
            balance_projection=BalanceProjectionConfig(
                mode="symmetric_tangent_green_response",
                symmetric_tangent_green_response={
                    "eta": 0.01,
                    "relative_lambda": 0.01,
                },
            ),
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
            canonical_energy=ComplexCanonicalEnergyConfig(boundary_weight=0.0),
        ),
        work_dir=tmp_path / "tangent_training",
        green_model=ConstantGreen(1.0),
    )
    batch = complex_coupling_collate_fn([dataset[0]])

    first = trainer._forward_batch(batch)
    first.loss.backward()
    second = trainer._forward_batch(batch)

    tangent = first.projection.symmetric_tangent_diagnostics
    assert tangent is not None
    torch.testing.assert_close(
        first.objective.energy_optimized,
        first.objective.energy.bulk,
    )
    torch.testing.assert_close(
        first.metrics["boundary_weight"],
        torch.zeros((), dtype=torch.float64),
    )
    assert first.objective.energy.boundary.item() > 0.0
    assert (
        "boundary_weight=0.000000e+00"
        in (tmp_path / "tangent_training" / "training.log").read_text()
    )
    assert trainer.symmetric_tangent_green_response_context_build_count == 1
    assert trainer.symmetric_tangent_green_response_context is not None
    assert all(
        key in first.metrics
        for key in (
            "tangent_response_mismatch_pre",
            "tangent_response_mismatch_post",
            "tangent_gradient_rms",
            "tangent_delta_rms",
        )
    )
    torch.testing.assert_close(
        first.reconstruction.u_phi_valid,
        tangent.projected_solution[:, 0],
    )
    torch.testing.assert_close(
        first.projection.projected_physical.sum(dim=1),
        batch.rhs_valid,
        atol=0.0,
        rtol=0.0,
    )
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert second.projection.symmetric_tangent_diagnostics is not None


def test_complex_adaptive_tangent_uses_scheduled_training_cap_and_final_validation_cap(
    tmp_path,
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
            balance_projection=BalanceProjectionConfig(
                mode="symmetric_tangent_green_response",
                symmetric_tangent_green_response={
                    "eta": 0.01,
                    "eta_strategy": "closed_loop_exact_line_search",
                    "line_search_relative_eps": 1e-12,
                    "relative_lambda": 0.01,
                },
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    work_dir = tmp_path / "adaptive_tangent_training"
    trainer = ComplexCouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=3,
            batch_size=1,
            log_interval=1,
            learning_rate=1e-3,
            use_lr_schedule=True,
            warmup_epochs=2,
            min_lr=1e-4,
            device="cpu",
            compile=CompileConfig(enabled=False),
        ),
        work_dir=work_dir,
        green_model=ConstantGreen(1.0),
    )

    trainer.train(dataset, dataset)

    train_rows = [row for row in trainer.metric_rows if row["split"] == "train"]
    val_rows = [row for row in trainer.metric_rows if row["split"] == "val"]
    assert [float(row["tangent_eta_cap"]) for row in train_rows] == pytest.approx(
        [0.005, 0.01, 0.01]
    )
    assert [float(row["tangent_eta_cap"]) for row in val_rows] == pytest.approx(
        [0.01, 0.01, 0.01]
    )
    for row in (*train_rows, *val_rows):
        assert "tangent_eta_star_mean" in row
        assert "tangent_eta_applied_mean" in row
        assert "tangent_eta_cap_fraction" in row
        assert float(row["tangent_eta_applied_mean"]) <= float(row["tangent_eta_cap"])
    assert trainer.symmetric_tangent_green_response_context_build_count == 1
    training_log = (work_dir / "training.log").read_text()
    assert "strategy=closed_loop_exact_line_search" in training_log
    assert "kind=closed_loop_half_cosine_warmup_hold" in training_log
    assert "training_cap=scheduled validation_cap=final" in training_log
    with (work_dir / "complex_training_metrics.csv").open(newline="") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert "tangent_eta_star_mean" in csv_rows[0]
    assert "tangent_eta_applied_mean" in csv_rows[0]
    assert "tangent_eta_cap_fraction" in csv_rows[0]


def test_complex_tangent_evaluator_reuses_context_and_reports_sample_metrics(tmp_path):
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
            balance_projection=BalanceProjectionConfig(
                mode="symmetric_tangent_green_response"
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )
    evaluator = ComplexCouplingEvaluator(
        model=model,
        green_model=ConstantGreen(1.0),
        config=CouplingTrainingConfig(batch_size=1, device="cpu"),
        device=torch.device("cpu"),
        work_dir=tmp_path / "tangent_evaluation",
    )
    batch = complex_coupling_collate_fn([dataset[0]])

    prediction = evaluator.predict_batch(batch)
    row = evaluator._sample_metric_row(prediction, 0)
    evaluator.predict_batch(batch)

    assert evaluator.symmetric_tangent_green_response_context_build_count == 1
    assert "tangent_response_mismatch_pre" in row
    assert "tangent_response_mismatch_post" in row
    assert "tangent_correction_rel_symmetric_pair" in row


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
