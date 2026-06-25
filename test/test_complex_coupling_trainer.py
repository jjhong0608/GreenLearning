from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_coupling_trainer import (
    ComplexCouplingTrainer,
    complex_metric_keys_are_safe,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    Axis1DTrunkConfig,
    CompileConfig,
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
            axis_1d_trunk=Axis1DTrunkConfig(
                num_frequencies=1,
                max_frequency=1.0,
                transverse_trunk=TransverseTrunkConfig(enabled=True),
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
        losses=CouplingLossesConfig(
            cross_consistency=CouplingLossTermConfig(enabled=True, weight=99.0),
            balance_loss=CouplingLossTermConfig(enabled=True, weight=99.0),
        ),
    )
    trainer = ComplexCouplingTrainer(
        model=model,
        config=training,
        work_dir=tmp_path / "work",
        green_model=ConstantGreen(1.0),
        terminal_width=120,
    )

    trainer.train(dataset)

    assert (tmp_path / "work" / "complex_coupling_model.safetensors").exists()
    assert (tmp_path / "work" / "complex_training_metrics.csv").exists()
    assert complex_metric_keys_are_safe(trainer.metric_rows[0].keys())
    assert "cross" not in (tmp_path / "work" / "training.log").read_text()
