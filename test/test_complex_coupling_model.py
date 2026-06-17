from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import CouplingModelConfig, CouplingTrunkPositionalEncodingConfig
from test.complex_fixtures import (
    write_coefficients,
    write_geometry_npz,
    write_sample_npz,
)


def test_complex_coupling_model_outputs_batch_axis_point_shape(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    item = dataset[0]
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            trunk_positional_encoding=CouplingTrunkPositionalEncodingConfig(
                num_frequencies=2,
                max_frequency=2.0,
            ),
        )
    )

    output = model(
        geometry=geometry,
        x_branch=item.x_branch.unsqueeze(0),
        y_branch=item.y_branch.unsqueeze(0),
    )

    assert output.shape == (1, 2, geometry.num_points)
    assert model.function_branch is model.function_branch
    assert model.geometry_branch is model.geometry_branch
    assert model.trunk is model.trunk
    assert model.geometry_feature_dim == 10
    assert not hasattr(model, "axis_one_hot")


def test_complex_geometry_feature_contract_excludes_raw_axis_label(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            trunk_positional_encoding=CouplingTrunkPositionalEncodingConfig(
                num_frequencies=1,
                max_frequency=1.0,
            ),
        )
    )

    features = model._geometry_features(geometry, "x")

    assert features.shape == (geometry.num_x_segments, 8)
    torch.testing.assert_close(features[:, -3], geometry.x_segment_length)
    torch.testing.assert_close(features[:, -2], geometry.x_segment_length.pow(2))
    torch.testing.assert_close(features[:, -1], 1.0 / geometry.x_segment_length)
