from __future__ import annotations

import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingBranchFusionConfig,
    CouplingCoefficientTermsConfig,
    CouplingModelConfig,
    CouplingTrunkPositionalEncodingConfig,
    TransverseTrunkConfig,
)
from greenonet.io import load_state_dict_auto, save_state_dict_safetensors
from test.complex_fixtures import (
    write_coefficients,
    write_geometry_npz,
    write_sample_npz,
)


def _build_item(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    return geometry, dataset[0]


def _model(
    *,
    fusion_mode: str = "product",
    transverse_trunk_enabled: bool = False,
    transverse_trunk_fusion: str = "product",
    coefficient_terms: CouplingCoefficientTermsConfig | None = None,
) -> ComplexCouplingNet:
    torch.manual_seed(0)
    return ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            coefficient_terms=coefficient_terms or CouplingCoefficientTermsConfig(),
            branch_fusion=CouplingBranchFusionConfig(mode=fusion_mode),
            axis_1d_trunk=Axis1DTrunkConfig(
                num_frequencies=2,
                max_frequency=2.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=transverse_trunk_enabled,
                    fusion=transverse_trunk_fusion,
                ),
            ),
        )
    )


def _forward(model: ComplexCouplingNet, geometry, item) -> torch.Tensor:
    return model(
        geometry=geometry,
        x_source_branch=item.x_source_branch.unsqueeze(0),
        y_source_branch=item.y_source_branch.unsqueeze(0),
        x_source_amplitude=item.x_source_amplitude.unsqueeze(0),
        y_source_amplitude=item.y_source_amplitude.unsqueeze(0),
        x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
        y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
    )


def test_complex_coupling_model_outputs_batch_axis_point_shape(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model()

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.branch_source is model.branch_source
    assert model.branch_transverse is model.branch_transverse
    assert model.branch_geometry is model.branch_geometry
    assert model.trunk is model.trunk
    assert model.trunk_transverse is None
    assert not any(key.startswith("trunk_transverse.") for key in model.state_dict())
    assert model.geometry_feature_dim == 6
    assert model.transverse_feature_dim == 4
    assert not hasattr(model, "axis_one_hot")
    assert int(model._output_contract_version.item()) == 2


def test_complex_pointwise_transverse_trunk_product_fusion_outputs_shape(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(transverse_trunk_enabled=True, transverse_trunk_fusion="product")

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.trunk_transverse is not None
    assert model.trunk_fuser is None
    assert any(key.startswith("trunk_transverse.") for key in model.state_dict())


def test_complex_pointwise_transverse_trunk_product_fuser_outputs_shape(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(
        transverse_trunk_enabled=True,
        transverse_trunk_fusion="product_fuser",
    )

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.trunk_transverse is not None
    assert model.trunk_fuser is not None


def test_complex_pointwise_transverse_trunk_uses_cross_axis_local_coordinates(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    model = _model(transverse_trunk_enabled=True)

    x_primary, x_transverse = model._trunk_coordinates(geometry, "x")
    y_primary, y_transverse = model._trunk_coordinates(geometry, "y")

    torch.testing.assert_close(x_primary, geometry.x_local_t)
    torch.testing.assert_close(x_transverse, geometry.y_local_t)
    torch.testing.assert_close(y_primary, geometry.y_local_t)
    torch.testing.assert_close(y_transverse, geometry.x_local_t)


def test_complex_coupling_model_is_source_conditioned(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model()

    original = _forward(model, geometry, item)
    changed = model(
        geometry=geometry,
        x_source_branch=item.x_source_branch.unsqueeze(0) * 0.0,
        y_source_branch=item.y_source_branch.unsqueeze(0) * 0.0,
        x_source_amplitude=item.x_source_amplitude.unsqueeze(0),
        y_source_amplitude=item.y_source_amplitude.unsqueeze(0),
        x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
        y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
    )

    assert not torch.allclose(original, changed)


def test_complex_model_scales_physical_output_by_source_amplitude(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model()

    original = _forward(model, geometry, item)
    doubled = model(
        geometry=geometry,
        x_source_branch=item.x_source_branch.unsqueeze(0),
        y_source_branch=item.y_source_branch.unsqueeze(0),
        x_source_amplitude=2.0 * item.x_source_amplitude.unsqueeze(0),
        y_source_amplitude=2.0 * item.y_source_amplitude.unsqueeze(0),
        x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
        y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
    )

    torch.testing.assert_close(doubled, 2.0 * original)


def test_complex_convection_term_adds_primary_and_transverse_channels(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    coefficient_terms = CouplingCoefficientTermsConfig(
        diffusion=True,
        convection=True,
        reaction=True,
    )
    dataset = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        coefficient_terms=coefficient_terms,
    )
    item = dataset[0]
    model = _model(coefficient_terms=coefficient_terms)

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.active_coefficient_terms == (
        "diffusion",
        "convection_primary",
        "convection_transverse",
        "reaction",
    )
    assert model.coefficient_branch_channels == 4
    assert model.coefficient_branch_input_dim == 16


def test_complex_model_supports_product_fuser_and_source_only_branch(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=False,
            convection=False,
            reaction=False,
        ),
    )
    item = dataset[0]
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            branch_fusion=CouplingBranchFusionConfig(mode="product_fuser"),
            coefficient_terms=dataset.coefficient_terms,
            axis_1d_trunk=Axis1DTrunkConfig(num_frequencies=1, max_frequency=1.0),
        )
    )

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.branch_coefficient is None


def test_complex_transverse_features_use_global_normalized_coordinate(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    model = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            axis_1d_trunk=Axis1DTrunkConfig(num_frequencies=1, max_frequency=1.0),
        )
    )

    features = model._transverse_features(geometry, "x")
    expected_phase = 2.0 * torch.pi * geometry.x_segment_y.unsqueeze(-1)
    expected = torch.cat((expected_phase.sin(), expected_phase.cos()), dim=-1)

    torch.testing.assert_close(features, expected)


def test_complex_geometry_feature_contract_excludes_raw_axis_label(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    model = _model()

    features = model._geometry_features(geometry, "x")

    assert features.shape == (geometry.num_x_segments, 6)
    torch.testing.assert_close(features[:, -3], geometry.x_segment_length)
    torch.testing.assert_close(features[:, -2], geometry.x_segment_length.pow(2))
    torch.testing.assert_close(features[:, -1], 1.0 / geometry.x_segment_length)


def test_complex_model_rejects_trunk_positional_encoding_enabled():
    with pytest.raises(ValueError, match="trunk_positional_encoding.enabled"):
        ComplexCouplingNet(
            CouplingModelConfig(
                branch_input_dim=4,
                hidden_dim=4,
                depth=1,
                dtype=torch.float64,
                trunk_positional_encoding=CouplingTrunkPositionalEncodingConfig(
                    enabled=True,
                ),
            )
        )


def test_complex_model_rejects_smooth_mask_balance_projection():
    with pytest.raises(ValueError, match="only symmetric"):
        ComplexCouplingNet(
            CouplingModelConfig(
                branch_input_dim=4,
                hidden_dim=4,
                depth=1,
                dtype=torch.float64,
                balance_projection=BalanceProjectionConfig(mode="smooth_mask"),
            )
        )


def test_complex_model_rejects_legacy_raw_unit_checkpoint(tmp_path):
    model = _model()
    legacy_state = dict(model.state_dict())
    legacy_state.pop("_output_contract_version")
    checkpoint = tmp_path / "legacy_complex_coupling.safetensors"
    save_state_dict_safetensors(legacy_state, checkpoint)

    with pytest.raises(ValueError, match="Legacy complex CouplingNet checkpoint"):
        load_state_dict_auto(_model(), checkpoint)


def test_complex_model_loads_matching_output_contract_checkpoint(tmp_path):
    model = _model()
    checkpoint = tmp_path / "complex_coupling_v2.safetensors"
    save_state_dict_safetensors(model.state_dict(), checkpoint)

    loaded = _model()
    load_state_dict_auto(loaded, checkpoint)

    assert int(loaded._output_contract_version.item()) == 2
