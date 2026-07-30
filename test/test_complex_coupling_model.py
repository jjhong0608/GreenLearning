from __future__ import annotations

import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_projection import apply_complex_balance_projection
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ComplexPreProjectionFusionConfig,
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
    transverse_trunk_enabled: bool = True,
    transverse_trunk_fusion: str = "product",
    coefficient_terms: CouplingCoefficientTermsConfig | None = None,
    pre_projection_fusion_enabled: bool = False,
    pre_projection_fusion_hidden_dim: int = 8,
    pre_projection_fusion_depth: int = 1,
) -> ComplexCouplingNet:
    torch.manual_seed(0)
    return ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            pre_projection_fusion=ComplexPreProjectionFusionConfig(
                enabled=pre_projection_fusion_enabled,
                hidden_dim=pre_projection_fusion_hidden_dim,
                depth=pre_projection_fusion_depth,
            ),
            coefficient_terms=coefficient_terms or CouplingCoefficientTermsConfig(),
            branch_fusion=CouplingBranchFusionConfig(mode=fusion_mode),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                num_frequencies=2,
                max_frequency=2.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=transverse_trunk_enabled,
                    fusion=transverse_trunk_fusion,
                    length_context=True,
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
        rhs_phys=item.rhs_valid.unsqueeze(0),
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
    assert model.trunk_transverse is not None
    assert any(key.startswith("trunk_transverse.") for key in model.state_dict())
    assert model.geometry_feature_dim == 6
    assert model.transverse_feature_dim == 4
    assert not hasattr(model, "axis_one_hot")
    assert int(model._output_contract_version.item()) == 6


def test_complex_pre_projection_fusion_disabled_preserves_state_surface(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(pre_projection_fusion_enabled=False)

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.pre_projection_fusion is None
    assert not any(
        key.startswith("pre_projection_fusion.") for key in model.state_dict()
    )


def test_complex_pre_projection_fusion_enabled_is_identity_initialized(tmp_path):
    geometry, item = _build_item(tmp_path)
    disabled = _model(pre_projection_fusion_enabled=False)
    enabled = _model(pre_projection_fusion_enabled=True)

    disabled_output = _forward(disabled, geometry, item)
    enabled_output, diagnostics = enabled.forward_with_fusion_diagnostics(
        geometry=geometry,
        x_source_branch=item.x_source_branch.unsqueeze(0),
        y_source_branch=item.y_source_branch.unsqueeze(0),
        x_source_amplitude=item.x_source_amplitude.unsqueeze(0),
        y_source_amplitude=item.y_source_amplitude.unsqueeze(0),
        x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
        y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
        rhs_phys=item.rhs_valid.unsqueeze(0),
    )

    assert diagnostics is not None
    config = BalanceProjectionConfig(enabled=True, mode="physical_symmetric")
    disabled_projection = apply_complex_balance_projection(
        disabled_output,
        item.rhs_valid.unsqueeze(0),
        geometry,
        config,
    )
    enabled_projection = apply_complex_balance_projection(
        enabled_output,
        item.rhs_valid.unsqueeze(0),
        geometry,
        config,
    )
    torch.testing.assert_close(
        enabled_projection.projected_response,
        disabled_projection.projected_response,
    )
    torch.testing.assert_close(
        diagnostics.physical_residual,
        torch.zeros_like(diagnostics.physical_residual),
    )
    assert enabled.pre_projection_fusion is not None
    assert any(key.startswith("pre_projection_fusion.") for key in enabled.state_dict())
    assert int(enabled._output_contract_version.item()) == 6


def test_complex_pre_projection_fusion_requires_rhs(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(pre_projection_fusion_enabled=True)

    with pytest.raises(ValueError, match="rhs_phys is required"):
        model(
            geometry=geometry,
            x_source_branch=item.x_source_branch.unsqueeze(0),
            y_source_branch=item.y_source_branch.unsqueeze(0),
            x_source_amplitude=item.x_source_amplitude.unsqueeze(0),
            y_source_amplitude=item.y_source_amplitude.unsqueeze(0),
            x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
            y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
        )


def test_complex_pre_projection_fusion_supports_torch_compile(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(pre_projection_fusion_enabled=True)
    compiled = torch.compile(model, backend="eager")

    output = compiled(
        geometry=geometry,
        x_source_branch=item.x_source_branch.unsqueeze(0),
        y_source_branch=item.y_source_branch.unsqueeze(0),
        x_source_amplitude=item.x_source_amplitude.unsqueeze(0),
        y_source_amplitude=item.y_source_amplitude.unsqueeze(0),
        x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
        y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
        rhs_phys=item.rhs_valid.unsqueeze(0),
    )
    output.square().mean().backward()

    assert output.shape == (1, 2, geometry.num_points)
    assert model.pre_projection_fusion is not None
    final_layer = model.pre_projection_fusion.residual_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0


def test_complex_pre_projection_residual_integrates_with_model(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(pre_projection_fusion_enabled=True)
    assert model.pre_projection_fusion is not None
    final_layer = model.pre_projection_fusion.residual_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        assert final_layer.bias is not None
        final_layer.bias.fill_(0.25)

    output, diagnostics = model.forward_with_fusion_diagnostics(
        geometry=geometry,
        x_source_branch=item.x_source_branch.unsqueeze(0),
        y_source_branch=item.y_source_branch.unsqueeze(0),
        x_source_amplitude=item.x_source_amplitude.unsqueeze(0),
        y_source_amplitude=item.y_source_amplitude.unsqueeze(0),
        x_coefficient_branch=item.x_coefficient_branch.unsqueeze(0),
        y_coefficient_branch=item.y_coefficient_branch.unsqueeze(0),
        rhs_phys=item.rhs_valid.unsqueeze(0),
    )

    assert output.shape == (1, 2, geometry.num_points)
    assert diagnostics is not None
    torch.testing.assert_close(
        diagnostics.fused_difference,
        diagnostics.base_difference + diagnostics.physical_residual,
    )
    torch.testing.assert_close(
        diagnostics.fused_physical.sum(dim=1),
        item.rhs_valid.unsqueeze(0),
    )


def test_complex_pre_projection_residual_uses_only_two_inputs(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(
        pre_projection_fusion_enabled=True,
        pre_projection_fusion_hidden_dim=7,
        pre_projection_fusion_depth=2,
    )

    assert model.pre_projection_fusion is not None
    first_layer = model.pre_projection_fusion.residual_mlp[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 2
    assert first_layer.out_features == 7
    output = _forward(model, geometry, item)
    assert output.shape == (1, 2, geometry.num_points)


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


def test_complex_pointwise_transverse_trunk_uses_cross_axis_length_context(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    model = _model()

    x_features = model.transverse_length_context_features(geometry, "x")
    y_features = model.transverse_length_context_features(geometry, "y")
    x_length = geometry.x_lengths_for_valid_points()
    y_length = geometry.y_lengths_for_valid_points()
    reference = torch.tensor(1.0, dtype=torch.float64)

    torch.testing.assert_close(x_features[:, 0], geometry.y_local_t)
    torch.testing.assert_close(y_features[:, 0], geometry.x_local_t)
    torch.testing.assert_close(x_features[:, 1], torch.log(y_length / reference))
    torch.testing.assert_close(y_features[:, 1], torch.log(x_length / reference))
    torch.testing.assert_close(x_features[:, 2], torch.log(x_length / y_length))
    torch.testing.assert_close(y_features[:, 2], torch.log(y_length / x_length))
    torch.testing.assert_close(x_features[:, 3], y_features[:, 3])


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


def test_complex_model_scales_response_output_by_source_amplitude(tmp_path):
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


def test_complex_model_scales_raw_response_by_primary_length_squared(
    tmp_path,
    monkeypatch,
):
    geometry, item = _build_item(tmp_path)
    model = _model()

    response = _forward(model, geometry, item)
    monkeypatch.setattr(
        model,
        "_primary_length_squared",
        lambda geometry, axis: torch.ones(  # noqa: ARG005
            geometry.num_points,
            dtype=geometry.coords_valid.dtype,
            device=geometry.coords_valid.device,
        ),
    )
    response_without_scale = _forward(model, geometry, item)
    expected_scale = torch.stack(
        (
            geometry.x_lengths_for_valid_points().square(),
            geometry.y_lengths_for_valid_points().square(),
        ),
        dim=0,
    ).unsqueeze(0)

    torch.testing.assert_close(response, response_without_scale * expected_scale)


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
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            coefficient_terms=dataset.coefficient_terms,
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
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
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
    with pytest.raises(ValueError, match="requires.*physical_symmetric"):
        ComplexCouplingNet(
            CouplingModelConfig(
                branch_input_dim=4,
                hidden_dim=4,
                depth=1,
                dtype=torch.float64,
                balance_projection=BalanceProjectionConfig(mode="smooth_mask"),
            )
        )


def test_complex_model_accepts_physical_symmetric_projection():
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

    assert model.config.balance_projection.mode == "physical_symmetric"


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
    checkpoint = tmp_path / "complex_coupling_v6.safetensors"
    save_state_dict_safetensors(model.state_dict(), checkpoint)

    loaded = _model()
    load_state_dict_auto(loaded, checkpoint)

    assert int(loaded._output_contract_version.item()) == 6


def test_complex_model_loads_single_residual_fuser_checkpoint_surface(tmp_path):
    model = _model(pre_projection_fusion_enabled=True)
    checkpoint = tmp_path / "complex_coupling_v6_single_residual_fuser.safetensors"
    save_state_dict_safetensors(model.state_dict(), checkpoint)

    loaded = _model(pre_projection_fusion_enabled=True)
    load_state_dict_auto(loaded, checkpoint)

    assert loaded.pre_projection_fusion is not None
    assert hasattr(loaded.pre_projection_fusion, "residual_mlp")
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_complex_model_rejects_legacy_enabled_fuser_checkpoint(tmp_path):
    model = _model(pre_projection_fusion_enabled=True)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("pre_projection_fusion.")
    }
    legacy_state.update(
        {
            "pre_projection_fusion.linear_correction.weight": torch.zeros(
                (1, 2), dtype=torch.float64
            ),
            "pre_projection_fusion.nonlinear_correction.0.weight": torch.zeros(
                (8, 8), dtype=torch.float64
            ),
            "pre_projection_fusion.nonlinear_correction.0.bias": torch.zeros(
                8, dtype=torch.float64
            ),
            "pre_projection_fusion.nonlinear_correction.2.weight": torch.zeros(
                (1, 8), dtype=torch.float64
            ),
            "pre_projection_fusion.nonlinear_correction.2.bias": torch.zeros(
                1, dtype=torch.float64
            ),
            "pre_projection_fusion.gate_logit": torch.zeros((), dtype=torch.float64),
        }
    )
    checkpoint = tmp_path / "complex_coupling_v6_legacy_fuser.safetensors"
    save_state_dict_safetensors(legacy_state, checkpoint)

    with pytest.raises(RuntimeError, match="Missing key.*residual_mlp"):
        load_state_dict_auto(_model(pre_projection_fusion_enabled=True), checkpoint)


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_complex_model_rejects_old_versioned_checkpoint(tmp_path, version):
    state = dict(_model().state_dict())
    state["_output_contract_version"] = torch.tensor(version, dtype=torch.int64)
    checkpoint = tmp_path / f"complex_coupling_v{version}.safetensors"
    save_state_dict_safetensors(state, checkpoint)

    with pytest.raises(ValueError, match="require retraining"):
        load_state_dict_auto(_model(), checkpoint)
