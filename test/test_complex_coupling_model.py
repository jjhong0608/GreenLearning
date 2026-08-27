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
    ComplexCrossAxisReconstructionConfig,
    ComplexPreProjectionFusionConfig,
    CouplingBranchFusionConfig,
    CouplingCoefficientTermsConfig,
    CouplingGeometryBranchConfig,
    CouplingModelConfig,
    CouplingTrunkPositionalEncodingConfig,
    FixedLineTransverseBranchConfig,
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
    geometry_branch_enabled: bool = True,
    fixed_line_transverse_branch_enabled: bool = True,
    transverse_trunk_enabled: bool = True,
    transverse_trunk_fusion: str = "product",
    coefficient_terms: CouplingCoefficientTermsConfig | None = None,
    pre_projection_fusion_enabled: bool = False,
    pre_projection_fusion_mode: str = "residual",
    pre_projection_fusion_hidden_dim: int = 8,
    pre_projection_fusion_depth: int = 1,
    pre_projection_fusion_final_layer_init_scale: float = 0.0,
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
                mode=pre_projection_fusion_mode,
                hidden_dim=pre_projection_fusion_hidden_dim,
                depth=pre_projection_fusion_depth,
                final_layer_init_scale=(pre_projection_fusion_final_layer_init_scale),
            ),
            coefficient_terms=coefficient_terms or CouplingCoefficientTermsConfig(),
            branch_fusion=CouplingBranchFusionConfig(mode=fusion_mode),
            geometry_branch=CouplingGeometryBranchConfig(
                enabled=geometry_branch_enabled
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                num_frequencies=2,
                max_frequency=2.0,
                fixed_line_transverse_branch=FixedLineTransverseBranchConfig(
                    enabled=fixed_line_transverse_branch_enabled
                ),
                transverse_trunk=TransverseTrunkConfig(
                    enabled=transverse_trunk_enabled,
                    fusion=transverse_trunk_fusion,
                    length_context=transverse_trunk_enabled,
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


def test_complex_all_on_defaults_preserve_branch_fuser_surface() -> None:
    model = _model(fusion_mode="product_fuser")

    assert model.active_branch_components == (
        "source",
        "coefficient",
        "fixed_line_transverse",
        "geometry",
    )
    assert model.branch_fuser is not None
    assert model.branch_fuser.in_features == 5 * model.hidden_dim
    architecture = model.architecture_provenance()
    assert architecture["branch_component_count"] == 4
    assert architecture["branch_fusion_includes_elementwise_product"] is True
    assert architecture["branch_fuser_features"] == [
        "source",
        "coefficient",
        "fixed_line_transverse",
        "geometry",
        "elementwise_product",
    ]


def test_complex_all_on_concat_fuser_uses_active_embeddings_only() -> None:
    model = _model(fusion_mode="concat_fuser")

    assert model.branch_fuser is not None
    assert model.branch_fuser.in_features == 4 * model.hidden_dim
    architecture = model.architecture_provenance()
    assert architecture["branch_fusion_configured"] == "concat_fuser"
    assert architecture["branch_fusion_effective"] == "concat_fuser"
    assert architecture["branch_fusion_includes_elementwise_product"] is False
    assert architecture["branch_fuser_features"] == [
        "source",
        "coefficient",
        "fixed_line_transverse",
        "geometry",
    ]


def test_complex_new_auxiliary_defaults_preserve_all_on_parameters() -> None:
    explicit = _model()
    torch.manual_seed(0)
    omitted = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                num_frequencies=2,
                max_frequency=2.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    fusion="product",
                    length_context=True,
                ),
            ),
        )
    )

    assert omitted.state_dict().keys() == explicit.state_dict().keys()
    for key, value in explicit.state_dict().items():
        torch.testing.assert_close(omitted.state_dict()[key], value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("geometry_enabled", "fixed_enabled", "pointwise_enabled"),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_complex_auxiliary_toggles_are_independent_and_differentiable(
    tmp_path,
    geometry_enabled,
    fixed_enabled,
    pointwise_enabled,
) -> None:
    geometry, item = _build_item(tmp_path)
    model = _model(
        fusion_mode="product_fuser",
        geometry_branch_enabled=geometry_enabled,
        fixed_line_transverse_branch_enabled=fixed_enabled,
        transverse_trunk_enabled=pointwise_enabled,
    )

    output = _forward(model, geometry, item)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert (model.branch_geometry is not None) is geometry_enabled
    assert (model.branch_transverse is not None) is fixed_enabled
    assert (model.trunk_transverse is not None) is pointwise_enabled
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_complex_minimal_network_uses_three_h_product_fuser(tmp_path) -> None:
    geometry, item = _build_item(tmp_path)
    model = _model(
        fusion_mode="product_fuser",
        geometry_branch_enabled=False,
        fixed_line_transverse_branch_enabled=False,
        transverse_trunk_enabled=False,
    )

    output = _forward(model, geometry, item)

    assert output.shape == (1, 2, geometry.num_points)
    assert model.active_branch_components == ("source", "coefficient")
    assert model.branch_geometry is None
    assert model.branch_transverse is None
    assert model.trunk_transverse is None
    assert model.trunk_fuser is None
    assert model.branch_fuser is not None
    assert model.branch_fuser.in_features == 3 * model.hidden_dim
    architecture = model.architecture_provenance()
    assert architecture["branch_fusion_includes_elementwise_product"] is True
    assert architecture["branch_fuser_features"] == [
        "source",
        "coefficient",
        "elementwise_product",
    ]
    captured: list[torch.Tensor] = []

    def capture_input(_module, args):
        captured.append(args[0])

    first = torch.randn(2, 3, model.hidden_dim, dtype=torch.float64, requires_grad=True)
    second = torch.randn(
        2, 3, model.hidden_dim, dtype=torch.float64, requires_grad=True
    )
    handle = model.branch_fuser.register_forward_pre_hook(capture_input)
    fused = model._fuse_branch_components([first, second])
    handle.remove()
    torch.testing.assert_close(
        captured[0],
        torch.cat((first, second, first * second), dim=-1),
    )
    fused.square().mean().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()
    assert not any(
        key.startswith(
            (
                "branch_geometry.",
                "branch_transverse.",
                "trunk_transverse.",
                "trunk_fuser.",
            )
        )
        for key in model.state_dict()
    )


def test_complex_minimal_concat_fuser_uses_two_h_without_product(tmp_path) -> None:
    geometry, item = _build_item(tmp_path)
    model = _model(
        fusion_mode="concat_fuser",
        geometry_branch_enabled=False,
        fixed_line_transverse_branch_enabled=False,
        transverse_trunk_enabled=False,
    )
    assert model.branch_fuser is not None
    captured: list[torch.Tensor] = []

    def capture_input(_module, args):
        captured.append(args[0])

    handle = model.branch_fuser.register_forward_pre_hook(capture_input)
    output = _forward(model, geometry, item)
    handle.remove()

    assert output.shape == (1, 2, geometry.num_points)
    assert torch.isfinite(output).all()
    assert model.branch_fuser.in_features == 2 * model.hidden_dim
    assert captured
    assert all(value.shape[-1] == 2 * model.hidden_dim for value in captured)
    architecture = model.architecture_provenance()
    assert architecture["branch_fusion_includes_elementwise_product"] is False
    assert architecture["branch_fuser_features"] == ["source", "coefficient"]

    first = torch.randn(2, 3, model.hidden_dim, dtype=torch.float64, requires_grad=True)
    second = torch.randn(
        2, 3, model.hidden_dim, dtype=torch.float64, requires_grad=True
    )
    captured.clear()
    handle = model.branch_fuser.register_forward_pre_hook(capture_input)
    fused = model._fuse_branch_components([first, second])
    handle.remove()
    torch.testing.assert_close(captured[0], torch.cat((first, second), dim=-1))

    pending = [fused.grad_fn]
    visited_nodes: set[int] = set()
    autograd_nodes: list[str] = []
    while pending:
        node = pending.pop()
        if node is None or id(node) in visited_nodes:
            continue
        visited_nodes.add(id(node))
        autograd_nodes.append(type(node).__name__)
        pending.extend(next_node for next_node, _index in node.next_functions)
    assert not any(name.startswith("MulBackward") for name in autograd_nodes)
    fused.square().mean().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


@pytest.mark.parametrize("fusion_mode", ["product_fuser", "concat_fuser"])
def test_complex_single_active_branch_bypasses_configured_fuser(fusion_mode) -> None:
    model = _model(
        fusion_mode=fusion_mode,
        geometry_branch_enabled=False,
        fixed_line_transverse_branch_enabled=False,
        transverse_trunk_enabled=False,
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=False,
            convection=False,
            reaction=False,
        ),
    )
    feature = torch.randn(2, 3, model.hidden_dim, dtype=torch.float64)

    fused = model._fuse_branch_components([feature])

    assert model.active_branch_components == ("source",)
    assert model.branch_fuser is None
    assert fused is feature
    assert model.architecture_provenance()["branch_fusion_effective"] == "identity"


def test_cross_axis_reconstruction_does_not_change_model_state_or_contract() -> None:
    disabled = _model()
    enabled = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            cross_axis_reconstruction=ComplexCrossAxisReconstructionConfig(
                enabled=True
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                num_frequencies=2,
                max_frequency=2.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )

    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    assert enabled.OUTPUT_CONTRACT_VERSION == disabled.OUTPUT_CONTRACT_VERSION == 6


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


def test_complex_pre_projection_absolute_zero_init_uses_symmetric_split(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(
        pre_projection_fusion_enabled=True,
        pre_projection_fusion_mode="absolute",
    )

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
        torch.zeros_like(diagnostics.fused_difference),
    )
    torch.testing.assert_close(
        diagnostics.fused_physical[:, 0],
        0.5 * item.rhs_valid.unsqueeze(0),
    )
    torch.testing.assert_close(
        diagnostics.fused_physical[:, 1],
        0.5 * item.rhs_valid.unsqueeze(0),
    )
    assert model.pre_projection_fusion is not None
    assert model.pre_projection_fusion.mode == "absolute"
    assert int(model._output_contract_version.item()) == 6


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


@pytest.mark.parametrize("mode", ["residual", "absolute"])
def test_complex_pre_projection_fusion_supports_torch_compile(tmp_path, mode):
    geometry, item = _build_item(tmp_path)
    model = _model(
        pre_projection_fusion_enabled=True,
        pre_projection_fusion_mode=mode,
    )
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
        diagnostics.base_difference + diagnostics.difference_delta,
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


def test_complex_pointwise_transverse_trunk_concat_fuser_outputs_shape(tmp_path):
    geometry, item = _build_item(tmp_path)
    model = _model(
        transverse_trunk_enabled=True,
        transverse_trunk_fusion="concat_fuser",
    )

    output = _forward(model, geometry, item)
    provenance = model.architecture_provenance()

    assert output.shape == (1, 2, geometry.num_points)
    assert model.trunk_transverse is not None
    assert model.trunk_fuser is not None
    assert model.trunk_fuser.in_features == 2 * model.hidden_dim
    assert provenance["pointwise_transverse_trunk_fuser_features"] == [
        "primary",
        "transverse",
    ]
    assert provenance["pointwise_transverse_trunk_fuser_input_dim"] == (
        2 * model.hidden_dim
    )
    assert not provenance[
        "pointwise_transverse_trunk_fusion_includes_elementwise_product"
    ]


@pytest.mark.parametrize(
    ("checkpoint_mode", "target_mode"),
    [("product_fuser", "concat_fuser"), ("concat_fuser", "product_fuser")],
)
def test_complex_model_rejects_transverse_fuser_checkpoint_mismatch(
    tmp_path,
    checkpoint_mode,
    target_mode,
):
    checkpoint = tmp_path / f"complex_coupling_trunk_{checkpoint_mode}.safetensors"
    save_state_dict_safetensors(
        _model(transverse_trunk_fusion=checkpoint_mode).state_dict(),
        checkpoint,
    )

    with pytest.raises(ValueError, match="trunk_fuser.weight shape"):
        load_state_dict_auto(
            _model(transverse_trunk_fusion=target_mode),
            checkpoint,
        )


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


@pytest.mark.parametrize("fusion_mode", ["product", "concat_fuser"])
@pytest.mark.parametrize("minimal", [False, True])
def test_complex_model_scales_response_output_by_source_amplitude(
    tmp_path,
    minimal,
    fusion_mode,
):
    geometry, item = _build_item(tmp_path)
    model = _model(
        fusion_mode=fusion_mode,
        geometry_branch_enabled=not minimal,
        fixed_line_transverse_branch_enabled=not minimal,
        transverse_trunk_enabled=not minimal,
    )

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


@pytest.mark.parametrize("fusion_mode", ["product", "concat_fuser"])
@pytest.mark.parametrize("minimal", [False, True])
def test_complex_model_scales_raw_response_by_primary_length_squared(
    tmp_path,
    monkeypatch,
    minimal,
    fusion_mode,
):
    geometry, item = _build_item(tmp_path)
    model = _model(
        fusion_mode=fusion_mode,
        geometry_branch_enabled=not minimal,
        fixed_line_transverse_branch_enabled=not minimal,
        transverse_trunk_enabled=not minimal,
    )

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


def test_complex_model_requires_length_context_only_for_enabled_transverse_trunk():
    with pytest.raises(ValueError, match="length_context=true"):
        ComplexCouplingNet(
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
                        length_context=False,
                    ),
                ),
            )
        )

    disabled = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=4,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=False,
                    length_context=False,
                ),
            ),
        )
    )
    assert disabled.trunk_transverse is None


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


def test_complex_model_accepts_column_diagonal_green_response_projection():
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
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )

    assert model.config.balance_projection.mode == "column_diagonal_green_response"
    assert model.OUTPUT_CONTRACT_VERSION == 6


def test_complex_model_accepts_symmetric_tangent_projection_without_state_change():
    torch.manual_seed(123)
    symmetric = _model()
    torch.manual_seed(123)
    tangent = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
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
                num_frequencies=2,
                max_frequency=2.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )

    assert tangent.config.balance_projection.mode == (
        "symmetric_tangent_green_response"
    )
    assert tangent.OUTPUT_CONTRACT_VERSION == 6
    assert symmetric.state_dict().keys() == tangent.state_dict().keys()
    for key, value in symmetric.state_dict().items():
        assert value.shape == tangent.state_dict()[key].shape


def test_column_diagonal_projection_does_not_change_model_state_dict_surface():
    torch.manual_seed(123)
    symmetric = _model()
    torch.manual_seed(123)
    column = ComplexCouplingNet(
        CouplingModelConfig(
            branch_input_dim=4,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            balance_projection=BalanceProjectionConfig(
                mode="column_diagonal_green_response",
                column_diagonal_green_response={"gain_exponent": 0.25},
            ),
            axis_1d_trunk=Axis1DTrunkConfig(
                enabled=True,
                num_frequencies=2,
                max_frequency=2.0,
                transverse_trunk=TransverseTrunkConfig(
                    enabled=True,
                    length_context=True,
                ),
            ),
        )
    )

    assert symmetric.state_dict().keys() == column.state_dict().keys()
    assert (
        column.config.balance_projection.column_diagonal_green_response.gain_exponent
        == 0.25
    )
    for key, value in symmetric.state_dict().items():
        assert value.shape == column.state_dict()[key].shape


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


def test_complex_model_loads_matching_concat_fuser_checkpoint(tmp_path):
    model = _model(
        fusion_mode="concat_fuser",
        geometry_branch_enabled=False,
        fixed_line_transverse_branch_enabled=False,
        transverse_trunk_enabled=False,
    )
    checkpoint = tmp_path / "complex_coupling_concat_fuser.safetensors"
    save_state_dict_safetensors(model.state_dict(), checkpoint)

    loaded = _model(
        fusion_mode="concat_fuser",
        geometry_branch_enabled=False,
        fixed_line_transverse_branch_enabled=False,
        transverse_trunk_enabled=False,
    )
    load_state_dict_auto(loaded, checkpoint)

    assert loaded.branch_fuser is not None
    assert loaded.branch_fuser.in_features == 2 * loaded.hidden_dim
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


@pytest.mark.parametrize(
    ("checkpoint_mode", "target_mode"),
    [("product_fuser", "concat_fuser"), ("concat_fuser", "product_fuser")],
)
def test_complex_model_rejects_concat_product_fuser_checkpoint_mismatch(
    tmp_path,
    checkpoint_mode,
    target_mode,
):
    kwargs = {
        "geometry_branch_enabled": False,
        "fixed_line_transverse_branch_enabled": False,
        "transverse_trunk_enabled": False,
    }
    checkpoint = tmp_path / f"complex_coupling_{checkpoint_mode}.safetensors"
    save_state_dict_safetensors(
        _model(fusion_mode=checkpoint_mode, **kwargs).state_dict(),
        checkpoint,
    )

    with pytest.raises(ValueError, match="branch_fuser.weight shape"):
        load_state_dict_auto(_model(fusion_mode=target_mode, **kwargs), checkpoint)


@pytest.mark.parametrize("checkpoint_is_minimal", [False, True])
def test_complex_model_rejects_full_minimal_checkpoint_mismatch(
    tmp_path,
    checkpoint_is_minimal,
):
    minimal_kwargs = {
        "fusion_mode": "product_fuser",
        "geometry_branch_enabled": False,
        "fixed_line_transverse_branch_enabled": False,
        "transverse_trunk_enabled": False,
    }
    full_kwargs = {"fusion_mode": "product_fuser"}
    checkpoint_kwargs = minimal_kwargs if checkpoint_is_minimal else full_kwargs
    target_kwargs = full_kwargs if checkpoint_is_minimal else minimal_kwargs
    checkpoint = tmp_path / "complex_coupling_architecture_mismatch.safetensors"
    save_state_dict_safetensors(_model(**checkpoint_kwargs).state_dict(), checkpoint)

    with pytest.raises(ValueError, match="checkpoint architecture mismatch"):
        load_state_dict_auto(_model(**target_kwargs), checkpoint)


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


def test_complex_model_loads_unmarked_v6_residual_fuser_checkpoint(tmp_path):
    model = _model(pre_projection_fusion_enabled=True)
    state = dict(model.state_dict())
    state.pop(ComplexCouplingNet.PRE_PROJECTION_FUSION_MODE_KEY)
    checkpoint = tmp_path / "complex_coupling_v6_unmarked_residual_fuser.safetensors"
    save_state_dict_safetensors(state, checkpoint)

    loaded = _model(pre_projection_fusion_enabled=True)
    load_state_dict_auto(loaded, checkpoint)

    assert loaded.pre_projection_fusion is not None
    assert loaded.pre_projection_fusion.mode == "residual"
    assert (
        int(
            loaded.state_dict()[
                ComplexCouplingNet.PRE_PROJECTION_FUSION_MODE_KEY
            ].item()
        )
        == 0
    )


def test_complex_model_rejects_unmarked_residual_checkpoint_in_absolute_mode(
    tmp_path,
):
    model = _model(pre_projection_fusion_enabled=True)
    state = dict(model.state_dict())
    state.pop(ComplexCouplingNet.PRE_PROJECTION_FUSION_MODE_KEY)
    checkpoint = tmp_path / "complex_coupling_v6_unmarked_residual_fuser.safetensors"
    save_state_dict_safetensors(state, checkpoint)

    with pytest.raises(ValueError, match="legacy residual-mode checkpoint"):
        load_state_dict_auto(
            _model(
                pre_projection_fusion_enabled=True,
                pre_projection_fusion_mode="absolute",
            ),
            checkpoint,
        )


@pytest.mark.parametrize(
    ("checkpoint_mode", "target_mode"),
    [
        ("residual", "absolute"),
        ("absolute", "residual"),
    ],
)
def test_complex_model_rejects_pre_projection_fusion_mode_mismatch(
    tmp_path,
    checkpoint_mode,
    target_mode,
):
    model = _model(
        pre_projection_fusion_enabled=True,
        pre_projection_fusion_mode=checkpoint_mode,
    )
    checkpoint = tmp_path / f"complex_coupling_v6_{checkpoint_mode}_fuser.safetensors"
    save_state_dict_safetensors(model.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="cannot be cross-loaded"):
        load_state_dict_auto(
            _model(
                pre_projection_fusion_enabled=True,
                pre_projection_fusion_mode=target_mode,
            ),
            checkpoint,
        )


def test_complex_model_loads_absolute_fuser_checkpoint(tmp_path):
    model = _model(
        pre_projection_fusion_enabled=True,
        pre_projection_fusion_mode="absolute",
    )
    checkpoint = tmp_path / "complex_coupling_v6_absolute_fuser.safetensors"
    save_state_dict_safetensors(model.state_dict(), checkpoint)

    loaded = _model(
        pre_projection_fusion_enabled=True,
        pre_projection_fusion_mode="absolute",
    )
    load_state_dict_auto(loaded, checkpoint)

    assert loaded.pre_projection_fusion is not None
    assert loaded.pre_projection_fusion.mode == "absolute"
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
