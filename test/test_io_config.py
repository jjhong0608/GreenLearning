import pytest
import torch

from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    ComplexCrossAxisReconstructionConfig,
    ComplexPostLineSearchStationarityConfig,
    ComplexPreProjectionFusionConfig,
    ComplexResponseTrustConfig,
    CouplingBranchFusionConfig,
    CouplingCoefficientTermsConfig,
    CouplingModelConfig,
    CouplingTrainingConfig,
    CouplingTrunkPositionalEncodingConfig,
    GreenResponseFeatureConfig,
    ModelConfig,
    SourceStencilLiftConfig,
    TransverseTrunkConfig,
    validate_complex_post_line_search_stationarity_config,
    validate_complex_response_trust_config,
    validate_unit_square_coupling_training_config,
)
from greenonet.coupling_model import CouplingNet
from greenonet.model import GreenONetModel


def _assert_state_dict_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def test_complex_pre_projection_fusion_config_round_trip():
    from greenonet.io import _deserialize_config, _serialize_config

    config = CouplingModelConfig(
        pre_projection_fusion=ComplexPreProjectionFusionConfig(
            enabled=True,
            mode="absolute",
            hidden_dim=12,
            depth=2,
            eps=1e-10,
            final_layer_init_scale=0.25,
        )
    )

    payload = _serialize_config(config)
    loaded = _deserialize_config(payload, CouplingModelConfig)

    assert isinstance(loaded, CouplingModelConfig)
    assert loaded == config
    assert isinstance(
        loaded.pre_projection_fusion,
        ComplexPreProjectionFusionConfig,
    )
    assert loaded.pre_projection_fusion.mode == "absolute"
    assert loaded.pre_projection_fusion.final_layer_init_scale == 0.25


def test_complex_cross_axis_reconstruction_config_round_trip():
    from greenonet.io import _deserialize_config, _serialize_config

    config = CouplingModelConfig(
        cross_axis_reconstruction=ComplexCrossAxisReconstructionConfig(
            enabled=True,
            gamma=0.75,
            smoothing_steps=3,
            smoothing_relaxation=0.25,
            relative_floor=0.2,
            eps=1.0e-10,
        )
    )

    payload = _serialize_config(config)
    loaded = _deserialize_config(payload, CouplingModelConfig)

    assert isinstance(loaded, CouplingModelConfig)
    assert loaded == config
    assert isinstance(
        loaded.cross_axis_reconstruction,
        ComplexCrossAxisReconstructionConfig,
    )
    assert loaded.cross_axis_reconstruction.enabled is True
    assert loaded.cross_axis_reconstruction.gamma == pytest.approx(0.75)


def test_complex_cross_axis_reconstruction_rejects_unknown_key() -> None:
    with pytest.raises(TypeError, match="cross_axis_reconstruction has unknown keys"):
        ComplexCrossAxisReconstructionConfig.from_raw(
            {"enabled": True, "geometry_ramp": True}
        )


def test_complex_canonical_energy_config_defaults_and_parses_boundary_weight():
    default = CouplingTrainingConfig()
    boundary_off = CouplingTrainingConfig(canonical_energy={"boundary_weight": 0.0})
    tempered = ComplexCanonicalEnergyConfig.from_raw({"boundary_weight": 0.25})

    assert isinstance(default.canonical_energy, ComplexCanonicalEnergyConfig)
    assert default.canonical_energy.boundary_weight == pytest.approx(1.0)
    assert boundary_off.canonical_energy.boundary_weight == pytest.approx(0.0)
    assert tempered.boundary_weight == pytest.approx(0.25)
    assert (
        ComplexCanonicalEnergyConfig.from_raw(
            {"boundary_weight": tempered.boundary_weight}
        )
        == tempered
    )


@pytest.mark.parametrize("boundary_weight", [0.0, 0.1, 1.0])
def test_complex_canonical_energy_config_round_trip(boundary_weight):
    from greenonet.io import _deserialize_config, _serialize_config

    config = CouplingTrainingConfig(
        canonical_energy={"boundary_weight": boundary_weight}
    )

    payload = _serialize_config(config)
    loaded = _deserialize_config(payload, CouplingTrainingConfig)

    assert isinstance(loaded, CouplingTrainingConfig)
    assert isinstance(loaded.canonical_energy, ComplexCanonicalEnergyConfig)
    assert loaded.canonical_energy.boundary_weight == pytest.approx(boundary_weight)


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        ("0", TypeError),
        (-0.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_complex_canonical_energy_rejects_invalid_boundary_weight(
    value,
    error_type,
):
    with pytest.raises(error_type, match="canonical_energy.boundary_weight"):
        ComplexCanonicalEnergyConfig.from_raw({"boundary_weight": value})

    with pytest.raises(TypeError, match="canonical_energy has unknown keys"):
        ComplexCanonicalEnergyConfig.from_raw({"boundary_enabled": False})


def test_unit_square_rejects_nondefault_complex_canonical_energy():
    with pytest.raises(ValueError, match="canonical_energy.*ComplexCouplingTrainer"):
        validate_unit_square_coupling_training_config(
            CouplingTrainingConfig(
                canonical_energy=ComplexCanonicalEnergyConfig(boundary_weight=0.0)
            )
        )

    validate_unit_square_coupling_training_config(CouplingTrainingConfig())


def test_post_line_search_stationarity_config_round_trip_and_defaults():
    from greenonet.io import _deserialize_config, _serialize_config

    default = CouplingTrainingConfig()
    configured = CouplingTrainingConfig(
        post_line_search_stationarity={
            "enabled": True,
            "weight": 1.0e-3,
            "eps": 2.0e-12,
        }
    )
    payload = _serialize_config(configured)
    loaded = _deserialize_config(payload, CouplingTrainingConfig)

    assert isinstance(
        default.post_line_search_stationarity,
        ComplexPostLineSearchStationarityConfig,
    )
    assert default.post_line_search_stationarity.enabled is False
    assert default.post_line_search_stationarity.weight == pytest.approx(1.0)
    assert default.post_line_search_stationarity.eps == pytest.approx(1.0e-12)
    assert isinstance(
        loaded.post_line_search_stationarity,
        ComplexPostLineSearchStationarityConfig,
    )
    assert loaded.post_line_search_stationarity.enabled is True
    assert loaded.post_line_search_stationarity.weight == pytest.approx(1.0e-3)
    assert loaded.post_line_search_stationarity.eps == pytest.approx(2.0e-12)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    (
        ("enabled", 1, TypeError),
        ("weight", True, TypeError),
        ("weight", -1.0, ValueError),
        ("weight", float("nan"), ValueError),
        ("eps", 0.0, ValueError),
        ("eps", float("inf"), ValueError),
    ),
)
def test_post_line_search_stationarity_rejects_invalid_values(
    field,
    value,
    error_type,
):
    with pytest.raises(
        error_type,
        match=rf"post_line_search_stationarity\.{field}",
    ):
        ComplexPostLineSearchStationarityConfig.from_raw({field: value})

    with pytest.raises(
        TypeError,
        match="post_line_search_stationarity has unknown keys",
    ):
        ComplexPostLineSearchStationarityConfig.from_raw({"unknown": 1})


def test_unit_square_rejects_enabled_post_line_search_stationarity():
    with pytest.raises(
        ValueError,
        match="post_line_search_stationarity.*ComplexCouplingTrainer",
    ):
        validate_unit_square_coupling_training_config(
            CouplingTrainingConfig(post_line_search_stationarity={"enabled": True})
        )


def test_post_line_search_stationarity_requires_closed_loop_tangent_projection():
    training = CouplingTrainingConfig(post_line_search_stationarity={"enabled": True})
    valid = BalanceProjectionConfig(
        mode="symmetric_tangent_green_response",
        symmetric_tangent_green_response={
            "eta_strategy": "closed_loop_exact_line_search"
        },
    )

    resolved = validate_complex_post_line_search_stationarity_config(
        training=training,
        balance_projection=valid,
    )
    assert resolved.enabled is True

    with pytest.raises(ValueError, match="symmetric_tangent_green_response"):
        validate_complex_post_line_search_stationarity_config(
            training=training,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        )
    with pytest.raises(ValueError, match="closed_loop_exact_line_search"):
        validate_complex_post_line_search_stationarity_config(
            training=training,
            balance_projection=BalanceProjectionConfig(
                mode="symmetric_tangent_green_response",
                symmetric_tangent_green_response={"eta_strategy": "fixed"},
            ),
        )


def test_response_trust_config_round_trip_and_defaults():
    from greenonet.io import _deserialize_config, _serialize_config

    default = CouplingTrainingConfig()
    configured = CouplingTrainingConfig(
        response_trust={
            "enabled": True,
            "weight": 1.0e-3,
            "trust_weight": 0.025,
            "eps": 2.0e-12,
        }
    )
    loaded = _deserialize_config(
        _serialize_config(configured),
        CouplingTrainingConfig,
    )

    assert isinstance(default.response_trust, ComplexResponseTrustConfig)
    assert default.response_trust == ComplexResponseTrustConfig()
    assert isinstance(loaded.response_trust, ComplexResponseTrustConfig)
    assert loaded.response_trust.enabled is True
    assert loaded.response_trust.weight == pytest.approx(1.0e-3)
    assert loaded.response_trust.trust_weight == pytest.approx(0.025)
    assert loaded.response_trust.eps == pytest.approx(2.0e-12)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    (
        ("enabled", 1, TypeError),
        ("weight", True, TypeError),
        ("weight", -1.0, ValueError),
        ("trust_weight", -1.0, ValueError),
        ("trust_weight", float("nan"), ValueError),
        ("eps", 0.0, ValueError),
        ("eps", float("inf"), ValueError),
    ),
)
def test_response_trust_rejects_invalid_values(field, value, error_type):
    with pytest.raises(error_type, match=rf"response_trust\.{field}"):
        ComplexResponseTrustConfig.from_raw({field: value})

    with pytest.raises(TypeError, match="response_trust has unknown keys"):
        ComplexResponseTrustConfig.from_raw({"unknown": 1})


def test_response_trust_requires_closed_loop_tangent_and_allows_stationarity():
    valid_projection = BalanceProjectionConfig(
        mode="symmetric_tangent_green_response",
        symmetric_tangent_green_response={
            "eta_strategy": "closed_loop_exact_line_search"
        },
    )
    training = CouplingTrainingConfig(response_trust={"enabled": True})

    resolved = validate_complex_response_trust_config(
        training=training,
        balance_projection=valid_projection,
    )
    assert resolved.enabled is True
    assert resolved.trust_weight == pytest.approx(0.01)

    with pytest.raises(ValueError, match="symmetric_tangent_green_response"):
        validate_complex_response_trust_config(
            training=training,
            balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        )
    with pytest.raises(ValueError, match="closed_loop_exact_line_search"):
        validate_complex_response_trust_config(
            training=training,
            balance_projection=BalanceProjectionConfig(
                mode="symmetric_tangent_green_response",
                symmetric_tangent_green_response={"eta_strategy": "fixed"},
            ),
        )
    joint_training = CouplingTrainingConfig(
        response_trust={"enabled": True},
        post_line_search_stationarity={"enabled": True},
    )
    assert validate_complex_response_trust_config(
        training=joint_training,
        balance_projection=valid_projection,
    ).enabled
    assert validate_complex_post_line_search_stationarity_config(
        training=joint_training,
        balance_projection=valid_projection,
    ).enabled


def test_unit_square_rejects_enabled_response_trust():
    with pytest.raises(ValueError, match="response_trust.*ComplexCouplingTrainer"):
        validate_unit_square_coupling_training_config(
            CouplingTrainingConfig(response_trust={"enabled": True})
        )


def test_column_diagonal_green_response_config_round_trip():
    from greenonet.io import _deserialize_config, _serialize_config

    config = CouplingModelConfig(
        balance_projection={
            "enabled": True,
            "mode": "column_diagonal_green_response",
            "column_diagonal_green_response": {
                "gain_squared_eps": 2.5e-11,
                "gain_exponent": 0.25,
            },
        }
    )

    payload = _serialize_config(config)
    loaded = _deserialize_config(payload, CouplingModelConfig)

    assert isinstance(loaded, CouplingModelConfig)
    assert loaded == config
    assert loaded.balance_projection.mode == "column_diagonal_green_response"
    assert (
        loaded.balance_projection.column_diagonal_green_response.gain_squared_eps
        == 2.5e-11
    )
    assert (
        loaded.balance_projection.column_diagonal_green_response.gain_exponent == 0.25
    )


def test_symmetric_tangent_green_response_config_round_trip():
    from greenonet.io import _deserialize_config, _serialize_config

    config = CouplingModelConfig(
        balance_projection={
            "enabled": True,
            "mode": "symmetric_tangent_green_response",
            "symmetric_tangent_green_response": {
                "eta": 0.025,
                "eta_strategy": "closed_loop_exact_line_search",
                "line_search_relative_eps": 4.0e-12,
                "relative_lambda": 0.2,
                "denominator_relative_eps": 2.5e-11,
            },
        }
    )

    payload = _serialize_config(config)
    loaded = _deserialize_config(payload, CouplingModelConfig)

    assert isinstance(loaded, CouplingModelConfig)
    assert loaded == config
    assert loaded.balance_projection.mode == "symmetric_tangent_green_response"
    tangent = loaded.balance_projection.symmetric_tangent_green_response
    assert tangent.eta == pytest.approx(0.025)
    assert tangent.eta_strategy == "closed_loop_exact_line_search"
    assert tangent.line_search_relative_eps == pytest.approx(4.0e-12)
    assert tangent.relative_lambda == pytest.approx(0.2)
    assert tangent.denominator_relative_eps == pytest.approx(2.5e-11)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("eta", -0.1, ValueError),
        ("eta", True, TypeError),
        ("eta_strategy", "adaptive", ValueError),
        ("eta_strategy", True, TypeError),
        ("line_search_relative_eps", 0.0, ValueError),
        ("line_search_relative_eps", "1e-12", TypeError),
        ("relative_lambda", float("inf"), ValueError),
        ("denominator_relative_eps", 0.0, ValueError),
        ("denominator_relative_eps", "1e-12", TypeError),
    ],
)
def test_symmetric_tangent_green_response_rejects_invalid_config(
    field,
    value,
    error_type,
):
    with pytest.raises(error_type, match="symmetric_tangent_green_response"):
        CouplingModelConfig(
            balance_projection={
                "mode": "symmetric_tangent_green_response",
                "symmetric_tangent_green_response": {field: value},
            }
        )

    with pytest.raises(TypeError, match="unknown keys"):
        CouplingModelConfig(
            balance_projection={
                "mode": "symmetric_tangent_green_response",
                "symmetric_tangent_green_response": {"learnable_eta": True},
            }
        )


def test_symmetric_tangent_green_response_defaults_to_fixed_eta_strategy():
    config = CouplingModelConfig(
        balance_projection={
            "mode": "symmetric_tangent_green_response",
            "symmetric_tangent_green_response": {"eta": 0.015},
        }
    )

    tangent = config.balance_projection.symmetric_tangent_green_response
    assert tangent.eta_strategy == "fixed"
    assert tangent.line_search_relative_eps == pytest.approx(1.0e-12)


def test_save_load_green_model_with_config(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(
        input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        branch_input_dim=5,
        dtype=torch.float64,
    )
    model = GreenONetModel(cfg)
    path = tmp_path / "green.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, GreenONetModel)
    assert loaded_cfg == cfg
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_green_compiled_model_with_config(tmp_path):
    if not hasattr(torch, "compile"):
        raise AssertionError("torch.compile is unavailable in this environment")
    torch.manual_seed(0)
    cfg = ModelConfig(
        input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        branch_input_dim=5,
        dtype=torch.float64,
    )
    model = torch.compile(GreenONetModel(cfg))
    path = tmp_path / "green_compiled.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, GreenONetModel)
    assert loaded_cfg == cfg
    _assert_state_dict_equal(
        model._orig_mod.state_dict(),
        loaded_model.state_dict(),
    )


def test_save_load_coupling_model_with_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.source_stencil_lift.enabled is False
    assert loaded_cfg.source_stencil_lift.coefficient_normalization == "rms"
    assert loaded_cfg.source_stencil_lift.coefficient_tanh_beta == 1.0
    assert loaded_cfg.coefficient_terms.diffusion is True
    assert loaded_cfg.coefficient_terms.convection is False
    assert loaded_cfg.coefficient_terms.reaction is False
    assert loaded_cfg.branch_fusion.mode == "product"
    assert loaded_cfg.green_response_feature.enabled is False
    assert loaded_cfg.trunk_positional_encoding.enabled is False
    assert loaded_cfg.trunk_positional_encoding.mode == "fourier"
    assert loaded_cfg.trunk_positional_encoding.num_frequencies == 4
    assert loaded_cfg.trunk_positional_encoding.max_frequency == 8.0
    assert loaded_cfg.trunk_positional_encoding.include_input is True
    assert not hasattr(loaded_cfg, "use_fourier")
    assert not hasattr(loaded_cfg, "fourier_dim")
    assert not hasattr(loaded_cfg, "fourier_scale")
    assert not hasattr(loaded_cfg, "fourier_include_input")
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_load_coupling_model_migrates_legacy_branch_a_state(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
    )
    model = CouplingNet(cfg)
    legacy_state = {}
    for key, value in model.state_dict().items():
        if key.startswith("branch_coefficient."):
            legacy_state["branch_a." + key.removeprefix("branch_coefficient.")] = value
            legacy_state["branch_b." + key.removeprefix("branch_coefficient.")] = (
                torch.zeros_like(value)
            )
            legacy_state["branch_c." + key.removeprefix("branch_coefficient.")] = (
                torch.ones_like(value)
            )
        else:
            legacy_state[key] = value
    path = tmp_path / "legacy_branch_a_coupling.pt"
    torch.save(
        {
            "state_dict": legacy_state,
            "model_type": "coupling",
            "model_config": {
                "branch_input_dim": 5,
                "trunk_input_dim": 2,
                "hidden_dim": 8,
                "depth": 2,
                "activation": "tanh",
                "use_bias": True,
                "dropout": 0.0,
                "dtype": "float64",
            },
        },
        path,
    )

    from greenonet.io import load_model_with_config

    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_model_with_source_stencil_lift_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        balance_projection="smooth_mask",
        smooth_mask_normalize=False,
        smooth_mask_eps=1e-9,
        smooth_mask_power=0.5,
        smooth_mask_diff_power=0.75,
        source_stencil_lift=SourceStencilLiftConfig(
            enabled=True,
            encoder_type="linear",
            coefficient_normalization="tanh",
            coefficient_tanh_beta=1.7,
            hidden_dim=32,
        ),
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_source_lift.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.balance_projection.enabled is True
    assert loaded_cfg.balance_projection.mode == "smooth_mask"
    assert loaded_cfg.balance_projection.mask == "quadratic"
    assert loaded_cfg.smooth_mask_normalize is False
    assert loaded_cfg.smooth_mask_eps == 1e-9
    assert loaded_cfg.smooth_mask_power == 0.5
    assert loaded_cfg.smooth_mask_diff_power == 0.75
    assert loaded_cfg.smooth_mask_diff_power_trainable is False
    assert loaded_cfg.smooth_mask_diff_power_min == 0.25
    assert loaded_cfg.smooth_mask_diff_power_max == 2.0
    assert loaded_cfg.source_stencil_lift.enabled is True
    assert loaded_cfg.source_stencil_lift.encoder_type == "linear"
    assert loaded_cfg.source_stencil_lift.coefficient_normalization == "tanh"
    assert loaded_cfg.source_stencil_lift.coefficient_tanh_beta == 1.7
    assert loaded_cfg.source_stencil_lift.hidden_dim == 32
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_model_with_balance_projection_object_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        dtype=torch.float64,
        balance_projection={
            "enabled": False,
            "mode": "smooth_mask",
            "mask": "sin",
        },
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_raw_projection.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.balance_projection.enabled is False
    assert loaded_cfg.balance_projection.mode == "smooth_mask"
    assert loaded_cfg.balance_projection.mask == "sin"
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_deserialize_rejects_retired_geometry_metadata():
    from greenonet.io import _deserialize_config

    payload = {
        "branch_input_dim": 5,
        "dtype": "float64",
        "balance_projection": {
            "enabled": True,
            "mode": "symmetric",
            "geometry_weighted_rule": "swapped_length_squared",
            "geometry_weighted_lambda": 0.5,
        },
    }

    with pytest.raises(ValueError, match="Retired balance_projection fields"):
        _deserialize_config(payload, CouplingModelConfig)


def test_save_load_coupling_model_with_green_response_feature_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        green_response_feature=GreenResponseFeatureConfig(enabled=True),
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_green_response.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.green_response_feature.enabled is True
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_model_with_coefficient_terms_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=True,
            convection=True,
            reaction=True,
        ),
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_coefficient_terms.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.coefficient_terms.diffusion is True
    assert loaded_cfg.coefficient_terms.convection is True
    assert loaded_cfg.coefficient_terms.reaction is True
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_model_with_trunk_positional_encoding_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        trunk_positional_encoding=CouplingTrunkPositionalEncodingConfig(
            enabled=True,
            mode="boundary_algebraic",
            num_frequencies=3,
            max_frequency=4.0,
            include_input=True,
        ),
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_trunk_positional.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.trunk_positional_encoding.enabled is True
    assert loaded_cfg.trunk_positional_encoding.mode == "boundary_algebraic"
    assert loaded_cfg.trunk_positional_encoding.num_frequencies == 3
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_model_with_axis_1d_trunk_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        axis_1d_trunk=Axis1DTrunkConfig(
            enabled=True,
            boundary_aware_modes=3,
            num_frequencies=5,
            max_frequency=16.0,
            transverse_trunk=TransverseTrunkConfig(
                enabled=True,
                fusion="product_fuser",
                length_context=True,
            ),
        ),
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_axis_1d_trunk.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.axis_1d_trunk.enabled is True
    assert loaded_cfg.axis_1d_trunk.boundary_aware_modes == 3
    assert loaded_cfg.axis_1d_trunk.num_frequencies == 5
    assert loaded_cfg.axis_1d_trunk.max_frequency == 16.0
    assert loaded_cfg.axis_1d_trunk.transverse_trunk.enabled is True
    assert loaded_cfg.axis_1d_trunk.transverse_trunk.fusion == "product_fuser"
    assert loaded_cfg.axis_1d_trunk.transverse_trunk.length_context is True
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_model_with_branch_fusion_config(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        branch_fusion=CouplingBranchFusionConfig(mode="product_fuser"),
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling_branch_fusion.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.branch_fusion.mode == "product_fuser"
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_trainable_smooth_mask_diff_power_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        balance_projection="smooth_mask",
        smooth_mask_diff_power=0.75,
        smooth_mask_diff_power_trainable=True,
        smooth_mask_diff_power_min=0.25,
        smooth_mask_diff_power_max=2.0,
    )
    model = CouplingNet(cfg)
    assert model.smooth_mask_diff_power_raw is not None
    with torch.no_grad():
        model.smooth_mask_diff_power_raw.add_(0.125)
    expected_q = model.effective_smooth_mask_diff_power()
    path = tmp_path / "coupling_trainable_q.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.smooth_mask_diff_power_trainable is True
    assert loaded_cfg.smooth_mask_diff_power_min == 0.25
    assert loaded_cfg.smooth_mask_diff_power_max == 2.0
    assert loaded_model.effective_smooth_mask_diff_power() == expected_q
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_save_load_coupling_compiled_model_with_config(tmp_path):
    if not hasattr(torch, "compile"):
        raise AssertionError("torch.compile is unavailable in this environment")
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
    )
    model = torch.compile(CouplingNet(cfg))
    path = tmp_path / "coupling_compiled.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    _assert_state_dict_equal(
        model._orig_mod.state_dict(),
        loaded_model.state_dict(),
    )


def test_load_coupling_model_with_legacy_removed_config_fields(tmp_path):
    torch.manual_seed(0)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=2,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
    )
    model = CouplingNet(cfg)
    path = tmp_path / "legacy_coupling.pt"

    legacy_payload = {
        "state_dict": model.state_dict(),
        "model_type": "coupling",
        "model_config": {
            "branch_input_dim": 5,
            "trunk_input_dim": 2,
            "hidden_dim": 8,
            "depth": 2,
            "activation": "tanh",
            "use_bias": True,
            "dropout": 0.0,
            "dtype": "float64",
            "use_fourier": False,
            "fourier_dim": 16,
            "fourier_scale": 1.0,
            "fourier_include_input": False,
            "axis_head_hidden_dim": 10,
            "axis_head_depth": 2,
            "axis_residual_blocks": 3,
            "use_learned_balance_coeff": True,
            "balance_hidden_dim": 64,
            "balance_depth": 2,
            "balance_eps": 1e-12,
            "coupler": {"enabled": True, "hidden_channels": 32},
        },
    }
    torch.save(legacy_payload, path)

    from greenonet.io import load_model_with_config

    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert loaded_cfg.balance_projection.enabled is True
    assert loaded_cfg.balance_projection.mode == "symmetric"
    assert loaded_cfg.balance_projection.mask == "quadratic"
    assert loaded_cfg.smooth_mask_normalize is True
    assert loaded_cfg.smooth_mask_eps == 1e-12
    assert loaded_cfg.smooth_mask_power == 1.0
    assert loaded_cfg.smooth_mask_diff_power == 1.0
    assert loaded_cfg.smooth_mask_diff_power_trainable is False
    assert loaded_cfg.smooth_mask_diff_power_min == 0.25
    assert loaded_cfg.smooth_mask_diff_power_max == 2.0
    assert loaded_cfg.source_stencil_lift.enabled is False
    assert loaded_cfg.source_stencil_lift.coefficient_normalization == "rms"
    assert loaded_cfg.source_stencil_lift.coefficient_tanh_beta == 1.0
    assert loaded_cfg.green_response_feature.enabled is False
    assert loaded_cfg.trunk_positional_encoding.enabled is False
    assert loaded_cfg.trunk_positional_encoding.mode == "fourier"
    assert not hasattr(loaded_cfg, "use_fourier")
    assert not hasattr(loaded_cfg, "fourier_dim")
    assert not hasattr(loaded_cfg, "fourier_scale")
    assert not hasattr(loaded_cfg, "fourier_include_input")
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())
