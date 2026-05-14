import torch

from greenonet.config import CouplingModelConfig, ModelConfig, SourceStencilLiftConfig
from greenonet.coupling_model import CouplingNet
from greenonet.model import GreenONetModel


def _assert_state_dict_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


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
    assert not hasattr(loaded_cfg, "use_fourier")
    assert not hasattr(loaded_cfg, "fourier_dim")
    assert not hasattr(loaded_cfg, "fourier_scale")
    assert not hasattr(loaded_cfg, "fourier_include_input")
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
    assert loaded_cfg.balance_projection == "smooth_mask"
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
    assert loaded_cfg.balance_projection == "symmetric"
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
    assert not hasattr(loaded_cfg, "use_fourier")
    assert not hasattr(loaded_cfg, "fourier_dim")
    assert not hasattr(loaded_cfg, "fourier_scale")
    assert not hasattr(loaded_cfg, "fourier_include_input")
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())
