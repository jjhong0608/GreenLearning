import torch

from greenonet.config import (
    CouplingLineEncoderConfig,
    CouplingLineEncoderHeadConfig,
    CouplingModelConfig,
    ModelConfig,
)
from greenonet.coupling_model import CouplingNet
from greenonet.model import GreenONetModel


def _assert_state_dict_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def _nested_line_encoder_config() -> CouplingLineEncoderConfig:
    return CouplingLineEncoderConfig(
        type="cnn1d",
        in_channels=3,
        include_position=True,
        include_boundary_distance=True,
        conv_channels=[32, 64, 64],
        kernel_size=5,
        dilations=[1, 2, 4],
        pooling="meanmax",
        activation="rational",
        mlp_head=CouplingLineEncoderHeadConfig(
            depth=2,
            hidden_dim=16,
            activation="rational",
            use_bias=True,
            dropout=0.0,
        ),
    )


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
        activation="rational",
        use_bias=True,
        dropout=0.0,
        line_encoder=_nested_line_encoder_config(),
        dtype=torch.float64,
    )
    model = CouplingNet(cfg)
    path = tmp_path / "coupling.safetensors"

    from greenonet.io import load_model_with_config, save_model_with_config

    save_model_with_config(model, cfg, path)
    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert not hasattr(loaded_cfg, "use_fourier")
    assert not hasattr(loaded_cfg, "fourier_dim")
    assert not hasattr(loaded_cfg, "fourier_scale")
    assert not hasattr(loaded_cfg, "fourier_include_input")
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
        activation="rational",
        use_bias=True,
        dropout=0.0,
        line_encoder=_nested_line_encoder_config(),
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
        },
    }
    torch.save(legacy_payload, path)

    from greenonet.io import load_model_with_config

    loaded_model, loaded_cfg = load_model_with_config(path)

    assert isinstance(loaded_model, CouplingNet)
    assert loaded_cfg == cfg
    assert not hasattr(loaded_cfg, "use_fourier")
    assert not hasattr(loaded_cfg, "fourier_dim")
    assert not hasattr(loaded_cfg, "fourier_scale")
    assert not hasattr(loaded_cfg, "fourier_include_input")
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())
