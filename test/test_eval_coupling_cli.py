import json

import pytest

from cli.eval_coupling import EvalCouplingCLI


def test_eval_cli_parses_q_head_and_evaluation_config(tmp_path):
    config_path = tmp_path / "config.json"
    payload = {
        "dataset": {"step_size": 0.25, "test_path": str(tmp_path)},
        "coupling_model": {
            "hidden_dim": 32,
            "depth": 3,
            "q_head": {
                "enabled": True,
                "s_branch_hidden_dim": 11,
                "s_branch_depth": 2,
                "m_branch_hidden_dim": 12,
                "m_branch_depth": 1,
                "latent_dim": 32,
                "share_trunk": True,
                "fusion": "add_transpose",
            },
        },
        "coupling_training": {
            "losses": {
                "q_split": {
                    "enabled": True,
                    "weight": 2.5,
                }
            }
        },
        "evaluation": {
            "posthoc_q_correction": {
                "enabled": True,
                "report_corrected_metrics": True,
            }
        },
    }
    config_path.write_text(json.dumps(payload))
    raw = json.loads(config_path.read_text())

    coupling_model_cfg = EvalCouplingCLI._build_coupling_model_config(
        raw["coupling_model"]
    )
    coupling_training_cfg = EvalCouplingCLI._build_coupling_training_config(
        raw["coupling_training"]
    )
    evaluation_cfg = EvalCouplingCLI._build_evaluation_config(raw["evaluation"])

    assert coupling_model_cfg.q_head.enabled is True
    assert coupling_model_cfg.q_head.s_branch_hidden_dim == 11
    assert coupling_model_cfg.q_head.m_branch_hidden_dim == 12
    assert coupling_model_cfg.q_head.latent_dim == 32
    assert coupling_training_cfg.losses.q_split.enabled is True
    assert coupling_training_cfg.losses.q_split.weight == 2.5
    assert evaluation_cfg.posthoc_q_correction.enabled is True
    assert evaluation_cfg.posthoc_q_correction.report_corrected_metrics is True


def test_eval_cli_rejects_invalid_q_head_config():
    with pytest.raises(TypeError, match="share_trunk"):
        EvalCouplingCLI._build_coupling_model_config(
            {
                "hidden_dim": 16,
                "depth": 2,
                "q_head": {"share_trunk": False, "fusion": "add_transpose"},
            }
        )

    with pytest.raises(TypeError, match="fusion"):
        EvalCouplingCLI._build_coupling_model_config(
            {
                "hidden_dim": 16,
                "depth": 2,
                "q_head": {"share_trunk": True, "fusion": "invalid"},
            }
        )
