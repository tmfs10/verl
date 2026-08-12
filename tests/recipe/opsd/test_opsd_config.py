from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.opsd import main_opsd
from verl.trainer.config.algorithm import (
    OPSDAdvantageShapingConfig,
    OPSDAuditConfig,
    OPSDConfig,
    OPSDGapDiagnosticsConfig,
    OPSDSteeringConfig,
    OPSDTokenKLLoggingConfig,
)
from verl.workers.config.actor import FSDPActorConfig


def test_opsd_config_defaults_to_sampled_reverse_kl_without_grad_balancing():
    config = OPSDConfig()

    assert config.distill_loss == "sampled_reverse_kl"
    assert config.actor_objective == "direct_reverse_kl"
    assert config.balance_mode == "none"
    assert config.topk is None
    assert config.distill_beta is None
    assert config.teacher_sft_weight == 0.0
    assert config.teacher_sft_target_scope == "thinking_and_answer"
    assert config.audit.enabled is False
    assert config.token_kl_logging.enabled is False
    assert config.sdpo_conditioning_mode == "prompt_append"
    assert config.steering.activation_aggregation == "per_rollout"
    assert config.steering.gradient_objective == "grpo_advantage"
    assert config.steering.gradient_aggregation == "per_rollout"
    assert config.steering.caa_scope == "same_prompt"
    assert config.steering.gap_diagnostics.enabled is False
    assert config.advantage_shaping.normalize is None
    assert config.advantage_shaping.clip_z is None
    assert config.advantage_shaping.max_delta_fraction is None


def test_gap_diagnostics_config_accepts_every_step_crossfit():
    config = OPSDSteeringConfig(
        gap_diagnostics={
            "enabled": True,
            "interval_steps": 1,
            "crossfit_enabled": True,
            "fold_seed": 1234,
        }
    )

    assert isinstance(config.gap_diagnostics, OPSDGapDiagnosticsConfig)
    assert config.gap_diagnostics.enabled is True
    assert config.gap_diagnostics.interval_steps == 1
    assert config.gap_diagnostics.crossfit_enabled is True


@pytest.mark.parametrize("override", [{"interval_steps": 0}, {"fold_seed": -1}])
def test_gap_diagnostics_config_rejects_invalid_values(override):
    with pytest.raises(ValueError):
        OPSDGapDiagnosticsConfig(**override)


def test_opsd_config_accepts_ground_truth_free_shared_actor_steering():
    config = OPSDConfig(
        mode="opsd",
        teacher_source="sdpo_success_rollout",
        teacher_model="actor",
        sdpo_conditioning_mode="steering",
        sdpo_distill_only_failed=False,
        steering={"layer_fractions": "0.31-0.37", "source_mode": "caa"},
    )
    assert config.steering.layer_fractions == "0.31-0.37"
    assert config.steering.source_mode == "caa"
    assert config.steering.detach_vectors is True


def test_opsd_config_accepts_strict_production_steering_contract():
    config = OPSDConfig(
        mode="opsd",
        teacher_source="sdpo_success_rollout",
        teacher_model="actor",
        sdpo_conditioning_mode="steering",
        sdpo_distill_only_failed=False,
        rlvr_backward_scale=0.0,
        steering={
            "strict_contract": True,
            "layer_fractions": "0.31-0.37",
            "expected_model_path": "/hf_models/Qwen3-1.7B",
            "actor_model_path": "/hf_models/Qwen3-1.7B",
            "expected_total_layers": 28,
            "expected_layer_indices": [9, 10],
            "normalize": "unit_norm",
            "apply_positions": "response_only",
        },
    )
    assert config.steering.strict_contract is True


@pytest.mark.parametrize(
    ("opsd_override", "steering_override"),
    [
        ({}, {"source_mode": "positive"}),
        ({}, {"activation_aggregation": "pooled_tokens"}),
        ({"sdpo_distill_only_failed": True}, {}),
        ({"distill_max_response_tokens": 128}, {}),
        ({"rlvr_backward_scale": 1.0}, {}),
    ],
)
def test_strict_production_steering_contract_rejects_semantic_drift(
    opsd_override, steering_override
):
    kwargs = {
        "mode": "opsd",
        "teacher_source": "sdpo_success_rollout",
        "teacher_model": "actor",
        "sdpo_conditioning_mode": "steering",
        "sdpo_distill_only_failed": False,
        "rlvr_backward_scale": 0.0,
        "steering": {
            "strict_contract": True,
            "layer_fractions": "0.31-0.37",
            "expected_model_path": "/hf_models/Qwen3-1.7B",
            "actor_model_path": "/hf_models/Qwen3-1.7B",
            "expected_total_layers": 28,
            "expected_layer_indices": [9, 10],
            "normalize": "unit_norm",
            "apply_positions": "response_only",
            **steering_override,
        },
        **opsd_override,
    }
    with pytest.raises(ValueError, match="strict_contract|outcome-symmetric"):
        OPSDConfig(**kwargs)


@pytest.mark.parametrize("field", ["expected_model_path", "actor_model_path"])
def test_strict_production_steering_contract_rejects_wrong_actor_identity(field):
    steering = {
        "strict_contract": True,
        "layer_fractions": "0.31-0.37",
        "expected_model_path": "/hf_models/Qwen3-1.7B",
        "actor_model_path": "/hf_models/Qwen3-1.7B",
        "expected_total_layers": 28,
        "expected_layer_indices": [9, 10],
        "normalize": "unit_norm",
        "apply_positions": "response_only",
    }
    steering[field] = "/hf_models/not-qwen3-1.7b"
    with pytest.raises(ValueError, match="strict_contract"):
        OPSDConfig(
            mode="opsd",
            teacher_source="sdpo_success_rollout",
            teacher_model="actor",
            sdpo_conditioning_mode="steering",
            sdpo_distill_only_failed=False,
            rlvr_backward_scale=0.0,
            steering=steering,
        )


def test_bind_opsd_actor_model_identity_overwrites_duplicate_assertion():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"path": "/hf_models/Qwen3-1.7B"}},
            "algorithm": {"opsd": {"steering": {"actor_model_path": "/misleading"}}},
        }
    )
    main_opsd._bind_opsd_actor_model_identity(config)
    assert config.algorithm.opsd.steering.actor_model_path == "/hf_models/Qwen3-1.7B"


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"teacher_source": "ground_truth"}, "teacher_source"),
        ({"teacher_model": "ema"}, "teacher_model=actor"),
        ({"mode": "opsd_rlvr"}, "requires opsd.mode=opsd"),
        ({"teacher_sft_weight": 1.0}, "teacher SFT"),
    ],
)
def test_opsd_config_rejects_privileged_or_trainable_steering_teacher(overrides, message):
    kwargs = {
        "mode": "opsd",
        "teacher_source": "sdpo_success_rollout",
        "teacher_model": "actor",
        "sdpo_conditioning_mode": "steering",
        "sdpo_distill_only_failed": False,
        "steering": {"layer_fractions": "0.31-0.37"},
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        OPSDConfig(**kwargs)


def test_steering_config_rejects_unsafe_vector_gradients():
    with pytest.raises(ValueError, match="detach_vectors"):
        OPSDSteeringConfig(layer_fractions="0.31-0.37", detach_vectors=False)


def test_steering_config_accepts_global_batch_caa_scope():
    config = OPSDSteeringConfig(
        layer_fractions="0.31-0.37", source_mode="caa", caa_scope="global_batch"
    )

    assert config.caa_scope == "global_batch"


def test_steering_config_accepts_global_batch_policy_gradient_scope():
    config = OPSDSteeringConfig(
        layer_fractions="0.31-0.37",
        source_mode="policy_gradient",
        caa_scope="global_batch",
        gradient_objective="grpo_advantage",
        gradient_aggregation="per_rollout",
    )

    assert config.source_mode == "policy_gradient"
    assert config.caa_scope == "global_batch"


@pytest.mark.parametrize(
    "override",
    [
        {"gradient_objective": "correctness_gap"},
        {"gradient_aggregation": "pooled_tokens"},
    ],
)
def test_steering_config_rejects_unaudited_policy_gradient_formulations(override):
    with pytest.raises(ValueError):
        OPSDSteeringConfig(
            layer_fractions="0.31-0.37",
            source_mode="policy_gradient",
            caa_scope="global_batch",
            **override,
        )


def test_steering_config_rejects_global_batch_positive_only_scope():
    with pytest.raises(ValueError, match="global_batch"):
        OPSDSteeringConfig(
            layer_fractions="0.31-0.37",
            source_mode="positive",
            caa_scope="global_batch",
        )


@pytest.mark.parametrize(
    ("actor_objective", "mode", "distill_scale", "shaping_enabled"),
    [
        ("direct_reverse_kl", "opsd", 1.0, False),
        ("negative_kl_advantage", "opsd", 1.0, False),
        ("grpo_advantage_reweighting", "opsd_rlvr", 0.0, True),
    ],
)
def test_strict_steering_accepts_all_three_actor_objectives(
    actor_objective, mode, distill_scale, shaping_enabled
):
    config = OPSDConfig(
        mode=mode,
        actor_objective=actor_objective,
        teacher_source="sdpo_success_rollout",
        teacher_model="actor",
        sdpo_conditioning_mode="steering",
        sdpo_distill_only_failed=False,
        mix_weight=1.0,
        distill_backward_scale=distill_scale,
        rlvr_backward_scale=0.0,
        advantage_shaping={"enable": shaping_enabled},
        steering={
            "strict_contract": True,
            "layer_fractions": "0.31-0.37",
            "expected_model_path": "/hf_models/Qwen3-1.7B",
            "actor_model_path": "/hf_models/Qwen3-1.7B",
            "expected_total_layers": 28,
            "expected_layer_indices": [9, 10],
            "normalize": "unit_norm",
            "apply_positions": "response_only",
        },
    )
    assert config.actor_objective == actor_objective


def test_actor_objective_rejects_conflicting_legacy_shaping_flag():
    with pytest.raises(ValueError, match="compatibility alias"):
        OPSDConfig(
            actor_objective="negative_kl_advantage",
            advantage_shaping={"enable": True},
        )


@pytest.mark.parametrize(
    ("legacy_shaping", "expected"),
    [
        (False, "direct_reverse_kl"),
        (True, "grpo_advantage_reweighting"),
    ],
)
def test_raw_recipe_resolves_legacy_actor_objective_once(legacy_shaping, expected):
    config = OmegaConf.create(
        {
            "algorithm": {
                "opsd": {
                    "actor_objective": None,
                    "advantage_shaping": {"enable": legacy_shaping},
                }
            }
        }
    )

    main_opsd._resolve_opsd_actor_objective(config)

    assert config.algorithm.opsd.actor_objective == expected


def test_opsd_config_accepts_teacher_sft_for_separate_ground_conditioned_teacher():
    config = OPSDConfig(
        mode="opsd_rlvr",
        teacher_model="separate",
        teacher_source="ground_truth",
        teacher_sft_weight=0.25,
        teacher_sft_target_scope="thinking_only",
    )

    assert config.teacher_sft_weight == 0.25
    assert config.teacher_sft_target_scope == "thinking_only"


@pytest.mark.parametrize("teacher_sft_weight", [-1.0, float("inf"), float("nan")])
def test_opsd_config_rejects_invalid_teacher_sft_weight(teacher_sft_weight):
    with pytest.raises(ValueError, match="teacher_sft_weight"):
        OPSDConfig(teacher_sft_weight=teacher_sft_weight)


def test_opsd_config_rejects_invalid_teacher_sft_scope():
    with pytest.raises(ValueError, match="teacher_sft_target_scope"):
        OPSDConfig(teacher_sft_target_scope="answer_only")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "opsd", "teacher_model": "actor"}, "opsd.mode=opsd_rlvr"),
        ({"mode": "opsd_rlvr", "teacher_model": "actor"}, "teacher_model=separate"),
        (
            {
                "mode": "opsd_rlvr",
                "teacher_model": "separate",
                "teacher_source": "sdpo_success_rollout",
            },
            "teacher_source=ground_truth",
        ),
    ],
)
def test_opsd_config_rejects_teacher_sft_without_isolated_ground_conditioned_teacher(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        OPSDConfig(teacher_sft_weight=1.0, **overrides)


def test_opsd_audit_config_rejects_invalid_steps_and_tolerances():
    with pytest.raises(ValueError, match="global_steps"):
        OPSDAuditConfig(enabled=True, global_steps=[0])


def test_opsd_audit_dense_cross_kernel_is_diagnostic_by_default():
    config = OPSDAuditConfig()

    assert config.dense_forward_fail_fast is False
    assert OPSDAuditConfig(dense_forward_fail_fast=True).dense_forward_fail_fast is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_step": 0},
        {"start_step": 3, "end_step": 2},
        {"interval_steps": 0},
        {"max_samples_per_rank": 0},
        {"max_tokens_per_sample": 0},
    ],
)
def test_token_kl_logging_config_rejects_unbounded_or_invalid_values(kwargs):
    with pytest.raises(ValueError, match="token_kl_logging"):
        OPSDTokenKLLoggingConfig(**kwargs)


def test_opsd_config_accepts_fail_closed_advantage_shaping():
    config = OPSDConfig(
        mode="opsd_rlvr",
        teacher_model="separate",
        mix_weight=1.0,
        balance_mode="none",
        distill_backward_scale=0.0,
        advantage_shaping={
            "enable": True,
            "allow_token_sign_flip": False,
            "max_response_tokens": None,
        },
    )

    assert config.advantage_shaping.enable is True
    assert config.advantage_shaping.score_source == "teacher_minus_student_logprob"
    assert config.advantage_shaping.allow_token_sign_flip is False


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mode": "opsd"}, "mode=opsd_rlvr"),
        ({"mix_weight": 0.5}, "mix_weight=1.0"),
        ({"balance_mode": "grad_norm"}, "balance_mode=none"),
        ({"distill_backward_scale": 1.0}, "distill_backward_scale=0.0"),
    ],
)
def test_opsd_config_rejects_unsafe_advantage_shaping(override, message):
    kwargs = {
        "mode": "opsd_rlvr",
        "teacher_model": "separate",
        "mix_weight": 1.0,
        "balance_mode": "none",
        "distill_backward_scale": 0.0,
        "advantage_shaping": {"enable": True},
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        OPSDConfig(**kwargs)


def test_shared_teacher_advantage_shaping_rejects_teacher_branch_training():
    with pytest.raises(ValueError, match="Shared-teacher advantage shaping"):
        OPSDConfig(
            mode="opsd_rlvr",
            teacher_model="actor",
            mix_weight=1.0,
            balance_mode="none",
            distill_backward_scale=0.0,
            rlvr_backward_scale=1.0,
            advantage_shaping={"enable": True},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_source": "reverse_kl_surrogate"},
        {"scale": -1.0},
        {"max_delta_fraction": 0.0},
        {"max_response_tokens": 0},
        {"student_rlvr_backward_scale": -1.0},
    ],
)
def test_advantage_shaping_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        OPSDAdvantageShapingConfig(**kwargs)


def test_opsd_config_rejects_legacy_topk_distillation_with_explanation():
    with pytest.raises(ValueError, match="double-counted"):
        OPSDConfig(distill_loss="topk_jsd", topk=16, distill_beta=0.5)


def test_opsd_config_rejects_full_jsd_under_reverse_kl_only_policy():
    with pytest.raises(ValueError, match="reverse-KL-only"):
        OPSDConfig(distill_loss="full_jsd")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("topk", 16),
        ("distill_beta", 0.5),
        ("distill_token_clip", 0.05),
        ("distill_token_clip_tail", False),
    ],
)
def test_opsd_config_rejects_legacy_jsd_controls(field, value):
    with pytest.raises(ValueError, match="Top-k/JSD-only"):
        OPSDConfig(**{field: value})


def test_opsd_config_rejects_non_positive_distill_max_response_tokens():
    with pytest.raises(ValueError, match="distill_max_response_tokens"):
        OPSDConfig(distill_max_response_tokens=0)


def test_opsd_config_accepts_fixed_teacher_for_distillation_mode():
    config = OPSDConfig(teacher_model="fixed", mode="opsd")

    assert config.teacher_model == "fixed"


def test_opsd_config_accepts_reference_solution_teacher_prompt_style():
    config = OPSDConfig(
        teacher_model="fixed",
        mode="opsd",
        teacher_prompt_style="reference_solution_single_user",
        teacher_apply_chat_template_kwargs={"enable_thinking": True},
    )

    assert config.teacher_prompt_style == "reference_solution_single_user"
    assert config.teacher_apply_chat_template_kwargs["enable_thinking"] is True


def test_opsd_config_rejects_fixed_teacher_for_opsd_rlvr():
    with pytest.raises(ValueError, match="teacher_model=fixed"):
        OPSDConfig(teacher_model="fixed", mode="opsd_rlvr")


def test_opsd_config_accepts_separate_trainable_teacher_for_opsd_rlvr():
    config = OPSDConfig(teacher_model="separate", mode="opsd_rlvr")

    assert config.teacher_model == "separate"


def test_fsdp_actor_config_accepts_separate_opsd_teacher():
    config = FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        opsd_teacher_model="separate",
    )

    assert config.opsd_teacher_model == "separate"


def test_opsd_config_rejects_separate_teacher_for_distillation_only_mode():
    with pytest.raises(ValueError, match="requires opsd.mode=opsd_rlvr"):
        OPSDConfig(teacher_model="separate", mode="opsd")


def test_opsd_config_rejects_grad_norm_balancing_for_separate_teacher():
    with pytest.raises(ValueError, match="separately optimized teacher"):
        OPSDConfig(teacher_model="separate", mode="opsd_rlvr", balance_mode="grad_norm")


def test_opsd_config_rejects_combined_backward_in_stabilized_path():
    with pytest.raises(ValueError, match="separate_backward=False is disabled"):
        OPSDConfig(separate_backward=False)


def test_opsd_recipe_allows_independent_teacher_optimizer_overrides():
    repo_root = Path(__file__).resolve().parents[3]
    recipe_config_dir = repo_root / "recipe" / "opsd" / "config"
    core_config_dir = repo_root / "verl" / "trainer" / "config"

    with initialize_config_dir(config_dir=str(recipe_config_dir), version_base=None):
        config = compose(
            config_name="opsd_trainer",
            overrides=[
                f"hydra.searchpath=[file://{core_config_dir}]",
                "actor_rollout_ref.opsd_teacher.optim.lr=2e-6",
                "actor_rollout_ref.opsd_teacher.fsdp_config.dtype=bfloat16",
            ],
        )

    assert OmegaConf.select(config, "actor_rollout_ref.opsd_teacher.optim.lr") == 2e-6
    assert OmegaConf.select(config, "actor_rollout_ref.opsd_teacher.optim._target_") == (
        "verl.workers.config.FSDPOptimizerConfig"
    )


def test_opsd_reward_loader_forwards_configured_reward_kwargs(monkeypatch):
    captured = {}

    def fake_load_reward_manager(config, tokenizer, **kwargs):
        del config, tokenizer
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(main_opsd, "load_reward_manager", fake_load_reward_manager)
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "use_response_logprob_reward_for_uniform_outcome_groups": True,
                    "uniform_outcome_group_success_threshold": 0.75,
                }
            }
        }
    )

    main_opsd.load_opsd_reward_manager(config, tokenizer=object(), num_examine=3)

    assert captured == {
        "num_examine": 3,
        "use_response_logprob_reward_for_uniform_outcome_groups": True,
        "uniform_outcome_group_success_threshold": 0.75,
    }
