import pytest

from verl.trainer.config.algorithm import OPSDConfig


def test_opsd_config_accepts_generalized_topk_divergence_controls():
    config = OPSDConfig(
        distill_loss="topk_jsd",
        distill_beta=0.0,
        distill_token_clip=0.05,
        distill_token_clip_tail=False,
    )

    assert config.distill_beta == 0.0
    assert config.distill_token_clip == 0.05
    assert config.distill_token_clip_tail is False


def test_opsd_config_accepts_full_jsd_distillation():
    config = OPSDConfig(
        distill_loss="full_jsd",
        topk=0,
        distill_beta=0.0,
        distill_token_clip=0.05,
    )

    assert config.distill_loss == "full_jsd"
    assert config.topk == 0


def test_opsd_config_rejects_invalid_distill_beta():
    with pytest.raises(ValueError, match="distill_beta"):
        OPSDConfig(distill_beta=1.1)


def test_opsd_config_rejects_non_positive_distill_token_clip():
    with pytest.raises(ValueError, match="distill_token_clip"):
        OPSDConfig(distill_token_clip=0.0)


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
