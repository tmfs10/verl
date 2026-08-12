import pytest
import torch

from recipe.opsd.audit import (
    build_sequence_gap_log_record,
    build_token_reverse_kl_log_record,
    global_aggregation_info,
    ppo_token_terms,
    summarize_sequence_gap_samples,
    token_kl_logging_is_enabled,
    validate_opsd_layout,
)


def test_sequence_gap_record_is_response_only_and_preserves_absolute_means():
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    attention = mask.long()
    actor = torch.tensor([[-2.0, -4.0, float("nan")], [-5.0, float("nan"), float("nan")]])
    teacher = torch.tensor([[-1.0, -3.0, float("nan")], [-7.0, float("nan"), float("nan")]])
    crossfit = torch.tensor([[-1.5, -2.5, float("nan")], [-6.0, float("nan"), float("nan")]])

    record = build_sequence_gap_log_record(
        response_attention_mask=attention,
        response_mask=mask,
        actor_log_probs=actor,
        teacher_log_probs=teacher,
        crossfit_teacher_log_probs=crossfit,
        crossfit_available=torch.tensor([True, True]),
        outcome_sign=torch.tensor([1.0, -1.0]),
        crossfit_fold=torch.tensor([0, 1]),
        prompt_group=torch.tensor([0, 0]),
        sample_indices=torch.tensor([10, 11]),
    )

    assert record["prompt_tokens_logged"] == 0
    assert record["masked_or_padded_response_tokens_logged"] == 0
    assert record["samples"][0]["actor_logprob_mean"] == -3.0
    assert record["samples"][0]["teacher_logprob_mean"] == -2.0
    assert record["samples"][1]["actor_logprob_mean"] == -5.0
    summary = summarize_sequence_gap_samples(record["samples"])
    assert summary["actor_gap"] == 2.0
    assert summary["teacher_gap"] == 5.0
    assert summary["teacher_gap_lift"] == 3.0
    assert summary["crossfit_teacher_gap"] == 4.0
    assert summary["crossfit_teacher_gap_lift"] == 2.0
    assert summary["prompt_balanced_teacher_gap_lift"] == 3.0


def test_sequence_gap_record_rejects_response_mask_escape():
    with pytest.raises(ValueError, match="response_mask escaped"):
        build_sequence_gap_log_record(
            response_attention_mask=torch.tensor([[1, 0]]),
            response_mask=torch.tensor([[1.0, 1.0]]),
            actor_log_probs=torch.tensor([[-1.0, -2.0]]),
            teacher_log_probs=torch.tensor([[-1.0, -2.0]]),
            crossfit_teacher_log_probs=None,
            crossfit_available=torch.tensor([False]),
            outcome_sign=torch.tensor([1.0]),
            crossfit_fold=torch.tensor([0]),
            prompt_group=torch.tensor([0]),
            sample_indices=torch.tensor([0]),
        )


def _valid_layout():
    responses = torch.tensor([[7, 2, 2], [8, 9, 2]])
    actor_prompt = torch.tensor([[2, 10, 11, 12], [2, 2, 20, 21]])
    teacher_prompt = torch.tensor([[2, 30, 31, 32], [2, 40, 41, 42]])
    actor_prompt_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    teacher_prompt_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])
    response_attention = torch.tensor([[1, 0, 0], [1, 1, 0]])
    response_mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.float32)
    distill_mask = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    teacher_sft_mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.float32)
    input_ids = torch.cat([actor_prompt, responses], dim=-1)
    teacher_input_ids = torch.cat([teacher_prompt, responses], dim=-1)
    attention = torch.cat([actor_prompt_mask, response_attention], dim=-1)
    teacher_attention = torch.cat([teacher_prompt_mask, response_attention], dim=-1)
    positions = torch.clamp(attention.long().cumsum(dim=-1) - 1, min=0)
    teacher_positions = torch.clamp(teacher_attention.long().cumsum(dim=-1) - 1, min=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention,
        "responses": responses,
        "response_attention_mask": response_attention,
        "response_mask": response_mask,
        "distill_mask": distill_mask,
        "teacher_sft_mask": teacher_sft_mask,
        "position_ids": positions,
        "teacher_position_ids": teacher_positions,
    }


def test_validate_opsd_layout_accepts_left_prompt_and_right_response_padding():
    summary = validate_opsd_layout(**_valid_layout())

    assert summary["all_layout_checks_passed"] is True
    assert summary["sampled_response_tokens"] == 3
    assert summary["distill_tokens"] == 2
    assert summary["teacher_sft_tokens"] == 3
    assert summary["teacher_sft_sequences"] == 2


def test_validate_opsd_layout_rejects_original_interior_pad_trap():
    values = _valid_layout()
    values["teacher_attention_mask"][0, :4] = torch.tensor([1, 1, 0, 0])
    values["teacher_position_ids"] = torch.clamp(
        values["teacher_attention_mask"].long().cumsum(dim=-1) - 1, min=0
    )

    with pytest.raises(ValueError, match="padding topology"):
        validate_opsd_layout(**values)


def test_validate_opsd_layout_uses_masks_when_pad_equals_eos():
    values = _valid_layout()
    # Token id 2 is both a valid first response token and padding in this
    # fixture. The explicit mask, not token identity, determines validity.
    values["responses"][0, 0] = 2
    values["input_ids"][0, -3:] = values["responses"][0]
    values["teacher_input_ids"][0, -3:] = values["responses"][0]

    assert validate_opsd_layout(**values)["all_layout_checks_passed"] is True


def test_validate_opsd_layout_requires_default_right_pad_position_plateau():
    values = _valid_layout()
    assert values["position_ids"][0, -2:].tolist() == [3, 3]
    values["position_ids"][0, -1] = 0

    with pytest.raises(ValueError, match="attended_mismatches=0"):
        validate_opsd_layout(**values)


def test_validate_opsd_layout_rejects_teacher_sft_beyond_response_attention():
    values = _valid_layout()
    values["teacher_sft_mask"][0, 1] = 1

    with pytest.raises(ValueError, match="subset of response_attention_mask"):
        validate_opsd_layout(**values)


def test_global_aggregation_info_uses_the_selected_branch_mask():
    info = global_aggregation_info(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))

    assert info["dp_size"] == 1
    assert info["batch_num_tokens"] == 3
    assert info["global_batch_size"] == 2


def test_ppo_token_ledger_matches_dual_clipped_formula():
    terms = ppo_token_terms(
        old_log_prob=torch.tensor([[0.0, 0.0]]),
        log_prob=torch.tensor([[0.5, -0.5]]),
        advantages=torch.tensor([[1.0, -1.0]]),
        is_weights=torch.tensor([[0.25, 2.0]]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        clip_ratio_c=3.0,
    )

    ratio = torch.exp(torch.tensor([[0.5, -0.5]]))
    clipped_ratio = ratio.clamp(0.8, 1.3)
    pg1 = -torch.tensor([[1.0, -1.0]]) * ratio
    pg2 = -torch.tensor([[1.0, -1.0]]) * clipped_ratio
    clipped = torch.maximum(pg1, pg2)
    selected = torch.where(
        torch.tensor([[1.0, -1.0]]) < 0,
        torch.minimum(torch.tensor([[-3.0, 3.0]]), clipped),
        clipped,
    )
    assert torch.allclose(terms["teacher_pg_after_is"], selected * torch.tensor([[0.25, 2.0]]))


def test_token_kl_logging_interval_is_explicit_and_one_indexed():
    config = {"enabled": True, "start_step": 2, "end_step": 6, "interval_steps": 2}

    assert not token_kl_logging_is_enabled(config, 1)
    assert token_kl_logging_is_enabled(config, 2)
    assert not token_kl_logging_is_enabled(config, 3)
    assert token_kl_logging_is_enabled(config, 4)
    assert not token_kl_logging_is_enabled(config, 7)


def test_token_reverse_kl_log_uses_masks_not_pad_or_eos_identity():
    response_ids = torch.tensor([[2, 9, 2], [8, 2, 2]])
    response_attention = torch.tensor([[1, 1, 0], [1, 0, 0]])
    response_mask = response_attention.float()
    distill_mask = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    student = torch.tensor([[-1.0, -2.0, 999.0], [-3.0, 999.0, 999.0]])
    teacher = torch.tensor([[-0.5, -3.0, -999.0], [-2.0, -999.0, -999.0]])

    record, selected_samples = build_token_reverse_kl_log_record(
        response_ids=response_ids,
        response_attention_mask=response_attention,
        response_mask=response_mask,
        distill_mask=distill_mask,
        student_log_probs=student,
        teacher_log_probs=teacher,
        sample_indices=torch.tensor([17, 18]),
        max_samples=2,
        max_tokens_per_sample=8,
    )

    assert selected_samples == 2
    assert record["axis_scope"] == "response_only"
    assert record["prompt_tokens_logged"] == 0
    assert record["masked_or_padded_response_tokens_logged"] == 0
    assert record["summary"]["actual_response_token_count"] == 3
    assert [token["response_position"] for token in record["samples"][0]["tokens"]] == [0, 1]
    # Token id 2 is a valid attended EOS/PAD-sharing token at position zero.
    assert record["samples"][0]["tokens"][0]["token_id"] == 2
    assert record["samples"][0]["tokens"][0]["sampled_reverse_kl"] == pytest.approx(-0.5)
    # The masked positions carry intentionally extreme values and must never leak.
    assert all(
        token["response_position"] < sample["actual_response_token_count"]
        for sample in record["samples"]
        for token in sample["tokens"]
    )


def test_token_reverse_kl_log_records_behavior_diagnostic_and_current_policy_advantage():
    response_ids = torch.tensor([[5, 6, 0]])
    response_attention = torch.tensor([[1, 1, 0]])
    response_mask = response_attention.float()
    distill_mask = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    current = torch.tensor([[-0.8, -1.2, 99.0]])
    behavior = torch.tensor([[-1.0, -1.1, -99.0]])
    teacher = torch.tensor([[-0.4, -1.4, 50.0]])
    # The objective evidence must be anchored to the recomputed current actor.
    # Behavior probabilities remain in the ledger only as a PPO diagnostic.
    policy_advantages = (teacher - current) * distill_mask

    record, _ = build_token_reverse_kl_log_record(
        response_ids=response_ids,
        response_attention_mask=response_attention,
        response_mask=response_mask,
        distill_mask=distill_mask,
        student_log_probs=current,
        teacher_log_probs=teacher,
        student_behavior_log_probs=behavior,
        policy_advantages=policy_advantages,
        sample_indices=torch.tensor([3]),
        max_samples=1,
        max_tokens_per_sample=3,
    )

    tokens = record["samples"][0]["tokens"]
    assert tokens[0]["student_behavior_log_prob"] == pytest.approx(-1.0)
    assert tokens[0]["behavior_sampled_reverse_kl"] == pytest.approx(-0.6)
    assert tokens[0]["policy_advantage"] == pytest.approx(0.4)
    # The second token is a real response token but not objective-eligible.
    assert tokens[1]["policy_advantage"] == 0.0
    # The deliberately extreme PAD values never enter the bounded ledger.
    assert len(tokens) == 2


def test_token_reverse_kl_log_selects_largest_absolute_values_but_preserves_trace_order():
    response_ids = torch.tensor([[10, 11, 12, 13, 0]])
    response_attention = torch.tensor([[1, 1, 1, 1, 0]])
    response_mask = response_attention.float()
    student = torch.tensor([[0.1, 0.2, 0.3, 0.4, 100.0]])
    teacher = torch.tensor([[0.0, -1.8, 0.0, 3.4, -100.0]])

    record, _ = build_token_reverse_kl_log_record(
        response_ids=response_ids,
        response_attention_mask=response_attention,
        response_mask=response_mask,
        distill_mask=response_mask,
        student_log_probs=student,
        teacher_log_probs=teacher,
        sample_indices=torch.tensor([0]),
        max_samples=1,
        max_tokens_per_sample=2,
    )

    assert [token["response_position"] for token in record["samples"][0]["tokens"]] == [1, 3]


def test_token_reverse_kl_log_rejects_any_mask_escape():
    with pytest.raises(ValueError, match="distill_mask escaped response_mask"):
        build_token_reverse_kl_log_record(
            response_ids=torch.tensor([[1, 0]]),
            response_attention_mask=torch.tensor([[1, 0]]),
            response_mask=torch.tensor([[1.0, 0.0]]),
            distill_mask=torch.tensor([[1.0, 1.0]]),
            student_log_probs=torch.tensor([[-1.0, -2.0]]),
            teacher_log_probs=torch.tensor([[-1.5, -2.5]]),
            sample_indices=torch.tensor([0]),
            max_samples=1,
            max_tokens_per_sample=2,
        )
