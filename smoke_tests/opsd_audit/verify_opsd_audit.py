#!/usr/bin/env python3
"""Independently verify structured OPSD audit artifacts.

This script intentionally uses only the Python standard library and does not
import VERL or ``recipe.opsd`` arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


WIDE_CSS = """<style>
body, main, article, .markdown-body, .rendered_html,
.jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: min(98vw, 1800px) !important;
}
table { width: 100% !important; }
</style>
"""


def close(actual: float, expected: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= atol + rtol * abs(expected)


def require_close(errors: list[str], name: str, actual: float, expected: float, *, atol: float, rtol: float):
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError):
        errors.append(f"{name}: non-numeric actual={actual!r}, expected={expected!r}")
        return
    if not close(actual_value, expected_value, atol=atol, rtol=rtol):
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}, atol={atol}, rtol={rtol}")


def require_equal(errors: list[str], name: str, actual, expected):
    if actual != expected:
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}")


def require_fields(errors: list[str], prefix: str, record: dict, fields: tuple[str, ...]):
    missing = [field for field in fields if field not in record]
    if missing:
        errors.append(f"{prefix}: missing fields {missing}")


def left_padded(mask: list[float]) -> bool:
    seen_valid = False
    for raw in mask:
        value = bool(raw)
        if value:
            seen_valid = True
        elif seen_valid:
            return False
    return True


def right_padded(mask: list[float]) -> bool:
    seen_padding = False
    for raw in mask:
        value = bool(raw)
        if not value:
            seen_padding = True
        elif seen_padding:
            return False
    return True


def expected_positions(mask: list[float]) -> list[int]:
    position = -1
    result = []
    for raw in mask:
        if bool(raw):
            position += 1
        # VERL clamps initial left-padding positions to zero, then retains the
        # final valid position across masked right-padding tokens.
        result.append(max(position, 0))
    return result


def flatten_masked(values, mask):
    return [value for row, mask_row in zip(values, mask) for value, active in zip(row, mask_row) if bool(active)]


def masked_mean(values, mask) -> float:
    selected = flatten_masked(values, mask)
    return sum(selected) / len(selected) if selected else 0.0


def masked_max(values, mask) -> float:
    selected = flatten_masked(values, mask)
    return max(selected) if selected else 0.0


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def compare_matrix(errors, prefix, actual, expected, *, atol=1e-5, rtol=1e-4):
    if actual is None or expected is None:
        if actual is not expected:
            errors.append(f"{prefix}: one matrix is None")
        return
    require_equal(errors, f"{prefix}:rows", len(actual), len(expected))
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        require_equal(errors, f"{prefix}:row={row_index}:columns", len(actual_row), len(expected_row))
        for token_index, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
            require_close(
                errors,
                f"{prefix}[{row_index},{token_index}]",
                actual_value,
                expected_value,
                atol=atol,
                rtol=rtol,
            )


def verify_logprob_stats(errors, prefix, metrics, values, mask, metric_prefix, *, atol, rtol):
    valid = flatten_masked(values, mask)
    if not valid:
        return
    sequence_means = []
    for row, mask_row in zip(values, mask):
        count = sum(mask_row)
        sequence_means.append(sum(value * active for value, active in zip(row, mask_row)) / max(count, 1))
    expected = {
        f"{metric_prefix}_token_logprob_mean": sum(valid) / len(valid),
        f"{metric_prefix}_token_logprob_std": population_std(valid),
        f"{metric_prefix}_seq_logprob_mean": sum(sequence_means) / len(sequence_means),
        f"{metric_prefix}_seq_logprob_std": population_std(sequence_means),
    }
    for name, value in expected.items():
        require_close(errors, f"{prefix}:{name}", metrics.get(name), value, atol=atol, rtol=rtol)


def single_island(mask: list[float]) -> bool:
    state = 0
    for raw in mask:
        value = bool(raw)
        if state == 0 and value:
            state = 1
        elif state == 1 and not value:
            state = 2
        elif state == 2 and value:
            return False
    return True


def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        raise ValueError("subsequence pattern must be non-empty")
    if len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def aggregate(values, masks, info, mode):
    dp_size = int(info["dp_size"])
    if mode == "token-mean":
        denominator = int(info["batch_num_tokens"])
        if denominator <= 0:
            return 0.0
        numerator = sum(value * mask for row, mask_row in zip(values, masks) for value, mask in zip(row, mask_row))
        return numerator / denominator * dp_size
    if mode in {"seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        denominator = int(info["global_batch_size"])
        if denominator <= 0:
            return 0.0
        result = sum(sum(value * mask for value, mask in zip(row, mask_row)) for row, mask_row in zip(values, masks))
        result = result / denominator * dp_size
        if mode == "seq-mean-token-sum-norm":
            result /= info.get("loss_scale_factor") or len(masks[0])
        return result
    if mode == "seq-mean-token-mean":
        denominator = int(info["global_batch_size"])
        if denominator <= 0:
            return 0.0
        result = 0.0
        for row, mask_row in zip(values, masks):
            token_count = sum(mask_row)
            if token_count:
                result += sum(value * mask for value, mask in zip(row, mask_row)) / token_count
        return result / denominator * dp_size
    raise ValueError(f"unsupported loss aggregation mode: {mode}")


def verify_layout_file(
    path: Path,
    errors: list[str],
    counters: dict[str, int],
    trainer_records: dict[tuple[int, int], dict],
    step_configs: dict[int, dict],
    observations: dict,
):
    payload = json.loads(path.read_text())
    step = int(payload.get("global_step", -1))
    prefix = f"{path}:step={step}"
    require_fields(
        errors,
        prefix,
        payload,
        (
            "optimizer_updates_completed",
            "trainer_updates_completed_before_step",
            "temperature",
            "top_p",
            "top_k",
            "generation_model",
            "generation_conditioning",
            "teacher_conditioning",
            "teacher_sft",
            "behavior_logprob_source",
            "advantage_estimator",
            "norm_adv_by_std_in_grpo",
            "use_kl_in_reward",
            "actor_model_path",
            "teacher_model_path",
            "actor_pad_token_id",
            "teacher_pad_token_id",
            "records",
        ),
    )
    require_equal(errors, f"{prefix}:optimizer_updates_completed", payload.get("optimizer_updates_completed"), step - 1)
    require_equal(
        errors,
        f"{prefix}:trainer_updates_completed_before_step",
        payload.get("trainer_updates_completed_before_step"),
        step - 1,
    )
    require_close(errors, f"{prefix}:temperature", payload.get("temperature"), 1.0, atol=0.0, rtol=0.0)
    require_close(errors, f"{prefix}:top_p", payload.get("top_p"), 1.0, atol=0.0, rtol=0.0)
    if payload.get("top_k") not in (-1, 0):
        errors.append(f"{prefix}: top-k sampling is not disabled: {payload.get('top_k')!r}")
    require_equal(errors, f"{prefix}:generation_model", payload.get("generation_model"), "actor")
    require_equal(errors, f"{prefix}:generation_conditioning", payload.get("generation_conditioning"), "original_prompt")
    require_equal(errors, f"{prefix}:teacher_conditioning", payload.get("teacher_conditioning"), "ground_truth")
    require_equal(errors, f"{prefix}:behavior_logprob_source", payload.get("behavior_logprob_source"), "rollout")
    if "grpo" not in str(payload.get("advantage_estimator", "")).lower():
        errors.append(f"{prefix}: expected GRPO advantage estimator, got {payload.get('advantage_estimator')!r}")
    require_equal(errors, f"{prefix}:use_kl_in_reward", payload.get("use_kl_in_reward"), False)
    require_equal(errors, f"{prefix}:actor_teacher_pad_id", payload.get("actor_pad_token_id"), payload.get("teacher_pad_token_id"))
    require_equal(errors, f"{prefix}:actor_teacher_eos_id", payload.get("actor_eos_token_id"), payload.get("teacher_eos_token_id"))

    opsd_config = payload.get("opsd_config", {})
    require_equal(errors, f"{prefix}:opsd_mode", opsd_config.get("mode"), "opsd_rlvr")
    require_equal(errors, f"{prefix}:teacher_model", opsd_config.get("teacher_model"), "separate")
    require_equal(errors, f"{prefix}:distill_loss", opsd_config.get("distill_loss"), "sampled_reverse_kl")
    require_equal(errors, f"{prefix}:balance_mode", opsd_config.get("balance_mode"), "none")
    for disabled_control in ("topk", "distill_beta", "distill_token_clip", "distill_token_clip_tail"):
        require_equal(errors, f"{prefix}:{disabled_control}", opsd_config.get(disabled_control), None)
    teacher_sft = payload.get("teacher_sft", {})
    sft_enabled = bool(teacher_sft.get("enabled", False))
    require_equal(
        errors,
        f"{prefix}:teacher_sft_enabled_from_weight",
        sft_enabled,
        float(opsd_config.get("teacher_sft_weight", 0.0) or 0.0) > 0.0,
    )
    if sft_enabled:
        require_equal(errors, f"{prefix}:teacher_sft_model", opsd_config.get("teacher_model"), "separate")
        require_equal(errors, f"{prefix}:teacher_sft_source", opsd_config.get("teacher_source"), "ground_truth")
        require_equal(
            errors,
            f"{prefix}:teacher_sft_scope",
            teacher_sft.get("target_scope"),
            opsd_config.get("teacher_sft_target_scope"),
        )
        require_close(
            errors,
            f"{prefix}:teacher_sft_weight",
            teacher_sft.get("weight"),
            opsd_config.get("teacher_sft_weight"),
            atol=0.0,
            rtol=0.0,
        )
        require_equal(
            errors,
            f"{prefix}:teacher_sft_success_field",
            teacher_sft.get("success_field"),
            opsd_config.get("teacher_sft_success_field"),
        )
        require_close(
            errors,
            f"{prefix}:teacher_sft_success_threshold",
            teacher_sft.get("success_threshold"),
            opsd_config.get("teacher_sft_success_threshold"),
            atol=0.0,
            rtol=0.0,
        )
        require_equal(
            errors,
            f"{prefix}:teacher_sft_think_end_tag",
            teacher_sft.get("think_end_tag"),
            opsd_config.get("teacher_sft_think_end_tag"),
        )
        if not teacher_sft.get("think_end_token_ids"):
            errors.append(f"{prefix}: teacher SFT closing-think token IDs are empty")
    step_configs[step] = {
        "opsd": opsd_config,
        "audit": opsd_config.get("audit", {}),
        "teacher_sft": teacher_sft,
        "norm_adv_by_std_in_grpo": bool(payload.get("norm_adv_by_std_in_grpo", True)),
    }

    if payload.get("summary", {}).get("all_layout_checks_passed") is not True:
        errors.append(f"{path}: missing successful trainer layout summary")
    records = payload.get("records", [])
    if sft_enabled:
        require_equal(
            errors,
            f"{prefix}:summary_teacher_sft_tokens",
            payload.get("summary", {}).get("teacher_sft_tokens"),
            sum(sum(record.get("teacher_sft_mask", [])) for record in records),
        )
        require_equal(
            errors,
            f"{prefix}:summary_teacher_sft_sequences",
            payload.get("summary", {}).get("teacher_sft_sequences"),
            sum(bool(sum(record.get("teacher_sft_mask", []))) for record in records),
        )
    for record in records:
        sample = int(record["audit_sample_index"])
        record_prefix = f"{prefix}:sample={sample}"
        require_fields(
            errors,
            record_prefix,
            record,
            (
                "actor_input_ids",
                "teacher_input_ids",
                "actor_prompt_ids",
                "teacher_prompt_ids",
                "response_ids",
                "actor_attention_mask",
                "teacher_attention_mask",
                "response_attention_mask",
                "response_mask",
                "distill_mask",
                "actor_position_ids",
                "teacher_position_ids",
                "ground_truth_answer",
                "actor_prompt_text",
                "teacher_prompt_text",
                "response_text",
                "token_level_scores",
                "token_level_rewards",
                "advantages",
                "rollout_log_probs",
                "teacher_old_log_probs",
                "student_behavior_log_probs",
            ),
        )
        if sft_enabled:
            require_fields(
                errors,
                record_prefix,
                record,
                (
                    "teacher_sft_mask",
                    "teacher_sft_success_value",
                    "teacher_sft_success",
                    "teacher_sft_think_end_exclusive",
                    "teacher_sft_target_text",
                ),
            )
        actor_mask = record["actor_attention_mask"]
        teacher_mask = record["teacher_attention_mask"]
        response_mask = record["response_attention_mask"]
        response_length = len(response_mask)
        actor_prompt_mask = actor_mask[:-response_length]
        teacher_prompt_mask = teacher_mask[:-response_length]
        if not left_padded(actor_prompt_mask):
            errors.append(f"{record_prefix}: actor prompt is not left padded")
        if not left_padded(teacher_prompt_mask):
            errors.append(f"{record_prefix}: teacher prompt is not left padded")
        if not right_padded(response_mask):
            errors.append(f"{record_prefix}: sampled response is not right padded")
        if not right_padded(record["response_mask"]):
            errors.append(f"{record_prefix}: RL loss response mask is not right padded")
        if not right_padded(record["distill_mask"]):
            errors.append(f"{record_prefix}: distillation mask is not right padded")
        if sft_enabled and not right_padded(record["teacher_sft_mask"]):
            errors.append(f"{record_prefix}: teacher SFT mask is not right padded")
        if not single_island(actor_mask):
            errors.append(f"{record_prefix}: actor interior padding")
        if not single_island(teacher_mask):
            errors.append(f"{record_prefix}: teacher interior padding")
        if actor_mask[-response_length:] != response_mask:
            errors.append(f"{record_prefix}: actor response mask suffix mismatch")
        if teacher_mask[-response_length:] != response_mask:
            errors.append(f"{record_prefix}: teacher response mask suffix mismatch")
        require_equal(
            errors,
            f"{record_prefix}:actor_input_concatenation",
            record["actor_input_ids"],
            record["actor_prompt_ids"] + record["response_ids"],
        )
        require_equal(
            errors,
            f"{record_prefix}:teacher_input_concatenation",
            record["teacher_input_ids"],
            record["teacher_prompt_ids"] + record["response_ids"],
        )
        require_equal(
            errors,
            f"{record_prefix}:actor_positions",
            record["actor_position_ids"],
            expected_positions(actor_mask),
        )
        require_equal(
            errors,
            f"{record_prefix}:teacher_positions",
            record["teacher_position_ids"],
            expected_positions(teacher_mask),
        )
        if any(d > r or r > a for d, r, a in zip(record["distill_mask"], record["response_mask"], response_mask)):
            errors.append(f"{record_prefix}: distill/loss/attention mask subset violation")
        if sft_enabled and any(s > a for s, a in zip(record["teacher_sft_mask"], response_mask)):
            errors.append(f"{record_prefix}: teacher SFT mask is not a subset of response attention")

        if sft_enabled:
            success_threshold = float(teacher_sft["success_threshold"])
            expected_success = float(record["teacher_sft_success_value"]) > success_threshold
            require_equal(
                errors,
                f"{record_prefix}:teacher_sft_success",
                bool(record["teacher_sft_success"]),
                expected_success,
            )
            success_field = str(teacher_sft["success_field"])
            reward_value_name = f"reward_extra_{success_field}"
            if reward_value_name in record:
                require_close(
                    errors,
                    f"{record_prefix}:teacher_sft_success_source",
                    record["teacher_sft_success_value"],
                    record[reward_value_name],
                    atol=0.0,
                    rtol=0.0,
                )
            elif success_field == "acc":
                errors.append(f"{record_prefix}: missing smoke success source {reward_value_name}")

            valid_length = int(sum(response_mask))
            valid_ids = record["response_ids"][:valid_length]
            delimiter_ids = [int(value) for value in teacher_sft["think_end_token_ids"]]
            delimiter_start = find_last_subsequence(valid_ids, delimiter_ids)
            expected_boundary = -1 if delimiter_start is None else delimiter_start + len(delimiter_ids)
            require_equal(
                errors,
                f"{record_prefix}:teacher_sft_think_end_exclusive",
                int(record["teacher_sft_think_end_exclusive"]),
                expected_boundary,
            )
            scope = teacher_sft["target_scope"]
            if expected_success and scope == "thinking_only" and expected_boundary < 0:
                errors.append(f"{record_prefix}: successful thinking-only target has no closing-think delimiter")
            if not expected_success:
                expected_sft_mask = [0.0] * response_length
            elif scope == "thinking_only":
                expected_sft_mask = [float(index < expected_boundary) for index in range(response_length)]
            else:
                expected_sft_mask = [float(value) for value in response_mask]
            require_equal(
                errors,
                f"{record_prefix}:teacher_sft_mask",
                [float(value) for value in record["teacher_sft_mask"]],
                expected_sft_mask,
            )
            target_text = str(record["teacher_sft_target_text"])
            if expected_success and any(expected_sft_mask) and not target_text:
                errors.append(f"{record_prefix}: successful teacher SFT target text is empty")
            if not expected_success and target_text:
                errors.append(f"{record_prefix}: unsuccessful rollout emitted teacher SFT target text")
            if (
                expected_success
                and scope == "thinking_only"
                and not target_text.rstrip().endswith(str(teacher_sft["think_end_tag"]))
            ):
                errors.append(f"{record_prefix}: thinking-only target text does not end at closing-think tag")
            counters["teacher_sft_target_tokens"] += int(sum(expected_sft_mask))
            counters["teacher_sft_target_sequences"] += int(any(expected_sft_mask))
            counters["teacher_sft_successful_rollouts"] += int(expected_success)
            counters["teacher_sft_rollouts_with_think_end"] += int(expected_boundary >= 0)

        actor_pad_id = payload.get("actor_pad_token_id")
        teacher_pad_id = payload.get("teacher_pad_token_id")
        for token_index, (token_id, active) in enumerate(zip(record["actor_prompt_ids"], actor_prompt_mask)):
            if not active and token_id != actor_pad_id:
                errors.append(f"{record_prefix}: actor prompt masked token {token_index} is not PAD")
        for token_index, (token_id, active) in enumerate(zip(record["teacher_prompt_ids"], teacher_prompt_mask)):
            if not active and token_id != teacher_pad_id:
                errors.append(f"{record_prefix}: teacher prompt masked token {token_index} is not PAD")
        for token_index, (token_id, active) in enumerate(zip(record["response_ids"], response_mask)):
            if not active and token_id != actor_pad_id:
                errors.append(f"{record_prefix}: masked response token {token_index} is not PAD")
            if active and token_id == actor_pad_id:
                counters["attended_pad_id_tokens"] += 1

        ground_truth = str(record.get("ground_truth_answer", ""))
        if not ground_truth:
            errors.append(f"{record_prefix}: empty ground-truth answer")
        elif ground_truth not in record.get("teacher_prompt_text", ""):
            errors.append(f"{record_prefix}: ground truth is absent from decoded teacher prompt")
        if record.get("actor_prompt_ids") == record.get("teacher_prompt_ids"):
            errors.append(f"{record_prefix}: actor and ground-conditioned teacher prompts are identical")

        if record.get("data_source") == "openthoughts_math_30k_opsd":
            require_equal(
                errors,
                f"{record_prefix}:source_dataset",
                record.get("source_dataset"),
                "siyanzhao/Openthoughts_math_30k_opsd",
            )
            require_fields(
                errors,
                record_prefix,
                record,
                (
                    "source_solution",
                    "source_answer",
                    "source_cot_reason_present",
                    "reward_ground_truth",
                    "ground_truth_field",
                ),
            )
            require_equal(errors, f"{record_prefix}:ground_truth_field", record["ground_truth_field"], "solution")
            require_equal(errors, f"{record_prefix}:solution_teacher_mapping", record["source_solution"], ground_truth)
            if ground_truth and ground_truth in record.get("actor_prompt_text", ""):
                errors.append(f"{record_prefix}: source solution leaked into decoded actor prompt")
            require_equal(
                errors,
                f"{record_prefix}:cot_reason_omitted",
                record["source_cot_reason_present"],
                False,
            )
            try:
                decoded_reward_ground_truth = json.loads(record["reward_ground_truth"])
            except (TypeError, json.JSONDecodeError) as exc:
                errors.append(f"{record_prefix}: reward ground truth is not valid JSON: {exc}")
            else:
                require_equal(
                    errors,
                    f"{record_prefix}:answer_reward_mapping",
                    decoded_reward_ground_truth,
                    record["source_answer"],
                )
            counters["openthoughts_data_contract_samples"] += 1

        if not payload.get("use_kl_in_reward", False):
            compare_matrix(
                errors,
                f"{record_prefix}:score_reward_identity",
                [record["token_level_rewards"]],
                [record["token_level_scores"]],
            )
        compare_matrix(
            errors,
            f"{record_prefix}:rollout_behavior_identity",
            [record["student_behavior_log_probs"]],
            [record["rollout_log_probs"]],
        )
        trainer_records[(step, sample)] = record
        counters["trainer_samples"] += 1

    grouped = {}
    for record in records:
        if "token_level_rewards" not in record or "advantages" not in record:
            continue
        grouped.setdefault(record.get("prompt_group_id"), []).append(record)
    for group_id, group_records in grouped.items():
        if group_id is None:
            errors.append(f"{prefix}: GRPO group ID is missing")
        if len(group_records) < 2:
            errors.append(f"{prefix}: GRPO group {group_id!r} has only {len(group_records)} sample")
        rewards = [sum(record["token_level_rewards"]) for record in group_records]
        if len(rewards) == 1:
            mean, std = 0.0, 1.0
        else:
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1))
        for record, reward in zip(group_records, rewards):
            expected_scalar = reward - mean
            if payload.get("norm_adv_by_std_in_grpo", True):
                expected_scalar /= std + 1e-6
            for token_index, (actual, mask) in enumerate(zip(record["advantages"], record["response_mask"])):
                expected = expected_scalar * mask
                require_close(
                    errors,
                    f"{path}:group={group_id}:sample={record['audit_sample_index']}:advantage[{token_index}]",
                    actual,
                    expected,
                    atol=1e-5,
                    rtol=1e-4,
                )

    observations.setdefault("steps", {}).setdefault(str(step), {}).update(
        {
            "trainer_updates_completed_at_value_capture": step - 1,
            "generation_model": payload.get("generation_model"),
            "generation_conditioning": payload.get("generation_conditioning"),
            "teacher_conditioning": payload.get("teacher_conditioning"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "top_k": payload.get("top_k"),
            "trainer_samples": len(records),
            "actor_model_path": payload.get("actor_model_path"),
            "teacher_model_path": payload.get("teacher_model_path"),
        }
    )


def verify_microbatch(
    record,
    path,
    line_number,
    errors,
    counters,
    health,
    aggregation,
    trainer_records,
    step_configs,
    observations,
    local_objectives,
    dispatch_counts,
):
    prefix = f"{path}:{line_number}:step={record['global_step']}:rank={record['rank']}:micro={record['microbatch_index']}"
    require_fields(
        errors,
        prefix,
        record,
        (
            "optimizer_updates_completed",
            "trainer_updates_completed_before_step",
            "warmup_active",
            "temperature",
            "teacher_model",
            "actor_use_remove_padding",
            "teacher_use_remove_padding",
            "loss_agg_mode",
            "sample_indices",
            "response_ids",
            "response_attention_mask",
            "response_mask",
            "distill_mask",
            "teacher_log_probs",
            "teacher_old_log_probs",
            "student_behavior_log_probs",
            "advantages",
            "teacher_is_mode",
            "teacher_is_upper_clip",
            "teacher_behavior_log_ratio",
            "teacher_behavior_sequence_log_ratio",
            "teacher_ppo_terms",
            "distill_global_info",
            "rlvr_global_info",
            "gradient_routing",
            "total_weighted_loss_production",
        ),
    )
    step = int(record["global_step"])
    require_equal(errors, f"{prefix}:optimizer_updates_completed", record["optimizer_updates_completed"], step - 1)
    require_equal(
        errors,
        f"{prefix}:trainer_updates_completed_before_step",
        record["trainer_updates_completed_before_step"],
        step - 1,
    )
    require_equal(errors, f"{prefix}:teacher_model", record["teacher_model"], "separate")
    require_close(errors, f"{prefix}:temperature", record["temperature"], 1.0, atol=0.0, rtol=0.0)
    config = step_configs.get(step, {})
    audit_config = config.get("audit", {})
    teacher_sft_config = config.get("teacher_sft", {})
    teacher_sft_enabled = bool(teacher_sft_config.get("enabled", False))
    if teacher_sft_enabled:
        require_fields(
            errors,
            prefix,
            record,
            (
                "teacher_sft_mask",
                "teacher_sft_success_values",
                "teacher_sft_success",
                "teacher_sft_think_end_exclusive",
                "teacher_sft_token_nll",
                "teacher_sft_loss_production",
                "teacher_sft_weight",
                "teacher_sft_global_info",
            ),
        )
    atol = float(audit_config.get("scalar_atol", 1e-5))
    rtol = float(audit_config.get("scalar_rtol", 1e-4))
    forward_max_tolerance = float(audit_config.get("forward_max_abs_error", 5e-2))
    forward_mean_tolerance = float(audit_config.get("forward_mean_abs_error", 5e-3))
    behavior_max_tolerance = float(audit_config.get("behavior_forward_max_abs_error", 0.75))
    behavior_mean_tolerance = float(audit_config.get("behavior_forward_mean_abs_error", 0.03))
    actor = record.get("actor_log_probs")
    teacher = record["teacher_log_probs"]
    distill_mask = record["distill_mask"]
    response_mask = record["response_mask"]
    sampling_mask = record["response_attention_mask"]
    teacher_sft_mask = record.get("teacher_sft_mask")
    mode = record["loss_agg_mode"]
    micro_metrics = record.get("micro_metrics", {})
    sample_indices = [int(value) for value in record["sample_indices"]]
    for local_index, sample_index in enumerate(sample_indices):
        dispatch_counts[(step, sample_index)] = dispatch_counts.get((step, sample_index), 0) + 1
        source = trainer_records.get((step, sample_index))
        if source is None:
            errors.append(f"{prefix}: no trainer record for sample index {sample_index}")
            continue
        require_equal(errors, f"{prefix}:sample={sample_index}:response_ids", record["response_ids"][local_index], source["response_ids"])
        require_equal(
            errors,
            f"{prefix}:sample={sample_index}:response_attention_mask",
            sampling_mask[local_index],
            source["response_attention_mask"],
        )
        require_equal(errors, f"{prefix}:sample={sample_index}:response_mask", response_mask[local_index], source["response_mask"])
        require_equal(errors, f"{prefix}:sample={sample_index}:distill_mask", distill_mask[local_index], source["distill_mask"])
        if teacher_sft_enabled:
            require_equal(
                errors,
                f"{prefix}:sample={sample_index}:teacher_sft_mask",
                teacher_sft_mask[local_index],
                source["teacher_sft_mask"],
            )
            require_close(
                errors,
                f"{prefix}:sample={sample_index}:teacher_sft_success_value",
                record["teacher_sft_success_values"][local_index],
                source["teacher_sft_success_value"],
                atol=0.0,
                rtol=0.0,
            )
            require_equal(
                errors,
                f"{prefix}:sample={sample_index}:teacher_sft_success",
                bool(record["teacher_sft_success"][local_index]),
                bool(source["teacher_sft_success"]),
            )
            require_equal(
                errors,
                f"{prefix}:sample={sample_index}:teacher_sft_boundary",
                int(record["teacher_sft_think_end_exclusive"][local_index]),
                int(source["teacher_sft_think_end_exclusive"]),
            )
        for tensor_name in ("teacher_old_log_probs", "student_behavior_log_probs", "advantages"):
            compare_matrix(
                errors,
                f"{prefix}:sample={sample_index}:{tensor_name}:dispatch_identity",
                [record[tensor_name][local_index]],
                [source[tensor_name]],
                atol=atol,
                rtol=rtol,
            )

    step_observation = observations.setdefault("steps", {}).setdefault(str(step), {})
    step_observation.setdefault("actor_behavior_logprob_max_abs_errors", [])
    step_observation.setdefault("_actor_behavior_signed_differences", [])
    step_observation.setdefault("teacher_old_current_logprob_max_abs_errors", [])
    step_observation.setdefault("strict_padding_reference_max_abs_errors", [])
    step_observation.setdefault("dense_cross_kernel_max_abs_errors", [])
    expected_distill_scalar = 0.0
    expected_estimate_scalar = 0.0
    expected_rlvr_scalar = 0.0
    expected_sft_scalar = 0.0
    require_close(
        errors,
        f"{prefix}:warmup_metric",
        micro_metrics.get("actor/rlvr_warmup_active"),
        float(bool(record["warmup_active"])),
        atol=0.0,
        rtol=0.0,
    )
    aggregation_branches = [
        ("distill", distill_mask, "distill_global_info"),
        ("rlvr", response_mask, "rlvr_global_info"),
    ]
    if teacher_sft_enabled:
        aggregation_branches.append(("teacher_sft", teacher_sft_mask, "teacher_sft_global_info"))
    for branch, mask, info_name in aggregation_branches:
        key = (int(record["global_step"]), branch)
        item = aggregation.setdefault(
            key,
            {
                "tokens": 0,
                "sequences": 0,
                "declared_tokens": set(),
                "declared_sequences": set(),
                "dp_sizes": set(),
            },
        )
        item["tokens"] += int(sum(sum(row) for row in mask))
        item["sequences"] += sum(1 for row in mask if sum(row) > 0)
        item["declared_tokens"].add(int(record[info_name]["batch_num_tokens"]))
        item["declared_sequences"].add(int(record[info_name]["global_batch_size"]))
        item["dp_sizes"].add(int(record[info_name]["dp_size"]))

    teacher_old = record.get("teacher_old_log_probs")
    if teacher_old is not None:
        teacher_errors = [
            abs(current - old)
            for current_row, old_row, mask_row in zip(teacher, teacher_old, sampling_mask)
            for current, old, active in zip(current_row, old_row, mask_row)
            if active
        ]
        if teacher_errors:
            teacher_max_error = max(teacher_errors)
            teacher_mean_error = sum(teacher_errors) / len(teacher_errors)
            step_observation["teacher_old_current_logprob_max_abs_errors"].append(teacher_max_error)
            if teacher_max_error > forward_max_tolerance or teacher_mean_error > forward_mean_tolerance:
                errors.append(
                    f"{prefix}: pre-update teacher current/old log-prob mismatch "
                    f"max_abs={teacher_max_error}, mean_abs={teacher_mean_error}"
                )

    if actor is not None:
        behavior = record["student_behavior_log_probs"]
        actor_behavior_errors = [
            abs(current - sampled)
            for current_row, sampled_row, mask_row in zip(actor, behavior, sampling_mask)
            for current, sampled, active in zip(current_row, sampled_row, mask_row)
            if active
        ]
        if actor_behavior_errors:
            actor_behavior_max_error = max(actor_behavior_errors)
            actor_behavior_mean_error = sum(actor_behavior_errors) / len(actor_behavior_errors)
            step_observation["actor_behavior_logprob_max_abs_errors"].append(actor_behavior_max_error)
            step_observation["_actor_behavior_signed_differences"].extend(
                current - sampled
                for current_row, sampled_row, mask_row in zip(actor, behavior, sampling_mask)
                for current, sampled, active in zip(current_row, sampled_row, mask_row)
                if active
            )
            if actor_behavior_max_error > behavior_max_tolerance or actor_behavior_mean_error > behavior_mean_tolerance:
                errors.append(
                    f"{prefix}: on-policy actor learner/rollout log-prob mismatch "
                    f"max_abs={actor_behavior_max_error} (limit {behavior_max_tolerance}), "
                    f"mean_abs={actor_behavior_mean_error} (limit {behavior_mean_tolerance})"
                )
        expected_estimate = []
        expected_surrogate = []
        for actor_row, teacher_row in zip(actor, teacher):
            estimate_row = []
            surrogate_row = []
            for actor_value, teacher_value in zip(actor_row, teacher_row):
                estimate = actor_value - teacher_value
                estimate_row.append(estimate)
                surrogate_row.append((estimate + 1.0) * actor_value)
            expected_estimate.append(estimate_row)
            expected_surrogate.append(surrogate_row)
        for row_index, (actual_row, expected_row) in enumerate(
            zip(record["reverse_kl_token_estimate"], expected_estimate)
        ):
            for token_index, (actual, expected) in enumerate(zip(actual_row, expected_row)):
                require_close(errors, f"{prefix}:reverse_kl[{row_index},{token_index}]", actual, expected, atol=atol, rtol=rtol)
        for row_index, (actual_row, expected_row) in enumerate(
            zip(record["distill_token_surrogate"], expected_surrogate)
        ):
            for token_index, (actual, expected) in enumerate(zip(actual_row, expected_row)):
                require_close(errors, f"{prefix}:surrogate[{row_index},{token_index}]", actual, expected, atol=atol, rtol=rtol)
        expected_distill = aggregate(expected_surrogate, distill_mask, record["distill_global_info"], mode)
        expected_distill_scalar = expected_distill
        expected_estimate_scalar = aggregate(expected_estimate, distill_mask, record["distill_global_info"], mode)
        require_close(errors, f"{prefix}:distill_loss", record["distill_loss_production"], expected_distill, atol=atol, rtol=rtol)
        require_close(
            errors,
            f"{prefix}:reverse_kl_estimate_scalar",
            record["reverse_kl_estimate_production"],
            expected_estimate_scalar,
            atol=atol,
            rtol=rtol,
        )
        require_close(
            errors,
            f"{prefix}:distill_active_rate",
            micro_metrics.get("actor/opsd_distill_active_rate"),
            sum(float(sum(row) > 0) for row in distill_mask) / len(distill_mask),
            atol=atol,
            rtol=rtol,
        )
        expected_distill_metrics = {
            "actor/opsd_distill_token_mean": masked_mean(expected_surrogate, distill_mask),
            "actor/opsd_distill_token_max": masked_max(expected_surrogate, distill_mask),
            "actor/opsd_reverse_kl_token_estimate_mean": masked_mean(expected_estimate, distill_mask),
            "actor/opsd_reverse_kl_token_estimate_max": masked_max(expected_estimate, distill_mask),
            "actor/opsd_student_logprob_mean": masked_mean(actor, distill_mask),
            "actor/opsd_teacher_logprob_mean": masked_mean(teacher, distill_mask),
        }
        for metric_name, expected in expected_distill_metrics.items():
            require_close(
                errors,
                f"{prefix}:{metric_name}",
                micro_metrics.get(metric_name),
                expected,
                atol=atol,
                rtol=rtol,
            )
        verify_logprob_stats(
            errors,
            prefix,
            micro_metrics,
            actor,
            distill_mask,
            "actor/opsd_student",
            atol=atol,
            rtol=rtol,
        )

    verify_logprob_stats(
        errors,
        prefix,
        micro_metrics,
        teacher,
        response_mask,
        "actor/opsd_teacher",
        atol=atol,
        rtol=rtol,
    )

    if teacher_sft_enabled:
        expected_sft_token_nll = [[-value for value in row] for row in teacher]
        compare_matrix(
            errors,
            f"{prefix}:teacher_sft_token_nll",
            record["teacher_sft_token_nll"],
            expected_sft_token_nll,
            atol=atol,
            rtol=rtol,
        )
        expected_sft_scalar = aggregate(
            expected_sft_token_nll,
            teacher_sft_mask,
            record["teacher_sft_global_info"],
            mode,
        )
        require_close(
            errors,
            f"{prefix}:teacher_sft_loss",
            record["teacher_sft_loss_production"],
            expected_sft_scalar,
            atol=atol,
            rtol=rtol,
        )
        expected_sft_metrics = {
            "actor/teacher_sft_success_rate": (
                sum(float(bool(value)) for value in record["teacher_sft_success"])
                / len(record["teacher_sft_success"])
            ),
            "actor/teacher_sft_active_rate": (
                sum(float(sum(row) > 0) for row in teacher_sft_mask) / len(teacher_sft_mask)
            ),
            "actor/teacher_sft_token_nll_mean": masked_mean(
                expected_sft_token_nll, teacher_sft_mask
            ),
            "actor/teacher_sft_token_nll_max": masked_max(
                expected_sft_token_nll, teacher_sft_mask
            ),
            "actor/teacher_sft_target_tokens": sum(sum(row) for row in teacher_sft_mask),
        }
        for metric_name, expected in expected_sft_metrics.items():
            require_close(
                errors,
                f"{prefix}:{metric_name}",
                micro_metrics.get(metric_name),
                expected,
                atol=atol,
                rtol=rtol,
            )

    if record.get("teacher_old_log_probs") is not None:
        teacher_old = record["teacher_old_log_probs"]
        behavior = record["student_behavior_log_probs"]
        advantages = record["advantages"]
        is_weights = record.get("is_weights")
        is_mode = record["teacher_is_mode"]
        clip = float(record["teacher_is_upper_clip"])
        raw_token_ratios = [
            [old - sampled for old, sampled in zip(old_row, behavior_row)]
            for old_row, behavior_row in zip(teacher_old, behavior)
        ]
        raw_sequence_ratios = [
            sum(raw * mask for raw, mask in zip(raw_row, sampling_row))
            for raw_row, sampling_row in zip(raw_token_ratios, sampling_mask)
        ]
        compare_matrix(
            errors,
            f"{prefix}:teacher_behavior_log_ratio",
            record["teacher_behavior_log_ratio"],
            raw_token_ratios,
            atol=atol,
            rtol=rtol,
        )
        compare_matrix(
            errors,
            f"{prefix}:teacher_behavior_sequence_log_ratio",
            [record["teacher_behavior_sequence_log_ratio"]],
            [raw_sequence_ratios],
            atol=atol,
            rtol=rtol,
        )

        raw_log_weights = []
        unclipped_weight_values = []
        expected_is_weights = []
        if is_mode == "sequence":
            for raw, sampling_row in zip(raw_sequence_ratios, sampling_mask):
                base = math.exp(max(-20.0, min(20.0, raw)))
                if any(sampling_row):
                    raw_log_weights.append(raw)
                    unclipped_weight_values.append(base)
                expected_is_weights.append([min(base, clip) * sampled for sampled in sampling_row])
        elif is_mode == "token":
            for raw_row, sampling_row in zip(raw_token_ratios, sampling_mask):
                expected_row = []
                for raw, sampled in zip(raw_row, sampling_row):
                    base = math.exp(max(-20.0, min(20.0, raw)))
                    if sampled:
                        raw_log_weights.append(raw)
                        unclipped_weight_values.append(base)
                    expected_row.append(min(base, clip) * sampled)
                expected_is_weights.append(expected_row)
        elif is_mode == "none":
            expected_is_weights = None
        else:
            errors.append(f"{prefix}: unsupported teacher IS mode {is_mode!r}")
            expected_is_weights = is_weights
        compare_matrix(errors, f"{prefix}:is_weights", is_weights, expected_is_weights, atol=atol, rtol=rtol)

        micro_metrics = record.get("micro_metrics", {})
        if expected_is_weights is not None:
            clipped_values = [min(value, clip) for value in unclipped_weight_values]
            weight_sum = sum(clipped_values)
            weight_square_sum = sum(value * value for value in clipped_values)
            ess = weight_sum * weight_sum / max(weight_square_sum, 1e-30) if clipped_values else 0.0
            expected_is_metrics = {
                "actor/is_weight_mean": (
                    sum(unclipped_weight_values) / len(unclipped_weight_values) if unclipped_weight_values else 0.0
                ),
                "actor/is_weight_max": max(clipped_values) if clipped_values else 0.0,
                "actor/is_weight_min": min(clipped_values) if clipped_values else 0.0,
                "actor/is_effective_sample_size": ess,
                "actor/is_effective_sample_size_fraction": ess / len(clipped_values) if clipped_values else 0.0,
                "actor/is_log_ratio_lower_floor_fraction": (
                    sum(raw <= -20.0 for raw in raw_log_weights) / len(raw_log_weights) if raw_log_weights else 0.0
                ),
                "actor/is_log_ratio_upper_floor_fraction": (
                    sum(raw >= 20.0 for raw in raw_log_weights) / len(raw_log_weights) if raw_log_weights else 0.0
                ),
                "actor/is_upper_clip_fraction": (
                    sum(value > clip for value in unclipped_weight_values) / len(unclipped_weight_values)
                    if unclipped_weight_values
                    else 0.0
                ),
            }
            for metric_name, expected in expected_is_metrics.items():
                require_close(
                    errors,
                    f"{prefix}:{metric_name}",
                    micro_metrics.get(metric_name),
                    expected,
                    atol=atol,
                    rtol=rtol,
                )
            health["ess_fractions"].append(expected_is_metrics["actor/is_effective_sample_size_fraction"])
            health["lower_floor_fractions"].append(expected_is_metrics["actor/is_log_ratio_lower_floor_fraction"])

        terms = record["teacher_ppo_terms"]
        expected_terms = {
            "teacher_current_old_log_ratio_clamped": [],
            "teacher_current_old_ratio": [],
            "teacher_ratio_clipped": [],
            "teacher_pg_unclipped": [],
            "teacher_pg_clipped_candidate": [],
            "teacher_pg_selected_before_is": [],
            "teacher_pg_after_is": [],
        }
        clip_low = float(record["ppo_clip_ratio_low"])
        clip_high = float(record["ppo_clip_ratio_high"])
        clip_c = float(record["ppo_clip_ratio_c"])
        pg_clip_indicators = []
        pg_lower_clip_indicators = []
        ppo_kl_values = []
        for row_index, (old_row, current_row, advantage_row, loss_mask_row) in enumerate(
            zip(teacher_old, teacher, advantages, response_mask)
        ):
            row_terms = {name: [] for name in expected_terms}
            for token_index, (old, current, advantage, loss_active) in enumerate(
                zip(old_row, current_row, advantage_row, loss_mask_row)
            ):
                log_ratio = max(-20.0, min(20.0, current - old))
                ratio = math.exp(log_ratio)
                pg1 = -advantage * ratio
                clipped_ratio = max(1.0 - clip_low, min(1.0 + clip_high, ratio))
                pg2 = -advantage * clipped_ratio
                clipped = max(pg1, pg2)
                selected = min(-advantage * clip_c, clipped) if advantage < 0 else clipped
                weight = 1.0 if is_weights is None else is_weights[row_index][token_index]
                row_terms["teacher_current_old_log_ratio_clamped"].append(log_ratio)
                row_terms["teacher_current_old_ratio"].append(ratio)
                row_terms["teacher_ratio_clipped"].append(clipped_ratio)
                row_terms["teacher_pg_unclipped"].append(pg1)
                row_terms["teacher_pg_clipped_candidate"].append(pg2)
                row_terms["teacher_pg_selected_before_is"].append(selected)
                row_terms["teacher_pg_after_is"].append(selected * weight)
                if loss_active:
                    pg_clip_indicators.append(float(pg2 > pg1))
                    pg_lower_clip_indicators.append(float(clipped > -advantage * clip_c and advantage < 0))
                    ppo_kl_values.append(-log_ratio)
            for name in expected_terms:
                expected_terms[name].append(row_terms[name])
        for term_name, expected in expected_terms.items():
            compare_matrix(errors, f"{prefix}:{term_name}", terms.get(term_name), expected, atol=atol, rtol=rtol)

        expected_weighted = expected_terms["teacher_pg_after_is"]
        expected_rlvr = aggregate(expected_weighted, response_mask, record["rlvr_global_info"], mode)
        expected_rlvr_scalar = expected_rlvr
        require_close(
            errors,
            f"{prefix}:teacher_rlvr_loss",
            record["teacher_rlvr_loss_production"],
            expected_rlvr,
            atol=atol,
            rtol=rtol,
        )
        expected_policy_metrics = {
            "actor/teacher_pg_clipfrac": sum(pg_clip_indicators) / len(pg_clip_indicators) if pg_clip_indicators else 0.0,
            "actor/teacher_ppo_kl": sum(ppo_kl_values) / len(ppo_kl_values) if ppo_kl_values else 0.0,
            "actor/teacher_pg_clipfrac_lower": (
                sum(pg_lower_clip_indicators) / len(pg_lower_clip_indicators) if pg_lower_clip_indicators else 0.0
            ),
        }
        for metric_name, expected in expected_policy_metrics.items():
            require_close(
                errors,
                f"{prefix}:{metric_name}",
                micro_metrics.get(metric_name),
                expected,
                atol=atol,
                rtol=rtol,
            )

    opsd = config.get("opsd", {})
    if record["warmup_active"]:
        expected_distill_weight = 0.0
        expected_rlvr_weight = float(opsd.get("rlvr_backward_scale", 1.0))
    else:
        expected_distill_weight = (1.0 - float(opsd.get("mix_weight", 0.5))) * float(
            opsd.get("distill_backward_scale", 1.0)
        )
        expected_rlvr_weight = float(opsd.get("mix_weight", 0.5)) * float(
            opsd.get("rlvr_backward_scale", 1.0)
        )
    require_close(
        errors,
        f"{prefix}:distill_weight",
        record["distill_weight"],
        expected_distill_weight,
        atol=atol,
        rtol=rtol,
    )
    require_close(
        errors,
        f"{prefix}:teacher_rlvr_weight",
        record["teacher_rlvr_weight"],
        expected_rlvr_weight,
        atol=atol,
        rtol=rtol,
    )
    expected_sft_weight = float(opsd.get("teacher_sft_weight", 0.0) or 0.0)
    require_close(
        errors,
        f"{prefix}:teacher_sft_weight",
        record.get("teacher_sft_weight", 0.0),
        expected_sft_weight,
        atol=atol,
        rtol=rtol,
    )
    expected_total = (
        float(record["distill_weight"]) * expected_distill_scalar
        + float(record["teacher_rlvr_weight"]) * expected_rlvr_scalar
        + expected_sft_weight * expected_sft_scalar
    )
    require_close(
        errors,
        f"{prefix}:total_weighted_loss",
        record["total_weighted_loss_production"],
        expected_total,
        atol=atol,
        rtol=rtol,
    )

    local_key = (step, int(record["rank"]))
    local_values = local_objectives.setdefault(
        local_key,
        {
            "reverse_kl_estimate": 0.0,
            "reverse_kl_surrogate": 0.0,
            "teacher_rlvr_loss": 0.0,
            "teacher_sft_loss": 0.0,
            "total": 0.0,
        },
    )
    local_values["reverse_kl_estimate"] += expected_estimate_scalar
    local_values["reverse_kl_surrogate"] += expected_distill_scalar
    local_values["teacher_rlvr_loss"] += expected_rlvr_scalar
    local_values["teacher_sft_loss"] += expected_sft_scalar
    local_values["total"] += expected_total

    routing = record.get("gradient_routing", {})
    if routing.get("distill_to_teacher", {}).get("hook_calls", 0) != 0:
        errors.append(f"{prefix}: detached distillation reached separate teacher")
    teacher_route_prefix = "teacher_joint" if teacher_sft_enabled else "teacher_rlvr"
    if routing.get(f"{teacher_route_prefix}_to_actor", {}).get("hook_calls", 0) != 0:
        errors.append(f"{prefix}: teacher RLVR/SFT objective reached separate actor")
    expected_active_routes = []
    if float(record["distill_weight"]) != 0.0 and record["distill_global_info"]["batch_num_tokens"] > 0:
        expected_active_routes.append("distill_to_actor")
    teacher_components = []
    if float(record["teacher_rlvr_weight"]) != 0.0 and record["rlvr_global_info"]["batch_num_tokens"] > 0:
        teacher_components.append("rlvr")
    if (
        teacher_sft_enabled
        and expected_sft_weight != 0.0
        and record["teacher_sft_global_info"]["batch_num_tokens"] > 0
    ):
        teacher_components.append("sft")
    if teacher_components:
        expected_active_routes.append(f"{teacher_route_prefix}_to_teacher")
    require_equal(
        errors,
        f"{prefix}:teacher_objective_components",
        routing.get("teacher_objective_components", []),
        teacher_components,
    )

    local_rlvr_nonzero_advantage_tokens = 0
    if float(record["teacher_rlvr_weight"]) != 0.0:
        local_rlvr_nonzero_advantage_tokens = sum(
            int(bool(active) and abs(float(advantage)) > 0.0)
            for advantage_row, mask_row in zip(record.get("advantages") or [], response_mask)
            for advantage, active in zip(advantage_row, mask_row)
        )
    local_sft_tokens = 0
    if teacher_sft_enabled and expected_sft_weight != 0.0:
        local_sft_tokens = sum(
            int(bool(value)) for row in (record.get("teacher_sft_mask") or []) for value in row
        )
    expected_teacher_nonzero_gradient = bool(
        local_rlvr_nonzero_advantage_tokens or local_sft_tokens
    )
    recorded_local_support = routing.get("teacher_local_objective_support")
    if recorded_local_support is not None:
        require_equal(
            errors,
            f"{prefix}:teacher_local_rlvr_nonzero_advantage_tokens",
            recorded_local_support.get("rlvr_nonzero_advantage_tokens"),
            local_rlvr_nonzero_advantage_tokens,
        )
        require_equal(
            errors,
            f"{prefix}:teacher_local_sft_tokens",
            recorded_local_support.get("sft_tokens"),
            local_sft_tokens,
        )
        require_equal(
            errors,
            f"{prefix}:teacher_local_expects_nonzero_gradient",
            recorded_local_support.get("expects_nonzero_gradient"),
            expected_teacher_nonzero_gradient,
        )

    for route_name in expected_active_routes:
        route = routing.get(route_name)
        if not route or int(route.get("hook_calls", 0)) <= 0:
            errors.append(f"{prefix}: active gradient route {route_name} has no gradient hook calls")
            continue
        if int(route.get("gradient_numel", 0)) <= 0:
            errors.append(f"{prefix}: active gradient route {route_name} has no gradient elements")
        gradient_norm = float(route.get("gradient_norm", float("nan")))
        if not math.isfinite(gradient_norm):
            errors.append(f"{prefix}: active gradient route {route_name} has non-finite norm {gradient_norm}")
        route_expects_nonzero = (
            expected_teacher_nonzero_gradient
            if route_name == f"{teacher_route_prefix}_to_teacher"
            else True
        )
        if route_expects_nonzero and gradient_norm <= 0.0:
            errors.append(f"{prefix}: supported gradient route {route_name} has invalid norm {gradient_norm}")
        if not route_expects_nonzero and gradient_norm != 0.0:
            errors.append(
                f"{prefix}: locally unsupported gradient route {route_name} "
                f"has unexpected nonzero norm {gradient_norm}"
            )

    for branch, comparison in record.get("reference_forwards", {}).items():
        if comparison.get("status") != "PASS":
            errors.append(f"{prefix}: {branch} compact/dense reference did not pass")
        production = comparison.get("production_log_probs")
        candidates = comparison.get("comparisons", {})
        require_equal(
            errors,
            f"{prefix}:{branch}:reference_candidate_names",
            sorted(candidates),
            ["compact", "dense", "extra_left_pad", "extra_right_pad"],
        )
        for candidate_name, candidate_record in candidates.items():
            candidate = candidate_record.get("candidate_log_probs")
            if production is None or candidate is None:
                errors.append(f"{prefix}:{branch}:{candidate_name}: missing forward arrays")
                continue
            differences = [
                abs(actual - expected)
                for production_row, candidate_row in zip(production, candidate)
                for actual, expected in zip(production_row, candidate_row)
            ]
            if not differences:
                errors.append(f"{prefix}:{branch}:{candidate_name}: empty forward comparison")
                continue
            max_error = max(differences)
            mean_error = sum(differences) / len(differences)
            if candidate_name == "dense":
                default_max_tolerance = float(audit_config.get("dense_forward_max_abs_error", 0.75))
                default_mean_tolerance = float(audit_config.get("dense_forward_mean_abs_error", 0.03))
            else:
                default_max_tolerance = forward_max_tolerance
                default_mean_tolerance = forward_mean_tolerance
            max_tolerance = float(candidate_record.get("max_abs_tolerance", default_max_tolerance))
            mean_tolerance = float(candidate_record.get("mean_abs_tolerance", default_mean_tolerance))
            require_close(
                errors,
                f"{prefix}:{branch}:{candidate_name}:reported_max_abs_error",
                candidate_record.get("max_abs_error"),
                max_error,
                atol=atol,
                rtol=rtol,
            )
            require_close(
                errors,
                f"{prefix}:{branch}:{candidate_name}:reported_mean_abs_error",
                candidate_record.get("mean_abs_error"),
                mean_error,
                atol=atol,
                rtol=rtol,
            )
            if candidate_name == "dense":
                step_observation["dense_cross_kernel_max_abs_errors"].append(max_error)
            else:
                step_observation["strict_padding_reference_max_abs_errors"].append(max_error)
            if max_error > max_tolerance or mean_error > mean_tolerance:
                message = (
                    f"{prefix}:{branch}:{candidate_name}: forward mismatch "
                    f"max_abs={max_error} (limit {max_tolerance}), "
                    f"mean_abs={mean_error} (limit {mean_tolerance})"
                )
                dense_required = bool(audit_config.get("dense_forward_fail_fast", False))
                if candidate_name == "dense" and not dense_required:
                    step_observation["dense_cross_kernel_diagnostic_mismatches"] = (
                        step_observation.get("dense_cross_kernel_diagnostic_mismatches", 0) + 1
                    )
                else:
                    errors.append(message)
        dense_candidate = candidates.get("dense", {}).get("candidate_log_probs")
        compact_candidate = candidates.get("compact", {}).get("candidate_log_probs")
        if dense_candidate is not None and compact_candidate is not None:
            oracle_differences = [
                abs(dense_value - compact_value)
                for dense_row, compact_row in zip(dense_candidate, compact_candidate)
                for dense_value, compact_value in zip(dense_row, compact_row)
            ]
            oracle_max = max(oracle_differences)
            oracle_mean = sum(oracle_differences) / len(oracle_differences)
            oracle_record = comparison.get("oracle_comparisons", {}).get("dense_vs_compact", {})
            require_close(
                errors,
                f"{prefix}:{branch}:dense_vs_compact:reported_max_abs_error",
                oracle_record.get("max_abs_error"),
                oracle_max,
                atol=atol,
                rtol=rtol,
            )
            oracle_max_tolerance = float(
                oracle_record.get("max_abs_tolerance", audit_config.get("dense_forward_max_abs_error", 0.75))
            )
            oracle_mean_tolerance = float(
                oracle_record.get("mean_abs_tolerance", audit_config.get("dense_forward_mean_abs_error", 0.03))
            )
            if oracle_max > oracle_max_tolerance or oracle_mean > oracle_mean_tolerance:
                if bool(audit_config.get("dense_forward_fail_fast", False)):
                    errors.append(
                        f"{prefix}:{branch}:dense_vs_compact cross-kernel mismatch "
                        f"max_abs={oracle_max} (limit {oracle_max_tolerance}), "
                        f"mean_abs={oracle_mean} (limit {oracle_mean_tolerance})"
                    )
                else:
                    step_observation["dense_cross_kernel_oracle_diagnostic_mismatches"] = (
                        step_observation.get("dense_cross_kernel_oracle_diagnostic_mismatches", 0) + 1
                    )
            require_close(
                errors,
                f"{prefix}:{branch}:dense_vs_compact:reported_mean_abs_error",
                oracle_record.get("mean_abs_error"),
                oracle_mean,
                atol=atol,
                rtol=rtol,
            )
        counters["reference_forwards"] += 1
    counters["microbatches"] += 1


def verify_optimizer_record(record, path, line_number, errors, counters, step_configs, observations, optimizer_records):
    step = int(record["global_step"])
    rank = int(record["rank"])
    prefix = f"{path}:{line_number}:step={step}:rank={rank}:optimizer"
    require_fields(
        errors,
        prefix,
        record,
        (
            "optimizer_updates_completed_before_step",
            "trainer_updates_completed_before_step",
            "actor_attempted",
            "actor_did_step",
            "actor_grad_norm",
            "actor_probe_before",
            "actor_probe_after",
            "teacher_attempted",
            "teacher_did_step",
            "teacher_grad_norm",
            "teacher_probe_before",
            "teacher_probe_after",
        ),
    )
    require_equal(
        errors,
        f"{prefix}:optimizer_updates_completed_before_step",
        record.get("optimizer_updates_completed_before_step"),
        step - 1,
    )
    require_equal(
        errors,
        f"{prefix}:trainer_updates_completed_before_step",
        record.get("trainer_updates_completed_before_step"),
        step - 1,
    )
    opsd = step_configs.get(step, {}).get("opsd", {})
    warmup = 1 <= step <= int(opsd.get("rlvr_warmup_steps", 0) or 0)
    require_equal(errors, f"{prefix}:actor_attempted", bool(record.get("actor_attempted")), not warmup)
    require_equal(errors, f"{prefix}:teacher_attempted", bool(record.get("teacher_attempted")), True)
    for branch in ("actor", "teacher"):
        attempted = bool(record.get(f"{branch}_attempted"))
        did_step = bool(record.get(f"{branch}_did_step"))
        if attempted and not did_step:
            errors.append(f"{prefix}: {branch} optimizer attempted but did not step")
        if not attempted and did_step:
            errors.append(f"{prefix}: {branch} optimizer stepped without a backward pass")
        grad_norm = record.get(f"{branch}_grad_norm")
        before = record.get(f"{branch}_probe_before")
        after = record.get(f"{branch}_probe_after")
        if not attempted:
            if grad_norm is not None or before is not None or after is not None:
                errors.append(f"{prefix}: inactive {branch} optimizer unexpectedly emitted gradient/probe values")
            continue
        grad_norm = float(grad_norm)
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            errors.append(f"{prefix}: {branch} grad norm must be finite and positive, got {grad_norm}")
        if not before or not after:
            errors.append(f"{prefix}: {branch} optimizer is missing parameter probes")
            continue
        require_equal(errors, f"{prefix}:{branch}:probe_name", after.get("parameter_name"), before.get("parameter_name"))
        require_equal(errors, f"{prefix}:{branch}:probe_index", after.get("local_index"), before.get("local_index"))
        gradient_value = before.get("gradient_value")
        if gradient_value is None or not math.isfinite(float(gradient_value)) or float(gradient_value) == 0.0:
            errors.append(f"{prefix}: {branch} probe gradient is invalid: {gradient_value!r}")
        parameter_before = float(before.get("parameter_value"))
        parameter_after = float(after.get("parameter_value"))
        parameter_delta = parameter_after - parameter_before
        observations.setdefault("steps", {}).setdefault(str(step), {}).setdefault(
            f"{branch}_parameter_deltas", []
        ).append(parameter_delta)
    optimizer_records[(step, rank)] = record
    counters["optimizer_records"] += 1


def verify_update_summaries(
    errors,
    summaries,
    optimizer_records,
    local_objectives,
    observations,
    step_configs,
    aggregation,
):
    ranks_by_step = {}
    for step, rank in local_objectives:
        ranks_by_step.setdefault(step, set()).add(rank)
    for (step, rank), summary in sorted(summaries.items()):
        prefix = f"step={step}:rank={rank}:summary"
        require_equal(errors, f"{prefix}:audit_status", summary.get("audit_status"), "PASS")
        require_equal(errors, f"{prefix}:local_reference_failures", summary.get("local_reference_failures"), [])
        expected_sft_enabled = bool(
            step_configs.get(step, {}).get("teacher_sft", {}).get("enabled", False)
        )
        require_equal(
            errors,
            f"{prefix}:teacher_sft_enabled",
            bool(summary.get("teacher_sft_enabled", False)),
            expected_sft_enabled,
        )
        if expected_sft_enabled:
            require_equal(
                errors,
                f"{prefix}:teacher_sft_target_scope",
                summary.get("teacher_sft_target_scope"),
                step_configs[step]["teacher_sft"].get("target_scope"),
            )
        require_equal(errors, f"{prefix}:updates_before", summary.get("optimizer_updates_completed_before_step"), step - 1)
        require_equal(errors, f"{prefix}:updates_after", summary.get("optimizer_updates_completed_after_step"), step)
        require_equal(
            errors,
            f"{prefix}:trainer_updates_before",
            summary.get("trainer_updates_completed_before_step"),
            step - 1,
        )
        require_equal(
            errors,
            f"{prefix}:trainer_updates_after",
            summary.get("trainer_updates_completed_after_step"),
            step,
        )
        optimizer = optimizer_records.get((step, rank))
        if optimizer is None:
            errors.append(f"{prefix}: missing optimizer record")
            continue
        for summary_name, optimizer_name in (
            ("actor_step_attempted", "actor_attempted"),
            ("actor_did_step", "actor_did_step"),
            ("teacher_step_attempted", "teacher_attempted"),
            ("teacher_did_step", "teacher_did_step"),
        ):
            require_equal(errors, f"{prefix}:{summary_name}", summary.get(summary_name), optimizer.get(optimizer_name))

        local = local_objectives.get((step, rank))
        if local is None:
            errors.append(f"{prefix}: missing recomputed local objectives")
            continue
        metrics = summary.get("metrics", {})
        audit = step_configs.get(step, {}).get("audit", {})
        atol = float(audit.get("scalar_atol", 1e-5))
        rtol = float(audit.get("scalar_rtol", 1e-4))
        for metric_name, local_name in (
            ("actor/opsd_reverse_kl_estimate", "reverse_kl_estimate"),
            ("actor/opsd_reverse_kl_surrogate_loss", "reverse_kl_surrogate"),
            ("actor/opsd_reverse_kl_loss", "reverse_kl_surrogate"),
            ("actor/teacher_rlvr_loss", "teacher_rlvr_loss"),
            ("actor/teacher_sft_loss", "teacher_sft_loss"),
            ("actor/opsd_loss", "total"),
        ):
            require_close(
                errors,
                f"{prefix}:{metric_name}",
                metrics.get(metric_name),
                local[local_name],
                atol=atol,
                rtol=rtol,
            )

        ranks = ranks_by_step.get(step, set())
        expected_global = {
            name: sum(local_objectives[(step, item_rank)][name] for item_rank in ranks) / len(ranks)
            for name in (
                "reverse_kl_estimate",
                "reverse_kl_surrogate",
                "teacher_rlvr_loss",
                "teacher_sft_loss",
            )
        }
        for name, expected in expected_global.items():
            require_close(
                errors,
                f"{prefix}:global_{name}",
                summary.get("global_audit_metrics", {}).get(name),
                expected,
                atol=atol,
                rtol=rtol,
            )
        if bool(step_configs.get(step, {}).get("teacher_sft", {}).get("enabled", False)):
            expected_sft_tokens = aggregation.get((step, "teacher_sft"), {}).get("tokens", 0)
            require_equal(
                errors,
                f"{prefix}:global_teacher_sft_tokens",
                summary.get("global_audit_metrics", {}).get("teacher_sft_tokens"),
                expected_sft_tokens,
            )
        step_observation = observations.setdefault("steps", {}).setdefault(str(step), {})
        step_observation.update(
            {
                "reverse_kl_estimate": expected_global["reverse_kl_estimate"],
                "reverse_kl_surrogate": expected_global["reverse_kl_surrogate"],
                "teacher_rlvr_loss": expected_global["teacher_rlvr_loss"],
                "teacher_sft_loss": expected_global["teacher_sft_loss"],
                "actor_did_step": all(
                    bool(summaries.get((step, item_rank), {}).get("actor_did_step")) for item_rank in ranks
                ),
                "teacher_did_step": all(
                    bool(summaries.get((step, item_rank), {}).get("teacher_did_step")) for item_rank in ranks
                ),
                "rank_count": len(ranks),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    report_json = args.report_json or audit_dir / "verification_report.json"
    report_md = args.report_md or audit_dir / "verification_report.md"
    errors: list[str] = []
    counters = {
        "trainer_samples": 0,
        "microbatches": 0,
        "optimizer_records": 0,
        "summaries": 0,
        "reference_forwards": 0,
        "attended_pad_id_tokens": 0,
        "teacher_sft_target_tokens": 0,
        "teacher_sft_target_sequences": 0,
        "teacher_sft_successful_rollouts": 0,
        "teacher_sft_rollouts_with_think_end": 0,
        "openthoughts_data_contract_samples": 0,
    }
    health = {"ess_fractions": [], "lower_floor_fractions": []}
    aggregation = {}
    trainer_records = {}
    step_configs = {}
    observations = {"steps": {}}
    local_objectives = {}
    optimizer_records = {}
    summaries = {}
    dispatch_counts = {}

    layout_files = sorted(audit_dir.glob("step_*/trainer_batch.json"))
    ledger_files = sorted(audit_dir.glob("step_*/rank_*.jsonl"))
    if not layout_files:
        errors.append(f"no trainer_batch.json files under {audit_dir}")
    if not ledger_files:
        errors.append(f"no rank JSONL ledgers under {audit_dir}")

    for index, path in enumerate(layout_files, start=1):
        print(f"[verify] layout {index}/{len(layout_files)} {path}", flush=True)
        verify_layout_file(path, errors, counters, trainer_records, step_configs, observations)
    for file_index, path in enumerate(ledger_files, start=1):
        print(f"[verify] ledger {file_index}/{len(ledger_files)} {path}", flush=True)
        with path.open() as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_type = record.get("record_type")
                if record_type == "microbatch":
                    verify_microbatch(
                        record,
                        path,
                        line_number,
                        errors,
                        counters,
                        health,
                        aggregation,
                        trainer_records,
                        step_configs,
                        observations,
                        local_objectives,
                        dispatch_counts,
                    )
                elif record_type == "optimizer_step":
                    verify_optimizer_record(
                        record,
                        path,
                        line_number,
                        errors,
                        counters,
                        step_configs,
                        observations,
                        optimizer_records,
                    )
                elif record_type == "update_summary":
                    counters["summaries"] += 1
                    summaries[(int(record["global_step"]), int(record["rank"]))] = record
                else:
                    errors.append(f"{path}:{line_number}: unknown record_type={record_type!r}")

    verify_update_summaries(
        errors,
        summaries,
        optimizer_records,
        local_objectives,
        observations,
        step_configs,
        aggregation,
    )

    for (step, branch), item in sorted(aggregation.items()):
        if item["declared_tokens"] != {item["tokens"]}:
            errors.append(
                f"step={step}:{branch}: global token denominator {sorted(item['declared_tokens'])} "
                f"does not match merged rank ledger count {item['tokens']}"
            )
        if item["declared_sequences"] != {item["sequences"]}:
            errors.append(
                f"step={step}:{branch}: global sequence denominator {sorted(item['declared_sequences'])} "
                f"does not match merged rank ledger count {item['sequences']}"
            )
        if len(item["dp_sizes"]) != 1:
            errors.append(f"step={step}:{branch}: inconsistent DP sizes {sorted(item['dp_sizes'])}")

    configured_step_sets = {
        tuple(sorted(int(value) for value in config.get("audit", {}).get("global_steps", [])))
        for config in step_configs.values()
    }
    if len(configured_step_sets) != 1:
        errors.append(f"inconsistent configured audit step sets: {sorted(configured_step_sets)}")
    expected_steps = list(next(iter(configured_step_sets))) if len(configured_step_sets) == 1 else sorted(step_configs)
    actual_layout_steps = sorted({int(path.parent.name.removeprefix("step_")) for path in layout_files})
    require_equal(errors, "audit step directories", actual_layout_steps, expected_steps)
    for step in expected_steps:
        expected_samples = {sample for record_step, sample in trainer_records if record_step == step}
        seen_samples = {sample for record_step, sample in dispatch_counts if record_step == step}
        require_equal(errors, f"step={step}:dispatched_sample_ids", seen_samples, expected_samples)
        for sample in expected_samples:
            require_equal(errors, f"step={step}:sample={sample}:dispatch_count", dispatch_counts.get((step, sample)), 1)
        dp_sizes = set()
        branches = ["distill", "rlvr"]
        if bool(step_configs.get(step, {}).get("teacher_sft", {}).get("enabled", False)):
            branches.append("teacher_sft")
        for branch in branches:
            dp_sizes.update(aggregation.get((step, branch), {}).get("dp_sizes", set()))
        if len(dp_sizes) != 1:
            errors.append(f"step={step}: cannot resolve one DP size from {sorted(dp_sizes)}")
            continue
        dp_size = next(iter(dp_sizes))
        observed_ranks = {rank for record_step, rank in local_objectives if record_step == step}
        require_equal(errors, f"step={step}:ledger_rank_count", len(observed_ranks), dp_size)
        require_equal(
            errors,
            f"step={step}:optimizer_record_count",
            sum(record_step == step for record_step, _ in optimizer_records),
            dp_size,
        )
        require_equal(
            errors,
            f"step={step}:summary_record_count",
            sum(record_step == step for record_step, _ in summaries),
            dp_size,
        )
        if bool(step_configs.get(step, {}).get("teacher_sft", {}).get("enabled", False)):
            sft_tokens = aggregation.get((step, "teacher_sft"), {}).get("tokens", 0)
            successful_rollouts = sum(
                int(bool(record.get("teacher_sft_success")))
                for (record_step, _), record in trainer_records.items()
                if record_step == step
            )
            if successful_rollouts > 0 and sft_tokens <= 0:
                errors.append(
                    f"step={step}: {successful_rollouts} successful rollouts emitted no teacher SFT tokens"
                )
            if successful_rollouts == 0:
                observations.setdefault("steps", {}).setdefault(str(step), {})[
                    "teacher_sft_zero_success_rollouts"
                ] = True

    health_warnings = []
    audit_warnings = []
    if health["ess_fractions"] and min(health["ess_fractions"]) < 0.20:
        health_warnings.append(f"minimum teacher IS ESS fraction is {min(health['ess_fractions']):.6g} < 0.20")
    if health["lower_floor_fractions"] and max(health["lower_floor_fractions"]) > 0.05:
        health_warnings.append(
            f"maximum sequence-IS lower-floor fraction is {max(health['lower_floor_fractions']):.6g} > 0.05"
        )

    for step_text, values in observations["steps"].items():
        step = int(step_text)
        dense_diagnostics = values.get("dense_cross_kernel_diagnostic_mismatches", 0)
        dense_oracle_diagnostics = values.get("dense_cross_kernel_oracle_diagnostic_mismatches", 0)
        if dense_diagnostics or dense_oracle_diagnostics:
            audit_warnings.append(
                f"step {step}: unused dense padded cross-kernel diagnostic exceeded tolerance "
                f"({dense_diagnostics} production comparisons, {dense_oracle_diagnostics} oracle comparisons); "
                "production compact/extra-PAD invariants remained fail-fast"
            )
        if values.get("teacher_sft_zero_success_rollouts"):
            audit_warnings.append(
                f"step {step}: no rollout passed the reward verifier, so teacher SFT correctly had no target tokens"
            )
        for branch in ("actor", "teacher"):
            deltas = values.get(f"{branch}_parameter_deltas", [])
            if deltas and not any(delta != 0.0 for delta in deltas):
                errors.append(f"step={step}: {branch} optimizer stepped but no distributed parameter probe changed")
            values[f"{branch}_unchanged_parameter_probe_count"] = sum(delta == 0.0 for delta in deltas)
        for list_name, summary_name in (
            ("actor_behavior_logprob_max_abs_errors", "actor_behavior_logprob_max_abs_error"),
            ("teacher_old_current_logprob_max_abs_errors", "teacher_old_current_logprob_max_abs_error"),
            ("strict_padding_reference_max_abs_errors", "padding_invariance_max_abs_error"),
            ("dense_cross_kernel_max_abs_errors", "dense_cross_kernel_max_abs_error"),
            ("actor_parameter_deltas", "actor_parameter_delta_max_abs"),
            ("teacher_parameter_deltas", "teacher_parameter_delta_max_abs"),
        ):
            observed = values.pop(list_name, [])
            values[summary_name] = max((abs(value) for value in observed), default=None)
        behavior_differences = values.pop("_actor_behavior_signed_differences", [])
        if behavior_differences:
            sorted_abs = sorted(abs(value) for value in behavior_differences)
            count = len(sorted_abs)
            values["actor_behavior_logprob_signed_mean_error"] = sum(behavior_differences) / count
            values["actor_behavior_logprob_mean_abs_error"] = sum(sorted_abs) / count
            values["actor_behavior_logprob_rms_error"] = math.sqrt(
                sum(value * value for value in behavior_differences) / count
            )
            values["actor_behavior_logprob_p95_abs_error"] = sorted_abs[int(0.95 * (count - 1))]
            values["actor_behavior_logprob_p99_abs_error"] = sorted_abs[int(0.99 * (count - 1))]
        values["distill_global_tokens"] = next(
            iter(aggregation.get((step, "distill"), {}).get("declared_tokens", [])), None
        )
        values["rlvr_global_tokens"] = next(
            iter(aggregation.get((step, "rlvr"), {}).get("declared_tokens", [])), None
        )
        values["teacher_sft_global_tokens"] = next(
            iter(aggregation.get((step, "teacher_sft"), {}).get("declared_tokens", [])), None
        )
        values["teacher_sft_target_scope"] = step_configs.get(step, {}).get("teacher_sft", {}).get(
            "target_scope"
        )
        values["teacher_sft_success_field"] = step_configs.get(step, {}).get("teacher_sft", {}).get(
            "success_field"
        )
        values["warmup_active"] = 1 <= step <= int(step_configs.get(step, {}).get("opsd", {}).get("rlvr_warmup_steps", 0) or 0)

    status = "PASS" if not errors else "FAIL"
    verified_checks = [
        "generation provenance and unconditioned actor vs ground-conditioned teacher setup",
        "full token/mask shapes, left-padded prompts, right-padded responses, no interior PAD island",
        "masked token IDs, response suffix identity, and attention-derived position IDs",
        "reward identity and independently recomputed grouped GRPO advantages",
        "trainer-to-worker dispatch identity for masks, log-probs, and advantages",
        "actor rollout-vs-learner and teacher old-vs-current pre-update log-prob parity",
        "sequence/token importance ratios, numerical floors, clipping, ESS, and all IS metrics",
        "every dual-clip PPO intermediate and policy diagnostic",
        "sampled reverse-KL estimate, score-function surrogate, global denominators, and weighted objective",
        "successful-rollout teacher SFT selection, closing-think boundary, target mask, token NLL, and global loss",
        "production remove-padding equivalence to compact input plus extra-left/right-PAD invariance; dense padded cross-kernel drift recorded diagnostically",
        "branch gradient isolation, finite active gradients, optimizer steps, parameter probes, and scheduler intent",
        "per-rank summaries and independently merged global values",
    ]
    report = {
        "status": status,
        "audit_dir": str(audit_dir),
        "counters": counters,
        "errors": errors,
        "warnings": audit_warnings,
        "verified_checks": verified_checks,
        "observations": observations,
        "training_health": {
            "status": "NOT_PRODUCTION_READY" if health_warnings else "PASS",
            "warnings": health_warnings,
            "ess_fractions": health["ess_fractions"],
            "lower_floor_fractions": health["lower_floor_fractions"],
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    step_rows = []
    for step_text, values in sorted(observations["steps"].items(), key=lambda item: int(item[0])):
        def display(name):
            value = values.get(name)
            if isinstance(value, float):
                return f"{value:.9g}"
            return str(value)

        step_rows.append(
            "| "
            + " | ".join(
                [
                    step_text,
                    display("trainer_updates_completed_at_value_capture"),
                    display("warmup_active"),
                    display("reverse_kl_estimate"),
                    display("reverse_kl_surrogate"),
                    display("teacher_rlvr_loss"),
                    display("teacher_sft_loss"),
                    display("teacher_sft_global_tokens"),
                    display("teacher_sft_target_scope"),
                    display("actor_did_step"),
                    display("teacher_did_step"),
                    display("actor_behavior_logprob_max_abs_error"),
                    display("actor_behavior_logprob_mean_abs_error"),
                    display("teacher_old_current_logprob_max_abs_error"),
                    display("padding_invariance_max_abs_error"),
                    display("dense_cross_kernel_max_abs_error"),
                ]
            )
            + " |"
        )
    step_table = (
        "| Global step | Pre-update optimizer count | Warmup | Reverse-KL estimate | Reverse-KL surrogate | "
        "Teacher RLVR loss | Teacher SFT loss | Teacher SFT tokens | Teacher SFT scope | "
        "Actor stepped | Teacher stepped | Actor rollout/learner max error | "
        "Actor rollout/learner MAE | Teacher old/current max error | PAD-invariance max error | "
        "Dense cross-kernel max error |\n"
        "|---:|---:|:---:|---:|---:|---:|---:|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(step_rows)
    )
    report_md.write_text(
        WIDE_CSS
        + f"\n# OPSD audit verification\n\n"
        + f"- Numerical status: **{status}**\n"
        + f"- Training-health status: **{report['training_health']['status']}**\n"
        + f"- Trainer samples: {counters['trainer_samples']}\n"
        + f"- Microbatches: {counters['microbatches']}\n"
        + f"- Reference forwards: {counters['reference_forwards']}\n\n"
        + f"- Teacher SFT target tokens: {counters['teacher_sft_target_tokens']}\n"
        + f"- Teacher SFT target sequences: {counters['teacher_sft_target_sequences']}\n\n"
        + f"- Teacher SFT successful rollouts: {counters['teacher_sft_successful_rollouts']}\n"
        + f"- Rollouts containing the closing-think delimiter: "
        + f"{counters['teacher_sft_rollouts_with_think_end']}\n\n"
        + "## Recomputed values\n\n"
        + step_table
        + "\n\n## Verified checks\n\n"
        + "\n".join(f"- {check}" for check in verified_checks)
        + "\n\n"
        + "## Health warnings\n\n"
        + ("\n".join(f"- {warning}" for warning in health_warnings) or "None.")
        + "\n\n## Audit warnings\n\n"
        + ("\n".join(f"- {warning}" for warning in audit_warnings) or "None.")
        + "\n\n## Numerical errors\n\n"
        + ("\n".join(f"- {error}" for error in errors) or "None.")
        + "\n"
    )
    print(
        f"[OPSD_AUDIT_VERIFY] status={status} health={report['training_health']['status']} "
        f"microbatches={counters['microbatches']} report={report_json}",
        flush=True,
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
