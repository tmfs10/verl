#!/usr/bin/env python3
"""Independently verify a live steering-vector OPSD audit ledger.

This verifier deliberately imports neither VERL nor ``recipe.opsd``. It
reconstructs CAA or policy-gradient vectors, masks, sampled reverse KL, the selected actor
objective, PPO clipping, GRPO advantage redistribution, loss aggregation,
gradient routes, and optimizer execution from JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


OBJECTIVES = {
    "direct_reverse_kl",
    "negative_kl_advantage",
    "grpo_advantage_reweighting",
}

WIDE_CSS = """<style>
body, main, article, .markdown-body, .rendered_html,
.jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: min(98vw, 1800px) !important;
}
table { width: 100% !important; }
</style>
"""


def as_bool(row):
    return [bool(value) for value in row]


def close(actual, expected, *, atol=3e-5, rtol=3e-4):
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected) <= atol + rtol * abs(expected)
    )


class Audit:
    def __init__(
        self,
        actor_objective: str,
        *,
        steering_scale: float,
        max_delta_fraction: float | None,
        caa_scope: str,
        steering_source_mode: str,
    ):
        self.actor_objective = actor_objective
        self.steering_scale = steering_scale
        self.max_delta_fraction = max_delta_fraction
        self.caa_scope = caa_scope
        self.steering_source_mode = steering_source_mode
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.counts = defaultdict(int)
        self.observations: dict[str, object] = {
            "actor_objective": actor_objective,
            "steering_source_mode": steering_source_mode,
        }
        self.global_caa_records = defaultdict(dict)
        self.global_caa_sources = defaultdict(lambda: defaultdict(list))
        self.global_policy_gradient_terms = defaultdict(
            lambda: defaultdict(list)
        )

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)

    def equal(self, actual, expected, message):
        self.require(
            actual == expected,
            f"{message}: actual={actual!r}, expected={expected!r}",
        )

    def near(self, actual, expected, message, *, atol=3e-5, rtol=3e-4):
        self.require(
            close(actual, expected, atol=atol, rtol=rtol),
            f"{message}: actual={actual!r}, expected={expected!r}, "
            f"atol={atol}, rtol={rtol}",
        )


def load_jsonl(path: Path):
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if line.strip():
            yield line_number, json.loads(line)


def compare_matrix(audit, actual, expected, message, *, mask=None, atol=3e-5, rtol=3e-4):
    if actual is None:
        audit.errors.append(f"{message}: missing matrix")
        return
    if len(actual) != len(expected):
        audit.errors.append(f"{message}: rows={len(actual)} expected={len(expected)}")
        return
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if len(actual_row) != len(expected_row):
            audit.errors.append(
                f"{message}[{row_index}]: cols={len(actual_row)} expected={len(expected_row)}"
            )
            continue
        for token_index, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row)
        ):
            if mask is not None and not bool(mask[row_index][token_index]):
                continue
            audit.near(
                actual_value,
                expected_value,
                f"{message}[{row_index},{token_index}]",
                atol=atol,
                rtol=rtol,
            )


def require_finite_matrix(audit, matrix, message):
    if matrix is None:
        return
    for row_index, row in enumerate(matrix):
        for token_index, value in enumerate(row):
            audit.require(
                math.isfinite(float(value)),
                f"{message}[{row_index},{token_index}] is non-finite: {value!r}",
            )


def aggregate(values, masks, info, mode):
    dp_size = int(info.get("dp_size", 0))
    if dp_size <= 0:
        return None
    if mode == "token-mean":
        denominator = int(info.get("batch_num_tokens", 0))
        if denominator <= 0:
            return 0.0
        numerator = sum(
            float(value) * float(active)
            for row, mask_row in zip(values, masks)
            for value, active in zip(row, mask_row)
        )
        return numerator / denominator * dp_size
    raise ValueError(f"The audited smoke expects token-mean, got {mode!r}")


def ppo_terms(old, current, advantages, clip_low, clip_high, clip_c):
    names = (
        "teacher_current_old_log_ratio_clamped",
        "teacher_current_old_ratio",
        "teacher_ratio_clipped",
        "teacher_pg_unclipped",
        "teacher_pg_clipped_candidate",
        "teacher_pg_selected_before_is",
        "teacher_pg_after_is",
    )
    result = {name: [] for name in names}
    for old_row, current_row, advantage_row in zip(old, current, advantages):
        rows = {name: [] for name in names}
        for old_value, current_value, advantage in zip(
            old_row, current_row, advantage_row
        ):
            log_ratio = max(-20.0, min(20.0, float(current_value) - float(old_value)))
            ratio = math.exp(log_ratio)
            unclipped = -float(advantage) * ratio
            clipped_ratio = max(1.0 - clip_low, min(1.0 + clip_high, ratio))
            clipped_candidate = -float(advantage) * clipped_ratio
            clipped = max(unclipped, clipped_candidate)
            selected = min(-float(advantage) * clip_c, clipped) if advantage < 0 else clipped
            rows["teacher_current_old_log_ratio_clamped"].append(log_ratio)
            rows["teacher_current_old_ratio"].append(ratio)
            rows["teacher_ratio_clipped"].append(clipped_ratio)
            rows["teacher_pg_unclipped"].append(unclipped)
            rows["teacher_pg_clipped_candidate"].append(clipped_candidate)
            rows["teacher_pg_selected_before_is"].append(selected)
            rows["teacher_pg_after_is"].append(selected)
        for name in names:
            result[name].append(rows[name])
    return result


def reshape_advantage_row(
    advantages,
    evidence,
    response_mask,
    shaping_mask,
    *,
    max_delta_fraction,
):
    """Independent reconstruction of the fixed smoke shaping configuration."""

    selected = [index for index, active in enumerate(shaping_mask) if bool(active)]
    if not selected:
        return list(advantages), None
    response_indices = [index for index, active in enumerate(response_mask) if bool(active)]
    score_mean = sum(float(evidence[index]) for index in selected) / len(selected)
    centered = [0.0] * len(evidence)
    for index in selected:
        centered[index] = float(evidence[index]) - score_mean
    z = centered

    response_advantage = (
        sum(float(advantages[index]) for index in response_indices) / len(response_indices)
        if response_indices
        else 0.0
    )
    scale = 1.0
    if max_delta_fraction is not None:
        z_abs_max = max(abs(z[index]) for index in selected)
        if z_abs_max > 0.0:
            scale = min(scale, float(max_delta_fraction) / max(z_abs_max, 1e-6))
    z_min = min(z[index] for index in selected)
    z_max = max(z[index] for index in selected)
    if response_advantage > 0.0 and z_min < 0.0:
        scale = min(scale, 1.0 / max(-z_min, 1e-6))
    elif response_advantage < 0.0 and z_max > 0.0:
        scale = min(scale, 1.0 / max(z_max, 1e-6))
    elif response_advantage == 0.0:
        scale = 0.0

    shaped = list(float(value) for value in advantages)
    for index in selected:
        shaped[index] += scale * abs(response_advantage) * z[index]
        if response_advantage > 0.0:
            shaped[index] = max(0.0, shaped[index])
        elif response_advantage < 0.0:
            shaped[index] = min(0.0, shaped[index])

    residual = sum(
        shaped[index] - float(advantages[index]) for index in response_indices
    )
    if max_delta_fraction is None:
        correction_index = max(selected, key=lambda index: abs(shaped[index]))
    else:
        cap_abs = abs(response_advantage) * float(max_delta_fraction)

        def correction_slack(index):
            delta = shaped[index] - float(advantages[index])
            lower = -cap_abs
            upper = cap_abs
            if response_advantage > 0.0:
                lower = max(lower, -response_advantage)
            elif response_advantage < 0.0:
                upper = min(upper, -response_advantage)
            else:
                lower = upper = 0.0
            return delta - lower if residual >= 0.0 else upper - delta

        correction_index = max(selected, key=correction_slack)
        if correction_slack(correction_index) + 1e-7 < abs(residual):
            audit_error = (
                "independent reconstruction found insufficient cap/sign slack "
                "for conservation correction"
            )
            raise AssertionError(audit_error)
    shaped[correction_index] -= residual
    return shaped, correction_index


def require_strict_config(audit: Audit, config: dict, prefix: str):
    objective = audit.actor_objective
    shaping_expected = objective == "grpo_advantage_reweighting"
    expected = {
        "enable": True,
        "actor_objective": objective,
        "mode": "opsd_rlvr" if shaping_expected else "opsd",
        "teacher_model": "actor",
        "teacher_source": "sdpo_success_rollout",
        "sdpo_conditioning_mode": "steering",
        "sdpo_distill_only_failed": False,
        "distill_max_response_tokens": None,
        "distill_loss": "sampled_reverse_kl",
        "topk": None,
        "distill_beta": None,
        "distill_token_clip": None,
        "distill_token_clip_tail": None,
        "balance_mode": "none",
        "mix_weight": 1.0,
        "distill_backward_scale": 0.0 if shaping_expected else 1.0,
        "rlvr_backward_scale": 0.0,
        "rlvr_warmup_steps": 0,
        "teacher_sft_weight": 0.0,
    }
    for name, wanted in expected.items():
        audit.equal(config.get(name), wanted, f"{prefix}:opsd.{name}")
    shaping = config.get("advantage_shaping") or {}
    audit.equal(shaping.get("enable"), shaping_expected, f"{prefix}:advantage_shaping.enable")
    if shaping_expected:
        for name, wanted in {
            "score_source": "teacher_minus_student_logprob",
            "scale": 1.0,
            "normalize": None,
            "clip_z": None,
            "use_distill_mask": True,
            "allow_token_sign_flip": False,
            "max_delta_fraction": audit.max_delta_fraction,
            "max_response_tokens": None,
            "student_rlvr_backward_scale": 1.0,
        }.items():
            audit.equal(shaping.get(name), wanted, f"{prefix}:advantage_shaping.{name}")

    steering = config.get("steering") or {}
    for name, wanted in {
        "strict_contract": True,
        "source_mode": audit.steering_source_mode,
        "correct_rollout_aggregation": "all",
        "activation_aggregation": "per_rollout",
        "caa_scope": audit.caa_scope,
        "scale": audit.steering_scale,
        "normalize": "unit_norm",
        "apply_positions": "response_only",
        "detach_vectors": True,
        "expected_model_path": "/hf_models/Qwen3-1.7B",
        "actor_model_path": "/hf_models/Qwen3-1.7B",
        "expected_total_layers": 28,
        "expected_layer_indices": [9, 10],
    }.items():
        audit.equal(steering.get(name), wanted, f"{prefix}:steering.{name}")
    if audit.steering_source_mode == "policy_gradient":
        audit.equal(
            steering.get("gradient_objective"),
            "grpo_advantage",
            f"{prefix}:steering.gradient_objective",
        )
        audit.equal(
            steering.get("gradient_aggregation"),
            "per_rollout",
            f"{prefix}:steering.gradient_aggregation",
        )
    audit.require(
        str(steering.get("layer_fractions", "")) in {"0.31-0.37", "0.31,0.37"},
        f"{prefix}:unexpected layer fractions {steering.get('layer_fractions')!r}",
    )


def verify_grpo_trainer_advantages(audit, payload, groups, indexed, prefix):
    for group, outcomes in groups.items():
        records = [indexed[index] for index, _ in outcomes]
        if any(record.get("token_level_rewards") is None for record in records):
            audit.errors.append(f"{prefix}:group {group} missing GRPO token rewards")
            continue
        rewards = [sum(float(value) for value in record["token_level_rewards"]) for record in records]
        mean = sum(rewards) / len(rewards)
        variance = (
            sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
            if len(rewards) > 1
            else 0.0
        )
        std = math.sqrt(variance)
        for record, reward in zip(records, rewards):
            expected = reward - mean
            if payload.get("norm_adv_by_std_in_grpo", True):
                expected /= std + 1e-6
            advantages = record.get("advantages")
            if advantages is None:
                audit.errors.append(
                    f"{prefix}:sample {record['audit_sample_index']} missing GRPO advantages"
                )
                continue
            for token_index, (actual, active) in enumerate(
                zip(advantages, record["response_mask"])
            ):
                audit.near(
                    actual,
                    expected * float(bool(active)),
                    f"{prefix}:group {group}:sample {record['audit_sample_index']} "
                    f"GRPO advantage[{token_index}]",
                    atol=1e-5,
                    rtol=1e-4,
                )
            audit.counts["grpo_sequences_verified"] += 1


def verify_trainer_file(audit: Audit, path: Path):
    payload = json.loads(path.read_text())
    step = int(payload["global_step"])
    prefix = f"trainer step={step}"
    audit.equal(payload.get("optimizer_updates_completed"), step - 1, f"{prefix}:pre-update counter")
    audit.equal(payload.get("generation_model"), "actor", f"{prefix}:generation model")
    audit.equal(payload.get("actor_model_path"), "/hf_models/Qwen3-1.7B", f"{prefix}:actor model")
    audit.equal(payload.get("generation_conditioning"), "original_prompt", f"{prefix}:generation conditioning")
    audit.equal(payload.get("teacher_conditioning"), "steering_vector", f"{prefix}:teacher conditioning")
    audit.equal(payload.get("teacher_prompt_identical_to_actor"), True, f"{prefix}:teacher identity")
    audit.near(payload.get("temperature"), 1.0, f"{prefix}:temperature", atol=0.0, rtol=0.0)
    audit.near(payload.get("top_p"), 1.0, f"{prefix}:top_p", atol=0.0, rtol=0.0)
    audit.equal(payload.get("top_k"), -1, f"{prefix}:top_k")
    audit.equal((payload.get("teacher_sft") or {}).get("enabled"), False, f"{prefix}:teacher SFT")
    audit.equal(payload.get("use_kl_in_reward"), False, f"{prefix}:KL-in-reward")
    require_strict_config(audit, payload.get("opsd_config") or {}, prefix)

    records = payload.get("records") or []
    audit.require(bool(records), f"{prefix}:no trainer records")
    groups = defaultdict(list)
    indexed = {}
    for record in records:
        index = int(record["audit_sample_index"])
        indexed[index] = record
        group = str(record.get("uid") or record.get("prompt_group_id"))
        correctness = float(
            record.get("reward_extra_opsd_correct", record.get("reward_extra_acc", 0.0))
        ) > 0.5
        groups[group].append((index, correctness))
        for suffix, actor_name, teacher_name in (
            ("input IDs", "actor_input_ids", "teacher_input_ids"),
            ("attention", "actor_attention_mask", "teacher_attention_mask"),
            ("positions", "actor_position_ids", "teacher_position_ids"),
            ("prompt IDs", "actor_prompt_ids", "teacher_prompt_ids"),
            ("prompt text", "actor_prompt_text", "teacher_prompt_text"),
        ):
            audit.equal(record.get(actor_name), record.get(teacher_name), f"{prefix}:sample {index} {suffix}")
        audit.equal(
            record.get("pre_opsd_actor_layout_sha256"),
            record.get("post_opsd_actor_layout_sha256"),
            f"{prefix}:sample {index} actor layout hash",
        )
        audit.require(
            len(str(record.get("pre_opsd_actor_layout_sha256", ""))) == 64,
            f"{prefix}:sample {index} missing layout SHA-256",
        )
        source_solution = str(record.get("source_solution") or "").strip()
        if source_solution:
            audit.require(
                source_solution not in str(record.get("actor_prompt_text") or ""),
                f"{prefix}:sample {index} source solution leaked into actor/teacher prompt",
            )
        audit.equal(record.get("source_cot_reason_present"), False, f"{prefix}:sample {index} COT_Reason")
        response_mask = as_bool(record["response_mask"])
        response_attention = as_bool(record["response_attention_mask"])
        distill_mask = as_bool(record["distill_mask"])
        audit.require(all(not d or r for d, r in zip(distill_mask, response_mask)), f"{prefix}:sample {index} distill escaped response")
        audit.require(all(not r or a for r, a in zip(response_mask, response_attention)), f"{prefix}:sample {index} response escaped attention")
        audit.counts["trainer_response_tokens"] += sum(response_mask)
        audit.counts["trainer_distill_tokens"] += sum(distill_mask)

    audit.equal(payload.get("actor_layout_unchanged_by_opsd"), True, f"{prefix}:actor layout")
    mixed_groups = 0
    global_positive = sum(correct for outcomes in groups.values() for _, correct in outcomes)
    global_negative = len(records) - global_positive
    global_mixed = bool(global_positive and global_negative)
    for group, outcomes in groups.items():
        positives = [index for index, correct in outcomes if correct]
        negatives = [index for index, correct in outcomes if not correct]
        mixed = bool(positives and negatives)
        mixed_groups += int(mixed)
        expected_sources = (
            [(index, 1.0) for index in positives]
            + [(index, -1.0) for index in negatives]
            if mixed
            else []
        )
        for target_index, _correct in outcomes:
            record = indexed[target_index]
            if audit.caa_scope == "global_batch":
                distill_active = any(as_bool(record["distill_mask"]))
                audit.equal(
                    distill_active,
                    global_mixed,
                    f"{prefix}:global target {target_index} symmetric mask",
                )
                audit.equal(
                    float(record.get("steering_source_outcome_sign")),
                    1.0 if _correct else -1.0,
                    f"{prefix}:global target {target_index} outcome sign",
                )
                audit.equal(
                    record.get("steering_source_candidate_mask"),
                    None,
                    f"{prefix}:global target {target_index} has no rectangular candidates",
                )
                if global_mixed:
                    audit.counts["outcome_symmetric_targets"] += 1
                continue
            candidate_mask = as_bool(record.get("steering_source_candidate_mask") or [])
            candidate_indices = record.get("steering_source_indices") or []
            signs = record.get("steering_source_signs") or []
            active = [
                (int(index), float(sign))
                for index, sign, keep in zip(candidate_indices, signs, candidate_mask)
                if keep
            ]
            distill_active = any(as_bool(record["distill_mask"]))
            audit.equal(distill_active, mixed, f"{prefix}:group {group} target {target_index} symmetric mask")
            audit.equal(active, expected_sources, f"{prefix}:group {group} target {target_index} CAA sources")
            source_masks = record.get("steering_source_response_mask") or []
            for slot, keep in enumerate(candidate_mask):
                source_mask = as_bool(source_masks[slot])
                if keep:
                    expected_mask = as_bool(indexed[int(candidate_indices[slot])]["response_mask"])
                    audit.equal(source_mask, expected_mask, f"{prefix}:target {target_index} source {slot} mask")
                else:
                    audit.require(not any(source_mask), f"{prefix}:target {target_index} padded source {slot} active")
            if mixed:
                audit.counts["outcome_symmetric_targets"] += 1
    audit.counts["mixed_prompt_groups"] += mixed_groups
    audit.counts["trainer_samples"] += len(records)
    if audit.caa_scope == "global_batch":
        audit.require(global_mixed, f"{prefix}:global batch does not contain both outcomes")
        audit.counts["global_mixed_batches"] += int(global_mixed)
    else:
        audit.require(mixed_groups > 0, f"{prefix}:no mixed-outcome prompt group")
    if audit.actor_objective == "grpo_advantage_reweighting":
        verify_grpo_trainer_advantages(audit, payload, groups, indexed, prefix)
    return step, indexed


def verify_steering_record(audit, record, response_mask, distill_mask, prefix):
    steering_config = record.get("steering_config") or {}
    for key, wanted in {
        "strict_contract": True,
        "source_mode": audit.steering_source_mode,
        "correct_rollout_aggregation": "all",
        "activation_aggregation": "per_rollout",
        "caa_scope": audit.caa_scope,
        "scale": audit.steering_scale,
        "normalize": "unit_norm",
        "apply_positions": "response_only",
        "detach_vectors": True,
        "expected_model_path": "/hf_models/Qwen3-1.7B",
        "actor_model_path": "/hf_models/Qwen3-1.7B",
        "expected_total_layers": 28,
        "expected_layer_indices": [9, 10],
    }.items():
        audit.equal(steering_config.get(key), wanted, f"{prefix}:steering.{key}")
    if audit.steering_source_mode == "policy_gradient":
        audit.equal(
            steering_config.get("gradient_objective"),
            "grpo_advantage",
            f"{prefix}:gradient objective",
        )
        audit.equal(
            steering_config.get("gradient_aggregation"),
            "per_rollout",
            f"{prefix}:gradient aggregation",
        )
    steering = record.get("steering_audit") or {}
    audit.equal(steering.get("caa_scope", "same_prompt"), audit.caa_scope, f"{prefix}:CAA scope")
    audit.equal(steering.get("activation_aggregation"), "per_rollout", f"{prefix}:activation aggregation")
    audit.equal(steering.get("accumulation_dtype"), "float32", f"{prefix}:accumulation dtype")
    audit.equal(
        steering.get("source_mode"),
        audit.steering_source_mode,
        f"{prefix}:steering source mode",
    )
    audit.equal(steering.get("selected_layers"), [9, 10], f"{prefix}:selected layers")
    if audit.caa_scope == "same_prompt":
        audit.equal(steering.get("candidate_mask"), record.get("steering_source_candidate_mask"), f"{prefix}:candidate mask ledger")
        audit.equal(steering.get("source_signs"), record.get("steering_source_signs"), f"{prefix}:source signs ledger")
        expected_counts = [
            [sum(float(value) for value in source_mask) for source_mask in target_masks]
            for target_masks in (record.get("steering_source_response_mask") or [])
        ]
        audit.equal(steering.get("source_response_token_counts"), expected_counts, f"{prefix}:source token counts")
    else:
        audit.equal(record.get("steering_source_indices"), None, f"{prefix}:no source index matrix")
        audit.equal(record.get("steering_source_candidate_mask"), None, f"{prefix}:no candidate matrix")
        audit.equal(steering.get("cross_rank_max_abs_error"), 0.0, f"{prefix}:cross-rank vector identity")
        step = int(record["global_step"])
        rank = int(record["rank"])
        audit.global_caa_records[step].setdefault(rank, steering)
        signs = record.get("steering_source_outcome_sign") or []
        for sign, mask in zip(signs, response_mask):
            audit.global_caa_sources[step][rank].append(
                (float(sign), sum(bool(value) for value in mask))
            )
    micro_metrics = record.get("micro_metrics") or {}
    audit.near(micro_metrics.get("actor/opsd_steering_per_rollout_normalization"), 1.0, f"{prefix}:per-rollout normalization flag", atol=0.0, rtol=0.0)
    audit.near(micro_metrics.get("actor/opsd_steering_fp32_accumulation"), 1.0, f"{prefix}:FP32 accumulation flag", atol=0.0, rtol=0.0)

    local_distill_tokens = sum(sum(row) for row in distill_mask)
    if local_distill_tokens:
        audit.equal(steering.get("vector_dtype"), "float32", f"{prefix}:vector dtype")
    norms = steering.get("vector_norms") or {}
    raw_norms = steering.get("raw_vector_norms") or {}
    applied_norms = steering.get("applied_vector_norms") or {}
    flat_norms = [float(value) for values in norms.values() for value in values]
    if local_distill_tokens:
        audit.require(bool(flat_norms), f"{prefix}:no steering vector norms")
    audit.counts["nonzero_steering_vectors"] += sum(abs(value) > 1e-8 for value in flat_norms)
    for layer, values in norms.items():
        raw_values = raw_norms.get(layer) or raw_norms.get(str(layer)) or []
        applied_values = applied_norms.get(layer) or applied_norms.get(str(layer)) or []
        for row_index, value in enumerate(values):
            raw_value = float(raw_values[row_index]) if row_index < len(raw_values) else None
            if raw_value is not None and raw_value > 1e-6:
                audit.near(value, 1.0, f"{prefix}:unit-L2 vector layer={layer} row={row_index}", atol=1e-5, rtol=1e-5)
            elif raw_value is not None:
                audit.near(value, 0.0, f"{prefix}:zero vector layer={layer} row={row_index}", atol=1e-8, rtol=0.0)
            if row_index < len(applied_values):
                audit.near(
                    applied_values[row_index],
                    float(value) * abs(audit.steering_scale),
                    f"{prefix}:applied vector norm layer={layer} row={row_index}",
                    atol=1e-5,
                    rtol=1e-5,
                )

    if audit.caa_scope == "global_batch" and audit.steering_source_mode == "caa":
        global_sums = steering.get("global_class_sums") or {}
        global_weights = steering.get("global_class_weights") or {}
        applied_vectors = steering.get("steering_vectors") or {}
        for layer in (9, 10):
            key = str(layer) if str(layer) in global_sums else layer
            class_sums = global_sums.get(key)
            class_weights = global_weights.get(key)
            actual_vectors = applied_vectors.get(key)
            audit.require(class_sums is not None, f"{prefix}:layer {layer} global sums missing")
            audit.require(class_weights is not None, f"{prefix}:layer {layer} global weights missing")
            audit.require(actual_vectors is not None, f"{prefix}:layer {layer} applied vector missing")
            if class_sums is None or class_weights is None or actual_vectors is None:
                continue
            positive_weight = float(class_weights[0][0])
            negative_weight = float(class_weights[1][0])
            audit.require(positive_weight > 0.0, f"{prefix}:layer {layer} has no positive weight")
            audit.require(negative_weight > 0.0, f"{prefix}:layer {layer} has no negative weight")
            expected_raw = [
                float(positive) / positive_weight - float(negative) / negative_weight
                for positive, negative in zip(class_sums[0], class_sums[1])
            ]
            expected_norm = math.sqrt(sum(value * value for value in expected_raw))
            expected_vector = [
                value / max(expected_norm, 1e-6) for value in expected_raw
            ]
            actual_vector = actual_vectors[0]
            audit.equal(len(actual_vector), len(expected_vector), f"{prefix}:layer {layer} vector width")
            for hidden_index, (actual, expected) in enumerate(
                zip(actual_vector, expected_vector)
            ):
                audit.near(
                    actual,
                    expected,
                    f"{prefix}:layer {layer} vector[{hidden_index}] from global sufficient stats",
                    atol=2e-5,
                    rtol=2e-4,
                )
            layer_raw_norms = raw_norms.get(key) or []
            if layer_raw_norms:
                audit.near(
                    layer_raw_norms[0],
                    expected_norm,
                    f"{prefix}:layer {layer} raw norm from global sufficient stats",
                    atol=2e-4,
                    rtol=2e-4,
                )
            audit.counts["global_vectors_recomputed"] += 1
    elif audit.caa_scope == "global_batch":
        global_gradient_sums = steering.get("global_gradient_sums") or {}
        applied_vectors = steering.get("steering_vectors") or {}
        rollout_count = float(steering.get("global_rollout_count", 0.0))
        audit.require(rollout_count > 0.0, f"{prefix}:global rollout count is zero")
        audit.equal(
            steering.get("gradient_objective"),
            "grpo_advantage",
            f"{prefix}:gradient objective audit",
        )
        audit.equal(
            steering.get("gradient_aggregation"),
            "per_rollout",
            f"{prefix}:gradient aggregation audit",
        )
        audit.equal(
            steering.get("nonzero_parameter_grad_count"),
            0,
            f"{prefix}:activation probe parameter-gradient route",
        )
        audit.require(
            float(steering.get("directional_derivative", 0.0)) > 0.0,
            f"{prefix}:policy-gradient direction is not an ascent direction",
        )
        caa_sums = steering.get("global_caa_class_sums") or {}
        caa_weights = steering.get("global_caa_class_weights") or {}
        caa_vectors = steering.get("caa_reference_vectors") or {}
        logged_cosines = steering.get("gradient_caa_cosines") or []
        recomputed_cosines = []
        for layer in (9, 10):
            key = str(layer) if str(layer) in global_gradient_sums else layer
            gradient_sum = global_gradient_sums.get(key)
            actual_vector_rows = applied_vectors.get(key)
            audit.require(
                gradient_sum is not None,
                f"{prefix}:layer {layer} global activation gradient missing",
            )
            audit.require(
                actual_vector_rows is not None,
                f"{prefix}:layer {layer} policy-gradient vector missing",
            )
            if gradient_sum is None or actual_vector_rows is None:
                continue
            expected_raw = [float(value) / rollout_count for value in gradient_sum]
            expected_norm = math.sqrt(sum(value * value for value in expected_raw))
            expected_vector = [
                value / max(expected_norm, 1e-6) for value in expected_raw
            ]
            actual_vector = actual_vector_rows[0]
            for hidden_index, (actual, expected) in enumerate(
                zip(actual_vector, expected_vector)
            ):
                audit.near(
                    actual,
                    expected,
                    f"{prefix}:layer {layer} vector[{hidden_index}] from global activation gradient",
                    atol=2e-5,
                    rtol=2e-4,
                )
            layer_raw_norms = raw_norms.get(key) or []
            if layer_raw_norms:
                audit.near(
                    layer_raw_norms[0],
                    expected_norm,
                    f"{prefix}:layer {layer} raw policy-gradient norm",
                    atol=2e-4,
                    rtol=2e-4,
                )

            class_sums = caa_sums.get(key)
            class_weights = caa_weights.get(key)
            actual_caa_rows = caa_vectors.get(key)
            audit.require(
                class_sums is not None and class_weights is not None and actual_caa_rows is not None,
                f"{prefix}:layer {layer} matched CAA reference is incomplete",
            )
            if class_sums is not None and class_weights is not None and actual_caa_rows is not None:
                positive_weight = float(class_weights[0][0])
                negative_weight = float(class_weights[1][0])
                caa_raw = [
                    float(positive) / positive_weight
                    - float(negative) / negative_weight
                    for positive, negative in zip(class_sums[0], class_sums[1])
                ]
                caa_norm = math.sqrt(sum(value * value for value in caa_raw))
                expected_caa = [value / max(caa_norm, 1e-6) for value in caa_raw]
                for hidden_index, (actual, expected) in enumerate(
                    zip(actual_caa_rows[0], expected_caa)
                ):
                    audit.near(
                        actual,
                        expected,
                        f"{prefix}:layer {layer} matched CAA vector[{hidden_index}]",
                        atol=2e-5,
                        rtol=2e-4,
                    )
                dot = sum(a * b for a, b in zip(expected_vector, expected_caa))
                recomputed_cosines.append(dot)
            audit.counts["global_vectors_recomputed"] += 1
        for layer_index, (actual, expected) in enumerate(
            zip(logged_cosines, recomputed_cosines)
        ):
            audit.near(
                actual,
                expected,
                f"{prefix}:gradient/CAA cosine layer_index={layer_index}",
                atol=2e-5,
                rtol=2e-4,
            )

    full_apply_mask = record.get("steering_apply_mask") or []
    for row_index, (apply_row, response_row) in enumerate(zip(full_apply_mask, record.get("response_attention_mask") or [])):
        response_width = len(response_row)
        audit.equal(
            [bool(value) for value in apply_row[-response_width:]],
            [bool(value) for value in response_row],
            f"{prefix}:response-only steering tail row={row_index}",
        )
        audit.require(
            not any(bool(value) for value in apply_row[:-response_width]),
            f"{prefix}:prompt steering leakage row={row_index}",
        )

    if audit.caa_scope == "global_batch":
        return

    # Targets with identical source membership must receive the same vector.
    memberships = {}
    source_indices = record.get("steering_source_indices") or []
    candidate_masks = record.get("steering_source_candidate_mask") or []
    source_signs = record.get("steering_source_signs") or []
    for row_index, (indices, candidates, signs) in enumerate(
        zip(source_indices, candidate_masks, source_signs)
    ):
        key = tuple(
            (int(index), float(sign))
            for index, sign, active in zip(indices, signs, candidates)
            if active
        )
        if not key:
            continue
        if key in memberships:
            prior = memberships[key]
            for layer, values in norms.items():
                audit.near(values[row_index], values[prior], f"{prefix}:shared CAA norm layer={layer} rows={prior},{row_index}", atol=1e-5, rtol=1e-5)
            audit.counts["shared_vector_target_pairs"] += 1
        else:
            memberships[key] = row_index


def verify_response_logprob_stats(audit, log_probs, response_mask, metrics, metric_prefix, prefix):
    """Recompute the response-only console statistics from raw ledger values."""

    token_values = [
        float(value)
        for row, mask_row in zip(log_probs, response_mask)
        for value, active in zip(row, mask_row)
        if active
    ]
    sequence_values = []
    for row, mask_row in zip(log_probs, response_mask):
        values = [float(value) for value, active in zip(row, mask_row) if active]
        if values:
            sequence_values.append(sum(values) / len(values))

    audit.require(token_values, f"{prefix}:{metric_prefix} has no response tokens")
    audit.require(
        all(math.isfinite(value) for value in token_values),
        f"{prefix}:{metric_prefix} has non-finite response log probabilities",
    )
    if not token_values:
        return

    token_mean = sum(token_values) / len(token_values)
    token_std = math.sqrt(
        sum((value - token_mean) ** 2 for value in token_values) / len(token_values)
    )
    sequence_mean = sum(sequence_values) / len(sequence_values)
    sequence_std = math.sqrt(
        sum((value - sequence_mean) ** 2 for value in sequence_values)
        / len(sequence_values)
    )
    expected = {
        f"{metric_prefix}_token_logprob_mean": token_mean,
        f"{metric_prefix}_token_logprob_std": token_std,
        f"{metric_prefix}_seq_logprob_mean": sequence_mean,
        f"{metric_prefix}_seq_logprob_std": sequence_std,
        f"{metric_prefix}_valid_token_count": float(len(token_values)),
        f"{metric_prefix}_valid_sequence_count": float(len(sequence_values)),
    }
    for name, wanted in expected.items():
        actual = metrics.get(name)
        audit.require(
            actual is not None and math.isfinite(float(actual)),
            f"{prefix}:{name} is missing or non-finite: {actual!r}",
        )
        if actual is not None:
            audit.near(actual, wanted, f"{prefix}:{name}", atol=5e-5, rtol=5e-4)
    audit.counts["response_logprob_stat_sets_verified"] += 1


def verify_microbatch(audit: Audit, record: dict, prefix: str, trainer_records):
    objective = audit.actor_objective
    audit.equal(record.get("actor_objective"), objective, f"{prefix}:actor objective")
    audit.equal(record.get("teacher_prompt_identical_to_actor"), True, f"{prefix}:teacher token identity")
    audit.equal(record.get("steering_conditioning"), True, f"{prefix}:steering active")
    audit.equal(record.get("loss_agg_mode"), "token-mean", f"{prefix}:loss aggregation")
    audit.equal(record.get("teacher_sft_loss_production"), None, f"{prefix}:teacher SFT loss")
    audit.near(record.get("teacher_sft_weight"), 0.0, f"{prefix}:teacher SFT weight", atol=0.0, rtol=0.0)
    audit.near(record.get("teacher_rlvr_weight"), 0.0, f"{prefix}:teacher RLVR weight", atol=0.0, rtol=0.0)

    response_mask = [[bool(value) for value in row] for row in record["response_mask"]]
    response_attention = [[bool(value) for value in row] for row in record["response_attention_mask"]]
    distill_mask = [[bool(value) for value in row] for row in record["distill_mask"]]
    for row_index, (response_row, attention_row, distill_row) in enumerate(
        zip(response_mask, response_attention, distill_mask)
    ):
        audit.require(all(not value or attention_row[index] for index, value in enumerate(response_row)), f"{prefix}:row {row_index} response escaped attention")
        audit.require(all(not value or response_row[index] for index, value in enumerate(distill_row)), f"{prefix}:row {row_index} distill escaped response")
    for row_index, sample_index in enumerate(record.get("sample_indices") or []):
        trainer = trainer_records.get(int(sample_index))
        audit.require(trainer is not None, f"{prefix}:unknown trainer sample {sample_index}")
        if trainer is not None:
            audit.equal(record["response_ids"][row_index], trainer["response_ids"], f"{prefix}:sample {sample_index} response IDs")
            audit.equal(record["response_mask"][row_index], trainer["response_mask"], f"{prefix}:sample {sample_index} response mask")
            audit.equal(record["distill_mask"][row_index], trainer["distill_mask"], f"{prefix}:sample {sample_index} distill mask")

    verify_steering_record(audit, record, response_mask, distill_mask, prefix)
    student = record["actor_log_probs"]
    teacher = record["teacher_log_probs"]
    if (
        audit.caa_scope == "global_batch"
        and audit.steering_source_mode == "policy_gradient"
    ):
        step = int(record["global_step"])
        rank = int(record["rank"])
        advantages = record.get("advantages") or []
        audit.equal(
            len(advantages),
            len(student),
            f"{prefix}:policy-gradient advantage rows",
        )
        for row_index, (log_prob_row, advantage_row, mask_row) in enumerate(
            zip(student, advantages, response_mask)
        ):
            valid_indices = [
                token_index
                for token_index, active in enumerate(mask_row)
                if active
            ]
            audit.require(
                bool(valid_indices),
                f"{prefix}:policy-gradient row {row_index} has no response tokens",
            )
            if not valid_indices:
                continue
            valid_log_probs = [
                float(log_prob_row[token_index]) for token_index in valid_indices
            ]
            valid_advantages = [
                float(advantage_row[token_index]) for token_index in valid_indices
            ]
            audit.require(
                all(math.isfinite(value) for value in valid_log_probs),
                f"{prefix}:policy-gradient row {row_index} has non-finite log probabilities",
            )
            audit.require(
                all(math.isfinite(value) for value in valid_advantages),
                f"{prefix}:policy-gradient row {row_index} has non-finite advantages",
            )
            sequence_advantage = sum(valid_advantages) / len(valid_advantages)
            max_deviation = max(
                abs(value - sequence_advantage) for value in valid_advantages
            )
            audit.require(
                max_deviation <= 1e-6,
                f"{prefix}:policy-gradient row {row_index} advantage deviation={max_deviation}",
            )
            sequence_log_prob = sum(valid_log_probs) / len(valid_log_probs)
            audit.global_policy_gradient_terms[step][rank].append(
                {
                    "term": sequence_advantage * sequence_log_prob,
                    "nonzero_advantage": abs(sequence_advantage) > 0.0,
                    "response_tokens": len(valid_indices),
                }
            )
    for matrix_name, matrix in (
        ("actor_log_probs", student),
        ("teacher_log_probs", teacher),
        ("student_behavior_log_probs", record.get("student_behavior_log_probs")),
        ("advantages", record.get("advantages")),
        ("advantages_shaped", record.get("advantages_shaped")),
    ):
        require_finite_matrix(audit, matrix, f"{prefix}:{matrix_name}")
    micro_metrics = record.get("micro_metrics") or {}
    verify_response_logprob_stats(
        audit, student, response_mask, micro_metrics, "actor/opsd_student", prefix
    )
    verify_response_logprob_stats(
        audit, teacher, response_mask, micro_metrics, "actor/opsd_teacher", prefix
    )
    expected_rkl = [
        [float(student_value) - float(teacher_value) for student_value, teacher_value in zip(student_row, teacher_row)]
        for student_row, teacher_row in zip(student, teacher)
    ]
    compare_matrix(audit, record.get("reverse_kl_token_estimate"), expected_rkl, f"{prefix}:sampled reverse KL", mask=response_mask)
    for row, mask_row in zip(expected_rkl, distill_mask):
        for value, active in zip(row, mask_row):
            if active:
                audit.counts["worker_distill_tokens"] += 1
                audit.counts["nonzero_reverse_kl_tokens"] += int(abs(value) > 1e-8)

    info = record.get("distill_global_info") or {}
    audit.require(int(info.get("batch_num_tokens", 0)) > 0, f"{prefix}:zero global distill denominator")
    routes = record.get("gradient_routing") or {}
    expected_total = None

    if objective == "direct_reverse_kl":
        expected_surrogate = [
            [rkl * float(logp) for rkl, logp in zip(rkl_row, student_row)]
            for rkl_row, student_row in zip(expected_rkl, student)
        ]
        compare_matrix(audit, record.get("distill_token_surrogate"), expected_surrogate, f"{prefix}:direct surrogate", mask=response_mask)
        expected_loss = aggregate(expected_surrogate, distill_mask, info, "token-mean")
        audit.near(record.get("distill_loss_production"), expected_loss, f"{prefix}:direct loss")
        audit.near(record.get("distill_weight"), 1.0, f"{prefix}:distill weight", atol=0.0, rtol=0.0)
        audit.near(record.get("student_rlvr_weight"), 0.0, f"{prefix}:student PPO weight", atol=0.0, rtol=0.0)
        audit.equal(record.get("student_ppo_terms"), None, f"{prefix}:student PPO absent")
        audit.equal(record.get("advantage_shaping_enabled"), False, f"{prefix}:shaping disabled")
        route_name = "distill_to_actor"
        expected_total = expected_loss
        audit.counts["direct_microbatches_verified"] += 1
    else:
        behavior = record.get("student_behavior_log_probs")
        audit.require(behavior is not None, f"{prefix}:missing behavior log probabilities")
        if behavior is None:
            behavior = student
        evidence = [
            [float(teacher_value) - float(student_value) for teacher_value, student_value in zip(teacher_row, student_row)]
            for teacher_row, student_row in zip(teacher, student)
        ]
        audit.equal(
            record.get("teacher_evidence_student_source"),
            "recomputed_current_actor",
            f"{prefix}:teacher evidence anchor",
        )
        compare_matrix(audit, record.get("teacher_evidence_scores"), evidence, f"{prefix}:fixed teacher evidence", mask=response_mask)
        audit.near(record.get("distill_weight"), 0.0, f"{prefix}:direct weight", atol=0.0, rtol=0.0)
        audit.near(record.get("student_rlvr_weight"), 1.0, f"{prefix}:student PPO weight", atol=0.0, rtol=0.0)
        audit.equal(record.get("distill_token_surrogate"), None, f"{prefix}:direct surrogate absent")
        audit.equal(record.get("distill_loss_production"), None, f"{prefix}:direct loss absent")

        if objective == "negative_kl_advantage":
            expected_advantages = [
                [value * float(active) for value, active in zip(evidence_row, mask_row)]
                for evidence_row, mask_row in zip(evidence, distill_mask)
            ]
            compare_matrix(audit, record.get("advantages_shaped"), expected_advantages, f"{prefix}:negative-KL advantages", mask=response_attention)
            audit.equal(record.get("advantage_shaping_mask"), record.get("distill_mask"), f"{prefix}:negative-KL objective mask")
            audit.equal(record.get("advantage_shaping_enabled"), False, f"{prefix}:shaping disabled")
            audit.equal(record.get("negative_kl_advantage_enabled"), True, f"{prefix}:negative-KL enabled")
            ppo_mask = distill_mask
            ppo_info = info
            route_name = "student_negative_kl_ppo_to_actor"
            audit.counts["negative_kl_microbatches_verified"] += 1
        else:
            original = record.get("advantages")
            actual_shaped = record.get("advantages_shaped")
            shaping_mask = record.get("advantage_shaping_mask")
            audit.require(original is not None and actual_shaped is not None, f"{prefix}:missing GRPO shaping tensors")
            audit.equal(shaping_mask, record.get("distill_mask"), f"{prefix}:GRPO shaping mask")
            expected_advantages = []
            if original is not None and actual_shaped is not None:
                for row_index, (advantage_row, evidence_row, response_row, shaping_row) in enumerate(
                    zip(original, evidence, response_mask, shaping_mask)
                ):
                    expected_row, correction_index = reshape_advantage_row(
                        advantage_row,
                        evidence_row,
                        response_row,
                        shaping_row,
                        max_delta_fraction=audit.max_delta_fraction,
                    )
                    expected_advantages.append(expected_row)
                    for token_index, (actual, expected, active) in enumerate(
                        zip(actual_shaped[row_index], expected_row, response_row)
                    ):
                        if not active:
                            audit.near(actual, advantage_row[token_index], f"{prefix}:row {row_index} PAD/outside response unchanged", atol=0.0, rtol=0.0)
                        elif correction_index is None or token_index != correction_index:
                            audit.near(actual, expected, f"{prefix}:row {row_index} shaped advantage[{token_index}]", atol=5e-5, rtol=5e-4)
                    conservation = sum(
                        (float(actual) - float(original_value)) * float(active)
                        for actual, original_value, active in zip(
                            actual_shaped[row_index], advantage_row, response_row
                        )
                    )
                    # Production corrects the represented response mass in
                    # float64 and stores one corrected token in float32.  A
                    # fixed 1e-6 threshold is invalid when an uncapped shaped
                    # token is large enough that one float32 ULP exceeds it.
                    # This independent bound is deliberately conservative but
                    # still scales only with the unavoidable single-token
                    # float32 store; it cannot hide prompt/PAD leakage or a
                    # missing conservation correction.
                    largest_magnitude = max(
                        1.0,
                        *(abs(float(value)) for value in advantage_row),
                        *(abs(float(value)) for value in actual_shaped[row_index]),
                    )
                    float32_roundoff_bound = 4.0 * (2.0 ** -23) * largest_magnitude
                    audit.near(
                        conservation,
                        0.0,
                        f"{prefix}:row {row_index} GRPO mass conservation",
                        atol=float32_roundoff_bound,
                        rtol=0.0,
                    )
                    if correction_index is not None:
                        ideal_correction = float(advantage_row[correction_index]) - sum(
                            float(actual_shaped[row_index][index]) - float(advantage_row[index])
                            for index, active in enumerate(response_row)
                            if active and index != correction_index
                        )
                        audit.near(actual_shaped[row_index][correction_index], ideal_correction, f"{prefix}:row {row_index} conservation correction", atol=1e-6, rtol=1e-5)
            audit.equal(record.get("advantage_shaping_enabled"), True, f"{prefix}:shaping enabled")
            audit.equal(record.get("negative_kl_advantage_enabled"), False, f"{prefix}:negative-KL disabled")
            ppo_mask = response_mask
            ppo_info = record.get("rlvr_global_info") or {}
            route_name = "student_shaped_rlvr_to_actor"
            audit.counts["grpo_reweight_microbatches_verified"] += 1

        clip_low = float(record["ppo_clip_ratio_low"])
        clip_high = float(record["ppo_clip_ratio_high"])
        clip_c = float(record["ppo_clip_ratio_c"])
        # Shaping is independently reconstructed and checked above. Rebuild
        # PPO from the exact FP32 advantage tensor that production consumed;
        # feeding the Python reconstruction into PPO compounds harmless
        # FP32-reduction differences at long (8K) sequence lengths.
        ppo_advantages = record.get("advantages_shaped") or expected_advantages
        expected_ppo = ppo_terms(behavior, student, ppo_advantages, clip_low, clip_high, clip_c)
        actual_ppo = record.get("student_ppo_terms") or {}
        for name, expected_values in expected_ppo.items():
            compare_matrix(audit, actual_ppo.get(name), expected_values, f"{prefix}:PPO {name}", mask=response_attention)
        audit.equal(record.get("student_ppo_response_mask"), [[float(value) for value in row] for row in ppo_mask], f"{prefix}:PPO mask")
        expected_student_loss = aggregate(
            expected_ppo["teacher_pg_after_is"], ppo_mask, ppo_info, "token-mean"
        )
        audit.near(record.get("student_rlvr_loss_production"), expected_student_loss, f"{prefix}:student PPO loss")
        expected_total = expected_student_loss

    audit.equal(record.get("teacher_rlvr_loss_production"), None, f"{prefix}:teacher RLVR loss absent")
    audit.equal(record.get("teacher_ppo_terms"), None, f"{prefix}:teacher PPO terms absent")
    audit.near(record.get("total_weighted_loss_production"), expected_total, f"{prefix}:weighted total loss")
    audit.require(route_name in routes, f"{prefix}:missing gradient route {route_name}")
    actor_route = routes.get(route_name) or {}
    audit.require(int(actor_route.get("hook_calls", 0)) > 0, f"{prefix}:{route_name} had no hooks")
    audit.require(not any("to_teacher" in key for key in routes), f"{prefix}:unexpected teacher gradient route {sorted(routes)}")
    for reference_name, reference in (record.get("reference_forwards") or {}).items():
        audit.equal(reference.get("status"), "PASS", f"{prefix}:{reference_name} reference forward")
    audit.counts["microbatches"] += 1


def verify_rank_file(audit: Audit, path: Path, trainer_steps):
    summaries = []
    optimizers = []
    for line_number, record in load_jsonl(path):
        prefix = f"{path.name}:line {line_number}:step {record.get('global_step')}"
        kind = record.get("record_type")
        if kind == "microbatch":
            trainer_records = trainer_steps.get(int(record["global_step"]), {})
            verify_microbatch(audit, record, prefix, trainer_records)
        elif kind == "optimizer_step":
            optimizers.append(record)
            audit.equal(record.get("actor_objective"), audit.actor_objective, f"{prefix}:objective")
            audit.equal(record.get("actor_attempted"), True, f"{prefix}:actor attempted")
            audit.equal(record.get("actor_did_step"), True, f"{prefix}:actor stepped")
            audit.equal(record.get("teacher_attempted"), False, f"{prefix}:teacher attempted")
            audit.equal(record.get("teacher_did_step"), False, f"{prefix}:teacher stepped")
            audit.counts["optimizer_records"] += 1
        elif kind == "update_summary":
            summaries.append(record)
            audit.equal(record.get("audit_status"), "PASS", f"{prefix}:runtime audit")
            audit.equal(record.get("actor_objective"), audit.actor_objective, f"{prefix}:objective")
            audit.equal(record.get("steering_signal_available"), True, f"{prefix}:steering signal")
            audit.equal(record.get("actor_step_attempted"), True, f"{prefix}:actor update attempted")
            audit.equal(record.get("actor_did_step"), True, f"{prefix}:actor update")
            audit.equal(record.get("teacher_step_attempted"), False, f"{prefix}:teacher update attempted")
            audit.equal(record.get("teacher_did_step"), False, f"{prefix}:teacher update")
            audit.equal(record.get("advantage_shaping_enabled"), audit.actor_objective == "grpo_advantage_reweighting", f"{prefix}:shaping flag")
            audit.equal(record.get("negative_kl_advantage_enabled"), audit.actor_objective == "negative_kl_advantage", f"{prefix}:negative-KL flag")
            audit.equal(record.get("teacher_sft_enabled"), False, f"{prefix}:teacher SFT")
            audit.equal(record.get("optimizer_updates_completed_after_step"), int(record["global_step"]), f"{prefix}:post-update counter")
            audit.counts["update_summaries"] += 1
    audit.require(bool(optimizers), f"{path}:missing optimizer record")
    audit.require(bool(summaries), f"{path}:missing update summary")


def verify_token_kl(audit: Audit, root: Path):
    paths = sorted(root.rglob("rank_*.jsonl"))
    micro_records = 0
    for path in paths:
        for line_number, record in load_jsonl(path):
            if record.get("record_type") != "token_reverse_kl_microbatch":
                continue
            prefix = f"token-KL {path.name}:{line_number}"
            audit.equal(record.get("actor_objective"), audit.actor_objective, f"{prefix}:objective")
            audit.equal(record.get("axis_scope"), "response_only", f"{prefix}:axis")
            audit.equal(record.get("prompt_tokens_logged"), 0, f"{prefix}:prompt tokens")
            audit.equal(record.get("masked_or_padded_response_tokens_logged"), 0, f"{prefix}:PAD tokens")
            audit.counts["token_kl_prompt_tokens_logged"] += int(
                record.get("prompt_tokens_logged", 0)
            )
            audit.counts["token_kl_masked_or_padded_tokens_logged"] += int(
                record.get("masked_or_padded_response_tokens_logged", 0)
            )
            for sample in record.get("samples") or []:
                audit.equal(sample.get("logged_token_count"), len(sample.get("tokens") or []), f"{prefix}:logged token count")
                for token in sample.get("tokens") or []:
                    current = float(token["student_log_prob"])
                    teacher = float(token["teacher_log_prob"])
                    audit.near(token["sampled_reverse_kl"], current - teacher, f"{prefix}:token reverse KL")
                    behavior = token.get("student_behavior_log_prob")
                    policy_advantage = token.get("policy_advantage")
                    distill_active = bool(token.get("distill_mask_active"))
                    if audit.actor_objective == "direct_reverse_kl":
                        audit.equal(behavior, None, f"{prefix}:direct behavior absent")
                        audit.equal(policy_advantage, None, f"{prefix}:direct policy advantage absent")
                    else:
                        audit.require(behavior is not None, f"{prefix}:behavior log probability missing")
                        if behavior is not None:
                            audit.near(token.get("behavior_sampled_reverse_kl"), float(behavior) - teacher, f"{prefix}:behavior reverse KL")
                        if audit.actor_objective == "negative_kl_advantage":
                            expected_advantage = (teacher - current) if distill_active else 0.0
                            audit.near(policy_advantage, expected_advantage, f"{prefix}:negative-KL token advantage")
                        else:
                            audit.near(policy_advantage, token.get("shaped_advantage"), f"{prefix}:GRPO policy/shaped advantage")
                    audit.counts["token_kl_tokens_verified"] += 1
            micro_records += 1
    audit.require(micro_records > 0, f"{root}:no token reverse-KL records")
    audit.counts["token_kl_microbatches"] = micro_records


def verify_global_caa_distributed_reduction(audit: Audit):
    """Rebuild every all-reduced steering statistic from rank-local ledgers."""

    if audit.caa_scope != "global_batch":
        return
    audit.require(bool(audit.global_caa_records), "no batch-global steering worker records")
    for step, rank_records in sorted(audit.global_caa_records.items()):
        prefix = f"global {audit.steering_source_mode} step={step}"
        audit.require(len(rank_records) > 1, f"{prefix}:distributed smoke used only one rank")
        if not rank_records:
            continue
        first_rank = min(rank_records)
        reference = rank_records[first_rank]
        global_rollouts = [float(value) for value in reference["global_rollout_counts"]]
        global_response_tokens = [
            float(value) for value in reference["global_response_token_counts"]
        ]
        summed_local_rollouts = [0.0, 0.0]
        summed_local_response_tokens = [0.0, 0.0]
        observed_rollouts = [0.0, 0.0]
        observed_response_tokens = [0.0, 0.0]
        for rank, steering in sorted(rank_records.items()):
            audit.equal(
                steering.get("global_rollout_counts"),
                reference.get("global_rollout_counts"),
                f"{prefix}:rank {rank} global rollout counts",
            )
            audit.equal(
                steering.get("global_response_token_counts"),
                reference.get("global_response_token_counts"),
                f"{prefix}:rank {rank} global response-token counts",
            )
            for class_index, value in enumerate(steering["local_rollout_counts"]):
                summed_local_rollouts[class_index] += float(value)
            for class_index, value in enumerate(steering["local_response_token_counts"]):
                summed_local_response_tokens[class_index] += float(value)
            for sign, token_count in audit.global_caa_sources[step][rank]:
                class_index = 0 if sign > 0 else 1
                observed_rollouts[class_index] += 1.0
                observed_response_tokens[class_index] += float(token_count)
            audit.equal(
                steering.get("cross_rank_max_abs_error"),
                0.0,
                f"{prefix}:rank {rank} vector all-reduce identity",
            )
        audit.equal(summed_local_rollouts, global_rollouts, f"{prefix}:summed local rollout counts")
        audit.equal(
            summed_local_response_tokens,
            global_response_tokens,
            f"{prefix}:summed local response-token counts",
        )
        audit.equal(observed_rollouts, global_rollouts, f"{prefix}:ledger outcome counts")
        audit.equal(
            observed_response_tokens,
            global_response_tokens,
            f"{prefix}:ledger response-token counts",
        )

        if audit.steering_source_mode == "policy_gradient":
            summed_local_rollout_count = sum(
                float(steering["local_rollout_count"])
                for steering in rank_records.values()
            )
            summed_local_nonzero_count = sum(
                float(steering["local_nonzero_advantage_count"])
                for steering in rank_records.values()
            )
            summed_local_response_count = sum(
                float(steering["local_response_token_count"])
                for steering in rank_records.values()
            )
            summed_local_objective = sum(
                float(steering["local_objective_sum"])
                for steering in rank_records.values()
            )
            reconstructed_terms = [
                term
                for rank_terms in audit.global_policy_gradient_terms[step].values()
                for term in rank_terms
            ]
            reconstructed_objective = sum(
                float(term["term"]) for term in reconstructed_terms
            )
            reconstructed_nonzero_count = sum(
                bool(term["nonzero_advantage"]) for term in reconstructed_terms
            )
            reconstructed_response_count = sum(
                int(term["response_tokens"]) for term in reconstructed_terms
            )
            reference_rollout_count = float(reference["global_rollout_count"])
            reference_nonzero_count = float(
                reference["global_nonzero_advantage_count"]
            )
            reference_response_count = float(reference["global_response_token_count"])
            reference_objective = float(reference["global_objective_sum"])
            for rank, steering in sorted(rank_records.items()):
                audit.near(
                    steering["global_rollout_count"],
                    reference_rollout_count,
                    f"{prefix}:rank {rank} global rollout count",
                    atol=0.0,
                    rtol=0.0,
                )
                audit.near(
                    steering["global_nonzero_advantage_count"],
                    reference_nonzero_count,
                    f"{prefix}:rank {rank} global nonzero-advantage count",
                    atol=0.0,
                    rtol=0.0,
                )
                audit.near(
                    steering["global_response_token_count"],
                    reference_response_count,
                    f"{prefix}:rank {rank} global response-token count",
                    atol=0.0,
                    rtol=0.0,
                )
                audit.near(
                    steering["global_objective_sum"],
                    reference_objective,
                    f"{prefix}:rank {rank} global objective",
                    atol=1e-5,
                    rtol=1e-6,
                )
            audit.near(
                summed_local_rollout_count,
                reference_rollout_count,
                f"{prefix}:summed local rollout count",
                atol=0.0,
                rtol=0.0,
            )
            audit.near(
                summed_local_nonzero_count,
                reference_nonzero_count,
                f"{prefix}:summed local nonzero-advantage count",
                atol=0.0,
                rtol=0.0,
            )
            audit.near(
                summed_local_response_count,
                reference_response_count,
                f"{prefix}:summed local response-token count",
                atol=0.0,
                rtol=0.0,
            )
            audit.near(
                summed_local_objective,
                reference_objective,
                f"{prefix}:summed local objective",
                atol=2e-3,
                rtol=2e-4,
            )
            audit.near(
                reconstructed_objective,
                reference_objective,
                f"{prefix}:objective from audited current-actor log probabilities",
                atol=2e-3,
                rtol=2e-4,
            )
            audit.equal(
                len(reconstructed_terms),
                int(reference_rollout_count),
                f"{prefix}:audited objective rollout count",
            )
            audit.equal(
                reconstructed_nonzero_count,
                int(reference_nonzero_count),
                f"{prefix}:audited nonzero-advantage count",
            )
            audit.equal(
                reconstructed_response_count,
                int(reference_response_count),
                f"{prefix}:audited objective response-token count",
            )
            audit.counts["policy_gradient_objectives_recomputed"] += 1

        for layer in (9, 10):
            key = str(layer)
            if audit.steering_source_mode == "caa":
                local_sums_key = "local_class_sums"
                global_sums_key = "global_class_sums"
                local_weights_key = "local_class_weights"
                global_weights_key = "global_class_weights"
            else:
                local_sums_key = "local_caa_class_sums"
                global_sums_key = "global_caa_class_sums"
                local_weights_key = "local_caa_class_weights"
                global_weights_key = "global_caa_class_weights"
            reference_global_sums = reference[global_sums_key].get(key)
            reference_global_weights = reference[global_weights_key].get(key)
            reference_vector = (
                reference["steering_vectors"].get(key)
                if audit.steering_source_mode == "caa"
                else reference["caa_reference_vectors"].get(key)
            )
            summed_sums = [
                [0.0 for _ in reference_global_sums[0]],
                [0.0 for _ in reference_global_sums[1]],
            ]
            summed_weights = [0.0, 0.0]
            max_global_sum_error = 0.0
            max_vector_error = 0.0
            for rank, steering in sorted(rank_records.items()):
                local_sums = steering[local_sums_key].get(key)
                local_weights = steering[local_weights_key].get(key)
                for class_index in (0, 1):
                    summed_weights[class_index] += float(local_weights[class_index][0])
                    for hidden_index, value in enumerate(local_sums[class_index]):
                        summed_sums[class_index][hidden_index] += float(value)
                global_sums = steering[global_sums_key].get(key)
                vector = (
                    steering["steering_vectors"].get(key)
                    if audit.steering_source_mode == "caa"
                    else steering["caa_reference_vectors"].get(key)
                )
                for expected_row, actual_row in zip(reference_global_sums, global_sums):
                    for expected, actual in zip(expected_row, actual_row):
                        max_global_sum_error = max(
                            max_global_sum_error, abs(float(actual) - float(expected))
                        )
                for expected_row, actual_row in zip(reference_vector, vector):
                    for expected, actual in zip(expected_row, actual_row):
                        max_vector_error = max(
                            max_vector_error, abs(float(actual) - float(expected))
                        )
            max_local_reduction_error = max(
                abs(actual - float(expected))
                for actual_row, expected_row in zip(summed_sums, reference_global_sums)
                for actual, expected in zip(actual_row, expected_row)
            )
            audit.require(
                max_local_reduction_error <= 2e-3,
                f"{prefix}:layer {layer} local-sum reduction max error={max_local_reduction_error}",
            )
            audit.require(
                max_global_sum_error <= 1e-6,
                f"{prefix}:layer {layer} global stats differ across ranks by {max_global_sum_error}",
            )
            audit.require(
                max_vector_error <= 1e-7,
                f"{prefix}:layer {layer} applied vectors differ across ranks by {max_vector_error}",
            )
            audit.equal(
                summed_weights,
                [float(row[0]) for row in reference_global_weights],
                f"{prefix}:layer {layer} summed local weights",
            )

            if audit.steering_source_mode == "policy_gradient":
                reference_global_gradient = reference["global_gradient_sums"].get(
                    key
                )
                reference_policy_vector = reference["steering_vectors"].get(key)
                summed_gradient = [
                    0.0 for _ in range(len(reference_global_gradient))
                ]
                max_global_gradient_error = 0.0
                max_policy_vector_error = 0.0
                for rank, steering in sorted(rank_records.items()):
                    local_gradient = steering["local_gradient_sums"].get(key)
                    for hidden_index, value in enumerate(local_gradient):
                        summed_gradient[hidden_index] += float(value)
                    global_gradient = steering["global_gradient_sums"].get(key)
                    for expected, actual in zip(
                        reference_global_gradient, global_gradient
                    ):
                        max_global_gradient_error = max(
                            max_global_gradient_error,
                            abs(float(actual) - float(expected)),
                        )
                    policy_vector = steering["steering_vectors"].get(key)
                    for expected_row, actual_row in zip(
                        reference_policy_vector, policy_vector
                    ):
                        for expected, actual in zip(expected_row, actual_row):
                            max_policy_vector_error = max(
                                max_policy_vector_error,
                                abs(float(actual) - float(expected)),
                            )
                max_gradient_reduction_error = max(
                    abs(actual - float(expected))
                    for actual, expected in zip(
                        summed_gradient, reference_global_gradient
                    )
                )
                audit.require(
                    max_gradient_reduction_error <= 2e-3,
                    f"{prefix}:layer {layer} local-gradient reduction max error="
                    f"{max_gradient_reduction_error}",
                )
                audit.require(
                    max_global_gradient_error <= 1e-6,
                    f"{prefix}:layer {layer} global gradients differ across ranks by "
                    f"{max_global_gradient_error}",
                )
                audit.require(
                    max_policy_vector_error <= 1e-7,
                    f"{prefix}:layer {layer} policy-gradient vectors differ across "
                    f"ranks by {max_policy_vector_error}",
                )
                expected_raw = [
                    float(value) / reference_rollout_count
                    for value in reference_global_gradient
                ]
                expected_norm = math.sqrt(
                    sum(value * value for value in expected_raw)
                )
                expected_policy_vector = [
                    value / max(expected_norm, 1e-6) for value in expected_raw
                ]
                for hidden_index, (actual, expected) in enumerate(
                    zip(reference_policy_vector[0], expected_policy_vector)
                ):
                    audit.near(
                        actual,
                        expected,
                        f"{prefix}:layer {layer} independently normalized gradient "
                        f"vector[{hidden_index}]",
                        atol=2e-5,
                        rtol=2e-4,
                    )
                audit.counts["global_policy_gradient_layers_verified"] += 1
            audit.counts["global_distributed_layers_verified"] += 1
        audit.counts["global_distributed_steps_verified"] += 1


def write_report(audit: Audit, report_dir: Path, audit_root: Path, token_root: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not audit.errors else "FAIL"
    payload = {
        "status": status,
        "actor_objective": audit.actor_objective,
        "audit_root": str(audit_root),
        "token_kl_root": str(token_root),
        "counts": dict(sorted(audit.counts.items())),
        "observations": audit.observations,
        "errors": audit.errors,
        "warnings": audit.warnings,
    }
    (report_dir / "audit_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        WIDE_CSS,
        "",
        "# Steering-vector OPSD live audit",
        "",
        f"Objective: `{audit.actor_objective}`",
        "",
        f"Status: **{status}**",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in sorted(audit.counts.items()))
    lines.extend(["", "## Errors", ""])
    lines.extend(["None."] if not audit.errors else [f"- {error}" for error in audit.errors])
    lines.extend(["", "## Warnings", ""])
    lines.extend(["None."] if not audit.warnings else [f"- {warning}" for warning in audit.warnings])
    (report_dir / "audit_report.md").write_text("\n".join(lines) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--token-kl-root", type=Path, required=True)
    parser.add_argument("--actor-objective", choices=sorted(OBJECTIVES), required=True)
    parser.add_argument("--steering-scale", type=float, default=1.0)
    parser.add_argument("--max-delta-fraction", default="1.0")
    parser.add_argument(
        "--caa-scope", choices=("same_prompt", "global_batch"), default="same_prompt"
    )
    parser.add_argument(
        "--steering-source-mode",
        choices=("caa", "policy_gradient"),
        default="caa",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    max_delta_fraction = (
        None
        if str(args.max_delta_fraction).lower() in {"none", "null"}
        else float(args.max_delta_fraction)
    )
    audit = Audit(
        args.actor_objective,
        steering_scale=args.steering_scale,
        max_delta_fraction=max_delta_fraction,
        caa_scope=args.caa_scope,
        steering_source_mode=args.steering_source_mode,
    )
    trainer_files = sorted(args.audit_root.rglob("trainer_batch.json"))
    audit.require(bool(trainer_files), f"{args.audit_root}:no trainer_batch.json")
    trainer_steps = {}
    for path in trainer_files:
        step, records = verify_trainer_file(audit, path)
        trainer_steps[step] = records
    rank_files = sorted(args.audit_root.rglob("rank_*.jsonl"))
    audit.require(bool(rank_files), f"{args.audit_root}:no worker rank ledgers")
    for path in rank_files:
        verify_rank_file(audit, path, trainer_steps)
    verify_global_caa_distributed_reduction(audit)
    verify_token_kl(audit, args.token_kl_root)
    audit.require(audit.counts["trainer_distill_tokens"] > 0, "trainer emitted no eligible tokens")
    audit.require(audit.counts["worker_distill_tokens"] > 0, "workers saw no eligible tokens")
    audit.require(audit.counts["nonzero_reverse_kl_tokens"] > 0, "all sampled reverse-KL values were zero")
    audit.require(audit.counts["nonzero_steering_vectors"] > 0, "all steering vectors were zero")
    status = write_report(audit, args.report_dir, args.audit_root, args.token_kl_root)
    print(
        json.dumps(
            {
                "status": status,
                "actor_objective": args.actor_objective,
                "counts": dict(audit.counts),
                "errors": len(audit.errors),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
