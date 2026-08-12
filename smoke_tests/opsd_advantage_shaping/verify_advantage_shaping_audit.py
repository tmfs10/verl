#!/usr/bin/env python3
"""Independently reconstruct OPSD advantage shaping from JSON audit ledgers.

This verifier intentionally imports neither VERL nor ``recipe.opsd``.  It
reimplements mask selection, centered evidence normalization, sign-preserving
scaling, PPO token arithmetic, aggregation, and weighted-loss composition with
the Python standard library.
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


PROFILE_EXPECTATIONS = {
    "generic": {
        "teacher_model": None,
        "teacher_objective": None,
        "is_mode": None,
        "require_positive_actor_grad": False,
        "require_positive_teacher_grad": False,
    },
    "shared": {
        "teacher_model": "actor",
        "teacher_objective": "none",
        "is_mode": None,
        "require_positive_actor_grad": True,
        "require_positive_teacher_grad": False,
    },
    "separate_sft": {
        "teacher_model": "separate",
        "teacher_objective": "sft",
        "is_mode": None,
        "require_positive_actor_grad": True,
        "require_positive_teacher_grad": True,
    },
    "separate_rlvr": {
        "teacher_model": "separate",
        "teacher_objective": "rlvr",
        "is_mode": "token",
        "require_positive_actor_grad": True,
        "require_positive_teacher_grad": True,
    },
    "separate_sft_warmup": {
        "teacher_model": "separate",
        "teacher_objective": "sft",
        "is_mode": None,
        "require_positive_actor_grad": True,
        "require_positive_teacher_grad": True,
    },
}


def close(actual: float, expected: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(float(actual)) and math.isfinite(float(expected)) and abs(
        float(actual) - float(expected)
    ) <= atol + rtol * abs(float(expected))


def require_close(errors, name, actual, expected, *, atol, rtol):
    if actual is None or not close(actual, expected, atol=atol, rtol=rtol):
        errors.append(
            f"{name}: actual={actual!r}, expected={expected!r}, atol={atol}, rtol={rtol}"
        )


def require_equal(errors, name, actual, expected):
    if actual != expected:
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}")


def compare_matrix(errors, name, actual, expected, *, atol, rtol):
    if actual is None:
        errors.append(f"{name}: missing matrix")
        return
    if len(actual) != len(expected):
        errors.append(f"{name}: row count {len(actual)} != {len(expected)}")
        return
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if len(actual_row) != len(expected_row):
            errors.append(
                f"{name}[{row_index}]: length {len(actual_row)} != {len(expected_row)}"
            )
            continue
        for token_index, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row)
        ):
            require_close(
                errors,
                f"{name}[{row_index},{token_index}]",
                actual_value,
                expected_value,
                atol=atol,
                rtol=rtol,
            )


def aggregate(values, masks, info, mode):
    dp_size = int(info["dp_size"])
    if mode == "token-mean":
        denominator = int(info["batch_num_tokens"])
        if denominator <= 0:
            return 0.0
        numerator = sum(
            value * mask
            for row, mask_row in zip(values, masks)
            for value, mask in zip(row, mask_row)
        )
        return numerator / denominator * dp_size
    if mode in {"seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        denominator = int(info["global_batch_size"])
        if denominator <= 0:
            return 0.0
        result = sum(
            sum(value * mask for value, mask in zip(row, mask_row))
            for row, mask_row in zip(values, masks)
        )
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
            count = sum(mask_row)
            if count:
                result += sum(value * mask for value, mask in zip(row, mask_row)) / count
        return result / denominator * dp_size
    raise ValueError(f"unsupported loss aggregation mode: {mode}")


def reshape_row(advantages, evidence, response_mask, shaping_mask, config):
    selected = [index for index, value in enumerate(shaping_mask) if bool(value)]
    if not selected:
        return (
            list(advantages),
            [0.0] * len(advantages),
            [0.0] * len(advantages),
            0.0,
            None,
        )

    score_mean = sum(evidence[index] for index in selected) / len(selected)
    centered = [0.0] * len(evidence)
    for index in selected:
        centered[index] = evidence[index] - score_mean

    normalize = config.get("normalize", "std")
    if normalize in (None, "none"):
        z = list(centered)
    elif normalize == "std":
        denominator = math.sqrt(
            sum(centered[index] ** 2 for index in selected) / len(selected)
        )
        z = [value / max(denominator, 1e-6) for value in centered]
    elif normalize == "mean_abs":
        denominator = sum(abs(centered[index]) for index in selected) / len(selected)
        z = [value / max(denominator, 1e-6) for value in centered]
    elif normalize == "range":
        denominator = max(evidence[index] for index in selected) - min(
            evidence[index] for index in selected
        )
        z = [value / max(denominator, 1e-6) for value in centered]
    else:
        raise ValueError(f"unsupported normalize={normalize!r}")

    clip_z = config.get("clip_z", 3.0)
    if clip_z is not None:
        z = [max(-float(clip_z), min(float(clip_z), value)) for value in z]
        clipped_mean = sum(z[index] for index in selected) / len(selected)
        z = [
            (value - clipped_mean) if index in selected else 0.0
            for index, value in enumerate(z)
        ]

    response_indices = [index for index, value in enumerate(response_mask) if bool(value)]
    response_advantage = (
        sum(advantages[index] for index in response_indices) / len(response_indices)
        if response_indices
        else 0.0
    )
    effective_scale = float(config.get("scale", 1.0))
    if not bool(config.get("allow_token_sign_flip", False)) and effective_scale > 0.0:
        z_min = min(z[index] for index in selected)
        z_max = max(z[index] for index in selected)
        if response_advantage > 0.0 and z_min < 0.0:
            effective_scale = min(effective_scale, 1.0 / max(-z_min, 1e-6))
        elif response_advantage < 0.0 and z_max > 0.0:
            effective_scale = min(effective_scale, 1.0 / max(z_max, 1e-6))
        elif response_advantage == 0.0:
            effective_scale = 0.0

    delta = [0.0] * len(advantages)
    shaped = list(advantages)
    for index in selected:
        delta[index] = effective_scale * abs(response_advantage) * z[index]
        shaped[index] += delta[index]
    if not bool(config.get("allow_token_sign_flip", False)):
        for index in selected:
            if response_advantage > 0.0:
                shaped[index] = max(0.0, shaped[index])
            elif response_advantage < 0.0:
                shaped[index] = min(0.0, shaped[index])
            delta[index] = shaped[index] - advantages[index]

    # Production removes the represented-token conservation residual from the
    # largest-magnitude selected token. Identify that token independently and
    # apply the binary64 reconstruction's residual here. The verifier compares
    # every other token directly. It verifies this correction token from the
    # exact zero-sum constraint below because a long PyTorch float32 reduction
    # can retain a much larger residual than this binary64 reconstruction.
    represented_residual = sum(
        shaped[index] - advantages[index] for index in response_indices
    )
    correction_index = max(selected, key=lambda index: abs(shaped[index]))
    shaped[correction_index] -= represented_residual
    delta[correction_index] -= represented_residual
    return shaped, z, delta, effective_scale, correction_index


def ppo_terms(old, current, advantages, clip_low, clip_high, clip_c, is_weights=None):
    terms = {
        "teacher_current_old_log_ratio_clamped": [],
        "teacher_current_old_ratio": [],
        "teacher_ratio_clipped": [],
        "teacher_pg_unclipped": [],
        "teacher_pg_clipped_candidate": [],
        "teacher_pg_selected_before_is": [],
        "teacher_pg_after_is": [],
    }
    if is_weights is None:
        is_weights = [[1.0] * len(row) for row in advantages]
    for old_row, current_row, advantage_row, is_row in zip(
        old, current, advantages, is_weights
    ):
        rows = {name: [] for name in terms}
        for old_value, current_value, advantage, is_weight in zip(
            old_row, current_row, advantage_row, is_row
        ):
            log_ratio = max(-20.0, min(20.0, current_value - old_value))
            ratio = math.exp(log_ratio)
            unclipped = -advantage * ratio
            clipped_ratio = max(1.0 - clip_low, min(1.0 + clip_high, ratio))
            clipped_candidate = -advantage * clipped_ratio
            clipped = max(unclipped, clipped_candidate)
            selected = min(-advantage * clip_c, clipped) if advantage < 0.0 else clipped
            rows["teacher_current_old_log_ratio_clamped"].append(log_ratio)
            rows["teacher_current_old_ratio"].append(ratio)
            rows["teacher_ratio_clipped"].append(clipped_ratio)
            rows["teacher_pg_unclipped"].append(unclipped)
            rows["teacher_pg_clipped_candidate"].append(clipped_candidate)
            rows["teacher_pg_selected_before_is"].append(selected)
            rows["teacher_pg_after_is"].append(selected * is_weight)
        for name in terms:
            terms[name].append(rows[name])
    return terms


def teacher_is_weights(old, behavior, sampling_mask, mode, clip):
    if mode == "none":
        return None
    output = []
    for old_row, behavior_row, mask_row in zip(old, behavior, sampling_mask):
        log_ratios = [old_value - behavior_value for old_value, behavior_value in zip(old_row, behavior_row)]
        if mode == "sequence":
            log_weight = sum(value * mask for value, mask in zip(log_ratios, mask_row))
            weight = min(math.exp(max(-20.0, min(20.0, log_weight))), float(clip))
            output.append([weight * float(bool(mask)) for mask in mask_row])
        elif mode == "token":
            output.append(
                [
                    min(math.exp(max(-20.0, min(20.0, value))), float(clip))
                    * float(bool(mask))
                    for value, mask in zip(log_ratios, mask_row)
                ]
            )
        else:
            raise ValueError(f"unsupported teacher IS mode: {mode}")
    return output


def verify(
    audit_dir: Path,
    *,
    profile: str = "generic",
    expected_warmup_steps: tuple[int, ...] = (),
    expected_joint_steps: tuple[int, ...] = (),
    required_response_axis: int | None = None,
):
    expectation = PROFILE_EXPECTATIONS[profile]
    expected_warmup_steps = set(expected_warmup_steps)
    expected_joint_steps = set(expected_joint_steps)
    errors = []
    observations = {
        "microbatches": 0,
        "shaped_microbatches": 0,
        "warmup_microbatches": 0,
        "response_tokens": 0,
        "shaped_tokens": 0,
        "nonzero_shaping_delta_tokens": 0,
        "pad_tokens_examined": 0,
        "prompt_tokens_examined": 0,
        "teacher_sft_tokens": 0,
        "teacher_rlvr_microbatches": 0,
        "teacher_rlvr_nonzero_advantage_tokens": 0,
        "teacher_is_nonunit_tokens": 0,
        "teacher_is_pad_abs_max": 0.0,
        "actor_prompt_valid_tokens_examined": 0,
        "teacher_prompt_valid_tokens_examined": 0,
        "actor_optimizer_steps": 0,
        "teacher_optimizer_steps": 0,
        "positive_actor_grad_records": 0,
        "positive_teacher_grad_records": 0,
        "max_pre_correction_error": 0.0,
        "correction_tokens_verified": 0,
        "max_conservation_error": 0.0,
        "max_outside_mask_delta": 0.0,
        "max_pad_delta": 0.0,
    }
    response_axis_lengths = set()
    teacher_models = set()
    teacher_is_modes = set()
    observed_warmup_steps = set()
    observed_joint_steps = set()
    configs = {}
    trainer_records = {}
    reverse_kl_by_step_rank = {}
    for path in sorted(audit_dir.glob("step_*/trainer_batch.json")):
        payload = json.loads(path.read_text())
        step = int(payload["global_step"])
        config = payload["opsd_config"]
        configs[step] = config
        shaping = config.get("advantage_shaping", {})
        require_equal(errors, f"step={step}:mode", config.get("mode"), "opsd_rlvr")
        require_equal(errors, f"step={step}:mix_weight", config.get("mix_weight"), 1.0)
        require_equal(errors, f"step={step}:distill_backward_scale", config.get("distill_backward_scale"), 0.0)
        require_equal(errors, f"step={step}:balance_mode", config.get("balance_mode"), "none")
        require_equal(errors, f"step={step}:shaping_enabled", shaping.get("enable"), True)
        require_equal(
            errors,
            f"step={step}:student_rlvr_backward_scale",
            shaping.get("student_rlvr_backward_scale"),
            1.0,
        )
        require_equal(
            errors,
            f"step={step}:score_source",
            shaping.get("score_source"),
            "teacher_minus_student_logprob",
        )
        teacher_model = config.get("teacher_model")
        teacher_models.add(teacher_model)
        if expectation["teacher_model"] is not None:
            require_equal(
                errors,
                f"step={step}:teacher_model",
                teacher_model,
                expectation["teacher_model"],
            )
        teacher_objective = expectation["teacher_objective"]
        if teacher_objective == "none":
            require_equal(errors, f"step={step}:teacher_sft_weight", config.get("teacher_sft_weight"), 0.0)
            require_equal(errors, f"step={step}:teacher_rlvr_weight", config.get("rlvr_backward_scale"), 0.0)
        elif teacher_objective == "sft":
            if float(config.get("teacher_sft_weight", 0.0) or 0.0) <= 0.0:
                errors.append(f"step={step}: expected a positive teacher SFT weight")
            require_equal(errors, f"step={step}:teacher_rlvr_weight", config.get("rlvr_backward_scale"), 0.0)
        elif teacher_objective == "rlvr":
            require_equal(errors, f"step={step}:teacher_sft_weight", config.get("teacher_sft_weight"), 0.0)
            if float(config.get("rlvr_backward_scale", 0.0) or 0.0) <= 0.0:
                errors.append(f"step={step}: expected a positive teacher RLVR weight")
            require_equal(
                errors,
                f"step={step}:offpolicy_is_mode",
                config.get("offpolicy_is_mode"),
                expectation["is_mode"],
            )
        for record in payload.get("records", []):
            sample = int(record["audit_sample_index"])
            trainer_records[(step, sample)] = record
            response_length = len(record["response_ids"])
            response_axis_lengths.add(response_length)
            actor_input = record["actor_input_ids"]
            actor_attention = record["actor_attention_mask"]
            response_attention = record["response_attention_mask"]
            if actor_input[-response_length:] != record["response_ids"]:
                errors.append(f"step={step}:sample={sample}: actor response suffix mismatch")
            if actor_attention[-response_length:] != response_attention:
                errors.append(f"step={step}:sample={sample}: response attention suffix mismatch")
            prompt_length = len(actor_input) - response_length
            if prompt_length <= 0:
                errors.append(f"step={step}:sample={sample}: no prompt axis to exclude")
            observations["prompt_tokens_examined"] += prompt_length
            actor_prompt_valid = sum(bool(value) for value in actor_attention[:prompt_length])
            observations["actor_prompt_valid_tokens_examined"] += actor_prompt_valid
            if actor_prompt_valid <= 0:
                errors.append(f"step={step}:sample={sample}: actor prompt has no valid token")
            teacher_input = record["teacher_input_ids"]
            teacher_attention = record["teacher_attention_mask"]
            if len(teacher_input) < response_length:
                errors.append(f"step={step}:sample={sample}: teacher input shorter than response axis")
            elif teacher_input[-response_length:] != record["response_ids"]:
                errors.append(f"step={step}:sample={sample}: teacher response suffix mismatch")
            if len(teacher_attention) < response_length:
                errors.append(f"step={step}:sample={sample}: teacher attention shorter than response axis")
            elif teacher_attention[-response_length:] != response_attention:
                errors.append(f"step={step}:sample={sample}: teacher response attention suffix mismatch")
            teacher_prompt_length = len(teacher_input) - response_length
            teacher_prompt_valid = sum(
                bool(value) for value in teacher_attention[:teacher_prompt_length]
            )
            observations["teacher_prompt_valid_tokens_examined"] += teacher_prompt_valid
            if teacher_prompt_valid <= 0:
                errors.append(f"step={step}:sample={sample}: teacher prompt has no valid token")
            for mask_name in ("response_mask", "distill_mask"):
                mask = record[mask_name]
                if len(mask) != response_length:
                    errors.append(
                        f"step={step}:sample={sample}: {mask_name} axis {len(mask)} != {response_length}"
                    )
                    continue
                for token_index, (mask_value, attention_value) in enumerate(
                    zip(mask, response_attention)
                ):
                    if bool(mask_value) and not bool(attention_value):
                        errors.append(
                            f"step={step}:sample={sample}: {mask_name} selected PAD token {token_index}"
                        )
            observations["pad_tokens_examined"] += sum(not bool(value) for value in response_attention)

    microbatch_records = []
    summaries = []
    optimizer_records = []
    for path in sorted(audit_dir.glob("step_*/rank_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "microbatch":
                microbatch_records.append((path, record))
            elif record.get("record_type") == "update_summary":
                summaries.append((path, record))
            elif record.get("record_type") == "optimizer_step":
                optimizer_records.append((path, record))

    for path, record in microbatch_records:
        observations["microbatches"] += 1
        step = int(record["global_step"])
        prefix = f"{path}:step={step}:microbatch={record['microbatch_index']}"
        config = configs.get(step, {})
        shaping_config = config.get("advantage_shaping", {})
        warmup = bool(record.get("warmup_active"))
        require_equal(
            errors,
            f"{prefix}:teacher_model",
            record.get("teacher_model"),
            config.get("teacher_model"),
        )
        response_mask = record["response_mask"]
        response_attention = record["response_attention_mask"]
        distill_mask = record["distill_mask"]
        teacher = record["teacher_log_probs"]
        observations["response_tokens"] += int(sum(sum(row) for row in response_mask))

        for row_index, (loss_row, attention_row, distill_row) in enumerate(
            zip(response_mask, response_attention, distill_mask)
        ):
            for token_index, (loss_value, attention_value, distill_value) in enumerate(
                zip(loss_row, attention_row, distill_row)
            ):
                if bool(loss_value) and not bool(attention_value):
                    errors.append(f"{prefix}: response mask selected PAD [{row_index},{token_index}]")
                if bool(distill_value) and not bool(loss_value):
                    errors.append(f"{prefix}: distill mask escaped response [{row_index},{token_index}]")

        # Reconstruct the independent teacher objectives before branching on
        # warmup. This covers both teacher-only warmup updates and joint
        # post-warmup updates rather than trusting a scalar copied from runtime.
        expected_sft_loss = 0.0
        teacher_sft_mask = record.get("teacher_sft_mask")
        if expectation["teacher_objective"] == "sft" and teacher_sft_mask is None:
            errors.append(f"{prefix}: expected teacher SFT tensors")
        if expectation["teacher_objective"] in {"none", "rlvr"} and teacher_sft_mask is not None:
            errors.append(f"{prefix}: unexpected teacher SFT tensors")
        if teacher_sft_mask is not None:
            observations["teacher_sft_tokens"] += int(
                sum(sum(row) for row in teacher_sft_mask)
            )
            expected_sft_nll = [[-value for value in row] for row in teacher]
            compare_matrix(
                errors,
                f"{prefix}:teacher_sft_token_nll",
                record.get("teacher_sft_token_nll"),
                expected_sft_nll,
                atol=1e-6,
                rtol=1e-6,
            )
            expected_sft_loss = aggregate(
                expected_sft_nll,
                teacher_sft_mask,
                record["teacher_sft_global_info"],
                record["loss_agg_mode"],
            )
            require_close(
                errors,
                f"{prefix}:teacher_sft_loss",
                record.get("teacher_sft_loss_production"),
                expected_sft_loss,
                atol=2e-5,
                rtol=1e-5,
            )
        else:
            require_equal(errors, f"{prefix}:teacher_sft_token_nll", record.get("teacher_sft_token_nll"), None)
            require_equal(errors, f"{prefix}:teacher_sft_loss", record.get("teacher_sft_loss_production"), None)

        expected_teacher_loss = 0.0
        if record.get("teacher_rlvr_loss_production") is not None:
            observations["teacher_rlvr_microbatches"] += 1
            teacher_is_mode = record.get("teacher_is_mode")
            teacher_is_modes.add(teacher_is_mode)
            if expectation["is_mode"] is not None:
                require_equal(
                    errors,
                    f"{prefix}:teacher_is_mode",
                    teacher_is_mode,
                    expectation["is_mode"],
                )
            observations["teacher_rlvr_nonzero_advantage_tokens"] += sum(
                int(bool(mask_value) and abs(float(advantage_value)) > 0.0)
                for advantage_row, mask_row in zip(record["advantages"], response_mask)
                for advantage_value, mask_value in zip(advantage_row, mask_row)
            )
            expected_is = teacher_is_weights(
                record["teacher_old_log_probs"],
                record["student_behavior_log_probs"],
                response_attention,
                record["teacher_is_mode"],
                record["teacher_is_upper_clip"],
            )
            if expected_is is None:
                require_equal(errors, f"{prefix}:teacher_is_weights", record.get("is_weights"), None)
            else:
                compare_matrix(
                    errors,
                    f"{prefix}:teacher_is_weights",
                    record.get("is_weights"),
                    expected_is,
                    atol=2e-5,
                    rtol=1e-5,
                )
                for weight_row, attention_row in zip(
                    record.get("is_weights") or [], response_attention
                ):
                    for weight_value, attention_value in zip(weight_row, attention_row):
                        if bool(attention_value):
                            if abs(float(weight_value) - 1.0) > 1e-6:
                                observations["teacher_is_nonunit_tokens"] += 1
                        else:
                            observations["teacher_is_pad_abs_max"] = max(
                                observations["teacher_is_pad_abs_max"],
                                abs(float(weight_value)),
                            )
            expected_teacher_terms = ppo_terms(
                record["teacher_old_log_probs"],
                teacher,
                record["advantages"],
                float(record["ppo_clip_ratio_low"]),
                float(record["ppo_clip_ratio_high"]),
                float(record["ppo_clip_ratio_c"]),
                expected_is,
            )
            for name, matrix in expected_teacher_terms.items():
                compare_matrix(
                    errors,
                    f"{prefix}:teacher_ppo:{name}",
                    record.get("teacher_ppo_terms", {}).get(name),
                    matrix,
                    atol=2e-5,
                    rtol=1e-5,
                )
            expected_teacher_loss = aggregate(
                expected_teacher_terms["teacher_pg_after_is"],
                response_mask,
                record["rlvr_global_info"],
                record["loss_agg_mode"],
            )
            require_close(
                errors,
                f"{prefix}:teacher_rlvr_loss",
                record.get("teacher_rlvr_loss_production"),
                expected_teacher_loss,
                atol=2e-5,
                rtol=1e-5,
            )
        else:
            require_equal(errors, f"{prefix}:teacher_ppo_terms", record.get("teacher_ppo_terms"), None)
            if expectation["teacher_objective"] == "rlvr":
                errors.append(f"{prefix}: expected a teacher RLVR objective")

        if expectation["teacher_objective"] in {"none", "sft"}:
            require_equal(
                errors,
                f"{prefix}:teacher_rlvr_loss",
                record.get("teacher_rlvr_loss_production"),
                None,
            )

        expected_teacher_objective = (
            float(record["teacher_sft_weight"]) * expected_sft_loss
            + float(record["teacher_rlvr_weight"]) * expected_teacher_loss
        )
        routing = record.get("gradient_routing", {})
        teacher_objective_active = (
            record.get("teacher_rlvr_loss_production") is not None
            and float(record["teacher_rlvr_weight"]) != 0.0
        ) or (
            record.get("teacher_sft_loss_production") is not None
            and float(record["teacher_sft_weight"]) != 0.0
            and int(record["teacher_sft_global_info"]["batch_num_tokens"]) > 0
        )
        if teacher_objective_active and config.get("teacher_model") == "separate":
            route_prefix = (
                "teacher_joint"
                if float(config.get("teacher_sft_weight", 0.0) or 0.0) > 0.0
                else "teacher_rlvr"
            )
            actor_teacher_route = routing.get(f"{route_prefix}_to_actor")
            teacher_route = routing.get(f"{route_prefix}_to_teacher")
            if actor_teacher_route is None or int(actor_teacher_route.get("hook_calls", -1)) != 0:
                errors.append(
                    f"{prefix}: independent teacher objective reached actor: {actor_teacher_route}"
                )
            if teacher_route is None or int(teacher_route.get("hook_calls", 0)) <= 0:
                errors.append(
                    f"{prefix}: missing teacher gradient route for independent objective: {teacher_route}"
                )

        if warmup:
            observations["warmup_microbatches"] += 1
            require_equal(errors, f"{prefix}:shaped_advantages", record.get("advantages_shaped"), None)
            require_equal(errors, f"{prefix}:student_loss", record.get("student_rlvr_loss_production"), None)
            require_close(
                errors,
                f"{prefix}:student_weight",
                record.get("student_rlvr_weight"),
                0.0,
                atol=0.0,
                rtol=0.0,
            )
            require_close(
                errors,
                f"{prefix}:warmup_weighted_total",
                record.get("total_weighted_loss_production"),
                expected_teacher_objective,
                atol=3e-5,
                rtol=1e-5,
            )
            require_equal(
                errors,
                f"{prefix}:warmup_reverse_kl_token_estimate",
                record.get("reverse_kl_token_estimate"),
                None,
            )
            continue

        observations["shaped_microbatches"] += 1
        actor = record["actor_log_probs"]
        advantages = record["advantages"]
        expected_reverse_kl = [
            [actor_value - teacher_value for actor_value, teacher_value in zip(actor_row, teacher_row)]
            for actor_row, teacher_row in zip(actor, teacher)
        ]
        # Older advantage-shaping ledgers intentionally left this diagnostic
        # empty. New ledgers must be checked whenever the field is present.
        reverse_kl_recorded = record.get("reverse_kl_token_estimate") is not None
        if config.get("token_kl_logging", {}).get("enabled", False) and not reverse_kl_recorded:
            errors.append(f"{prefix}: enabled token reverse-KL logging produced no token estimate")
        if reverse_kl_recorded:
            compare_matrix(
                errors,
                f"{prefix}:reverse_kl_token_estimate",
                record.get("reverse_kl_token_estimate"),
                expected_reverse_kl,
                atol=1e-6,
                rtol=1e-6,
            )
            expected_reverse_kl_scalar = aggregate(
                expected_reverse_kl,
                distill_mask,
                record["distill_global_info"],
                record["loss_agg_mode"],
            )
            require_close(
                errors,
                f"{prefix}:reverse_kl_estimate_production",
                record.get("reverse_kl_estimate_production"),
                expected_reverse_kl_scalar,
                atol=2e-5,
                rtol=1e-5,
            )
            step_rank = (step, int(record["rank"]))
            reverse_kl_by_step_rank[step_rank] = (
                reverse_kl_by_step_rank.get(step_rank, 0.0) + expected_reverse_kl_scalar
            )
        expected_evidence = [
            [teacher_value - actor_value for teacher_value, actor_value in zip(teacher_row, actor_row)]
            for teacher_row, actor_row in zip(teacher, actor)
        ]
        compare_matrix(
            errors,
            f"{prefix}:teacher_evidence",
            record.get("teacher_evidence_scores"),
            expected_evidence,
            atol=1e-6,
            rtol=1e-6,
        )

        base_mask = distill_mask if shaping_config.get("use_distill_mask", True) else response_mask
        cap = shaping_config.get("max_response_tokens")
        expected_mask = [
            [
                float(bool(value) and (cap is None or token_index < int(cap)))
                for token_index, value in enumerate(row)
            ]
            for row in base_mask
        ]
        compare_matrix(
            errors,
            f"{prefix}:shaping_mask",
            record.get("advantage_shaping_mask"),
            expected_mask,
            atol=0.0,
            rtol=0.0,
        )
        observations["shaped_tokens"] += int(sum(sum(row) for row in expected_mask))

        expected_shaped = []
        expected_z = []
        expected_delta = []
        effective_scales = []
        correction_indices = []
        for advantage_row, evidence_row, response_row, shaping_row in zip(
            advantages, expected_evidence, response_mask, expected_mask
        ):
            shaped_row, z_row, delta_row, effective_scale, correction_index = reshape_row(
                advantage_row,
                evidence_row,
                response_row,
                shaping_row,
                shaping_config,
            )
            expected_shaped.append(shaped_row)
            expected_z.append(z_row)
            expected_delta.append(delta_row)
            effective_scales.append(effective_scale)
            correction_indices.append(correction_index)
        actual_shaped = record.get("advantages_shaped")
        if not isinstance(actual_shaped, list):
            errors.append(f"{prefix}: missing shaped-advantage ledger matrix")
            continue
        if len(actual_shaped) != len(expected_shaped):
            errors.append(
                f"{prefix}:shaped_advantages: row count "
                f"{len(actual_shaped)} != {len(expected_shaped)}"
            )
        else:
            correction_magnitudes = []
            for row_index, (
                actual_row,
                expected_row,
                original_row,
                response_row,
                shaping_row,
                correction_index,
            ) in enumerate(
                zip(
                    actual_shaped,
                    expected_shaped,
                    advantages,
                    response_mask,
                    expected_mask,
                    correction_indices,
                )
            ):
                if len(actual_row) != len(expected_row):
                    errors.append(
                        f"{prefix}:shaped_advantages[{row_index}]: length "
                        f"{len(actual_row)} != {len(expected_row)}"
                    )
                    continue
                for token_index, (actual_value, expected_value) in enumerate(
                    zip(actual_row, expected_row)
                ):
                    if token_index == correction_index:
                        continue
                    require_close(
                        errors,
                        f"{prefix}:shaped_advantages[{row_index},{token_index}]",
                        actual_value,
                        expected_value,
                        atol=2e-5,
                        rtol=1e-5,
                    )
                if correction_index is None:
                    continue
                if not bool(shaping_row[correction_index]) or not bool(
                    response_row[correction_index]
                ):
                    errors.append(
                        f"{prefix}: correction index {correction_index} is not "
                        f"an actual selected response token in row {row_index}"
                    )
                    continue
                # Once all non-correction tokens match the independently
                # reconstructed redistribution, conservation uniquely fixes
                # the correction token. This checks the production float32
                # correction without pretending Python's binary64 reduction
                # reproduces its long-reduction residual bit for bit.
                other_response_delta = sum(
                    (actual_value - original_value) * response_token
                    for token_index, (actual_value, original_value, response_token) in enumerate(
                        zip(actual_row, original_row, response_row)
                    )
                    if token_index != correction_index
                )
                conservation_fixed_value = (
                    original_row[correction_index] - other_response_delta
                )
                require_close(
                    errors,
                    f"{prefix}:shaped_advantages_correction"
                    f"[{row_index},{correction_index}]",
                    actual_row[correction_index],
                    conservation_fixed_value,
                    atol=1e-6,
                    rtol=0.0,
                )
                correction_magnitudes.append(
                    abs(
                        float(actual_row[correction_index])
                        - float(expected_row[correction_index])
                    )
                )
                observations["correction_tokens_verified"] += 1
            correction_metric = record.get("micro_metrics", {}).get(
                "actor/advantage_shaping_correction_abs_max"
            )
            if correction_metric is None:
                errors.append(f"{prefix}: missing per-microbatch correction metric")
            else:
                require_close(
                    errors,
                    f"{prefix}:correction_token_magnitude",
                    max(correction_magnitudes, default=0.0),
                    correction_metric,
                    atol=1e-6,
                    rtol=1e-5,
                )

        max_conservation = 0.0
        max_outside = 0.0
        max_pad = 0.0
        max_sign_flip = 0.0
        for original_row, shaped_row, shape_row, response_row in zip(
            advantages, actual_shaped, expected_mask, response_mask
        ):
            response_delta = sum(
                (shaped - original) * response_token
                for original, shaped, response_token in zip(
                    original_row, shaped_row, response_row
                )
            )
            max_conservation = max(max_conservation, abs(response_delta))
            response_values = [
                original
                for original, response_token in zip(original_row, response_row)
                if bool(response_token)
            ]
            sequence_advantage = (
                sum(response_values) / len(response_values) if response_values else 0.0
            )
            for original, shaped, shaped_token, response_token in zip(
                original_row, shaped_row, shape_row, response_row
            ):
                delta_value = shaped - original
                if bool(shaped_token) and abs(delta_value) > 1e-8:
                    observations["nonzero_shaping_delta_tokens"] += 1
                if not bool(shaped_token):
                    max_outside = max(max_outside, abs(delta_value))
                if not bool(response_token):
                    max_pad = max(max_pad, abs(delta_value))
                if bool(shaped_token):
                    if sequence_advantage > 0.0 and shaped < 0.0:
                        max_sign_flip = max(max_sign_flip, abs(shaped))
                    if sequence_advantage < 0.0 and shaped > 0.0:
                        max_sign_flip = max(max_sign_flip, abs(shaped))
        observations["max_conservation_error"] = max(
            observations["max_conservation_error"], max_conservation
        )
        observations["max_outside_mask_delta"] = max(
            observations["max_outside_mask_delta"], max_outside
        )
        observations["max_pad_delta"] = max(observations["max_pad_delta"], max_pad)
        if max_conservation > 1e-6:
            errors.append(f"{prefix}: conservation error {max_conservation}")
        if max_outside != 0.0 or max_pad != 0.0:
            errors.append(f"{prefix}: changed outside={max_outside} pad={max_pad}")
        if not shaping_config.get("allow_token_sign_flip", False) and max_sign_flip != 0.0:
            errors.append(f"{prefix}: verifier-advantage sign flip magnitude={max_sign_flip}")

        terms = ppo_terms(
            record["student_behavior_log_probs"],
            actor,
            actual_shaped,
            float(record["ppo_clip_ratio_low"]),
            float(record["ppo_clip_ratio_high"]),
            float(record["ppo_clip_ratio_c"]),
        )
        for name, matrix in terms.items():
            compare_matrix(
                errors,
                f"{prefix}:student_ppo:{name}",
                record.get("student_ppo_terms", {}).get(name),
                matrix,
                atol=2e-5,
                rtol=1e-5,
            )
        expected_student_loss = aggregate(
            terms["teacher_pg_after_is"],
            response_mask,
            record["rlvr_global_info"],
            record["loss_agg_mode"],
        )
        require_close(
            errors,
            f"{prefix}:student_rlvr_loss",
            record.get("student_rlvr_loss_production"),
            expected_student_loss,
            atol=2e-5,
            rtol=1e-5,
        )
        require_close(errors, f"{prefix}:distill_weight", record.get("distill_weight"), 0.0, atol=0.0, rtol=0.0)
        require_equal(errors, f"{prefix}:distill_loss", record.get("distill_loss_production"), None)

        expected_total = (
            float(record["student_rlvr_weight"]) * expected_student_loss
            + expected_teacher_objective
        )
        require_close(
            errors,
            f"{prefix}:weighted_total",
            record.get("total_weighted_loss_production"),
            expected_total,
            atol=3e-5,
            rtol=1e-5,
        )

        actor_route = routing.get("student_shaped_rlvr_to_actor")
        teacher_route = routing.get("student_shaped_rlvr_to_teacher")
        if actor_route is None or int(actor_route.get("hook_calls", 0)) <= 0:
            errors.append(f"{prefix}: missing actor gradient route for shaped student RLVR")
        if config.get("teacher_model") == "separate":
            if teacher_route is None:
                errors.append(f"{prefix}: missing isolated teacher gradient audit")
            elif int(teacher_route.get("hook_calls", -1)) != 0:
                errors.append(f"{prefix}: shaped student RLVR reached teacher: {teacher_route}")

        metrics = record.get("micro_metrics", {})
        for name, expected in (
            ("actor/advantage_shaping_prompt_token_count", 0.0),
            ("actor/advantage_shaping_pad_token_count", 0.0),
            ("actor/advantage_shaping_outside_mask_delta_max", 0.0),
            ("actor/advantage_shaping_pad_delta_max", 0.0),
            ("actor/advantage_shaping_neg_to_pos_rate", 0.0),
            ("actor/advantage_shaping_pos_to_neg_rate", 0.0),
            ("actor/opsd_distill_active_rate", 0.0),
            ("actor/opsd_distill_weight", 0.0),
        ):
            require_close(errors, f"{prefix}:{name}", metrics.get(name), expected, atol=1e-8, rtol=0.0)
        sequence_deviation = metrics.get(
            "actor/advantage_shaping_sequence_advantage_deviation_max"
        )
        if sequence_deviation is None or float(sequence_deviation) > 1e-6:
            errors.append(
                f"{prefix}: sequence-level GRPO advantage deviation={sequence_deviation}"
            )
        pre_correction_error = metrics.get(
            "actor/advantage_shaping_pre_correction_error_max"
        )
        correction_abs = metrics.get("actor/advantage_shaping_correction_abs_max")
        if pre_correction_error is None or correction_abs is None:
            errors.append(f"{prefix}: missing conservation-correction metrics")
        else:
            observations["max_pre_correction_error"] = max(
                observations["max_pre_correction_error"],
                abs(float(pre_correction_error)),
            )
            require_close(
                errors,
                f"{prefix}:correction_matches_pre_error",
                correction_abs,
                pre_correction_error,
                atol=1e-8,
                rtol=1e-6,
            )
        require_close(
            errors,
            f"{prefix}:conservation_metric",
            metrics.get("actor/advantage_shaping_total_error_max"),
            max_conservation,
            atol=1e-7,
            rtol=1e-5,
        )
        require_close(
            errors,
            f"{prefix}:response_conservation_metric",
            metrics.get("actor/advantage_shaping_response_total_error_max"),
            max_conservation,
            atol=1e-7,
            rtol=1e-5,
        )

        for local_index, sample in enumerate(record["sample_indices"]):
            source = trainer_records.get((step, int(sample)))
            if source is None:
                errors.append(f"{prefix}: missing trainer record for sample={sample}")
                continue
            require_equal(
                errors,
                f"{prefix}:sample={sample}:response_ids",
                record["response_ids"][local_index],
                source["response_ids"],
            )
            require_equal(
                errors,
                f"{prefix}:sample={sample}:response_mask",
                record["response_mask"][local_index],
                source["response_mask"],
            )

    optimizer_by_key = {}
    for path, optimizer in optimizer_records:
        key = (int(optimizer["global_step"]), int(optimizer["rank"]))
        if key in optimizer_by_key:
            errors.append(f"{path}: duplicate optimizer record for step/rank={key}")
        optimizer_by_key[key] = (path, optimizer)

    summary_by_key = {}
    for path, summary in summaries:
        key = (int(summary["global_step"]), int(summary["rank"]))
        if key in summary_by_key:
            errors.append(f"{path}: duplicate update summary for step/rank={key}")
        summary_by_key[key] = (path, summary)

        step = key[0]
        prefix = f"{path}:step={step}:rank={key[1]}"
        warmup = bool(summary.get("warmup_active"))
        if warmup:
            observed_warmup_steps.add(step)
        else:
            observed_joint_steps.add(step)
        if profile in {"shared", "separate_sft", "separate_rlvr"} and warmup:
            errors.append(f"{prefix}: unexpected warmup update for profile={profile}")
        if step in expected_warmup_steps and not warmup:
            errors.append(f"{prefix}: expected warmup_active=true")
        if step in expected_joint_steps and warmup:
            errors.append(f"{prefix}: expected warmup_active=false")
        if summary.get("audit_status") != "PASS":
            errors.append(f"{prefix}: summary audit_status={summary.get('audit_status')!r}")
        if summary.get("advantage_shaping_enabled") is not True:
            errors.append(f"{prefix}: summary missing advantage_shaping_enabled")
        require_equal(
            errors,
            f"{prefix}:optimizer_before",
            summary.get("optimizer_updates_completed_before_step"),
            step - 1,
        )
        require_equal(
            errors,
            f"{prefix}:optimizer_after",
            summary.get("optimizer_updates_completed_after_step"),
            step,
        )
        require_equal(
            errors,
            f"{prefix}:trainer_before",
            summary.get("trainer_updates_completed_before_step"),
            step - 1,
        )
        require_equal(
            errors,
            f"{prefix}:trainer_after",
            summary.get("trainer_updates_completed_after_step"),
            step,
        )
        global_metrics = summary.get("global_audit_metrics", {})
        step_rank_values = [
            value
            for (value_step, _), value in reverse_kl_by_step_rank.items()
            if value_step == step
        ]
        expected_global_reverse_kl = (
            sum(step_rank_values) / len(step_rank_values) if step_rank_values else 0.0
        )
        require_close(
            errors,
            f"{prefix}:reverse_kl_estimate",
            global_metrics.get("reverse_kl_estimate"),
            expected_global_reverse_kl,
            atol=2e-5,
            rtol=1e-5,
        )
        require_close(
            errors,
            f"{prefix}:reverse_kl_surrogate",
            global_metrics.get("reverse_kl_surrogate"),
            0.0,
            atol=0.0,
            rtol=0.0,
        )
        if expectation["teacher_objective"] == "sft":
            require_equal(errors, f"{prefix}:teacher_sft_enabled", summary.get("teacher_sft_enabled"), True)
        elif expectation["teacher_objective"] in {"none", "rlvr"}:
            require_equal(errors, f"{prefix}:teacher_sft_enabled", summary.get("teacher_sft_enabled"), False)

    for key, (path, optimizer) in optimizer_by_key.items():
        step, rank = key
        prefix = f"{path}:step={step}:rank={rank}"
        summary_entry = summary_by_key.get(key)
        if summary_entry is None:
            errors.append(f"{prefix}: missing matching update summary")
            continue
        _, summary = summary_entry
        require_equal(
            errors,
            f"{prefix}:optimizer_before",
            optimizer.get("optimizer_updates_completed_before_step"),
            step - 1,
        )
        require_equal(
            errors,
            f"{prefix}:trainer_before",
            optimizer.get("trainer_updates_completed_before_step"),
            step - 1,
        )
        for branch in ("actor", "teacher"):
            attempted = bool(optimizer.get(f"{branch}_attempted"))
            did_step = bool(optimizer.get(f"{branch}_did_step"))
            require_equal(
                errors,
                f"{prefix}:{branch}_summary_attempted",
                summary.get(f"{branch}_step_attempted"),
                attempted,
            )
            require_equal(
                errors,
                f"{prefix}:{branch}_summary_did_step",
                summary.get(f"{branch}_did_step"),
                did_step,
            )
            grad_norm = optimizer.get(f"{branch}_grad_norm")
            if attempted:
                if not did_step:
                    errors.append(f"{prefix}: {branch} optimizer was attempted but did not step")
                if grad_norm is None or not math.isfinite(float(grad_norm)) or float(grad_norm) < 0.0:
                    errors.append(f"{prefix}: invalid {branch} grad norm {grad_norm!r}")
                elif float(grad_norm) > 0.0:
                    observations[f"positive_{branch}_grad_records"] += 1
                if optimizer.get(f"{branch}_probe_before") is None:
                    errors.append(f"{prefix}: missing {branch} pre-step parameter probe")
                if optimizer.get(f"{branch}_probe_after") is None:
                    errors.append(f"{prefix}: missing {branch} post-step parameter probe")
            else:
                require_equal(errors, f"{prefix}:{branch}_did_step", did_step, False)
                require_equal(errors, f"{prefix}:{branch}_grad_norm", grad_norm, None)
            if did_step:
                observations[f"{branch}_optimizer_steps"] += 1

        warmup = bool(summary.get("warmup_active"))
        expected_actor_step = not warmup
        require_equal(
            errors,
            f"{prefix}:actor_attempted_contract",
            bool(optimizer.get("actor_attempted")),
            expected_actor_step,
        )
        teacher_objective = expectation["teacher_objective"]
        if teacher_objective is not None:
            if teacher_objective == "none":
                expected_teacher_step = False
            elif teacher_objective == "rlvr":
                expected_teacher_step = True
            else:
                expected_teacher_step = int(
                    summary.get("global_audit_metrics", {}).get("teacher_sft_tokens", 0)
                ) > 0
            require_equal(
                errors,
                f"{prefix}:teacher_attempted_contract",
                bool(optimizer.get("teacher_attempted")),
                expected_teacher_step,
            )

    missing_optimizer_summaries = sorted(set(summary_by_key) - set(optimizer_by_key))
    for key in missing_optimizer_summaries:
        errors.append(f"missing optimizer record for step/rank={key}")

    if not microbatch_records:
        errors.append("no microbatch audit records found")
    if observations["shaped_microbatches"] <= 0:
        errors.append("no post-warmup advantage-shaping microbatch found")
    if observations["shaped_tokens"] <= 0:
        errors.append("advantage shaping selected zero response tokens")
    if observations["nonzero_shaping_delta_tokens"] <= 0:
        errors.append(
            "all audited GRPO groups had zero advantage; no nonzero shaping delta was exercised"
        )
    if observations["prompt_tokens_examined"] <= 0:
        errors.append("audit did not establish a non-empty prompt region")
    if observations["actor_prompt_valid_tokens_examined"] <= 0:
        errors.append("audit did not observe a valid actor prompt token")
    if observations["teacher_prompt_valid_tokens_examined"] <= 0:
        errors.append("audit did not observe a valid teacher prompt token")
    if observations["pad_tokens_examined"] <= 0:
        errors.append("audit did not observe any natural right-padding tokens")
    if not optimizer_records:
        errors.append("no optimizer-step audit records found")
    if not summaries:
        errors.append("no update-summary audit records found")
    if required_response_axis is not None and response_axis_lengths != {required_response_axis}:
        errors.append(
            f"response axes {sorted(response_axis_lengths)} != required [{required_response_axis}]"
        )
    if expected_warmup_steps and not expected_warmup_steps.issubset(observed_warmup_steps):
        errors.append(
            f"missing expected warmup steps {sorted(expected_warmup_steps - observed_warmup_steps)}"
        )
    if expected_joint_steps and not expected_joint_steps.issubset(observed_joint_steps):
        errors.append(
            f"missing expected joint steps {sorted(expected_joint_steps - observed_joint_steps)}"
        )
    if expectation["teacher_objective"] == "sft" and observations["teacher_sft_tokens"] <= 0:
        errors.append("teacher SFT profile audited no successful-rollout target token")
    if expectation["teacher_objective"] == "rlvr":
        if observations["teacher_rlvr_microbatches"] <= 0:
            errors.append("teacher RLVR profile audited no teacher RLVR microbatch")
        if observations["teacher_rlvr_nonzero_advantage_tokens"] <= 0:
            errors.append("teacher RLVR profile audited no nonzero verifier-advantage token")
        if teacher_is_modes != {expectation["is_mode"]}:
            errors.append(
                f"teacher IS modes {sorted(teacher_is_modes)} != [{expectation['is_mode']!r}]"
            )
        if observations["teacher_is_nonunit_tokens"] <= 0:
            errors.append("token-IS path produced no non-unit response-token weight")
        if observations["teacher_is_pad_abs_max"] != 0.0:
            errors.append(
                f"token-IS assigned nonzero PAD weight {observations['teacher_is_pad_abs_max']}"
            )
    if expectation["require_positive_actor_grad"] and observations["positive_actor_grad_records"] <= 0:
        errors.append("profile did not exercise a positive shaped-student actor gradient")
    if expectation["require_positive_teacher_grad"] and observations["positive_teacher_grad_records"] <= 0:
        errors.append("profile did not exercise a positive independent-teacher gradient")

    observations["response_axis_lengths"] = sorted(response_axis_lengths)
    observations["teacher_models"] = sorted(str(value) for value in teacher_models)
    observations["teacher_is_modes"] = sorted(str(value) for value in teacher_is_modes)
    observations["warmup_steps"] = sorted(observed_warmup_steps)
    observations["joint_steps"] = sorted(observed_joint_steps)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "audit_dir": str(audit_dir),
        "profile": profile,
        "expectations": {
            **expectation,
            "warmup_steps": sorted(expected_warmup_steps),
            "joint_steps": sorted(expected_joint_steps),
            "required_response_axis": required_response_axis,
        },
        "errors": errors,
        "observations": observations,
        "layout_files": len(configs),
        "microbatch_records": len(microbatch_records),
        "optimizer_records": len(optimizer_records),
        "summary_records": len(summaries),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_EXPECTATIONS),
        default="generic",
        help="Fail-closed runtime contract to verify in addition to the arithmetic ledger.",
    )
    parser.add_argument(
        "--expect-warmup-step",
        action="append",
        type=int,
        default=[],
        help="Audited global step that must be teacher-only warmup; may be repeated.",
    )
    parser.add_argument(
        "--expect-joint-step",
        action="append",
        type=int,
        default=[],
        help="Audited global step that must update the shaped student; may be repeated.",
    )
    parser.add_argument(
        "--require-response-axis",
        type=int,
        help="Require every audited padded response tensor to have this exact width.",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    report = verify(
        audit_dir,
        profile=args.profile,
        expected_warmup_steps=tuple(args.expect_warmup_step),
        expected_joint_steps=tuple(args.expect_joint_step),
        required_response_axis=args.require_response_axis,
    )
    # Keep the established smoke-test artifact names so the common post-check
    # cannot accidentally inspect an older verifier's report.
    report_json = args.report_json or audit_dir / "verification_report.json"
    report_md = args.report_md or audit_dir / "verification_report.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        WIDE_CSS,
        "",
        "# OPSD advantage-shaping audit verification",
        "",
        f"Status: **{report['status']}**",
        "",
        "```json",
        json.dumps(report["observations"], indent=2, sort_keys=True),
        "```",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in report["errors"])
    if not report["errors"]:
        lines.append("None.")
    report_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
