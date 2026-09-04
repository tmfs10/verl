# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation-only random-prefix continuation baseline for the Ray PPO driver."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.experimental.agent_loop.random_continuation_agent_loop import (
    RANDOM_CONTINUATION_AGENT_NAME,
    RANDOM_CONTINUATION_RECORD_FIELD,
    RandomContinuationGeneration,
    RandomContinuationRecord,
)
from verl.trainer.config import RandomContinuationBaselineConfig
from verl.trainer.ppo.branch_revision_grpo import (
    branch_prefix_open_block_reason,
    decode_exact,
    strip_terminal_eos,
    validate_binary_reward_row,
)
from verl.trainer.ppo.random_continuation_baseline import RandomMarkSelection, clustered_rate, descriptive
from verl.trainer.ppo.reward import compute_reward
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.model import compute_position_id_with_mask


def validate_random_continuation_runtime_config(config) -> None:
    feature = omega_conf_to_dataclass(
        config.algorithm.random_continuation_baseline,
        dataclass_type=RandomContinuationBaselineConfig,
    )
    if not feature.enable:
        return
    if bool(OmegaConf.select(config, "algorithm.intermediate_mc_value.enable", default=False)):
        raise ValueError("random-continuation baseline and intermediate MC are mutually exclusive")
    if bool(OmegaConf.select(config, "algorithm.branch_revision_grpo.enable", default=False)):
        raise ValueError("random-continuation baseline and branch revision are mutually exclusive")
    if config.actor_rollout_ref.rollout.name != "vllm":
        raise ValueError("random-continuation baseline requires the vLLM rollout engine")
    if int(config.actor_rollout_ref.rollout.n) != 1:
        raise ValueError("random-continuation baseline requires actor_rollout_ref.rollout.n=1")
    exact_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }
    for name, expected in exact_sampling.items():
        actual = config.actor_rollout_ref.rollout.get(name)
        if actual != expected:
            raise ValueError(f"random-continuation baseline requires rollout.{name}={expected}, got {actual!r}")
    if config.critic.get("enable", None) is not False:
        raise ValueError("random-continuation baseline requires critic.enable=false")
    if bool(config.algorithm.use_kl_in_reward):
        raise ValueError("random-continuation baseline requires algorithm.use_kl_in_reward=false")
    if bool(config.actor_rollout_ref.actor.use_kl_loss):
        raise ValueError("random-continuation baseline requires actor.use_kl_loss=false")
    if str(config.algorithm.adv_estimator).lower() != "grpo":
        raise ValueError("random-continuation baseline requires algorithm.adv_estimator=grpo")
    if bool(OmegaConf.select(config, "reward.reward_model.launch_reward_fn_async", default=False)):
        raise ValueError("random-continuation baseline requires synchronous reward evaluation")
    if config.trainer.total_training_steps != 1:
        raise ValueError("random-continuation baseline requires trainer.total_training_steps=1")
    if int(config.trainer.save_freq) != -1 or int(config.trainer.test_freq) != -1:
        raise ValueError("random-continuation baseline requires save_freq=test_freq=-1")
    if bool(config.trainer.val_before_train):
        raise ValueError("random-continuation baseline requires trainer.val_before_train=false")
    if not feature.structural_boundaries_only:
        raise ValueError("this baseline must match production structural-boundary filtering")
    response_length = int(config.data.max_response_length)
    if feature.min_continuation_tokens >= response_length:
        raise ValueError("min_continuation_tokens must be smaller than data.max_response_length")
    if not feature.audit_output_dir:
        raise ValueError("random-continuation baseline requires audit_output_dir")
    if str(config.actor_rollout_ref.rollout.get("logprobs_mode", "")) != "processed_logprobs":
        raise ValueError("random-continuation baseline requires rollout.logprobs_mode=processed_logprobs")
    if bool(config.actor_rollout_ref.rollout.multi_turn.enable):
        raise ValueError("random-continuation baseline supports only single-turn rollouts")
    if bool(config.reward.reward_model.enable):
        raise ValueError("random-continuation baseline requires a synchronous rule-based reward")
    if bool(OmegaConf.select(config, "algorithm.opsd.enable", default=False)):
        raise ValueError("random-continuation baseline and OPSD training are mutually exclusive")
    with open_dict(config):
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.actor.use_rollout_log_probs = True


@dataclass
class _Original:
    row: int
    dataset_index: object
    prompt_group_id: str
    rollout_id: str
    prompt_ids: list[int]
    solution_ids: list[int]
    editable_solution_ids: list[int]
    original_reward: float
    record: RandomContinuationRecord


class RandomContinuationBaselineController:
    """Own random-continuation evaluation and evidence; never update a model."""

    evaluation_only = True
    _REWARD_BATCH_SIZE = 256

    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.tokenizer = trainer.tokenizer
        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.random_continuation_baseline,
            dataclass_type=RandomContinuationBaselineConfig,
        )
        if trainer.processor is not None:
            raise ValueError("random-continuation baseline supports text-only inputs")
        if trainer.reward_fn is None:
            raise ValueError("random-continuation baseline requires a synchronous reward function")
        audit_dir = os.path.abspath(os.path.expanduser(str(self.feature.audit_output_dir)))
        os.makedirs(audit_dir, exist_ok=True)
        self.audit_path = os.path.join(audit_dir, "random_continuation_baseline.jsonl")
        self.summary_path = os.path.join(audit_dir, "summary.json")
        if os.path.exists(self.audit_path) or os.path.exists(self.summary_path):
            raise FileExistsError(f"random-continuation evidence already exists under {audit_dir}")

    def _audit(self, event: str, **payload: object) -> None:
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            record = {"event": event, "global_step": self.trainer.global_steps, **payload}
            handle.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _object_array(values: list[Any]) -> np.ndarray:
        result = np.empty(len(values), dtype=object)
        result[:] = values
        return result

    def _pad_token_id(self) -> int:
        token_id = self.tokenizer.pad_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("random-continuation baseline requires a tokenizer pad or EOS token")
        return int(token_id)

    @staticmethod
    def _valid_prompt_ids(batch: DataProto, row: int) -> list[int]:
        width = batch.batch["prompts"].shape[1]
        mask = batch.batch["attention_mask"][row, :width].bool()
        return [int(token) for token in batch.batch["prompts"][row][mask].tolist()]

    @staticmethod
    def _valid_solution_ids(batch: DataProto, row: int) -> list[int]:
        mask = batch.batch["response_mask"][row].bool()
        length = int(mask.sum().item())
        if length <= 0 or not torch.all(mask[:length]) or torch.any(mask[length:]):
            raise ValueError("random-continuation baseline requires a nonempty contiguous response")
        return [int(token) for token in batch.batch["responses"][row, :length].tolist()]

    def prepare_generation_batch(self, batch: DataProto) -> None:
        if "index" not in batch.non_tensor_batch:
            raise ValueError("random-continuation baseline requires stable dataset indices")
        group_values = batch.non_tensor_batch.get("prompt_group_id", batch.non_tensor_batch.get("uid"))
        if group_values is None:
            group_values = np.arange(len(batch), dtype=object)
        counts: dict[str, int] = {}
        rollout_ids: list[str] = []
        for value in group_values:
            key = str(value)
            index = counts.get(key, 0)
            counts[key] = index + 1
            rollout_ids.append(f"{key}:{index}")
        batch.non_tensor_batch["agent_name"] = np.array(
            [RANDOM_CONTINUATION_AGENT_NAME] * len(batch), dtype=object
        )
        batch.non_tensor_batch["random_continuation_rollout_id"] = np.array(rollout_ids, dtype=object)

    @staticmethod
    def _coerce_generation(value: Any) -> RandomContinuationGeneration:
        if isinstance(value, RandomContinuationGeneration):
            return value
        return RandomContinuationGeneration(
            mark=int(value["mark"]),
            sample_index=int(value["sample_index"]),
            token_ids=tuple(int(token) for token in value["token_ids"]),
            log_probs=tuple(float(item) for item in value["log_probs"]),
            finish_reason=None if value.get("finish_reason") is None else str(value["finish_reason"]),
        )

    @classmethod
    def _coerce_record(cls, value: Any) -> RandomContinuationRecord:
        if isinstance(value, RandomContinuationRecord):
            return value
        selection = value["selection"]
        if not isinstance(selection, RandomMarkSelection):
            selection = RandomMarkSelection(
                marks=tuple(int(mark) for mark in selection["marks"]),
                candidate_low=int(selection["candidate_low"]),
                candidate_high=int(selection["candidate_high"]),
                inspected=int(selection["inspected"]),
                rejection_counts={str(k): int(v) for k, v in selection["rejection_counts"].items()},
            )
        return RandomContinuationRecord(
            rollout_id=str(value["rollout_id"]),
            editable_solution_length=int(value["editable_solution_length"]),
            selection=selection,
            continuations=tuple(cls._coerce_generation(item) for item in value["continuations"]),
            failures=tuple((int(a), int(b), str(c)) for a, b, c in value["failures"]),
        )

    def _extract_originals(self, source: DataProto, reward_tensor: torch.Tensor) -> list[_Original]:
        raw_records = source.non_tensor_batch.pop(RANDOM_CONTINUATION_RECORD_FIELD, None)
        if raw_records is None or len(raw_records) != len(source):
            raise RuntimeError("random-continuation agent did not return one record per original")
        dataset_values = source.non_tensor_batch.get("index", np.arange(len(source), dtype=object))
        group_values = source.non_tensor_batch.get("prompt_group_id", np.arange(len(source), dtype=object))
        rollout_values = source.non_tensor_batch.get("random_continuation_rollout_id")
        if rollout_values is None:
            raise RuntimeError("random-continuation rollout IDs were lost")
        originals: list[_Original] = []
        for row, raw_record in enumerate(raw_records):
            record = self._coerce_record(raw_record)
            rollout_id = str(rollout_values[row])
            if record.rollout_id != rollout_id:
                raise RuntimeError("random-continuation record changed rollout identity")
            prompt_ids = self._valid_prompt_ids(source, row)
            solution_ids = self._valid_solution_ids(source, row)
            editable = strip_terminal_eos(solution_ids, self.tokenizer)
            if len(editable) != record.editable_solution_length:
                raise RuntimeError("worker and driver disagree on terminal-EOS-stripped length")
            marks = list(record.selection.marks)
            if marks != sorted(set(marks)):
                raise RuntimeError("random marks must be sorted and unique")
            if len(marks) > self.feature.points_per_rollout:
                raise RuntimeError("worker selected too many random marks")
            for mark in marks:
                if not record.selection.candidate_low <= mark <= record.selection.candidate_high:
                    raise RuntimeError("random mark lies outside audited numeric eligibility bounds")
                if mark / len(editable) <= self.feature.min_prefix_fraction:
                    raise RuntimeError("random mark does not satisfy the strict prefix-fraction bound")
                reason = branch_prefix_open_block_reason(decode_exact(editable[:mark], self.tokenizer))
                if reason is not None:
                    raise RuntimeError(f"random mark violates production structural check: {reason}")
            generated_keys = {(item.mark, item.sample_index) for item in record.continuations}
            failed_keys = {(mark, sample) for mark, sample, _ in record.failures}
            expected_keys = {(mark, 0) for mark in marks}
            if generated_keys & failed_keys or generated_keys | failed_keys != expected_keys:
                raise RuntimeError("random continuation outcomes do not conserve selected marks")
            for item in record.continuations:
                if not item.token_ids or len(item.token_ids) != len(item.log_probs):
                    raise RuntimeError("random continuation token/logprob alignment is invalid")
                if not all(math.isfinite(value) for value in item.log_probs):
                    raise RuntimeError("random continuation contains non-finite log probabilities")
                if item.mark + len(item.token_ids) > int(self.config.data.max_response_length):
                    raise RuntimeError("random continuation exceeds the response budget")
            original_reward = validate_binary_reward_row(
                reward_tensor[row].detach().cpu().tolist(), tolerance=1e-6
            )
            originals.append(
                _Original(
                    row=row,
                    dataset_index=dataset_values[row],
                    prompt_group_id=str(group_values[row]),
                    rollout_id=rollout_id,
                    prompt_ids=prompt_ids,
                    solution_ids=solution_ids,
                    editable_solution_ids=editable,
                    original_reward=original_reward,
                    record=record,
                )
            )
        return originals

    def _make_reward_batch(
        self,
        source: DataProto,
        items: list[tuple[_Original, RandomContinuationGeneration]],
    ) -> DataProto:
        prompt_width = int(self.config.actor_rollout_ref.rollout.prompt_length)
        response_width = int(self.config.actor_rollout_ref.rollout.response_length)
        pad_id = self._pad_token_id()
        prompt_tensor = torch.full((len(items), prompt_width), pad_id, dtype=torch.long)
        response_tensor = torch.full((len(items), response_width), pad_id, dtype=torch.long)
        prompt_mask = torch.zeros_like(prompt_tensor)
        response_mask = torch.zeros_like(response_tensor)
        rows: list[int] = []
        for out_row, (original, continuation) in enumerate(items):
            response = [*original.editable_solution_ids[: continuation.mark], *continuation.token_ids]
            prompt_tensor[out_row, -len(original.prompt_ids) :] = torch.tensor(original.prompt_ids)
            prompt_mask[out_row, -len(original.prompt_ids) :] = 1
            response_tensor[out_row, : len(response)] = torch.tensor(response)
            response_mask[out_row, : len(response)] = 1
            rows.append(original.row)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        non_tensors = {
            key: np.take(values, rows, axis=0).copy()
            for key, values in source.non_tensor_batch.items()
            if key != RANDOM_CONTINUATION_RECORD_FIELD
        }
        return DataProto.from_dict(
            tensors={
                "prompts": prompt_tensor,
                "responses": response_tensor,
                "response_mask": response_mask,
                "input_ids": torch.cat([prompt_tensor, response_tensor], dim=1),
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
            },
            non_tensors=non_tensors,
        )

    def _evaluate(
        self,
        source: DataProto,
        originals: list[_Original],
    ) -> list[dict[str, Any]]:
        pairs = [(original, item) for original in originals for item in original.record.continuations]
        results: list[dict[str, Any]] = []
        for start in range(0, len(pairs), self._REWARD_BATCH_SIZE):
            chunk = pairs[start : start + self._REWARD_BATCH_SIZE]
            reward_batch = self._make_reward_batch(source, chunk)
            reward_tensor, _ = compute_reward(
                reward_batch,
                self.trainer.reward_fn,
                actor_wg=self.trainer.actor_rollout_wg,
            )
            if len(reward_tensor) != len(chunk):
                raise RuntimeError("random-continuation reward count mismatch")
            for (original, continuation), row in zip(chunk, reward_tensor.detach().cpu().tolist(), strict=True):
                reward = validate_binary_reward_row(row, tolerance=1e-6)
                results.append(
                    {
                        "original": original,
                        "continuation": continuation,
                        "reward": reward,
                    }
                )
        return results

    def _summary(self, originals: list[_Original], attempts: list[dict[str, Any]]) -> dict[str, Any]:
        by_prompt: dict[str, list[float]] = defaultdict(list)
        recovery: dict[str, list[float]] = defaultdict(list)
        retention: dict[str, list[float]] = defaultdict(list)
        deciles: dict[str, list[float]] = defaultdict(list)
        finish_reasons: Counter[str] = Counter()
        rejection_counts: Counter[str] = Counter()
        for original in originals:
            rejection_counts.update(original.record.selection.rejection_counts)
        for attempt in attempts:
            original = attempt["original"]
            continuation = attempt["continuation"]
            reward = float(attempt["reward"])
            by_prompt[original.prompt_group_id].append(reward)
            (retention if original.original_reward == 1.0 else recovery)[original.prompt_group_id].append(reward)
            fraction = continuation.mark / len(original.editable_solution_ids)
            lower = min(9, int(fraction * 10))
            deciles[f"{lower / 10:.1f}-{(lower + 1) / 10:.1f}"].append(reward)
            finish_reasons[str(continuation.finish_reason)] += 1
        seed = int(self.feature.selection_seed)
        original_rewards = [original.original_reward for original in originals]
        selected = sum(len(original.record.selection.marks) for original in originals)
        failed = sum(len(original.record.failures) for original in originals)
        prefix_fractions = [
            attempt["continuation"].mark / len(attempt["original"].editable_solution_ids) for attempt in attempts
        ]
        continuation_lengths = [len(attempt["continuation"].token_ids) for attempt in attempts]
        completed_lengths = [
            attempt["continuation"].mark + len(attempt["continuation"].token_ids) for attempt in attempts
        ]
        return {
            "schema_version": 1,
            "originals": len(originals),
            "original_successes": int(sum(original_rewards)),
            "original_pass_at_1": float(np.mean(original_rewards)) if original_rewards else None,
            "requested_attempts": len(originals) * self.feature.points_per_rollout,
            "selected_points": selected,
            "generated_attempts": len(attempts),
            "failed_generations": failed,
            "selection_shortfall": len(originals) * self.feature.points_per_rollout - selected,
            "overall": clustered_rate(
                list(by_prompt.values()), bootstrap_samples=self.feature.bootstrap_samples, seed=seed + 1
            ),
            "recovery_original_incorrect": clustered_rate(
                list(recovery.values()), bootstrap_samples=self.feature.bootstrap_samples, seed=seed + 2
            ),
            "retention_original_correct": clustered_rate(
                list(retention.values()), bootstrap_samples=self.feature.bootstrap_samples, seed=seed + 3
            ),
            "prefix_fraction": descriptive(prefix_fractions),
            "continuation_length": descriptive(continuation_lengths),
            "completed_response_length": descriptive(completed_lengths),
            "success_by_prefix_decile": {
                key: {"successes": int(sum(values)), "attempts": len(values), "rate": float(np.mean(values))}
                for key, values in sorted(deciles.items())
            },
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "structural_rejections_while_scanning": dict(sorted(rejection_counts.items())),
        }

    def run_evaluation(self, source: DataProto, reward_tensor: torch.Tensor, metrics: dict[str, Any]) -> bool:
        originals = self._extract_originals(source, reward_tensor)
        attempts = self._evaluate(source, originals)
        self._audit(
            "configuration",
            schema_version=1,
            points_per_rollout=self.feature.points_per_rollout,
            min_prefix_fraction=self.feature.min_prefix_fraction,
            min_continuation_tokens=self.feature.min_continuation_tokens,
            structural_boundaries_only=self.feature.structural_boundaries_only,
            selection_seed=self.feature.selection_seed,
            temperature=float(self.config.actor_rollout_ref.rollout.temperature),
            max_prompt_length=int(self.config.data.max_prompt_length),
            max_response_length=int(self.config.data.max_response_length),
            max_model_len=int(self.config.actor_rollout_ref.rollout.max_model_len),
        )
        for original in originals:
            self._audit(
                "original",
                dataset_index=original.dataset_index,
                prompt_group_id=original.prompt_group_id,
                rollout_id=original.rollout_id,
                prompt_ids=original.prompt_ids,
                solution_ids=original.solution_ids,
                editable_solution_length=len(original.editable_solution_ids),
                original_reward=original.original_reward,
                selection=asdict(original.record.selection),
                failures=[list(item) for item in original.record.failures],
            )
        for attempt in attempts:
            original = attempt["original"]
            continuation = attempt["continuation"]
            self._audit(
                "continuation",
                prompt_group_id=original.prompt_group_id,
                rollout_id=original.rollout_id,
                original_correct=bool(original.original_reward),
                mark=continuation.mark,
                prefix_fraction=continuation.mark / len(original.editable_solution_ids),
                prefix_ids=original.editable_solution_ids[: continuation.mark],
                prefix_text=decode_exact(original.editable_solution_ids[: continuation.mark], self.tokenizer),
                continuation_max_tokens=int(self.config.data.max_response_length) - continuation.mark,
                continuation_ids=list(continuation.token_ids),
                continuation_log_probs=list(continuation.log_probs),
                finish_reason=continuation.finish_reason,
                reward=attempt["reward"],
            )
        summary = self._summary(originals, attempts)
        self._audit("summary", **summary)
        with open(self.summary_path, "x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        source.batch["advantages"] = torch.zeros_like(source.batch["response_mask"], dtype=torch.float32)
        source.batch["returns"] = torch.zeros_like(source.batch["response_mask"], dtype=torch.float32)
        metrics.update(
            {
                "random_continuation/original_pass_at_1": summary["original_pass_at_1"],
                "random_continuation/overall_success": summary["overall"]["attempt_weighted"],
                "random_continuation/recovery_success": summary["recovery_original_incorrect"]["attempt_weighted"],
                "random_continuation/retention_success": summary["retention_original_correct"]["attempt_weighted"],
                "random_continuation/generated_attempts": summary["generated_attempts"],
                "random_continuation/failed_generations": summary["failed_generations"],
            }
        )
        return False
