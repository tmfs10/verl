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
"""Native RayPPOTrainer integration for synchronous branch-revision GRPO."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from transformers import AutoConfig

from verl import DataProto
from verl.experimental.agent_loop.branch_revision_agent_loop import (
    BRANCH_REVISION_AGENT_NAME,
    BRANCH_REVISION_CHILD_FIELD,
    BRANCH_REVISION_CONTINUATION_FIELD,
    BRANCH_REVISION_SCORE_FIELD,
    BranchRevisionContinuationGeneration,
    BranchRevisionCritiqueGeneration,
    BranchRevisionGenerationRecord,
    BranchRevisionScoreGeneration,
)
from verl.trainer.config import BranchRevisionGRPOConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.branch_revision_grpo import (
    LearnabilityScore,
    aggregate_log_probs,
    build_learnability_reference,
    build_rollout_logprob_prefixes,
    encode_followup_user_turn,
    normalize_log_probs_float32,
    score_seed_learnability,
    strip_terminal_eos,
    validate_binary_reward_row,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import marked_timer

_AUDIT_SCHEMA_VERSION = 5


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        error.__notes__ = notes
    notes.append(note)


def _is_moe_model(model_config) -> bool:
    for name in ("num_experts", "num_local_experts", "n_routed_experts"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and value > 1:
            return True
    return getattr(model_config, "moe_intermediate_size", None) is not None


def _canonical_sha256(values: Any, *, dtype: str) -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _float32_list(values: Any) -> list[float]:
    return [float(value) for value in normalize_log_probs_float32(values).tolist()]


def validate_branch_revision_runtime_config(config, actor_tokenizer=None, actor_model_path: str | None = None) -> None:
    """Fail closed before allocation for unsupported or ambiguous combinations."""

    feature = omega_conf_to_dataclass(
        config.algorithm.branch_revision_grpo,
        dataclass_type=BranchRevisionGRPOConfig,
    )
    if not feature.enable:
        return
    if bool(OmegaConf.select(config, "algorithm.intermediate_mc_value.enable", default=False)):
        raise ValueError("branch-revision GRPO and intermediate MC are mutually exclusive")
    if bool(OmegaConf.select(config, "algorithm.opsd.enable", default=False)):
        raise ValueError("branch-revision GRPO and OPSD cannot be enabled together")
    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError("branch-revision GRPO currently supports only VeRL's legacy FSDP/FSDP2 workers")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("branch-revision actor strategy must be fsdp or fsdp2")
    if config.actor_rollout_ref.rollout.name != "vllm":
        raise ValueError("branch-revision GRPO currently supports only the dense vLLM rollout engine")
    admission_capacity = config.actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens
    if admission_capacity is not None and (
        isinstance(admission_capacity, bool) or not isinstance(admission_capacity, int) or admission_capacity <= 0
    ):
        raise ValueError("branch-revision GRPO prompt-logprob token budget must be null or a positive integer")
    if config.critic.get("enable", None) is not False:
        raise ValueError("branch-revision GRPO is actor-only and requires critic.enable=false")
    if str(config.algorithm.adv_estimator).lower() != "grpo":
        raise ValueError("branch-revision GRPO requires algorithm.adv_estimator=grpo")
    rollout_n = config.actor_rollout_ref.rollout.n
    if isinstance(rollout_n, bool) or not isinstance(rollout_n, int) or rollout_n < 2:
        raise ValueError("branch-revision solution GRPO requires actor_rollout_ref.rollout.n>=2")
    loss_mode = str(config.actor_rollout_ref.actor.policy_loss.loss_mode)
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError("branch-revision GRPO supports policy loss modes dppo_tv and vanilla")
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    get_policy_loss_fn(loss_mode)
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        raise ValueError("branch-revision GRPO does not support reference-policy or reward KL")
    rollout_correction = config.algorithm.get("rollout_correction", None)
    if rollout_correction is not None and any(
        (
            rollout_correction.get("rollout_is", None) is not None,
            rollout_correction.get("rollout_rs", None) is not None,
            bool(rollout_correction.get("bypass_mode", False)),
        )
    ):
        raise ValueError("branch-revision GRPO rejects rollout correction and uses recorded behavior log probabilities")
    rollout = config.actor_rollout_ref.rollout
    if float(rollout.temperature) != 1.0 or float(rollout.val_kwargs.temperature) != 1.0:
        raise ValueError("all branch-revision generation, including validation, requires temperature=1.0")
    if rollout.max_model_len is None or int(rollout.max_model_len) <= 0:
        raise ValueError("branch-revision GRPO requires an explicit positive rollout.max_model_len")
    if str(rollout.get("logprobs_mode", "")) != "processed_logprobs":
        raise ValueError("branch-revision GRPO requires rollout.logprobs_mode=processed_logprobs")
    if float(rollout.top_p) != 1.0 or int(rollout.top_k) != -1 or float(rollout.repetition_penalty) != 1.0:
        raise ValueError("branch-revision learnability comparisons require top_p=1, top_k=-1, and repetition_penalty=1")
    val_kwargs = rollout.val_kwargs
    if float(val_kwargs.top_p) != 1.0 or int(val_kwargs.top_k) != -1:
        raise ValueError("branch-revision validation requires val_kwargs.top_p=1 and val_kwargs.top_k=-1")
    if int(config.data.max_prompt_length) + int(config.data.max_response_length) >= int(rollout.max_model_len):
        raise ValueError("configured prompt plus response lengths must leave branch-critique context headroom")
    if int(config.data.max_response_length) < feature.min_continuation_tokens:
        raise ValueError("data.max_response_length is smaller than branch-revision min_continuation_tokens")
    if bool(rollout.multi_turn.enable):
        raise ValueError("branch-revision GRPO supports only text-only single-turn rollouts")
    if bool(rollout.get("skip_rollout", False)):
        raise ValueError("branch-revision GRPO cannot use precomputed or skipped rollouts")
    if bool(rollout.get("enable_rollout_routing_replay", False)):
        raise ValueError("branch-revision GRPO does not support rollout routing replay")
    router_replay = config.actor_rollout_ref.actor.get("router_replay", {})
    if str(router_replay.get("mode", "none")).lower() not in {"none", "disabled"}:
        raise ValueError("branch-revision GRPO does not support actor router replay")
    if bool(config.actor_rollout_ref.actor.get("use_prefix_grouper", False)):
        raise ValueError("branch-revision GRPO does not yet support actor prefix grouping")
    if bool(config.reward.reward_model.get("launch_reward_fn_async", False)):
        raise ValueError("branch-revision reward evaluation is a blocking iteration barrier")
    if bool(config.reward.reward_model.get("enable", False)):
        raise ValueError("branch-revision GRPO requires a synchronous environment reward, not a reward model")
    reward_loop_keys = ("reward_loop_source", "reward_loop_module_path", "reward_loop_class_name")
    if any(config.reward.reward_model.get(key, None) is not None for key in reward_loop_keys):
        raise ValueError("branch-revision GRPO does not support rollout-time reward loops")
    reward_source = str(OmegaConf.select(config, "reward.reward_manager.source", default="register"))
    reward_name = str(OmegaConf.select(config, "reward.reward_manager.name", default=""))
    if reward_source == "register" and reward_name == "conditional_logprob":
        raise ValueError("branch-revision GRPO does not support conditional_logprob training rewards")
    grouped_reward_keys = (
        "use_response_logprob_reward_for_uniform_outcome_groups",
        "use_shortest_success_reward",
        "use_longest_success_penalty_reward",
    )
    reward_kwargs = config.reward.get("reward_kwargs", {}) or {}
    enabled_grouped_rewards = [key for key in grouped_reward_keys if bool(reward_kwargs.get(key, False))]
    if enabled_grouped_rewards:
        raise ValueError(
            "branch-revision GRPO rejects native grouped-reward transformations: " + repr(enabled_grouped_rewards)
        )
    if config.data.get("use_dataset_responses", False):
        raise ValueError("branch-revision GRPO does not support off-policy dataset responses")

    if actor_tokenizer is not None:
        if not callable(getattr(actor_tokenizer, "apply_chat_template", None)):
            raise ValueError("branch-revision GRPO requires an actor tokenizer chat template")
        raw_template_kwargs = OmegaConf.select(config, "data.train_apply_chat_template_kwargs", default=None)
        if raw_template_kwargs is None:
            raw_template_kwargs = OmegaConf.select(config, "data.apply_chat_template_kwargs", default=None)
        if OmegaConf.is_config(raw_template_kwargs):
            template_kwargs = OmegaConf.to_container(raw_template_kwargs, resolve=True)
        else:
            template_kwargs = raw_template_kwargs
        if template_kwargs is not None and not isinstance(template_kwargs, dict):
            raise ValueError("branch-revision training chat-template kwargs must be a mapping or null")
        instructions = [("recovery", feature.critique_prompt)]
        if feature.enable_positive_compression:
            instructions.append(("compression", feature.positive_critique_prompt))
        critique_cap = int(feature.critique_max_response_length or config.data.max_response_length)
        for objective, instruction in instructions:
            followup_tokens = len(
                encode_followup_user_turn(
                    instruction,
                    actor_tokenizer,
                    chat_template_kwargs=template_kwargs,
                )
            )
            components = {
                "max_prompt": int(config.data.max_prompt_length),
                "max_response": int(config.data.max_response_length),
                "followup": followup_tokens,
                "critique_cap": critique_cap,
            }
            required = sum(components.values())
            if required > int(rollout.max_model_len):
                rendered = ", ".join(f"{name}={value}" for name, value in components.items())
                raise ValueError(
                    f"branch-revision {objective} critique exceeds rollout.max_model_len: "
                    f"{rendered}, required={required}, limit={int(rollout.max_model_len)}"
                )
    if actor_model_path is not None:
        actor_hf_config = AutoConfig.from_pretrained(
            actor_model_path,
            trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False),
        )
        if _is_moe_model(actor_hf_config):
            raise ValueError("branch-revision GRPO currently supports only dense actor models")
        override = config.actor_rollout_ref.model.get("override_config", {})
        actor_limit = override.get(
            "max_position_embeddings",
            getattr(actor_hf_config, "max_position_embeddings", None),
        )
        if isinstance(actor_limit, int) and int(rollout.max_model_len) > actor_limit:
            raise ValueError("rollout.max_model_len exceeds the actor model's effective context limit")

    with open_dict(config):
        config.critic.enable = False
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.actor.use_rollout_log_probs = True


@dataclass
class _Bundle:
    source_row: int
    rollout_id: str
    prompt_group_id: str
    prompt_ids: list[int]
    solution_ids: list[int]
    solution_log_probs: list[float]
    original_reward: float
    record: BranchRevisionGenerationRecord | None = None
    learnability: dict[int, LearnabilityScore] = field(default_factory=dict)
    score_admissions: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    continuation_rewards: dict[int, float] = field(default_factory=dict)
    compression_fractions: dict[int, float] = field(default_factory=dict)
    compression_credits: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _ActorRow:
    audit_row_id: str
    full_ids: list[int]
    train_start: int
    behavior_log_probs: list[float]
    reward: float
    group_id: str
    kind: str


class BranchRevisionGRPOController:
    """Own one synchronous post-reward child phase and one native actor update."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.tokenizer = trainer.tokenizer
        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.branch_revision_grpo,
            dataclass_type=BranchRevisionGRPOConfig,
        )
        if not self.feature.enable:
            raise ValueError("BranchRevisionGRPOController requires enable=true")
        if trainer.processor is not None:
            raise ValueError("branch-revision GRPO currently supports only text-only models and datasets")
        if trainer.reward_fn is None:
            raise ValueError("branch-revision GRPO requires a synchronous environment reward function")
        self.audit_root = None
        self.audit_dir = None
        self.audit_attempt_id = None
        self._initialized_audit_steps: set[int] = set()
        if self.feature.audit_output_dir:
            self.audit_root = os.path.abspath(os.path.expanduser(self.feature.audit_output_dir))
            os.makedirs(self.audit_root, exist_ok=True)

    def _encode(self, text: str) -> list[int]:
        result = [int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)]
        if not result:
            raise ValueError(f"branch-revision boundary tokenized empty: {text!r}")
        return result

    def _ensure_audit_attempt(self) -> None:
        if self.audit_root is None or self.audit_dir is not None:
            return
        self.audit_attempt_id = uuid.uuid4().hex
        self.audit_dir = os.path.join(self.audit_root, f"attempt_{self.audit_attempt_id}")
        os.mkdir(self.audit_dir)
        resolved_config = OmegaConf.to_container(self.config, resolve=True)
        config_json = json.dumps(resolved_config, sort_keys=True, default=str, ensure_ascii=False)
        metadata = {
            "schema_version": _AUDIT_SCHEMA_VERSION,
            "attempt_id": self.audit_attempt_id,
            "starting_global_step": int(self.trainer.global_steps),
            "resolved_config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            "resolved_config": resolved_config,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        with open(os.path.join(self.audit_dir, "attempt.json"), "x", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, sort_keys=True, ensure_ascii=False) + "\n")

    def _audit(self, event: str, **payload: object) -> None:
        if self.audit_root is None:
            return
        self._ensure_audit_attempt()
        if self.audit_dir is None or self.audit_attempt_id is None:
            raise RuntimeError("branch-revision audit attempt initialization failed")
        step = int(self.trainer.global_steps)
        path = os.path.join(self.audit_dir, f"step_{step:08d}.jsonl")
        if step not in self._initialized_audit_steps:
            with open(path, "x", encoding="utf-8") as handle:
                handle.write("")
            self._initialized_audit_steps.add(step)
        record = {
            "schema_version": _AUDIT_SCHEMA_VERSION,
            "attempt_id": self.audit_attempt_id,
            "event": event,
            "global_step": step,
            **payload,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str, ensure_ascii=False) + "\n")

    def _pad_token_id(self) -> int:
        token_id = self.tokenizer.pad_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("branch-revision GRPO requires a tokenizer pad or EOS token")
        return int(token_id)

    @staticmethod
    def _object_array(values: list[Any]) -> np.ndarray:
        result = np.empty(len(values), dtype=object)
        result[:] = values
        return result

    @staticmethod
    def _valid_prompt_ids(batch: DataProto, row: int) -> list[int]:
        prompt_width = batch.batch["prompts"].shape[1]
        mask = batch.batch["attention_mask"][row, :prompt_width].bool()
        result = [int(token) for token in batch.batch["prompts"][row][mask].tolist()]
        if not result:
            raise ValueError("branch-revision prompt must contain at least one token")
        return result

    @staticmethod
    def _valid_solution(batch: DataProto, row: int) -> tuple[list[int], list[float]]:
        mask = batch.batch["response_mask"][row].bool()
        length = int(mask.sum().item())
        if length <= 0 or not torch.all(mask[:length]) or torch.any(mask[length:]):
            raise ValueError("branch-revision requires a non-empty contiguous single-turn response mask")
        if "rollout_log_probs" not in batch.batch:
            raise RuntimeError("branch-revision requires sampling-time processed rollout_log_probs")
        tokens = [int(token) for token in batch.batch["responses"][row, :length].tolist()]
        log_probs = [float(value) for value in batch.batch["rollout_log_probs"][row, :length].tolist()]
        if len(log_probs) != len(tokens) or not all(math.isfinite(value) for value in log_probs):
            raise RuntimeError("branch-revision solution behavior log probabilities are missing or non-finite")
        return tokens, log_probs

    def prepare_original_generation_batch(self, batch: DataProto) -> None:
        """Attach stable rollout ids without replacing VeRL's native initial agent."""

        group_values = batch.non_tensor_batch.get("prompt_group_id", batch.non_tensor_batch.get("uid"))
        if group_values is None:
            raise RuntimeError("branch-revision generation requires prompt_group_id or uid")
        counts: dict[str, int] = {}
        rollout_ids: list[str] = []
        for value in group_values:
            key = str(value)
            index = counts.get(key, 0)
            counts[key] = index + 1
            rollout_ids.append(f"{key}:{index}")
        batch.non_tensor_batch["branch_revision_rollout_id"] = np.array(rollout_ids, dtype=object)

    def _original_rewards(self, reward_tensor: torch.Tensor) -> list[float]:
        return [
            validate_binary_reward_row(row, tolerance=self.feature.reward_tolerance)
            for row in reward_tensor.detach().cpu().tolist()
        ]

    def _build_bundles(self, source: DataProto, rewards: list[float]) -> list[_Bundle]:
        if len(source) != len(rewards):
            raise RuntimeError("branch-revision original reward count does not match rollout count")
        group_values = source.non_tensor_batch.get("prompt_group_id")
        rollout_values = source.non_tensor_batch.get("branch_revision_rollout_id")
        if group_values is None or rollout_values is None:
            raise RuntimeError("branch-revision rollout identity columns were lost")
        bundles: list[_Bundle] = []
        for row, reward in enumerate(rewards):
            solution_ids, solution_log_probs = self._valid_solution(source, row)
            bundles.append(
                _Bundle(
                    source_row=row,
                    rollout_id=str(rollout_values[row]),
                    prompt_group_id=str(group_values[row]),
                    prompt_ids=self._valid_prompt_ids(source, row),
                    solution_ids=solution_ids,
                    solution_log_probs=solution_log_probs,
                    original_reward=reward,
                )
            )
        return bundles

    def _audit_originals(self, bundles: list[_Bundle]) -> None:
        for bundle in bundles:
            editable_length = len(strip_terminal_eos(bundle.solution_ids, self.tokenizer))
            self._audit(
                "original",
                rollout_id=bundle.rollout_id,
                prompt_group_id=bundle.prompt_group_id,
                source_row=bundle.source_row,
                prompt_ids=bundle.prompt_ids,
                solution_ids=bundle.solution_ids,
                solution_log_probs=_float32_list(bundle.solution_log_probs),
                editable_solution_length=editable_length,
                reward=bundle.original_reward,
            )

    def _make_child_request(self, source: DataProto, bundles: list[_Bundle]) -> DataProto | None:
        selected = [
            bundle
            for bundle in bundles
            if bundle.original_reward == 0.0
            or (self.feature.enable_positive_compression and bundle.original_reward == 1.0)
        ]
        if not selected:
            return None
        rows = [bundle.source_row for bundle in selected]
        non_tensors = {
            key: np.take(values, rows, axis=0).copy()
            for key, values in source.non_tensor_batch.items()
            if key != BRANCH_REVISION_CHILD_FIELD
        }
        non_tensors["agent_name"] = np.array([BRANCH_REVISION_AGENT_NAME] * len(selected), dtype=object)
        non_tensors["branch_revision_phase"] = np.array(["critique"] * len(selected), dtype=object)
        non_tensors["branch_revision_rollout_id"] = np.array([bundle.rollout_id for bundle in selected], dtype=object)
        non_tensors["branch_revision_parent_prompt_ids"] = self._object_array(
            [bundle.prompt_ids for bundle in selected]
        )
        non_tensors["branch_revision_parent_solution_ids"] = self._object_array(
            [bundle.solution_ids for bundle in selected]
        )
        non_tensors["branch_revision_parent_solution_log_probs"] = self._object_array(
            [bundle.solution_log_probs for bundle in selected]
        )
        objectives = ["recovery" if bundle.original_reward == 0.0 else "compression" for bundle in selected]
        non_tensors["branch_revision_parent_objective"] = np.array(objectives, dtype=object)
        non_tensors["branch_revision_num_critiques"] = np.array(
            [
                self.feature.num_critiques if objective == "recovery" else self.feature.num_positive_critiques
                for objective in objectives
            ],
            dtype=np.int64,
        )
        return DataProto.from_dict(
            non_tensors=non_tensors,
            meta_info={"global_steps": self.trainer.global_steps},
        )

    def _run_with_manager_lifecycle(
        self,
        callback,
        *,
        checkpoint_manager,
        rollout_manager,
        profile_rollout: bool,
    ):
        primary_error: BaseException | None = None
        profile_cleanup_required = False
        try:
            checkpoint_manager.update_weights(self.trainer.global_steps)
            if profile_rollout:
                profile_cleanup_required = True
                rollout_manager.start_profile()
            return callback()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_errors: list[tuple[str, BaseException]] = []
            try:
                checkpoint_manager.sleep_replicas()
            except BaseException as error:
                cleanup_errors.append(("rollout sleep cleanup", error))
            if profile_cleanup_required:
                try:
                    rollout_manager.stop_profile()
                except BaseException as error:
                    cleanup_errors.append(("rollout profile cleanup", error))
            if cleanup_errors:
                if primary_error is not None:
                    for label, error in cleanup_errors:
                        _add_exception_note(primary_error, f"{label} also failed: {error!r}")
                else:
                    _, first_error = cleanup_errors[0]
                    for label, error in cleanup_errors[1:]:
                        _add_exception_note(first_error, f"{label} also failed: {error!r}")
                    raise first_error

    def _run_with_rollout_lifecycle(self, callback, *, profile_rollout: bool):
        return self._run_with_manager_lifecycle(
            callback,
            checkpoint_manager=self.trainer.checkpoint_manager,
            rollout_manager=self.trainer.async_rollout_manager,
            profile_rollout=profile_rollout,
        )

    def _run_with_critique_rollout_lifecycle(self, callback, *, profile_rollout: bool):
        if not self.feature.separate_critique_model:
            return self._run_with_rollout_lifecycle(callback, profile_rollout=profile_rollout)
        required = ("critique_checkpoint_manager", "critique_async_rollout_manager")
        missing = [name for name in required if not hasattr(self.trainer, name)]
        if missing:
            raise RuntimeError(f"separate critique policy rollout is not initialized: {missing}")
        return self._run_with_manager_lifecycle(
            callback,
            checkpoint_manager=self.trainer.critique_checkpoint_manager,
            rollout_manager=self.trainer.critique_async_rollout_manager,
            profile_rollout=profile_rollout,
        )

    @staticmethod
    def _coerce_critique(value: Any) -> BranchRevisionCritiqueGeneration:
        if isinstance(value, BranchRevisionCritiqueGeneration):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"invalid branch-revision critique record {type(value)!r}")
        seed_log_probs = value.get("new_continuation_log_probs", ())
        return BranchRevisionCritiqueGeneration(
            token_ids=tuple(int(token) for token in value["token_ids"]),
            log_probs=tuple(float(item) for item in value["log_probs"]),
            finish_reason=value.get("finish_reason"),
            parse_reason=str(value["parse_reason"]),
            prefix_text=str(value.get("prefix_text", "")),
            prefix_plus_new_continuation_text=str(value.get("prefix_plus_new_continuation_text", "")),
            new_continuation_text=str(value.get("new_continuation_text", "")),
            branch_prefix_ids=tuple(int(token) for token in value.get("branch_prefix_ids", ())),
            prefix_ids=tuple(int(token) for token in value.get("prefix_ids", ())),
            continuation_prefix_ids=tuple(int(token) for token in value.get("continuation_prefix_ids", ())),
            new_continuation_ids=tuple(int(token) for token in value.get("new_continuation_ids", ())),
            new_continuation_log_probs=tuple(_float32_list(seed_log_probs)) if seed_log_probs else (),
            revised_prefix_ids=tuple(int(token) for token in value.get("revised_prefix_ids", ())),
            continuation_ids=tuple(int(token) for token in value.get("continuation_ids", ())),
            continuation_log_probs=tuple(float(item) for item in value.get("continuation_log_probs", ())),
            continuation_finish_reason=value.get("continuation_finish_reason"),
            continuation_max_tokens=int(value.get("continuation_max_tokens", 0)),
        )

    @classmethod
    def _coerce_record(cls, value: Any) -> BranchRevisionGenerationRecord:
        if isinstance(value, BranchRevisionGenerationRecord):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"invalid branch-revision child record {type(value)!r}")
        return BranchRevisionGenerationRecord(
            rollout_id=str(value["rollout_id"]),
            objective=str(value["objective"]),
            critiques=tuple(cls._coerce_critique(item) for item in value["critiques"]),
            critique_prompt_ids=tuple(int(token) for token in value["critique_prompt_ids"]),
        )

    def _extract_records(self, output: DataProto) -> dict[str, BranchRevisionGenerationRecord]:
        raw = output.non_tensor_batch.pop(BRANCH_REVISION_CHILD_FIELD, None)
        if raw is None or len(raw) != len(output):
            raise RuntimeError("branch-revision child rollout did not return one record per selected solution")
        records: dict[str, BranchRevisionGenerationRecord] = {}
        for value in raw:
            record = self._coerce_record(value)
            if record.rollout_id in records:
                raise RuntimeError(f"duplicate branch-revision rollout id {record.rollout_id!r}")
            expected_critiques = (
                self.feature.num_critiques
                if record.objective == "recovery"
                else self.feature.num_positive_critiques
                if record.objective == "compression"
                else None
            )
            if expected_critiques is None:
                raise RuntimeError(f"rollout {record.rollout_id!r} returned unknown objective {record.objective!r}")
            if len(record.critiques) != expected_critiques:
                raise RuntimeError(
                    f"rollout {record.rollout_id!r} returned {len(record.critiques)} critiques; "
                    f"expected {expected_critiques}"
                )
            records[record.rollout_id] = record
        return records

    def _attach_records(self, bundles: list[_Bundle], records: dict[str, BranchRevisionGenerationRecord]) -> None:
        expected = {
            bundle.rollout_id
            for bundle in bundles
            if bundle.original_reward == 0.0
            or (self.feature.enable_positive_compression and bundle.original_reward == 1.0)
        }
        if set(records) != expected:
            raise RuntimeError("branch-revision child stage returned an unexpected rollout-id set")
        for bundle in bundles:
            if bundle.rollout_id in expected:
                bundle.record = records[bundle.rollout_id]
                expected_objective = "recovery" if bundle.original_reward == 0.0 else "compression"
                if bundle.record.objective != expected_objective:
                    raise RuntimeError(
                        f"rollout {bundle.rollout_id!r} returned {bundle.record.objective!r}; "
                        f"expected {expected_objective!r}"
                    )
                editable_solution = strip_terminal_eos(bundle.solution_ids, self.tokenizer)
                expected_prefix = [*bundle.prompt_ids, *editable_solution]
                critique_prompt = list(bundle.record.critique_prompt_ids)
                if (
                    len(critique_prompt) <= len(expected_prefix)
                    or critique_prompt[: len(expected_prefix)] != expected_prefix
                ):
                    raise RuntimeError(
                        "branch-revision worker critique prompt does not preserve the exact "
                        "original prompt/solution prefix"
                    )
            elif bundle.rollout_id in records:
                raise RuntimeError("unselected original solution unexpectedly received branch critiques")

    def _phase_non_tensors(self, source: DataProto, rows: list[int]) -> dict[str, np.ndarray]:
        return {
            key: np.take(values, rows, axis=0).copy()
            for key, values in source.non_tensor_batch.items()
            if key
            not in {
                BRANCH_REVISION_CHILD_FIELD,
                BRANCH_REVISION_SCORE_FIELD,
                BRANCH_REVISION_CONTINUATION_FIELD,
            }
        }

    def _make_score_request(self, source: DataProto, bundles: list[_Bundle]) -> DataProto | None:
        items: list[tuple[_Bundle, int, BranchRevisionCritiqueGeneration]] = []
        for bundle in bundles:
            if bundle.record is None:
                continue
            items.extend(
                (bundle, critique_index, critique)
                for critique_index, critique in enumerate(bundle.record.critiques)
                if critique.valid
            )
        if not items:
            return None
        rows = [bundle.source_row for bundle, _, _ in items]
        non_tensors = self._phase_non_tensors(source, rows)
        non_tensors["agent_name"] = np.array([BRANCH_REVISION_AGENT_NAME] * len(items), dtype=object)
        non_tensors["branch_revision_phase"] = np.array(["score"] * len(items), dtype=object)
        non_tensors["branch_revision_rollout_id"] = np.array(
            [bundle.rollout_id for bundle, _, _ in items],
            dtype=object,
        )
        non_tensors["branch_revision_critique_index"] = np.array(
            [critique_index for _, critique_index, _ in items],
            dtype=np.int64,
        )
        non_tensors["branch_revision_route_key"] = np.array(
            [f"{bundle.rollout_id}:revision:{critique_index}" for bundle, critique_index, _ in items],
            dtype=object,
        )
        non_tensors["branch_revision_parent_prompt_ids"] = self._object_array(
            [bundle.prompt_ids for bundle, _, _ in items]
        )
        non_tensors["branch_revision_continuation_prefix_ids"] = self._object_array(
            [list(critique.continuation_prefix_ids) for _, _, critique in items]
        )
        non_tensors["branch_revision_new_continuation_ids"] = self._object_array(
            [list(critique.new_continuation_ids) for _, _, critique in items]
        )
        return DataProto.from_dict(
            non_tensors=non_tensors,
            meta_info={"global_steps": self.trainer.global_steps},
        )

    @staticmethod
    def _coerce_score(value: Any) -> BranchRevisionScoreGeneration:
        if isinstance(value, BranchRevisionScoreGeneration):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"invalid branch-revision score record {type(value)!r}")
        admission = value.get("admission")
        if admission is not None and not isinstance(admission, dict):
            raise TypeError("branch-revision score admission evidence must be null or a mapping")
        return BranchRevisionScoreGeneration(
            rollout_id=str(value["rollout_id"]),
            critique_index=int(value["critique_index"]),
            prompt_logprob_start=int(value["prompt_logprob_start"]),
            scored_token_ids=tuple(int(token) for token in value["scored_token_ids"]),
            scored_token_log_probs=tuple(_float32_list(value["scored_token_log_probs"])),
            admission=None if admission is None else dict(admission),
        )

    def _extract_scores(self, output: DataProto) -> dict[tuple[str, int], BranchRevisionScoreGeneration]:
        raw = output.non_tensor_batch.pop(BRANCH_REVISION_SCORE_FIELD, None)
        if raw is None or len(raw) != len(output):
            raise RuntimeError("branch-revision score stage did not return one record per proposal")
        records: dict[tuple[str, int], BranchRevisionScoreGeneration] = {}
        for value in raw:
            record = self._coerce_score(value)
            key = (record.rollout_id, record.critique_index)
            if key in records:
                raise RuntimeError(f"duplicate branch-revision score key {key!r}")
            records[key] = record
        return records

    def _attach_scores(
        self,
        bundles: list[_Bundle],
        scores: dict[tuple[str, int], BranchRevisionScoreGeneration],
    ) -> None:
        expected = {
            (bundle.rollout_id, critique_index)
            for bundle in bundles
            if bundle.record is not None
            for critique_index, critique in enumerate(bundle.record.critiques)
            if critique.valid
        }
        if set(scores) != expected:
            raise RuntimeError("branch-revision score stage returned an unexpected proposal set")
        for bundle in bundles:
            if bundle.record is None:
                continue
            critiques = list(bundle.record.critiques)
            for critique_index, critique in enumerate(critiques):
                if not critique.valid:
                    continue
                score = scores[(bundle.rollout_id, critique_index)]
                expected_start = len(bundle.prompt_ids) + len(critique.continuation_prefix_ids)
                if score.prompt_logprob_start != expected_start:
                    raise RuntimeError("branch-revision score changed the prompt-logprob slice boundary")
                if list(score.scored_token_ids) != list(critique.new_continuation_ids):
                    raise RuntimeError("branch-revision score changed the replacement-token sequence")
                if len(score.scored_token_log_probs) != len(critique.new_continuation_ids):
                    raise RuntimeError("branch-revision score omitted replacement-token log probabilities")
                critiques[critique_index] = replace(
                    critique,
                    new_continuation_log_probs=score.scored_token_log_probs,
                )
                bundle.score_admissions[critique_index] = None if score.admission is None else dict(score.admission)
            bundle.record = replace(bundle.record, critiques=tuple(critiques))

    def _make_continuation_request(self, source: DataProto, bundles: list[_Bundle]) -> DataProto | None:
        items: list[tuple[_Bundle, int, BranchRevisionCritiqueGeneration]] = []
        for bundle in bundles:
            if bundle.record is None:
                continue
            items.extend(
                (bundle, critique_index, critique)
                for critique_index, critique in enumerate(bundle.record.critiques)
                if critique.valid
                and bundle.learnability.get(critique_index, None) is not None
                and bundle.learnability[critique_index].accepted
            )
        if not items:
            return None
        rows = [bundle.source_row for bundle, _, _ in items]
        non_tensors = self._phase_non_tensors(source, rows)
        non_tensors["agent_name"] = np.array([BRANCH_REVISION_AGENT_NAME] * len(items), dtype=object)
        non_tensors["branch_revision_phase"] = np.array(["continuation"] * len(items), dtype=object)
        non_tensors["branch_revision_rollout_id"] = np.array(
            [bundle.rollout_id for bundle, _, _ in items],
            dtype=object,
        )
        non_tensors["branch_revision_critique_index"] = np.array(
            [critique_index for _, critique_index, _ in items],
            dtype=np.int64,
        )
        non_tensors["branch_revision_route_key"] = np.array(
            [f"{bundle.rollout_id}:revision:{critique_index}" for bundle, critique_index, _ in items],
            dtype=object,
        )
        non_tensors["branch_revision_parent_prompt_ids"] = self._object_array(
            [bundle.prompt_ids for bundle, _, _ in items]
        )
        non_tensors["branch_revision_revised_prefix_ids"] = self._object_array(
            [list(critique.revised_prefix_ids) for _, _, critique in items]
        )
        non_tensors["branch_revision_continuation_max_tokens"] = np.array(
            [critique.continuation_max_tokens for _, _, critique in items],
            dtype=np.int64,
        )
        return DataProto.from_dict(
            non_tensors=non_tensors,
            meta_info={"global_steps": self.trainer.global_steps},
        )

    @staticmethod
    def _coerce_continuation(value: Any) -> BranchRevisionContinuationGeneration:
        if isinstance(value, BranchRevisionContinuationGeneration):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"invalid branch-revision continuation record {type(value)!r}")
        return BranchRevisionContinuationGeneration(
            rollout_id=str(value["rollout_id"]),
            critique_index=int(value["critique_index"]),
            token_ids=tuple(int(token) for token in value["token_ids"]),
            log_probs=tuple(float(item) for item in value["log_probs"]),
            finish_reason=value.get("finish_reason"),
            max_tokens=int(value["max_tokens"]),
        )

    def _extract_continuations(
        self,
        output: DataProto,
    ) -> dict[tuple[str, int], BranchRevisionContinuationGeneration]:
        raw = output.non_tensor_batch.pop(BRANCH_REVISION_CONTINUATION_FIELD, None)
        if raw is None or len(raw) != len(output):
            raise RuntimeError("branch-revision continuation stage did not return one record per accepted edit")
        records: dict[tuple[str, int], BranchRevisionContinuationGeneration] = {}
        for value in raw:
            record = self._coerce_continuation(value)
            key = (record.rollout_id, record.critique_index)
            if key in records:
                raise RuntimeError(f"duplicate branch-revision continuation key {key!r}")
            records[key] = record
        return records

    def _attach_continuations(
        self,
        bundles: list[_Bundle],
        continuations: dict[tuple[str, int], BranchRevisionContinuationGeneration],
    ) -> None:
        expected = {
            (bundle.rollout_id, critique_index)
            for bundle in bundles
            if bundle.record is not None
            for critique_index, critique in enumerate(bundle.record.critiques)
            if critique.valid and critique_index in bundle.learnability and bundle.learnability[critique_index].accepted
        }
        if set(continuations) != expected:
            raise RuntimeError("branch-revision continuation stage returned an unexpected accepted-edit set")
        for bundle in bundles:
            if bundle.record is None:
                continue
            critiques = list(bundle.record.critiques)
            for critique_index, critique in enumerate(critiques):
                key = (bundle.rollout_id, critique_index)
                if key not in expected:
                    continue
                continuation = continuations[key]
                if continuation.max_tokens != critique.continuation_max_tokens:
                    raise RuntimeError("branch-revision continuation changed its configured token budget")
                if not continuation.token_ids or len(continuation.token_ids) != len(continuation.log_probs):
                    raise RuntimeError("branch-revision continuation tokens and behavior log probabilities misalign")
                critiques[critique_index] = replace(
                    critique,
                    continuation_ids=continuation.token_ids,
                    continuation_log_probs=continuation.log_probs,
                    continuation_finish_reason=continuation.finish_reason,
                )
            bundle.record = replace(bundle.record, critiques=tuple(critiques))

    def _score_seed_learnability(self, bundles: list[_Bundle]) -> None:
        """Score proposal seeds from vLLM prompt logprobs at their actual context."""

        editable_log_probs: list[list[float]] = []
        for bundle in bundles:
            editable_length = len(strip_terminal_eos(bundle.solution_ids, self.tokenizer))
            editable_log_probs.append(bundle.solution_log_probs[:editable_length])
        prefixes = build_rollout_logprob_prefixes(
            [bundle.rollout_id for bundle in bundles],
            editable_log_probs,
        )
        statistic = self.feature.learnability_logprob_statistic
        admission_records: list[dict[str, Any]] = []
        proposals_by_length: defaultdict[
            int,
            list[tuple[_Bundle, int, BranchRevisionCritiqueGeneration, list[float]]],
        ] = defaultdict(list)
        for bundle in bundles:
            if bundle.record is None:
                continue
            for critique_index, critique in enumerate(bundle.record.critiques):
                if not critique.valid:
                    continue
                if (
                    not critique.new_continuation_ids
                    or not critique.revised_prefix_ids
                    or not critique.prefix_ids
                    or [*critique.branch_prefix_ids, *critique.prefix_ids] != list(critique.continuation_prefix_ids)
                    or [*critique.continuation_prefix_ids, *critique.new_continuation_ids]
                    != list(critique.revised_prefix_ids)
                ):
                    raise RuntimeError("valid branch revision has inconsistent prefix/joint token boundaries")
                seed_length = len(critique.new_continuation_ids)
                seed_values = _float32_list(critique.new_continuation_log_probs)
                if len(seed_values) != seed_length:
                    raise RuntimeError(
                        "valid branch revision does not have one vLLM prompt log probability per replacement token"
                    )
                if not all(math.isfinite(value) for value in seed_values):
                    raise RuntimeError("vLLM returned non-finite replacement-seed prompt log probabilities")
                proposals_by_length[seed_length].append((bundle, critique_index, critique, seed_values))

        for seed_length in sorted(proposals_by_length):
            reference = build_learnability_reference(
                prefixes,
                window_size=seed_length,
                logprob_statistic=statistic,
            )
            self._audit(
                "learnability_reference",
                reference_key=f"{statistic}:{seed_length}",
                logprob_statistic=statistic,
                seed_tokens=seed_length,
                window_weighting="uniform_per_window",
                eligible_rollouts=reference.eligible_rollouts,
                total_windows=reference.total_windows,
                rollout_window_counts=[
                    {"rollout_id": rollout_id, "windows": count}
                    for rollout_id, count in reference.rollout_window_counts
                ],
                window_scores_sha256=reference.window_scores_sha256,
                population_mean=reference.population_mean,
                population_stddev=reference.population_stddev,
            )
            for bundle, critique_index, critique, seed_values in proposals_by_length[seed_length]:
                if critique_index not in bundle.score_admissions:
                    raise RuntimeError("valid branch revision is missing its prompt-logprob admission result")
                admission = bundle.score_admissions[critique_index]
                configured_capacity = self.config.actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens
                expected_prompt_tokens = (
                    len(bundle.prompt_ids) + len(critique.continuation_prefix_ids) + len(critique.new_continuation_ids)
                )
                if configured_capacity is None:
                    if admission is not None:
                        raise RuntimeError("unbounded prompt-logprob scoring unexpectedly returned admission evidence")
                else:
                    if not isinstance(admission, dict):
                        raise RuntimeError("valid branch revision is missing prompt-logprob admission evidence")
                    capacity = int(configured_capacity)
                    required_admission = {
                        "server_id",
                        "capacity",
                        "request_sequence",
                        "prompt_tokens",
                        "charged_tokens",
                        "wait_seconds",
                        "inflight_prompt_tokens_at_grant",
                        "inflight_charged_tokens_at_grant",
                        "high_water_prompt_tokens",
                        "high_water_charged_tokens",
                        "oversized",
                    }
                    if not required_admission.issubset(admission):
                        raise RuntimeError("prompt-logprob admission evidence is incomplete")
                    try:
                        admitted_capacity = int(admission["capacity"])
                        request_sequence = int(admission["request_sequence"])
                        prompt_tokens = int(admission["prompt_tokens"])
                        charged_tokens = int(admission["charged_tokens"])
                        wait_seconds = float(admission["wait_seconds"])
                        inflight_prompt_tokens = int(admission["inflight_prompt_tokens_at_grant"])
                        inflight_charged_tokens = int(admission["inflight_charged_tokens_at_grant"])
                        high_water_prompt_tokens = int(admission["high_water_prompt_tokens"])
                        high_water_charged_tokens = int(admission["high_water_charged_tokens"])
                    except (TypeError, ValueError) as error:
                        raise RuntimeError("prompt-logprob admission evidence has invalid numeric fields") from error
                    oversized = expected_prompt_tokens > capacity
                    if (
                        not isinstance(admission["server_id"], str)
                        or not admission["server_id"]
                        or admitted_capacity != capacity
                        or request_sequence <= 0
                        or prompt_tokens != expected_prompt_tokens
                        or charged_tokens != min(expected_prompt_tokens, capacity)
                        or not math.isfinite(wait_seconds)
                        or wait_seconds < 0.0
                        or inflight_prompt_tokens < prompt_tokens
                        or inflight_charged_tokens < charged_tokens
                        or inflight_charged_tokens > capacity
                        or high_water_prompt_tokens < inflight_prompt_tokens
                        or high_water_charged_tokens < inflight_charged_tokens
                        or high_water_charged_tokens > capacity
                        or bool(admission["oversized"]) != oversized
                    ):
                        raise RuntimeError("prompt-logprob admission evidence violates its configured token budget")
                    if oversized and (inflight_prompt_tokens != prompt_tokens or inflight_charged_tokens != capacity):
                        raise RuntimeError("oversized prompt-logprob request did not run alone")
                    admission_records.append(dict(admission))
                seed_score = aggregate_log_probs(seed_values, statistic=statistic)
                score = score_seed_learnability(
                    seed_score,
                    reference,
                    threshold_mode=self.feature.learnability_threshold_mode,
                    max_seed_window_stddevs=self.feature.max_seed_window_stddevs,
                    minimum_percentile=self.feature.min_seed_window_percentile,
                    full_credit_percentile=self.feature.full_credit_seed_window_percentile,
                )
                bundle.learnability[critique_index] = score
                self._audit(
                    "learnability",
                    score_source="vllm_prompt_logprobs",
                    reference_key=f"{statistic}:{seed_length}",
                    rollout_id=bundle.rollout_id,
                    objective=bundle.record.objective,
                    critique_index=critique_index,
                    seed_tokens=seed_length,
                    logprob_statistic=score.logprob_statistic,
                    threshold_mode=score.threshold_mode,
                    seed_score=score.seed_score,
                    scoring_prompt_ids=[
                        *bundle.prompt_ids,
                        *critique.continuation_prefix_ids,
                        *critique.new_continuation_ids,
                    ],
                    prompt_logprob_start=len(bundle.prompt_ids) + len(critique.continuation_prefix_ids),
                    scored_token_ids=list(critique.new_continuation_ids),
                    scored_token_log_probs=seed_values,
                    prompt_logprob_admission=admission,
                    percentile=score.percentile,
                    reference_mean=score.reference_mean,
                    reference_stddev=score.reference_stddev,
                    stddevs_below_mean=score.stddevs_below_mean,
                    acceptance_floor=score.acceptance_floor,
                    max_seed_window_stddevs=score.max_seed_window_stddevs,
                    reward_weight=score.reward_weight,
                    accepted=score.accepted,
                    eligible_rollouts=score.eligible_rollouts,
                    total_windows=score.total_windows,
                )
        if admission_records:
            per_server: dict[str, dict[str, int | float]] = {}
            for admission in admission_records:
                server_id = str(admission["server_id"])
                summary = per_server.setdefault(
                    server_id,
                    {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "max_inflight_prompt_tokens": 0,
                        "max_inflight_charged_tokens": 0,
                        "max_wait_seconds": 0.0,
                    },
                )
                summary["requests"] = int(summary["requests"]) + 1
                summary["prompt_tokens"] = int(summary["prompt_tokens"]) + int(admission["prompt_tokens"])
                summary["max_inflight_prompt_tokens"] = max(
                    int(summary["max_inflight_prompt_tokens"]),
                    int(admission["high_water_prompt_tokens"]),
                )
                summary["max_inflight_charged_tokens"] = max(
                    int(summary["max_inflight_charged_tokens"]),
                    int(admission["high_water_charged_tokens"]),
                )
                summary["max_wait_seconds"] = max(
                    float(summary["max_wait_seconds"]),
                    float(admission["wait_seconds"]),
                )
            self._audit(
                "prompt_logprob_admission_summary",
                capacity=int(self.config.actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens),
                requests=len(admission_records),
                prompt_tokens=sum(int(item["prompt_tokens"]) for item in admission_records),
                per_server=per_server,
            )

    def _make_reward_batch(
        self,
        source: DataProto,
        rows: list[int],
        prompts: list[list[int]],
        responses: list[list[int]],
    ) -> DataProto:
        prompt_width = int(self.config.actor_rollout_ref.rollout.prompt_length)
        response_width = int(self.config.actor_rollout_ref.rollout.response_length)
        if any(len(tokens) > prompt_width for tokens in prompts):
            raise ValueError("branch-revision reward prompt exceeds rollout.prompt_length")
        if any(not tokens or len(tokens) > response_width for tokens in responses):
            raise ValueError("branch-revision completed response must fit rollout.response_length")
        pad_id = self._pad_token_id()
        prompt_tensor = torch.full((len(rows), prompt_width), pad_id, dtype=torch.long)
        response_tensor = torch.full((len(rows), response_width), pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((len(rows), prompt_width), dtype=torch.long)
        response_mask = torch.zeros((len(rows), response_width), dtype=torch.long)
        for row, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            prompt_tensor[row, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            prompt_mask[row, -len(prompt) :] = 1
            response_tensor[row, : len(response)] = torch.tensor(response, dtype=torch.long)
            response_mask[row, : len(response)] = 1
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        non_tensors = {
            key: np.take(values, rows, axis=0).copy()
            for key, values in source.non_tensor_batch.items()
            if key != BRANCH_REVISION_CHILD_FIELD
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

    def _evaluate_continuations(self, source: DataProto, bundles: list[_Bundle]) -> None:
        rows: list[int] = []
        prompts: list[list[int]] = []
        responses: list[list[int]] = []
        mapping: list[tuple[int, int]] = []
        for bundle_index, bundle in enumerate(bundles):
            if bundle.record is None:
                continue
            for critique_index, critique in enumerate(bundle.record.critiques):
                if not critique.token_ids or len(critique.token_ids) != len(critique.log_probs):
                    raise RuntimeError("branch-revision critique tokens and behavior log probabilities misalign")
                if not all(math.isfinite(value) for value in critique.log_probs):
                    raise RuntimeError("branch-revision critique contains non-finite behavior log probabilities")
                if critique.valid:
                    if (
                        not critique.new_continuation_ids
                        or not critique.revised_prefix_ids
                        or critique.continuation_max_tokens < self.feature.min_continuation_tokens
                    ):
                        raise RuntimeError("valid branch revision lacks its parsed replacement record")
                    learnability = bundle.learnability.get(critique_index)
                    if learnability is None:
                        raise RuntimeError("valid branch revision is missing its learnability assessment")
                    if learnability.accepted:
                        if not critique.continuation_ids or len(critique.continuation_ids) != len(
                            critique.continuation_log_probs
                        ):
                            raise RuntimeError("accepted branch revision lacks one complete continuation record")
                        if not all(math.isfinite(value) for value in critique.continuation_log_probs):
                            raise RuntimeError(
                                "branch-revision continuation contains non-finite behavior log probabilities"
                            )
                        rows.append(bundle.source_row)
                        prompts.append(bundle.prompt_ids)
                        responses.append([*critique.revised_prefix_ids, *critique.continuation_ids])
                        mapping.append((bundle_index, critique_index))
                    elif critique.continuation_ids or critique.continuation_log_probs:
                        raise RuntimeError("learnability-rejected branch revision unexpectedly generated a suffix")
                elif (
                    critique.branch_prefix_ids
                    or critique.prefix_ids
                    or critique.continuation_prefix_ids
                    or critique.new_continuation_ids
                    or critique.revised_prefix_ids
                    or critique.continuation_ids
                    or critique.continuation_log_probs
                ):
                    raise RuntimeError("invalid branch revision unexpectedly launched or retained a continuation")
        if not mapping:
            return
        reward_batch = self._make_reward_batch(source, rows, prompts, responses)
        reward_tensor, _ = compute_reward(
            reward_batch,
            self.trainer.reward_fn,
            actor_wg=self.trainer.actor_rollout_wg,
        )
        reward_rows = reward_tensor.detach().cpu().tolist()
        if len(reward_rows) != len(mapping):
            raise RuntimeError("branch-revision continuation reward count does not match generated continuations")
        for (bundle_index, critique_index), row in zip(mapping, reward_rows, strict=True):
            reward = validate_binary_reward_row(row, tolerance=self.feature.reward_tolerance)
            bundle = bundles[bundle_index]
            bundle.continuation_rewards[critique_index] = reward
            critique = bundle.record.critiques[critique_index]
            if bundle.record.objective == "compression":
                original_length = len(strip_terminal_eos(bundle.solution_ids, self.tokenizer))
                revised_length = len(
                    strip_terminal_eos([*critique.revised_prefix_ids, *critique.continuation_ids], self.tokenizer)
                )
                compression_fraction = max(0.0, (original_length - revised_length) / original_length)
                compression_credit = reward * min(
                    compression_fraction / self.feature.positive_compression_target,
                    1.0,
                )
                bundle.compression_fractions[critique_index] = compression_fraction
                bundle.compression_credits[critique_index] = compression_credit
            self._audit(
                "continuation",
                actor_row_id=f"continuation:{bundle.rollout_id}:{critique_index}",
                rollout_id=bundle.rollout_id,
                objective=bundle.record.objective,
                critique_index=critique_index,
                reward=reward,
                compression_fraction=bundle.compression_fractions.get(critique_index),
                compression_credit=bundle.compression_credits.get(critique_index),
                revised_prefix_ids=list(critique.revised_prefix_ids),
                continuation_ids=list(critique.continuation_ids),
                continuation_log_probs=_float32_list(critique.continuation_log_probs),
                continuation_max_tokens=critique.continuation_max_tokens,
                finish_reason=critique.continuation_finish_reason,
            )

    @staticmethod
    def _prompt_rewards(bundles: list[_Bundle]) -> dict[str, list[float]]:
        prompt_rewards: defaultdict[str, list[float]] = defaultdict(list)
        for bundle in bundles:
            prompt_rewards[bundle.prompt_group_id].append(bundle.original_reward)
        return dict(prompt_rewards)

    def _actor_rows(self, bundles: list[_Bundle]) -> list[_ActorRow]:
        rows: list[_ActorRow] = []
        prompt_rewards = self._prompt_rewards(bundles)
        prompt_pass_at_1 = {
            prompt_group_id: sum(rewards) / len(rewards) for prompt_group_id, rewards in prompt_rewards.items()
        }
        for bundle in bundles:
            solution_group = f"solution:{bundle.prompt_group_id}"
            critique_group = (
                "critique:batch" if self.feature.critique_grpo_grouping == "batch" else f"critique:{bundle.rollout_id}"
            )
            rows.append(
                _ActorRow(
                    audit_row_id=f"original:{bundle.rollout_id}",
                    full_ids=[*bundle.prompt_ids, *bundle.solution_ids],
                    train_start=len(bundle.prompt_ids),
                    behavior_log_probs=bundle.solution_log_probs,
                    reward=bundle.original_reward,
                    group_id=solution_group,
                    kind="original",
                )
            )
            if bundle.record is None:
                continue
            critique_prompt = list(bundle.record.critique_prompt_ids)
            for critique_index, critique in enumerate(bundle.record.critiques):
                continuation_outcome = bundle.continuation_rewards.get(critique_index, 0.0)
                baseline = prompt_pass_at_1[bundle.prompt_group_id]
                learnability = bundle.learnability.get(critique_index)
                learnability_weight = learnability.reward_weight if learnability is not None else 0.0
                accepted = bool(learnability is not None and learnability.accepted)
                if bundle.record.objective == "recovery":
                    objective_credit = continuation_outcome
                    critique_reward = objective_credit * learnability_weight - baseline
                elif bundle.record.objective == "compression":
                    objective_credit = bundle.compression_credits.get(critique_index, 0.0)
                    critique_reward = objective_credit * learnability_weight
                else:
                    raise RuntimeError(f"unknown branch-revision objective {bundle.record.objective!r}")
                rows.append(
                    _ActorRow(
                        audit_row_id=f"critique:{bundle.rollout_id}:{critique_index}",
                        full_ids=[*critique_prompt, *critique.token_ids],
                        train_start=len(critique_prompt),
                        behavior_log_probs=list(critique.log_probs),
                        reward=critique_reward,
                        group_id=critique_group,
                        kind="critique",
                    )
                )
                if critique.valid and accepted:
                    rows.append(
                        _ActorRow(
                            audit_row_id=f"continuation:{bundle.rollout_id}:{critique_index}",
                            full_ids=[*bundle.prompt_ids, *critique.revised_prefix_ids, *critique.continuation_ids],
                            train_start=len(bundle.prompt_ids) + len(critique.revised_prefix_ids),
                            behavior_log_probs=list(critique.continuation_log_probs),
                            reward=continuation_outcome,
                            group_id=solution_group,
                            kind="continuation",
                        )
                    )
                self._audit(
                    "critique",
                    actor_row_id=f"critique:{bundle.rollout_id}:{critique_index}",
                    continuation_actor_row_id=f"continuation:{bundle.rollout_id}:{critique_index}",
                    rollout_id=bundle.rollout_id,
                    prompt_group_id=bundle.prompt_group_id,
                    objective=bundle.record.objective,
                    critique_index=critique_index,
                    reward=critique_reward,
                    objective_credit=objective_credit,
                    continuation_outcome=continuation_outcome,
                    prompt_pass_at_1=baseline,
                    learnability_accepted=accepted,
                    learnability_percentile=learnability.percentile if learnability is not None else None,
                    learnability_weight=learnability_weight,
                    compression_fraction=bundle.compression_fractions.get(critique_index),
                    compression_credit=bundle.compression_credits.get(critique_index),
                    generated_continuation_tokens=len(critique.continuation_ids),
                    continuation_reward_evaluated=critique_index in bundle.continuation_rewards,
                    continuation_wasted_by_learnability=False,
                    continuation_skipped_by_learnability=bool(critique.valid and not accepted),
                    parse_reason=critique.parse_reason,
                    prefix=critique.prefix_text,
                    prefix_plus_new_continuation=critique.prefix_plus_new_continuation_text,
                    new_continuation=critique.new_continuation_text,
                    branch_prefix_ids=list(critique.branch_prefix_ids),
                    prefix_ids=list(critique.prefix_ids),
                    continuation_prefix_ids=list(critique.continuation_prefix_ids),
                    new_continuation_ids=list(critique.new_continuation_ids),
                    new_continuation_log_probs=_float32_list(critique.new_continuation_log_probs)
                    if critique.new_continuation_log_probs
                    else [],
                    revised_prefix_ids=list(critique.revised_prefix_ids),
                    generated_continuation_ids=list(critique.continuation_ids),
                    generated_continuation_log_probs=_float32_list(critique.continuation_log_probs)
                    if critique.continuation_log_probs
                    else [],
                    critique_ids=list(critique.token_ids),
                    critique_log_probs=_float32_list(critique.log_probs),
                    critique_prompt_ids=critique_prompt,
                    finish_reason=critique.finish_reason,
                )
        return rows

    def _make_policy_batch(self, rows: list[_ActorRow], *, worker_group) -> tuple[DataProto, int]:
        if not rows:
            raise RuntimeError("branch-revision policy batch has no trainable rows")
        max_sequence = max(len(row.full_ids) for row in rows)
        if max_sequence > int(self.config.actor_rollout_ref.rollout.max_model_len):
            raise ValueError("branch-revision packed actor sequence exceeds rollout.max_model_len")
        response_width = max_sequence - 1
        if response_width <= 0:
            raise ValueError("branch-revision actor sequence needs at least two context tokens")
        dp_size = self.trainer._get_dp_size(worker_group, "actor")
        padding_rows = (-len(rows)) % dp_size
        total_rows = len(rows) + padding_rows
        pad_id = self._pad_token_id()
        prompts = torch.full((total_rows, 1), pad_id, dtype=torch.long)
        responses = torch.full((total_rows, response_width), pad_id, dtype=torch.long)
        response_mask = torch.zeros((total_rows, response_width), dtype=torch.long)
        attention_mask = torch.zeros((total_rows, max_sequence), dtype=torch.long)
        old_log_probs = torch.zeros((total_rows, response_width), dtype=torch.float32)
        rollout_log_probs = torch.zeros_like(old_log_probs)
        token_level_rewards = torch.zeros_like(old_log_probs)
        group_ids: list[str] = []
        kinds: list[str] = []
        audit_row_ids: list[str] = []
        scalar_rewards: list[float] = []
        for row_index, row in enumerate(rows):
            behavior = row.behavior_log_probs
            if not behavior or row.train_start <= 0 or row.train_start + len(behavior) != len(row.full_ids):
                raise RuntimeError("branch-revision actor behavior span is inconsistent with packed context")
            if not all(math.isfinite(value) for value in behavior):
                raise RuntimeError("branch-revision actor row contains non-finite behavior log probabilities")
            prompts[row_index, 0] = row.full_ids[0]
            responses[row_index, : len(row.full_ids) - 1] = torch.tensor(row.full_ids[1:], dtype=torch.long)
            attention_mask[row_index, : len(row.full_ids)] = 1
            start = row.train_start - 1
            stop = start + len(behavior)
            response_mask[row_index, start:stop] = 1
            values = torch.tensor(behavior, dtype=torch.float32)
            old_log_probs[row_index, start:stop] = values
            rollout_log_probs[row_index, start:stop] = values
            token_level_rewards[row_index, stop - 1] = row.reward
            group_ids.append(row.group_id)
            kinds.append(row.kind)
            audit_row_ids.append(row.audit_row_id)
            scalar_rewards.append(row.reward)
        for padding_index in range(padding_rows):
            row_index = len(rows) + padding_index
            attention_mask[row_index, 0] = 1
            group_ids.append(f"padding:{padding_index}")
            kinds.append("padding")
            audit_row_ids.append(f"padding:{padding_index}")
            scalar_rewards.append(0.0)

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=np.array(group_ids, dtype=object),
            norm_adv_by_std_in_grpo=bool(self.config.algorithm.norm_adv_by_std_in_grpo),
            config=self.config.algorithm,
        )
        input_ids = torch.cat([prompts, responses], dim=1)
        actor_batch = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
                "old_log_probs": old_log_probs,
                "rollout_log_probs": rollout_log_probs,
                "token_level_rewards": token_level_rewards,
                "advantages": advantages,
                "returns": returns,
            },
            non_tensors={
                "uid": np.array(group_ids, dtype=object),
                "branch_revision_actor_kind": np.array(kinds, dtype=object),
                "branch_revision_audit_row_id": np.array(audit_row_ids, dtype=object),
                "branch_revision_reward": np.array(scalar_rewards, dtype=np.float32),
            },
        )
        actor_batch.meta_info.update(
            {
                "temperature": 1.0,
                "global_token_num": attention_mask.sum(dim=-1).tolist(),
                "use_global_loss_normalization": True,
            }
        )
        return actor_batch, padding_rows

    def _make_actor_batch(self, bundles: list[_Bundle]) -> tuple[DataProto, int]:
        """Backwards-compatible packer for the shared-policy path and focused tests."""
        return self._make_policy_batch(
            self._actor_rows(bundles),
            worker_group=self.trainer.actor_rollout_wg,
        )

    def _audit_actor_batch(self, actor_batch: DataProto, *, padding_rows: int, policy: str = "actor") -> None:
        kinds = actor_batch.non_tensor_batch["branch_revision_actor_kind"]
        group_ids = actor_batch.non_tensor_batch["uid"]
        audit_row_ids = actor_batch.non_tensor_batch["branch_revision_audit_row_id"]
        scalar_rewards = actor_batch.non_tensor_batch["branch_revision_reward"]
        if len(set(str(value) for value in audit_row_ids)) != len(actor_batch):
            raise RuntimeError("branch-revision actor audit row IDs must be unique after balancing")
        actor_rows: list[dict[str, Any]] = []
        for row_index in range(len(actor_batch)):
            attention_mask = actor_batch.batch["attention_mask"][row_index].bool()
            input_ids = actor_batch.batch["input_ids"][row_index][attention_mask].detach().cpu().tolist()
            response_mask = actor_batch.batch["response_mask"][row_index].to(dtype=torch.uint8)
            trained = response_mask.bool()
            trained_positions = trained.nonzero(as_tuple=False).flatten()
            train_start = int(trained_positions[0].item() + 1) if trained_positions.numel() else None
            train_stop = int(trained_positions[-1].item() + 2) if trained_positions.numel() else None
            old_log_probs = actor_batch.batch["old_log_probs"][row_index][trained]
            rollout_log_probs = actor_batch.batch["rollout_log_probs"][row_index][trained]
            actor_rows.append(
                {
                    "balanced_row_index": row_index,
                    "actor_row_id": str(audit_row_ids[row_index]),
                    "kind": str(kinds[row_index]),
                    "group_id": str(group_ids[row_index]),
                    "reward": float(scalar_rewards[row_index]),
                    "sequence_length": len(input_ids),
                    "response_width": int(response_mask.numel()),
                    "train_start": train_start,
                    "train_stop": train_stop,
                    "input_ids_sha256": _canonical_sha256(input_ids, dtype="<i8"),
                    "response_mask_sha256": _canonical_sha256(response_mask.detach().cpu().tolist(), dtype="u1"),
                    "old_log_probs_sha256": _canonical_sha256(old_log_probs.detach().cpu().tolist(), dtype="<f4"),
                    "rollout_log_probs_sha256": _canonical_sha256(
                        rollout_log_probs.detach().cpu().tolist(), dtype="<f4"
                    ),
                }
            )
        self._audit(
            "actor_batch",
            policy=policy,
            rows=len(actor_batch) - padding_rows,
            original=int(np.sum(kinds == "original")),
            critiques=int(np.sum(kinds == "critique")),
            continuations=int(np.sum(kinds == "continuation")),
            padding=padding_rows,
            pad_token_id=self._pad_token_id(),
            policy_loss_mode=str(self.config.actor_rollout_ref.actor.policy_loss.loss_mode),
            clip_ratio=float(self.config.actor_rollout_ref.actor.clip_ratio),
            clip_ratio_low=self.config.actor_rollout_ref.actor.clip_ratio_low,
            clip_ratio_high=self.config.actor_rollout_ref.actor.clip_ratio_high,
            clip_ratio_c=float(self.config.actor_rollout_ref.actor.clip_ratio_c),
            actor_rows=actor_rows,
        )

    @staticmethod
    def _set_source_metric_advantages(
        source: DataProto,
        bundles: list[_Bundle],
        actor_batch: DataProto,
    ) -> None:
        """Expose the actual original-row GRPO values to native post-step metrics."""

        kinds = actor_batch.non_tensor_batch["branch_revision_actor_kind"]
        original_rows = [index for index, kind in enumerate(kinds) if kind == "original"]
        if len(original_rows) != len(bundles):
            raise RuntimeError("branch-revision actor batch lost or duplicated original rows")
        source_advantages = torch.zeros_like(source.batch["response_mask"], dtype=torch.float32)
        source_returns = torch.zeros_like(source_advantages)
        for bundle, actor_row in zip(bundles, original_rows, strict=True):
            actor_mask = actor_batch.batch["response_mask"][actor_row].bool()
            row_advantages = actor_batch.batch["advantages"][actor_row][actor_mask]
            row_returns = actor_batch.batch["returns"][actor_row][actor_mask]
            token_count = len(bundle.solution_ids)
            if row_advantages.numel() != token_count or row_returns.numel() != token_count:
                raise RuntimeError("branch-revision original metric span does not match its solution")
            source_mask = source.batch["response_mask"][bundle.source_row].bool()
            if int(source_mask.sum().item()) != token_count:
                raise RuntimeError("branch-revision source response mask changed after bundle construction")
            source_advantages[bundle.source_row, source_mask] = row_advantages.to(source_advantages.device)
            source_returns[bundle.source_row, source_mask] = row_returns.to(source_returns.device)
        source.batch["advantages"] = source_advantages
        source.batch["returns"] = source_returns

    @staticmethod
    def _set_zero_source_metric_advantages(source: DataProto) -> None:
        source.batch["advantages"] = torch.zeros_like(source.batch["response_mask"], dtype=torch.float32)
        source.batch["returns"] = torch.zeros_like(source.batch["response_mask"], dtype=torch.float32)

    def _critique_warmup_active(self) -> bool:
        return int(self.trainer.global_steps) <= self.feature.critique_warmup_steps

    def _policy_rows(self, bundles: list[_Bundle]) -> tuple[list[_ActorRow], list[_ActorRow]]:
        rows = self._actor_rows(bundles)
        critique_rows = [row for row in rows if row.kind == "critique"]
        if self.feature.separate_critique_model:
            actor_rows = [] if self._critique_warmup_active() else [row for row in rows if row.kind != "critique"]
            return actor_rows, critique_rows
        if self._critique_warmup_active():
            return critique_rows, []
        return rows, []

    @staticmethod
    def _critique_policy_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        renamed: dict[str, Any] = {}
        for key, value in metrics.items():
            if key.startswith("actor/"):
                key = f"critique_actor/{key.removeprefix('actor/')}"
            elif key == "perf/mfu/actor":
                key = "perf/mfu/critique_actor"
            renamed[key] = value
        return renamed

    def _metrics(
        self,
        bundles: list[_Bundle],
        actor_batch: DataProto | None,
        padding_rows: int,
        *,
        critique_batch: DataProto | None = None,
        critique_padding_rows: int = 0,
    ) -> dict[str, float]:
        originals = [bundle.original_reward for bundle in bundles]
        prompt_rewards = self._prompt_rewards(bundles)
        prompt_pass_at_1 = {
            prompt_group_id: sum(rewards) / len(rewards) for prompt_group_id, rewards in prompt_rewards.items()
        }
        incorrect = [bundle for bundle in bundles if bundle.original_reward == 0.0]
        correct = [bundle for bundle in bundles if bundle.original_reward == 1.0]
        selected = [bundle for bundle in bundles if bundle.record is not None]
        critiques = [critique for bundle in selected for critique in bundle.record.critiques]
        incorrect_critiques = [critique for bundle in incorrect for critique in bundle.record.critiques]
        correct_critiques = [critique for bundle in correct if bundle.record for critique in bundle.record.critiques]
        valid_count = sum(critique.valid for critique in critiques)
        accepted_count = sum(score.accepted for bundle in selected for score in bundle.learnability.values())
        incorrect_accepted = sum(score.accepted for bundle in incorrect for score in bundle.learnability.values())
        incorrect_valid = sum(
            critique.valid for bundle in incorrect if bundle.record is not None for critique in bundle.record.critiques
        )
        successes = sum(sum(bundle.continuation_rewards.values()) for bundle in incorrect)
        incorrect_any = sum(any(value == 1.0 for value in bundle.continuation_rewards.values()) for bundle in incorrect)
        prompts_with_incorrect = {bundle.prompt_group_id for bundle in incorrect}
        prompts_with_success = {
            bundle.prompt_group_id
            for bundle in incorrect
            if any(value == 1.0 for value in bundle.continuation_rewards.values())
        }
        all_prompts = {bundle.prompt_group_id for bundle in bundles}
        parse_counts = Counter(critique.parse_reason for critique in critiques)
        learnability_scores = [score for bundle in selected for score in bundle.learnability.values()]
        finite_stddev_distances = [
            score.stddevs_below_mean
            for score in learnability_scores
            if score.stddevs_below_mean is not None and math.isfinite(score.stddevs_below_mean)
        ]
        compression_fractions = [value for bundle in correct for value in bundle.compression_fractions.values()]
        compression_credits = [value for bundle in correct for value in bundle.compression_credits.values()]
        self_critique_rewards = [
            bundle.continuation_rewards.get(critique_index, 0.0) - prompt_pass_at_1[bundle.prompt_group_id]
            for bundle in selected
            for critique_index, _critique in enumerate(bundle.record.critiques)
        ]
        generated_continuation_tokens = [
            len(critique.continuation_ids)
            for bundle in selected
            for critique in bundle.record.critiques
            if critique.valid
        ]
        rejected_continuation_tokens = [
            len(critique.continuation_ids)
            for bundle in selected
            for critique_index, critique in enumerate(bundle.record.critiques)
            if critique.valid and not bundle.learnability[critique_index].accepted
        ]
        policy_batches = [batch for batch in (actor_batch, critique_batch) if batch is not None]
        actor_kinds = (
            np.concatenate([batch.non_tensor_batch["branch_revision_actor_kind"] for batch in policy_batches])
            if policy_batches
            else np.array([], dtype=object)
        )
        global_minibatch = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(
            self.config.actor_rollout_ref.rollout.n
        )
        total_padding_rows = padding_rows + critique_padding_rows
        total_policy_rows = sum(len(batch) for batch in policy_batches)
        optimizer_minibatches = sum(
            math.ceil(len(batch) / global_minibatch) * int(self.config.actor_rollout_ref.actor.ppo_epochs)
            for batch in policy_batches
        )
        actor_input_tokens = sum(int(batch.batch["attention_mask"].sum().item()) for batch in policy_batches)
        actor_train_tokens = sum(int(batch.batch["response_mask"].sum().item()) for batch in policy_batches)
        critique_group_ids: list[str] = []
        critique_rewards: list[float] = []
        critique_advantages: list[float] = []
        for batch in policy_batches:
            kinds = batch.non_tensor_batch["branch_revision_actor_kind"]
            group_ids = batch.non_tensor_batch["uid"]
            rewards = batch.non_tensor_batch["branch_revision_reward"]
            for row_index, kind in enumerate(kinds):
                if kind != "critique":
                    continue
                mask = batch.batch["response_mask"][row_index].bool()
                row_advantages = batch.batch["advantages"][row_index][mask]
                if row_advantages.numel() == 0:
                    raise RuntimeError("branch-revision critique row has no trained advantage tokens")
                first_advantage = float(row_advantages[0].item())
                if not torch.allclose(row_advantages, row_advantages.new_full(row_advantages.shape, first_advantage)):
                    raise RuntimeError("branch-revision GRPO critique advantage is not constant across trained tokens")
                critique_group_ids.append(str(group_ids[row_index]))
                critique_rewards.append(float(rewards[row_index]))
                critique_advantages.append(first_advantage)
        critique_group_sizes = list(Counter(critique_group_ids).values())

        def distribution_metrics(values: list[float], prefix: str) -> dict[str, float]:
            if not values:
                return {
                    f"{prefix}/mean": 0.0,
                    f"{prefix}/std": 0.0,
                    f"{prefix}/min": 0.0,
                    f"{prefix}/max": 0.0,
                }
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}/mean": float(np.mean(array)),
                f"{prefix}/std": float(np.std(array, ddof=0)),
                f"{prefix}/min": float(np.min(array)),
                f"{prefix}/max": float(np.max(array)),
            }

        metrics = {
            "branch_revision/original/pass_at_1": float(sum(originals) / len(originals)),
            "branch_revision/self_critique_reward/mean": (
                float(sum(self_critique_rewards) / len(self_critique_rewards)) if self_critique_rewards else 0.0
            ),
            "branch_revision/flip/success_per_all_critiques": (
                float(successes / len(incorrect_critiques)) if incorrect_critiques else 0.0
            ),
            "branch_revision/flip/success_per_valid_continuation": (
                float(successes / incorrect_accepted) if incorrect_accepted else 0.0
            ),
            "branch_revision/flip/success_per_continuation": (
                float(successes / incorrect_valid) if incorrect_valid else 0.0
            ),
            "branch_revision/flip/incorrect_originals_with_any_success": (
                float(incorrect_any / len(incorrect)) if incorrect else 0.0
            ),
            "branch_revision/flip/prompts_with_any_success": (
                float(len(prompts_with_success) / len(prompts_with_incorrect)) if prompts_with_incorrect else 0.0
            ),
            "branch_revision/flip/prompts_with_any_success_all_prompts": float(
                len(prompts_with_success) / len(all_prompts)
            ),
            "branch_revision/originals": float(len(bundles)),
            "branch_revision/incorrect_originals": float(len(incorrect)),
            "branch_revision/correct_originals": float(len(correct)),
            "branch_revision/critiques": float(len(critiques)),
            "branch_revision/recovery_critiques": float(len(incorrect_critiques)),
            "branch_revision/compression_critiques": float(len(correct_critiques)),
            "branch_revision/valid_edits": float(valid_count),
            "branch_revision/learnability_accepted_edits": float(accepted_count),
            "branch_revision/learnability_rejected_edits": float(valid_count - accepted_count),
            "branch_revision/continuations": float(accepted_count),
            "branch_revision/learnability/mean_percentile": (
                float(sum(score.percentile for score in learnability_scores) / len(learnability_scores))
                if learnability_scores
                else 0.0
            ),
            "branch_revision/learnability/mean_reward_weight": (
                float(sum(score.reward_weight for score in learnability_scores) / len(learnability_scores))
                if learnability_scores
                else 0.0
            ),
            "branch_revision/learnability/mean_stddevs_below_mean": (
                float(sum(finite_stddev_distances) / len(finite_stddev_distances)) if finite_stddev_distances else 0.0
            ),
            "branch_revision/learnability/max_stddevs_below_mean": (
                float(max(finite_stddev_distances)) if finite_stddev_distances else 0.0
            ),
            "branch_revision/compression/mean_fraction": (
                float(sum(compression_fractions) / len(compression_fractions)) if compression_fractions else 0.0
            ),
            "branch_revision/compression/mean_credit": (
                float(sum(compression_credits) / len(compression_credits)) if compression_credits else 0.0
            ),
            "branch_revision/tokens/generated_continuations": float(sum(generated_continuation_tokens)),
            "branch_revision/tokens/learnability_rejected_continuations": float(sum(rejected_continuation_tokens)),
            "branch_revision/actor_rows": float(total_policy_rows - total_padding_rows),
            "branch_revision/padding_rows": float(total_padding_rows),
            "branch_revision/actor_optimizer_minibatches": float(optimizer_minibatches),
            "branch_revision/tokens/actor_input": float(actor_input_tokens),
            "branch_revision/tokens/actor_train": float(actor_train_tokens),
            "branch_revision/actor_original_rows": float(np.sum(actor_kinds == "original")),
            "branch_revision/actor_critique_rows": float(np.sum(actor_kinds == "critique")),
            "branch_revision/actor_continuation_rows": float(np.sum(actor_kinds == "continuation")),
            "branch_revision/main_actor_rows": float(0.0 if actor_batch is None else len(actor_batch) - padding_rows),
            "branch_revision/critique_model_rows": float(
                0.0 if critique_batch is None else len(critique_batch) - critique_padding_rows
            ),
            "branch_revision/critique_warmup_active": float(self._critique_warmup_active()),
            "branch_revision/separate_critique_model": float(self.feature.separate_critique_model),
            "branch_revision/critique_grpo_grouping_is_batch": float(self.feature.critique_grpo_grouping == "batch"),
            "branch_revision/critique_grpo_group_count": float(len(critique_group_sizes)),
            "branch_revision/critique_grpo_group_size_mean": (
                float(sum(critique_group_sizes) / len(critique_group_sizes)) if critique_group_sizes else 0.0
            ),
            "branch_revision/critique_grpo_group_size_max": float(max(critique_group_sizes, default=0)),
            "branch_revision/critique_advantage/positive_fraction": (
                float(sum(value > 0.0 for value in critique_advantages) / len(critique_advantages))
                if critique_advantages
                else 0.0
            ),
            "branch_revision/critique_advantage/negative_fraction": (
                float(sum(value < 0.0 for value in critique_advantages) / len(critique_advantages))
                if critique_advantages
                else 0.0
            ),
            "branch_revision/critique_advantage/zero_fraction": (
                float(sum(value == 0.0 for value in critique_advantages) / len(critique_advantages))
                if critique_advantages
                else 0.0
            ),
            "branch_revision/policy_loss_is_dppo_tv": float(
                str(self.config.actor_rollout_ref.actor.policy_loss.loss_mode) == "dppo_tv"
            ),
        }
        metrics.update(distribution_metrics(critique_rewards, "branch_revision/critique_reward"))
        metrics.update(distribution_metrics(critique_advantages, "branch_revision/critique_advantage"))
        denominator = len(critiques) or 1
        for reason, count in sorted(parse_counts.items()):
            metrics[f"branch_revision/parser/{reason}"] = float(count / denominator)
        return metrics

    def run_update(
        self,
        source: DataProto,
        reward_tensor: torch.Tensor,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
        *,
        profile_rollout: bool = False,
    ) -> bool:
        rewards = self._original_rewards(reward_tensor)
        bundles = self._build_bundles(source, rewards)
        self._audit_originals(bundles)
        request = self._make_child_request(source, bundles)
        if request is not None:
            critique_rollout_manager = (
                self.trainer.critique_async_rollout_manager
                if self.feature.separate_critique_model
                else self.trainer.async_rollout_manager
            )

            def generate_critiques() -> None:
                output = critique_rollout_manager.generate_sequences(request)
                timing_raw.update(
                    {
                        f"branch_revision_critique/{key}": value
                        for key, value in output.meta_info.get("timing", {}).items()
                    }
                )
                self._attach_records(bundles, self._extract_records(output))

            def generate_actor_followups() -> None:
                score_request = self._make_score_request(source, bundles)
                if score_request is None:
                    self._attach_scores(bundles, {})
                else:
                    score_output = self.trainer.async_rollout_manager.generate_sequences(score_request)
                    timing_raw.update(
                        {
                            f"branch_revision_score/{key}": value
                            for key, value in score_output.meta_info.get("timing", {}).items()
                        }
                    )
                    self._attach_scores(bundles, self._extract_scores(score_output))

                with marked_timer("branch_revision_learnability", timing_raw, color="blue"):
                    self._score_seed_learnability(bundles)

                continuation_request = self._make_continuation_request(source, bundles)
                if continuation_request is None:
                    self._attach_continuations(bundles, {})
                else:
                    continuation_output = self.trainer.async_rollout_manager.generate_sequences(continuation_request)
                    timing_raw.update(
                        {
                            f"branch_revision_continuation/{key}": value
                            for key, value in continuation_output.meta_info.get("timing", {}).items()
                        }
                    )
                    self._attach_continuations(bundles, self._extract_continuations(continuation_output))

            with marked_timer("branch_revision_children", timing_raw, color="red"):
                if self.feature.separate_critique_model:
                    self._run_with_critique_rollout_lifecycle(
                        generate_critiques,
                        profile_rollout=profile_rollout,
                    )
                    self._run_with_rollout_lifecycle(
                        generate_actor_followups,
                        profile_rollout=profile_rollout,
                    )
                else:

                    def run_shared_child_pipeline() -> None:
                        generate_critiques()
                        generate_actor_followups()

                    self._run_with_rollout_lifecycle(
                        run_shared_child_pipeline,
                        profile_rollout=profile_rollout,
                    )
        else:
            self._attach_records(bundles, {})
            self._attach_scores(bundles, {})
            with marked_timer("branch_revision_learnability", timing_raw, color="blue"):
                self._score_seed_learnability(bundles)
            self._attach_continuations(bundles, {})
        with marked_timer("branch_revision_rewards", timing_raw, color="yellow"):
            self._evaluate_continuations(source, bundles)

        actor_rows, critique_rows = self._policy_rows(bundles)
        actor_batch: DataProto | None = None
        critique_batch: DataProto | None = None
        padding_rows = 0
        critique_padding_rows = 0
        if actor_rows:
            actor_batch, padding_rows = self._make_policy_batch(
                actor_rows,
                worker_group=self.trainer.actor_rollout_wg,
            )
        if critique_rows:
            if not self.feature.separate_critique_model:
                raise RuntimeError("shared critique rows must be routed through the main actor batch")
            if not hasattr(self.trainer, "critique_actor_rollout_wg"):
                raise RuntimeError("separate critique policy worker group is not initialized")
            critique_batch, critique_padding_rows = self._make_policy_batch(
                critique_rows,
                worker_group=self.trainer.critique_actor_rollout_wg,
            )

        if actor_batch is not None and np.any(actor_batch.non_tensor_batch["branch_revision_actor_kind"] == "original"):
            self._set_source_metric_advantages(source, bundles, actor_batch)
        else:
            self._set_zero_source_metric_advantages(source)

        if self.config.trainer.balance_batch:
            if actor_batch is not None:
                self.trainer._balance_batch(
                    actor_batch,
                    metrics=metrics,
                    logging_prefix="branch_revision_actor_global_seqlen",
                    worker_group=self.trainer.actor_rollout_wg,
                    role="actor",
                )
            if critique_batch is not None:
                self.trainer._balance_batch(
                    critique_batch,
                    metrics=metrics,
                    logging_prefix="branch_revision_critique_actor_global_seqlen",
                    worker_group=self.trainer.critique_actor_rollout_wg,
                    role="actor",
                )
        if actor_batch is not None:
            self._audit_actor_batch(actor_batch, padding_rows=padding_rows, policy="actor")
        if critique_batch is not None:
            self._audit_actor_batch(
                critique_batch,
                padding_rows=critique_padding_rows,
                policy="critique_actor",
            )

        critique_actor_updated = False
        if critique_batch is not None:
            with marked_timer("update_critique_actor", timing_raw, color="red"):
                critique_output = self.trainer._update_actor(
                    critique_batch,
                    worker_group=self.trainer.critique_actor_rollout_wg,
                )
            critique_metrics = reduce_metrics(critique_output.meta_info["metrics"])
            metrics.update(self._critique_policy_metrics(critique_metrics))
            critique_actor_updated = True

        actor_updated = False
        if actor_batch is not None:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self.trainer._update_actor(actor_batch)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
            actor_updated = True

        metrics["branch_revision/main_actor_updated"] = float(actor_updated)
        metrics["branch_revision/critique_actor_updated"] = float(critique_actor_updated)
        metrics.update(
            self._metrics(
                bundles,
                actor_batch,
                padding_rows,
                critique_batch=critique_batch,
                critique_padding_rows=critique_padding_rows,
            )
        )
        self._audit(
            "iteration",
            originals=len(bundles),
            incorrect=sum(bundle.original_reward == 0.0 for bundle in bundles),
            correct=sum(bundle.original_reward == 1.0 for bundle in bundles),
            positive_compression_enabled=self.feature.enable_positive_compression,
            critique_grpo_grouping=self.feature.critique_grpo_grouping,
            learnability_logprob_statistic=self.feature.learnability_logprob_statistic,
            learnability_threshold_mode=self.feature.learnability_threshold_mode,
            max_seed_window_stddevs=self.feature.max_seed_window_stddevs,
            separate_critique_model=self.feature.separate_critique_model,
            critique_warmup_steps=self.feature.critique_warmup_steps,
            critique_warmup_active=self._critique_warmup_active(),
            main_actor_updated=actor_updated,
            critique_actor_updated=critique_actor_updated,
            original_rewards=rewards,
            prompt_pass_at_1={
                prompt_group_id: sum(group_rewards) / len(group_rewards)
                for prompt_group_id, group_rewards in self._prompt_rewards(bundles).items()
            },
        )
        self._audit("step_complete")
        return actor_updated
