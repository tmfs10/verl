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
"""Native-RayPPO integration for synchronous intermediate Monte Carlo values."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from transformers import AutoConfig

from verl import DataProto
from verl.experimental.agent_loop.intermediate_mc_agent_loop import (
    INTERMEDIATE_MC_AGENT_NAME,
    INTERMEDIATE_MC_CHILD_FIELD,
    ContinuationGeneration,
    CritiqueGeneration,
    IntermediateMCGenerationRecord,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.config import INTERMEDIATE_MC_CRITIQUE_PROMPT, IntermediateMCValueConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.intermediate_mc_value import (
    CRITIQUE_DELIMITER,
    SOLUTION_DELIMITER,
    CriticContext,
    VarianceCandidate,
    aggregate_mark_targets,
    build_critic_context,
    build_unconditioned_critic_context,
    candidate_bounds,
    critique_accuracy_reward,
    critique_group_advantages,
    initial_state_target,
    select_ema_marks,
    select_variance_marks,
    stable_rng,
    validate_reward,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fs import is_non_local
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import marked_timer


def _tokenizer_fingerprint(tokenizer) -> str:
    payload = {
        "vocab": sorted((str(token), int(index)) for token, index in tokenizer.get_vocab().items()),
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _positive_model_limit(model_path: str, config, *, role: str) -> int:
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=config.model.get("trust_remote_code", False),
    )
    if _is_moe_model(model_config):
        raise ValueError(f"intermediate MC initially supports only dense {role} models")
    override = config.model.get("override_config", {})
    limit = override.get("max_position_embeddings", getattr(model_config, "max_position_embeddings", None))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"{role} model must declare a positive max_position_embeddings")
    return int(limit)


def _is_moe_model(model_config) -> bool:
    for name in ("num_experts", "num_local_experts", "n_routed_experts"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and value > 1:
            return True
    return getattr(model_config, "moe_intermediate_size", None) is not None


def configured_optimizer_rows(config, feature: IntermediateMCValueConfig) -> dict[str, int]:
    """Return deterministic global row counts for one intermediate-MC update."""

    configured_generation_batch = config.data.get("gen_batch_size", config.data.train_batch_size)
    if (
        isinstance(configured_generation_batch, bool)
        or not isinstance(configured_generation_batch, int)
        or configured_generation_batch <= 0
    ):
        raise ValueError("intermediate MC requires a positive integer data.gen_batch_size or data.train_batch_size")
    rollout_n = config.actor_rollout_ref.rollout.n
    if isinstance(rollout_n, bool) or not isinstance(rollout_n, int) or rollout_n <= 0:
        raise ValueError("intermediate MC requires actor_rollout_ref.rollout.n to be a positive integer")
    solution_rows = configured_generation_batch * rollout_n
    return {
        "solutions": solution_rows,
        "critic": solution_rows * feature.num_critic_streams,
        "actor": solution_rows * (1 + feature.num_critiques),
        "actor_critiques": solution_rows * feature.num_critiques,
    }


def _validate_configured_optimizer_rows(config, feature: IntermediateMCValueConfig) -> None:
    rows = configured_optimizer_rows(config, feature)
    total_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    if total_gpus <= 0:
        raise ValueError("intermediate MC requires a positive trainer GPU count")
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    role_configs = {
        "actor": config.actor_rollout_ref.actor,
        "critic": config.critic,
    }
    for role, role_config in role_configs.items():
        sequence_parallel = int(role_config.get("ulysses_sequence_parallel_size", 1))
        if sequence_parallel <= 0 or total_gpus % sequence_parallel != 0:
            raise ValueError(
                f"intermediate MC {role} Ulysses size ({sequence_parallel}) must divide total GPUs ({total_gpus})"
            )
        dp_size = total_gpus // sequence_parallel
        configured_minibatch = role_config.ppo_mini_batch_size
        if (
            isinstance(configured_minibatch, bool)
            or not isinstance(configured_minibatch, int)
            or configured_minibatch <= 0
        ):
            raise ValueError(f"intermediate MC {role} PPO minibatch must be a positive integer")
        global_minibatch = configured_minibatch * rollout_n
        if global_minibatch % dp_size != 0:
            raise ValueError(
                f"intermediate MC {role} global PPO minibatch ({global_minibatch}) must be divisible by "
                f"{role} DP size ({dp_size})"
            )
        role_rows = rows[role]
        if role_rows % dp_size != 0:
            raise ValueError(
                f"intermediate MC configured {role} rows ({role_rows}) must be divisible by "
                f"{role} DP size ({dp_size}); optimizer padding is forbidden"
            )
        if role_rows % global_minibatch != 0:
            raise ValueError(
                f"intermediate MC configured {role} rows ({role_rows}) must be divisible by "
                f"global PPO minibatch ({global_minibatch}); optimizer padding is forbidden"
            )


def validate_intermediate_mc_runtime_config(
    config,
    actor_tokenizer=None,
    critic_tokenizer=None,
    actor_model_path: str | None = None,
) -> None:
    """Fail closed before worker allocation for unsupported combinations."""

    feature = omega_conf_to_dataclass(
        config.algorithm.intermediate_mc_value,
        dataclass_type=IntermediateMCValueConfig,
    )
    if not feature.enable:
        return
    reward_source = str(OmegaConf.select(config, "reward.reward_manager.source", default="register"))
    reward_name = str(OmegaConf.select(config, "reward.reward_manager.name", default=""))
    if reward_source == "register" and reward_name == "conditional_logprob":
        raise ValueError("intermediate MC does not support the registered conditional_logprob training reward manager")
    critic_paths = {
        "critic.model.path": config.critic.model.path,
        "critic.model.tokenizer_path": config.critic.model.tokenizer_path,
    }
    for name, raw_path in critic_paths.items():
        if raw_path is not None and is_non_local(str(raw_path)):
            raise ValueError(f"intermediate MC requires a local or Hugging Face {name}; HDFS is unsupported")
    _validate_configured_optimizer_rows(config, feature)
    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError("intermediate MC currently supports only VeRL's legacy FSDP/FSDP2 workers")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("intermediate MC actor strategy must be fsdp or fsdp2")
    if config.critic.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("intermediate MC critic strategy must be fsdp or fsdp2")
    if config.actor_rollout_ref.rollout.name != "vllm":
        raise ValueError("intermediate MC initially supports only the dense vLLM rollout engine")
    if config.critic.get("enable", None) is False:
        raise ValueError("intermediate MC requires critic.enable=true")
    if str(config.algorithm.adv_estimator).lower() != "gae":
        raise ValueError("intermediate MC requires algorithm.adv_estimator=gae")
    if float(config.algorithm.gamma) != 1.0:
        raise ValueError("intermediate MC raw terminal-reward targets require algorithm.gamma=1")
    warmup = config.trainer.critic_warmup
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("trainer.critic_warmup must be a non-negative integer")
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    get_policy_loss_fn(config.actor_rollout_ref.actor.policy_loss.loss_mode)
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        raise ValueError("intermediate MC does not support actor KL or KL-in-reward")
    rollout_correction = config.algorithm.get("rollout_correction", None)
    if rollout_correction is not None and any(
        (
            rollout_correction.get("rollout_is", None) is not None,
            rollout_correction.get("rollout_rs", None) is not None,
            bool(rollout_correction.get("bypass_mode", False)),
        )
    ):
        raise ValueError("intermediate MC rejects rollout correction and uses recorded behavior log probabilities")
    rollout = config.actor_rollout_ref.rollout
    if float(rollout.temperature) != 1.0 or float(rollout.val_kwargs.temperature) != 1.0:
        raise ValueError("all intermediate MC generation, including validation, requires temperature=1.0")
    if rollout.max_model_len is None or int(rollout.max_model_len) <= 0:
        raise ValueError("intermediate MC requires an explicit positive actor_rollout_ref.rollout.max_model_len")
    if str(rollout.get("logprobs_mode", "")) != "processed_logprobs":
        raise ValueError("intermediate MC requires rollout.logprobs_mode=processed_logprobs")
    if int(config.data.max_prompt_length) + int(config.data.max_response_length) > int(rollout.max_model_len):
        raise ValueError("configured prompt plus response lengths exceed rollout.max_model_len")
    if bool(rollout.multi_turn.enable):
        raise ValueError("intermediate MC supports only text-only single-turn rollouts")
    if bool(rollout.get("skip_rollout", False)):
        raise ValueError("intermediate MC cannot use precomputed or skipped rollouts")
    if bool(rollout.get("enable_rollout_routing_replay", False)):
        raise ValueError("intermediate MC does not support rollout routing replay")
    router_replay = config.actor_rollout_ref.actor.get("router_replay", {})
    if str(router_replay.get("mode", "none")).lower() not in {"none", "disabled"}:
        raise ValueError("intermediate MC does not support actor router replay")
    if bool(config.actor_rollout_ref.actor.get("use_prefix_grouper", False)):
        raise ValueError("intermediate MC does not yet support actor prefix grouping")
    if bool(config.reward.reward_model.get("launch_reward_fn_async", False)):
        raise ValueError("intermediate MC reward evaluation is a blocking iteration barrier")
    if bool(config.reward.reward_model.get("enable", False)):
        raise ValueError("intermediate MC requires a synchronous environment reward function, not a reward model")
    reward_loop_keys = ("reward_loop_source", "reward_loop_module_path", "reward_loop_class_name")
    if any(config.reward.reward_model.get(key, None) is not None for key in reward_loop_keys):
        raise ValueError("intermediate MC does not support rollout-time reward loops")
    grouped_reward_keys = (
        "use_response_logprob_reward_for_uniform_outcome_groups",
        "use_shortest_success_reward",
        "use_longest_success_penalty_reward",
    )
    reward_kwargs = config.reward.get("reward_kwargs", {}) or {}
    enabled_grouped_rewards = [key for key in grouped_reward_keys if bool(reward_kwargs.get(key, False))]
    if enabled_grouped_rewards:
        raise ValueError(
            "intermediate MC does not support grouped reward transformations because continuation batches "
            f"do not preserve native rollout groups: {enabled_grouped_rewards}"
        )
    if config.data.get("use_dataset_responses", False):
        raise ValueError("intermediate MC does not support off-policy dataset responses")
    if OmegaConf.select(config, "algorithm.opsd.enable", default=False):
        raise ValueError("intermediate MC and OPSD cannot be enabled together")
    cliprange = float(config.critic.cliprange_value)
    if not math.isfinite(cliprange) or not 0.0 <= cliprange <= 1.0:
        raise ValueError("intermediate MC interprets critic.cliprange_value as a normalized epsilon in [0, 1]")
    if actor_tokenizer is not None and critic_tokenizer is not None:
        if _tokenizer_fingerprint(actor_tokenizer) != _tokenizer_fingerprint(critic_tokenizer):
            raise ValueError("actor and critic tokenizers must have identical vocabularies and special-token IDs")
    if actor_model_path is not None:
        actor_hf_config = AutoConfig.from_pretrained(
            actor_model_path,
            trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False),
        )
        if _is_moe_model(actor_hf_config):
            raise ValueError("intermediate MC initially supports only dense actor models")
        actor_override = config.actor_rollout_ref.model.get("override_config", {})
        actor_limit = actor_override.get(
            "max_position_embeddings",
            getattr(actor_hf_config, "max_position_embeddings", None),
        )
        if isinstance(actor_limit, int) and int(rollout.max_model_len) > actor_limit:
            raise ValueError("rollout.max_model_len exceeds the actor model's effective context limit")

    with open_dict(config):
        config.critic.enable = True
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.actor.use_rollout_log_probs = True


@dataclass
class _Bundle:
    order: int
    dataset_index: object
    rollout_id: str
    prompt_group_id: str
    source_row: int
    prompt_ids: list[int]
    solution_ids: list[int]
    solution_log_probs: list[float]
    terminal_reward: float
    critique_ids: list[list[int]] = field(default_factory=list)
    critique_log_probs: list[list[float]] = field(default_factory=list)
    contexts: list[CriticContext] = field(default_factory=list)
    critic_values: list[list[float] | None] = field(default_factory=list)
    critic_variances: list[list[float] | None] = field(default_factory=list)
    marks: list[int] = field(default_factory=list)
    continuations: list[ContinuationGeneration] = field(default_factory=list)
    failed_continuations: list[tuple[int, int]] = field(default_factory=list)
    per_mark_targets: dict[int, float] = field(default_factory=dict)
    dense_targets: dict[int, float] = field(default_factory=dict)


class IntermediateMCValueController:
    """Feature controller called from VeRL's unmodified trainer lifecycle."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.tokenizer = trainer.tokenizer
        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.intermediate_mc_value,
            dataclass_type=IntermediateMCValueConfig,
        )
        if not self.feature.enable:
            raise ValueError("IntermediateMCValueController requires enable=true")
        if trainer.processor is not None:
            raise ValueError("intermediate MC currently supports only text-only models and datasets")
        if trainer.reward_fn is None:
            raise ValueError("intermediate MC requires a synchronous environment reward function")
        self.solution_delimiter_ids = self._encode(SOLUTION_DELIMITER)
        if self.feature.num_critiques > 0:
            self.critique_delimiter_ids = self._encode(CRITIQUE_DELIMITER)
            self.critique_instruction_ids = self._encode("\n\n" + INTERMEDIATE_MC_CRITIQUE_PROMPT)
        else:
            self.critique_delimiter_ids = []
            self.critique_instruction_ids = []
        self.critic_context_limit = _positive_model_limit(
            os.path.expanduser(self.config.critic.model.path),
            self.config.critic,
            role="critic",
        )
        self.audit_path = None
        if self.feature.audit_output_dir:
            audit_dir = os.path.abspath(os.path.expanduser(self.feature.audit_output_dir))
            os.makedirs(audit_dir, exist_ok=True)
            self.audit_path = os.path.join(audit_dir, "intermediate_mc_value.jsonl")

    def _encode(self, text: str) -> list[int]:
        result = [int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)]
        if not result:
            raise ValueError(f"intermediate MC boundary tokenized empty: {text!r}")
        return result

    def _audit(self, event: str, **payload: object) -> None:
        if self.audit_path is None:
            return
        record = {"event": event, "global_step": self.trainer.global_steps, **payload}
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _pad_token_id(self) -> int:
        token_id = self.tokenizer.pad_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("intermediate MC requires a tokenizer pad or EOS token")
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
            raise ValueError("intermediate MC prompt must contain at least one token")
        return result

    @staticmethod
    def _valid_solution(batch: DataProto, row: int) -> tuple[list[int], list[float]]:
        mask = batch.batch["response_mask"][row].bool()
        length = int(mask.sum().item())
        if length <= 0 or not torch.all(mask[:length]) or torch.any(mask[length:]):
            raise ValueError("intermediate MC requires a non-empty contiguous single-turn response mask")
        if "rollout_log_probs" not in batch.batch:
            raise RuntimeError("intermediate MC requires sampling-time processed rollout_log_probs")
        tokens = [int(token) for token in batch.batch["responses"][row, :length].tolist()]
        log_probs = [float(value) for value in batch.batch["rollout_log_probs"][row, :length].tolist()]
        if len(log_probs) != len(tokens) or not all(math.isfinite(value) for value in log_probs):
            raise RuntimeError("solution behavior log probabilities are missing or non-finite")
        return tokens, log_probs

    def prepare_generation_batch(self, batch: DataProto) -> None:
        """Select the composite agent loop and attach compact per-rollout controls."""

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
        warmup = self.trainer.global_steps <= int(self.config.trainer.critic_warmup)
        batch.non_tensor_batch["agent_name"] = np.array([INTERMEDIATE_MC_AGENT_NAME] * len(batch), dtype=object)
        batch.non_tensor_batch["intermediate_mc_stage"] = np.array(["solution"] * len(batch), dtype=object)
        batch.non_tensor_batch["intermediate_mc_rollout_id"] = np.array(rollout_ids, dtype=object)
        batch.non_tensor_batch["intermediate_mc_warmup"] = np.array([warmup] * len(batch), dtype=object)
        batch.non_tensor_batch["intermediate_mc_global_step"] = np.array(
            [self.trainer.global_steps] * len(batch), dtype=object
        )
        batch.non_tensor_batch["intermediate_mc_critic_context_limit"] = np.array(
            [self.critic_context_limit] * len(batch), dtype=object
        )

    @staticmethod
    def _coerce_critique(value: Any) -> CritiqueGeneration:
        if isinstance(value, CritiqueGeneration):
            return value
        if isinstance(value, dict):
            return CritiqueGeneration(tuple(value["token_ids"]), tuple(value["log_probs"]))
        raise TypeError(f"invalid critique child record {type(value)!r}")

    @classmethod
    def _coerce_record(cls, value: Any) -> IntermediateMCGenerationRecord:
        if isinstance(value, IntermediateMCGenerationRecord):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"invalid intermediate MC child record {type(value)!r}")
        continuations = tuple(
            item
            if isinstance(item, ContinuationGeneration)
            else ContinuationGeneration(int(item["mark"]), int(item["sample_index"]), tuple(item["token_ids"]))
            for item in value["continuations"]
        )
        return IntermediateMCGenerationRecord(
            rollout_id=str(value["rollout_id"]),
            critiques=tuple(cls._coerce_critique(item) for item in value["critiques"]),
            selected_marks=tuple(int(mark) for mark in value["selected_marks"]),
            continuations=continuations,
            failed_continuations=tuple(tuple(map(int, item)) for item in value["failed_continuations"]),
            selector_diagnostics=tuple(dict(item) for item in value["selector_diagnostics"]),
        )

    def extract_generation_records(self, output: DataProto) -> dict[str, IntermediateMCGenerationRecord]:
        raw = output.non_tensor_batch.pop(INTERMEDIATE_MC_CHILD_FIELD, None)
        if raw is None or len(raw) != len(output):
            raise RuntimeError("composite rollout did not return one intermediate MC child record per solution")
        records: dict[str, IntermediateMCGenerationRecord] = {}
        for value in raw:
            record = self._coerce_record(value)
            if record.rollout_id in records:
                raise RuntimeError(f"duplicate intermediate MC rollout id {record.rollout_id!r}")
            records[record.rollout_id] = record
        return records

    def _build_bundles(
        self,
        source: DataProto,
        records: dict[str, IntermediateMCGenerationRecord],
        terminal_rewards: list[float],
    ) -> list[_Bundle]:
        if len(source) != len(terminal_rewards):
            raise RuntimeError("solution reward count does not match rollout count")
        dataset_values = source.non_tensor_batch.get("index", np.arange(len(source), dtype=object))
        group_values = source.non_tensor_batch.get("prompt_group_id", np.arange(len(source), dtype=object))
        rollout_values = source.non_tensor_batch.get("intermediate_mc_rollout_id")
        if rollout_values is None:
            raise RuntimeError("rollout ids were lost before intermediate MC construction")
        bundles: list[_Bundle] = []
        for row, reward in enumerate(terminal_rewards):
            rollout_id = str(rollout_values[row])
            record = records.get(rollout_id)
            if record is None:
                raise RuntimeError(f"missing composite child record for rollout {rollout_id!r}")
            if len(record.critiques) != self.feature.num_critiques:
                raise RuntimeError(
                    f"rollout {rollout_id!r} has {len(record.critiques)} critiques; "
                    f"expected {self.feature.num_critiques}"
                )
            prompt_ids = self._valid_prompt_ids(source, row)
            solution_ids, solution_log_probs = self._valid_solution(source, row)
            critique_ids = [[int(token) for token in item.token_ids] for item in record.critiques]
            critique_log_probs = [[float(value) for value in item.log_probs] for item in record.critiques]
            if any(
                not tokens or len(tokens) != len(log_probs) or not all(math.isfinite(value) for value in log_probs)
                for tokens, log_probs in zip(critique_ids, critique_log_probs, strict=True)
            ):
                raise RuntimeError(f"rollout {rollout_id!r} contains an invalid critique child")
            if self.feature.num_critiques > 0:
                contexts = [
                    build_critic_context(
                        prompt_ids,
                        tokens,
                        solution_ids,
                        critique_delimiter_ids=self.critique_delimiter_ids,
                        solution_delimiter_ids=self.solution_delimiter_ids,
                    )
                    for tokens in critique_ids
                ]
            else:
                contexts = [
                    build_unconditioned_critic_context(
                        prompt_ids,
                        solution_ids,
                        solution_delimiter_ids=self.solution_delimiter_ids,
                    )
                ]
            if any(len(context.token_ids) > self.critic_context_limit for context in contexts):
                raise ValueError("intermediate MC critic sequence exceeds its effective context limit")
            marks = sorted(int(mark) for mark in record.selected_marks)
            if self.feature.mark_selector in {"ema", "variance"} and marks:
                raise RuntimeError(f"{self.feature.mark_selector} marks must be selected only after critic inference")
            bundle = _Bundle(
                order=row,
                dataset_index=dataset_values[row],
                rollout_id=rollout_id,
                prompt_group_id=str(group_values[row]),
                source_row=row,
                prompt_ids=prompt_ids,
                solution_ids=solution_ids,
                solution_log_probs=solution_log_probs,
                terminal_reward=validate_reward(reward, self.feature.max_reward),
                critique_ids=critique_ids,
                critique_log_probs=critique_log_probs,
                contexts=contexts,
                critic_values=[None] * self.feature.num_critic_streams,
                critic_variances=[None] * self.feature.num_critic_streams,
                marks=marks,
                continuations=list(record.continuations),
                failed_continuations=list(record.failed_continuations),
            )
            for diagnostic in record.selector_diagnostics:
                self._audit("mark_selection", rollout_id=rollout_id, **diagnostic)
            bundles.append(bundle)
        if set(records) != {bundle.rollout_id for bundle in bundles}:
            raise RuntimeError("composite rollout returned extra or missing child records")
        return bundles

    def _make_critic_batch(self, bundles: list[_Bundle]) -> DataProto:
        contexts = [context for bundle in bundles for context in bundle.contexts]
        max_sequence = max(len(context.token_ids) for context in contexts)
        max_positions = max(len(context.value_positions) for context in contexts)
        pad_id = self._pad_token_id()
        input_ids = torch.full((len(contexts), max_sequence), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(contexts), max_sequence), dtype=torch.long)
        positions = torch.zeros((len(contexts), max_positions), dtype=torch.long)
        position_mask = torch.zeros((len(contexts), max_positions), dtype=torch.float32)
        bundle_indices: list[int] = []
        critique_indices: list[int] = []
        row = 0
        for bundle_index, bundle in enumerate(bundles):
            for critique_index, context in enumerate(bundle.contexts):
                length = len(context.token_ids)
                value_count = len(context.value_positions)
                input_ids[row, :length] = torch.tensor(context.token_ids, dtype=torch.long)
                attention_mask[row, :length] = 1
                positions[row, :value_count] = torch.tensor(context.value_positions, dtype=torch.long)
                position_mask[row, :value_count] = 1.0
                bundle_indices.append(bundle_index)
                critique_indices.append(critique_index)
                row += 1
        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "critic_positions": positions,
            "critic_position_mask": position_mask,
            "critic_targets": torch.zeros_like(position_mask),
            "critic_target_mask": torch.zeros_like(position_mask),
            "critic_old_values": torch.zeros_like(position_mask),
        }
        batch = DataProto.from_dict(
            tensors=tensors,
            non_tensors={
                "intermediate_mc_bundle_index": np.array(bundle_indices, dtype=np.int64),
                "intermediate_mc_critique_index": np.array(critique_indices, dtype=np.int64),
            },
        )
        batch.meta_info.update(
            {
                "global_token_num": attention_mask.sum(dim=-1).tolist(),
                "micro_batch_size": self.config.critic.forward_micro_batch_size_per_gpu,
                "max_token_len": self.config.critic.forward_max_token_len_per_gpu,
                "use_dynamic_bsz": self.config.critic.use_dynamic_bsz,
            }
        )
        self._audit(
            "critic_batch",
            solutions=len(bundles),
            contexts=len(contexts),
            critiques=len(bundles) * self.feature.num_critiques,
        )
        return batch

    def _global_minibatch_size(self, role: str) -> int:
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if role == "actor":
            return int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * rollout_n
        return int(self.config.critic.ppo_mini_batch_size) * rollout_n

    def _validate_optimizer_batch(self, batch: DataProto, *, role: str, worker_group) -> None:
        dp_size = self.trainer._get_dp_size(worker_group, role)
        global_minibatch = self._global_minibatch_size(role)
        if len(batch) % dp_size != 0:
            raise ValueError(
                f"intermediate MC {role} rows ({len(batch)}) must be divisible by {role} DP size ({dp_size}); "
                "optimizer padding is forbidden"
            )
        if len(batch) % global_minibatch != 0:
            raise ValueError(
                f"intermediate MC {role} rows ({len(batch)}) must be divisible by global PPO minibatch "
                f"size ({global_minibatch}); optimizer padding is forbidden"
            )

    def _score_contexts(self, critic_batch: DataProto, bundles: list[_Bundle]) -> None:
        dp_size = self.trainer._get_dp_size(self.trainer.critic_wg, "critic")
        inference_batch, pad_size = pad_dataproto_to_divisor(critic_batch, dp_size)
        output = self.trainer._compute_values(inference_batch)
        if pad_size:
            output = unpad_dataproto(output, pad_size=pad_size)
        values = output.batch["values"].detach().cpu()
        variances = output.batch.get("variances")
        if variances is not None:
            variances = variances.detach().cpu()
        expected = critic_batch.batch["critic_position_mask"].shape
        if tuple(values.shape) != tuple(expected) or not torch.isfinite(values).all():
            raise RuntimeError(
                f"critic returned invalid values: expected finite {tuple(expected)}, got {tuple(values.shape)}"
            )
        if self.feature.critic_head == "beta":
            if variances is None or tuple(variances.shape) != tuple(expected):
                raise RuntimeError("Beta critic must return one variance for every requested value")
            if not torch.isfinite(variances).all() or torch.any(variances < 0):
                raise RuntimeError("Beta critic returned invalid variances")
        elif variances is not None:
            raise RuntimeError("scalar critic unexpectedly returned variances")
        critic_batch.batch["critic_old_values"] = values.clone()
        bundle_indices = critic_batch.non_tensor_batch["intermediate_mc_bundle_index"]
        critique_indices = critic_batch.non_tensor_batch["intermediate_mc_critique_index"]
        for row, (bundle_index, critique_index) in enumerate(zip(bundle_indices, critique_indices, strict=True)):
            bundle = bundles[int(bundle_index)]
            critique_index = int(critique_index)
            count = len(bundle.contexts[critique_index].value_positions)
            bundle.critic_values[critique_index] = values[row, :count].tolist()
            if variances is not None:
                bundle.critic_variances[critique_index] = variances[row, :count].tolist()
        if any(any(values is None for values in bundle.critic_values) for bundle in bundles):
            raise RuntimeError("critic scoring did not cover every intermediate MC context")
        self._audit("critic_scored", contexts=len(critic_batch), solutions=len(bundles))

    @staticmethod
    def _average_streams(streams: list[list[float] | None], *, name: str) -> list[float]:
        available = [stream for stream in streams if stream is not None]
        if not available or len(available) != len(streams):
            raise RuntimeError(f"{name} requires every critic stream")
        try:
            return [sum(items) / len(items) for items in zip(*available, strict=True)]
        except ValueError as error:
            raise RuntimeError(f"{name} critic streams have inconsistent lengths") from error

    def _select_ema_marks(self, bundles: list[_Bundle]) -> None:
        for bundle in bundles:
            averaged = self._average_streams(bundle.critic_values, name="EMA selection")
            if len(averaged) != len(bundle.solution_ids) + 1:
                raise RuntimeError("EMA selection requires V(s0) plus one value per solution token")
            selections, ema_values = select_ema_marks(
                averaged[1:],
                k=self.feature.resolved_max_marks,
                min_gap=self.feature.min_mark_gap,
                start_fraction=self.feature.mark_start_fraction,
                end_fraction=self.feature.mark_end_fraction,
                alpha=self.feature.ema_alpha,
                baseline_token=self.feature.ema_baseline_token,
                floor=self.feature.ema_floor,
                ratio_up=self.feature.ema_ratio_up,
                ratio_down=self.feature.ema_ratio_down,
            )
            bundle.marks = [selection.token for selection in selections]
            for selection in selections:
                self._audit(
                    "mark_selection",
                    rollout_id=bundle.rollout_id,
                    token=selection.token,
                    reason=f"ema_{selection.direction}",
                    value=selection.value,
                    ema=selection.ema,
                    reference=selection.reference,
                    ratio=selection.ratio,
                )
            if ema_values:
                self._audit(
                    "mark_selection",
                    rollout_id=bundle.rollout_id,
                    reason="ema_summary",
                    first=ema_values[0],
                    last=ema_values[-1],
                    count=len(ema_values),
                )

    def _select_variance_marks(self, bundles: list[_Bundle]) -> None:
        scopes: dict[str, list[_Bundle]] = {}
        for bundle in bundles:
            if self.feature.variance_scope == "rollout":
                scope = bundle.rollout_id
            elif self.feature.variance_scope == "prompt":
                scope = bundle.prompt_group_id
            else:
                scope = "batch"
            scopes.setdefault(scope, []).append(bundle)
        by_rollout = {bundle.rollout_id: bundle for bundle in bundles}
        for scope, scope_bundles in scopes.items():
            candidates: list[VarianceCandidate] = []
            for bundle in scope_bundles:
                averaged = self._average_streams(bundle.critic_variances, name="variance selection")
                low, high = candidate_bounds(
                    len(bundle.solution_ids),
                    self.feature.mark_start_fraction,
                    self.feature.mark_end_fraction,
                )
                for token in range(low, high + 1):
                    candidates.append(VarianceCandidate(bundle.order, bundle.rollout_id, token, averaged[token]))
            rng = stable_rng(self.feature.selection_seed, self.trainer.global_steps, scope)
            selections = select_variance_marks(
                candidates,
                k=self.feature.resolved_max_marks,
                min_gap=self.feature.min_mark_gap,
                random_probability=self.feature.variance_random_probability,
                rng=rng,
            )
            for selection in selections:
                bundle = by_rollout[selection.candidate.rollout_id]
                bundle.marks.append(selection.candidate.token)
                self._audit(
                    "mark_selection",
                    rollout_id=bundle.rollout_id,
                    token=selection.candidate.token,
                    variance=selection.candidate.variance,
                    reason=selection.reason,
                    draw=selection.draw,
                    scope=scope,
                )
        for bundle in bundles:
            bundle.marks.sort()

    def _make_deferred_continuation_request(
        self,
        source: DataProto,
        bundles: list[_Bundle],
    ) -> DataProto | None:
        selected = [bundle for bundle in bundles if bundle.marks]
        if not selected:
            return None
        rows = [bundle.source_row for bundle in selected]
        non_tensors = {
            key: np.take(values, rows, axis=0).copy()
            for key, values in source.non_tensor_batch.items()
            if key != INTERMEDIATE_MC_CHILD_FIELD
        }
        non_tensors["agent_name"] = np.array([INTERMEDIATE_MC_AGENT_NAME] * len(selected), dtype=object)
        non_tensors["intermediate_mc_stage"] = np.array(["continuations"] * len(selected), dtype=object)
        non_tensors["intermediate_mc_rollout_id"] = np.array([bundle.rollout_id for bundle in selected], dtype=object)
        non_tensors["intermediate_mc_critic_context_limit"] = np.array(
            [self.critic_context_limit] * len(selected), dtype=object
        )
        non_tensors["intermediate_mc_parent_prompt_ids"] = self._object_array(
            [bundle.prompt_ids for bundle in selected]
        )
        non_tensors["intermediate_mc_parent_solution_ids"] = self._object_array(
            [bundle.solution_ids for bundle in selected]
        )
        non_tensors["intermediate_mc_parent_solution_log_probs"] = self._object_array(
            [bundle.solution_log_probs for bundle in selected]
        )
        non_tensors["intermediate_mc_selected_marks"] = self._object_array([bundle.marks for bundle in selected])
        return DataProto.from_dict(
            non_tensors=non_tensors,
            meta_info={"global_steps": self.trainer.global_steps},
        )

    def _generate_deferred_continuations(
        self,
        source: DataProto,
        bundles: list[_Bundle],
        timing_raw: dict[str, float],
        profile_rollout: bool,
    ) -> None:
        request = self._make_deferred_continuation_request(source, bundles)
        if request is None:
            return
        with marked_timer("intermediate_mc_continuations", timing_raw, color="red"):
            output = self._generate_sequences_with_lifecycle(
                request,
                profile_rollout=profile_rollout,
                wake_up_replicas=True,
            )
            records = self.extract_generation_records(output)
        expected = {bundle.rollout_id for bundle in bundles if bundle.marks}
        if set(records) != expected:
            raise RuntimeError("deferred continuation stage returned an unexpected rollout-id set")
        for bundle in bundles:
            if not bundle.marks:
                continue
            record = records[bundle.rollout_id]
            if record.critiques:
                raise RuntimeError("deferred continuation stage unexpectedly regenerated critiques")
            if list(record.selected_marks) != bundle.marks:
                raise RuntimeError("deferred continuation stage changed the selected marks")
            bundle.continuations = list(record.continuations)
            bundle.failed_continuations = list(record.failed_continuations)

    def _generate_sequences_with_lifecycle(
        self,
        request: DataProto,
        *,
        profile_rollout: bool,
        wake_up_replicas: bool,
    ) -> DataProto:
        """Run one blocking feature rollout and independently attempt every cleanup."""

        primary_error: BaseException | None = None
        profile_cleanup_required = False
        try:
            if wake_up_replicas:
                self.trainer.checkpoint_manager.wake_up_replicas()
            if profile_rollout:
                # Mark cleanup required before entry because a multi-replica
                # gather can partially succeed and then raise.
                profile_cleanup_required = True
                self.trainer.async_rollout_manager.start_profile()
            return self.trainer.async_rollout_manager.generate_sequences(request)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_errors: list[tuple[str, BaseException]] = []
            try:
                # This is required even when wake-up partially fails, and for
                # the already-awake first rollout stage.
                self.trainer.checkpoint_manager.sleep_replicas()
            except BaseException as error:
                cleanup_errors.append(("rollout sleep cleanup", error))

            if profile_cleanup_required:
                try:
                    self.trainer.async_rollout_manager.stop_profile()
                except BaseException as error:
                    cleanup_errors.append(("rollout profile cleanup", error))

            if cleanup_errors:
                if primary_error is not None:
                    for label, error in cleanup_errors:
                        primary_error.add_note(f"{label} also failed: {error!r}")
                else:
                    _, first_error = cleanup_errors[0]
                    for label, error in cleanup_errors[1:]:
                        first_error.add_note(f"{label} also failed: {error!r}")
                    raise first_error

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
            raise ValueError("continuation reward prompt exceeds rollout.prompt_length")
        if any(not tokens or len(tokens) > response_width for tokens in responses):
            raise ValueError("completed continuation must fit rollout.response_length")
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
            if key != INTERMEDIATE_MC_CHILD_FIELD
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
        mapping: list[tuple[int, int, int]] = []
        for bundle_index, bundle in enumerate(bundles):
            for continuation in bundle.continuations:
                if continuation.mark not in bundle.marks:
                    raise RuntimeError("continuation record refers to an unselected mark")
                full_response = [*bundle.solution_ids[: continuation.mark], *continuation.token_ids]
                rows.append(bundle.source_row)
                prompts.append(bundle.prompt_ids)
                responses.append(full_response)
                mapping.append((bundle_index, continuation.mark, continuation.sample_index))
        rewards_by_mark: dict[tuple[int, int], list[float]] = {}
        if mapping:
            reward_batch = self._make_reward_batch(source, rows, prompts, responses)
            reward_tensor, _ = compute_reward(
                reward_batch,
                self.trainer.reward_fn,
                actor_wg=self.trainer.actor_rollout_wg,
            )
            raw_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
            if len(raw_rewards) != len(mapping):
                raise RuntimeError("continuation reward count does not match successful generations")
            for (bundle_index, mark, sample_index), raw_reward in zip(mapping, raw_rewards, strict=True):
                reward = validate_reward(raw_reward, self.feature.max_reward)
                rewards_by_mark.setdefault((bundle_index, mark), []).append(reward)
                self._audit(
                    "continuation",
                    rollout_id=bundles[bundle_index].rollout_id,
                    mark=mark,
                    sample=sample_index,
                    reward=reward,
                )
        for bundle_index, bundle in enumerate(bundles):
            mark_rewards = {
                mark: rewards_by_mark[(bundle_index, mark)]
                for mark in bundle.marks
                if (bundle_index, mark) in rewards_by_mark
            }
            bundle.per_mark_targets, bundle.dense_targets = aggregate_mark_targets(mark_rewards)

    def _set_critic_targets(self, critic_batch: DataProto, bundles: list[_Bundle]) -> None:
        bundle_indices = critic_batch.non_tensor_batch["intermediate_mc_bundle_index"]
        for row, raw_bundle_index in enumerate(bundle_indices):
            bundle = bundles[int(raw_bundle_index)]
            critic_batch.batch["critic_targets"][row, 0] = initial_state_target(
                bundle.terminal_reward,
                bundle.per_mark_targets,
            )
            critic_batch.batch["critic_target_mask"][row, 0] = 1.0
            for token, target in bundle.dense_targets.items():
                critic_batch.batch["critic_targets"][row, token] = target
                critic_batch.batch["critic_target_mask"][row, token] = 1.0
            terminal = len(bundle.solution_ids)
            critic_batch.batch["critic_targets"][row, terminal] = bundle.terminal_reward
            critic_batch.batch["critic_target_mask"][row, terminal] = 1.0
        for bundle in bundles:
            self._audit(
                "critic_targets",
                rollout_id=bundle.rollout_id,
                selected_marks=bundle.marks,
                surviving_marks=sorted(bundle.per_mark_targets),
                dense_token_labels=len(bundle.dense_targets),
                initial_state_target=initial_state_target(bundle.terminal_reward, bundle.per_mark_targets),
                terminal_token=len(bundle.solution_ids),
            )

    def _add_solution_gae(self, source: DataProto, bundles: list[_Bundle]) -> None:
        response_width = source.batch["responses"].shape[1]
        values = torch.zeros((len(bundles), response_width), dtype=torch.float32)
        rewards = torch.zeros_like(values)
        for row, bundle in enumerate(bundles):
            averaged = self._average_streams(bundle.critic_values, name="solution GAE")
            token_count = len(bundle.solution_ids)
            if len(averaged) != token_count + 1:
                raise RuntimeError("solution GAE requires V(s0) plus one value per solution token")
            values[row, :token_count] = torch.tensor(averaged[:token_count], dtype=torch.float32)
            rewards[row, token_count - 1] = bundle.terminal_reward
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=source.batch["response_mask"].float(),
            gamma=float(self.config.algorithm.gamma),
            lam=float(self.config.algorithm.lam),
        )
        source.batch["values"] = values
        source.batch["token_level_rewards"] = rewards
        source.batch["advantages"] = advantages
        source.batch["returns"] = returns

    def _critique_advantages(self, bundle: _Bundle) -> list[float]:
        if self.feature.num_critiques <= 0:
            raise RuntimeError("critique advantages are undefined when self-critique is disabled")
        if len(bundle.critic_values) != self.feature.num_critiques:
            raise RuntimeError("critique credit requires one critic stream per critique")
        points = [len(bundle.solution_ids), *sorted(bundle.per_mark_targets)]
        targets = [
            bundle.terminal_reward,
            *(bundle.per_mark_targets[mark] for mark in sorted(bundle.per_mark_targets)),
        ]
        rewards = [
            critique_accuracy_reward(
                [float(values[point]) for point in points],
                targets,
                max_reward=self.feature.max_reward,
            )
            for values in bundle.critic_values
            if values is not None
        ]
        advantages = critique_group_advantages(rewards, self.feature.critique_normalization_epsilon)
        self._audit(
            "critique_credit",
            rollout_id=bundle.rollout_id,
            rewards=rewards,
            advantages=advantages,
            points=points,
            targets=targets,
        )
        return advantages

    def _make_actor_batch(self, source: DataProto, bundles: list[_Bundle]) -> DataProto:
        rows: list[tuple[list[int], int, list[float], list[float], str]] = []
        for bundle in bundles:
            solution_advantages = source.batch["advantages"][bundle.source_row, : len(bundle.solution_ids)].tolist()
            solution_full = [*bundle.prompt_ids, *bundle.solution_ids]
            rows.append(
                (
                    solution_full,
                    len(bundle.prompt_ids),
                    bundle.solution_log_probs,
                    [float(value) for value in solution_advantages],
                    "solution",
                )
            )
            if self.feature.num_critiques == 0:
                if bundle.critique_ids or bundle.critique_log_probs:
                    raise RuntimeError("self-critique-disabled bundle unexpectedly contains critique outputs")
                continue
            critique_advantages = self._critique_advantages(bundle)
            critique_prompt = [*bundle.prompt_ids, *bundle.solution_ids, *self.critique_instruction_ids]
            for critique_ids, log_probs, advantage in zip(
                bundle.critique_ids,
                bundle.critique_log_probs,
                critique_advantages,
                strict=True,
            ):
                rows.append(
                    (
                        [*critique_prompt, *critique_ids],
                        len(critique_prompt),
                        log_probs,
                        [float(advantage)] * len(critique_ids),
                        "critique",
                    )
                )
        max_sequence = max(len(full) for full, *_ in rows)
        if max_sequence > int(self.config.actor_rollout_ref.rollout.max_model_len):
            raise ValueError("packed actor sequence exceeds the actor's effective context limit")
        response_width = max_sequence - 1
        pad_id = self._pad_token_id()
        prompts = torch.full((len(rows), 1), pad_id, dtype=torch.long)
        responses = torch.full((len(rows), response_width), pad_id, dtype=torch.long)
        response_mask = torch.zeros((len(rows), response_width), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_sequence), dtype=torch.long)
        old_log_probs = torch.zeros((len(rows), response_width), dtype=torch.float32)
        rollout_log_probs = torch.zeros_like(old_log_probs)
        advantages = torch.zeros_like(old_log_probs)
        kinds: list[str] = []
        for row, (full, train_start, behavior, row_advantages, kind) in enumerate(rows):
            if len(full) < 2 or train_start <= 0:
                raise ValueError("packed actor sequence needs one real prompt token and non-empty context")
            if len(behavior) != len(row_advantages) or train_start + len(behavior) != len(full):
                raise RuntimeError("packed actor behavior/advantage alignment is inconsistent")
            prompts[row, 0] = full[0]
            responses[row, : len(full) - 1] = torch.tensor(full[1:], dtype=torch.long)
            attention_mask[row, : len(full)] = 1
            offset = train_start - 1
            stop = offset + len(behavior)
            response_mask[row, offset:stop] = 1
            old_log_probs[row, offset:stop] = torch.tensor(behavior, dtype=torch.float32)
            rollout_log_probs[row, offset:stop] = torch.tensor(behavior, dtype=torch.float32)
            advantages[row, offset:stop] = torch.tensor(row_advantages, dtype=torch.float32)
            kinds.append(kind)
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
                "advantages": advantages,
            },
            non_tensors={"intermediate_mc_actor_kind": np.array(kinds, dtype=object)},
        )
        actor_batch.meta_info.update(
            {
                "temperature": 1.0,
                "global_token_num": attention_mask.sum(dim=-1).tolist(),
            }
        )
        self._audit(
            "actor_batch",
            solutions=len(bundles),
            critiques=len(bundles) * self.feature.num_critiques,
            continuations=0,
            padding=0,
        )
        return actor_batch

    def _token_metrics(
        self,
        bundles: list[_Bundle],
        *,
        critic_batch: DataProto,
        actor_batch: DataProto | None,
    ) -> dict[str, float]:
        """Return aggregate token-volume counters for throughput comparisons.

        These counters intentionally avoid per-example audit output. They make it
        possible to distinguish a genuinely faster iteration from one that only
        sampled shorter solutions, critiques, or continuations.
        """

        prompt_tokens = sum(len(bundle.prompt_ids) for bundle in bundles)
        solution_tokens = sum(len(bundle.solution_ids) for bundle in bundles)
        critique_tokens = sum(len(tokens) for bundle in bundles for tokens in bundle.critique_ids)
        continuation_tokens = sum(
            len(continuation.token_ids) for bundle in bundles for continuation in bundle.continuations
        )
        critique_input_tokens = sum(
            len(bundle.prompt_ids) + len(bundle.solution_ids) + len(self.critique_instruction_ids)
            for bundle in bundles
            for _ in bundle.critique_ids
        )
        continuation_input_tokens = 0
        continuation_attempts = 0
        for bundle in bundles:
            attempted_marks = [continuation.mark for continuation in bundle.continuations]
            attempted_marks.extend(mark for mark, _ in bundle.failed_continuations)
            continuation_attempts += len(attempted_marks)
            continuation_input_tokens += sum(len(bundle.prompt_ids) + mark for mark in attempted_marks)

        critic_input_tokens = int(critic_batch.batch["attention_mask"].sum().item())
        actor_input_tokens = 0
        actor_train_tokens = 0
        if actor_batch is not None:
            actor_input_tokens = int(actor_batch.batch["attention_mask"].sum().item())
            actor_train_tokens = int(actor_batch.batch["response_mask"].sum().item())

        generation_input_tokens = prompt_tokens + critique_input_tokens + continuation_input_tokens
        generation_output_tokens = solution_tokens + critique_tokens + continuation_tokens
        return {
            "intermediate_mc/tokens/prompt": float(prompt_tokens),
            "intermediate_mc/tokens/solution_output": float(solution_tokens),
            "intermediate_mc/tokens/critique_input": float(critique_input_tokens),
            "intermediate_mc/tokens/critique_output": float(critique_tokens),
            "intermediate_mc/tokens/continuation_input": float(continuation_input_tokens),
            "intermediate_mc/tokens/continuation_output": float(continuation_tokens),
            "intermediate_mc/tokens/generation_input": float(generation_input_tokens),
            "intermediate_mc/tokens/generation_output": float(generation_output_tokens),
            "intermediate_mc/tokens/critic_input": float(critic_input_tokens),
            "intermediate_mc/tokens/actor_input": float(actor_input_tokens),
            "intermediate_mc/tokens/actor_train": float(actor_train_tokens),
            "intermediate_mc/continuation_attempts": float(continuation_attempts),
        }

    def run_update(
        self,
        source: DataProto,
        records: dict[str, IntermediateMCGenerationRecord],
        reward_tensor: torch.Tensor,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
        profile_rollout: bool = False,
    ) -> bool:
        terminal_rewards = [
            validate_reward(value, self.feature.max_reward)
            for value in reward_tensor.sum(dim=-1).detach().cpu().tolist()
        ]
        bundles = self._build_bundles(source, records, terminal_rewards)
        critic_batch = self._make_critic_batch(bundles)
        self._validate_optimizer_batch(critic_batch, role="critic", worker_group=self.trainer.critic_wg)
        if self.config.trainer.balance_batch:
            self.trainer._balance_batch(
                critic_batch,
                metrics=metrics,
                logging_prefix="critic_global_seqlen",
                worker_group=self.trainer.critic_wg,
                role="critic",
            )
        with marked_timer("values", timing_raw, color="cyan"):
            self._score_contexts(critic_batch, bundles)
        in_warmup = self.trainer.global_steps <= int(self.config.trainer.critic_warmup)
        if not in_warmup:
            if self.feature.mark_selector in {"ema", "variance"}:
                if self.feature.mark_selector == "ema":
                    self._select_ema_marks(bundles)
                else:
                    self._select_variance_marks(bundles)
                self._generate_deferred_continuations(
                    source,
                    bundles,
                    timing_raw,
                    profile_rollout=profile_rollout,
                )
            self._evaluate_continuations(source, bundles)
        else:
            self._audit("warmup", continuations=0)
        self._set_critic_targets(critic_batch, bundles)
        with marked_timer("update_critic", timing_raw, color="pink"):
            critic_output = self.trainer._update_critic(critic_batch)
        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
        self._add_solution_gae(source, bundles)

        actor_updated = False
        actor_batch = None
        if not in_warmup:
            actor_batch = self._make_actor_batch(source, bundles)
            self._validate_optimizer_batch(actor_batch, role="actor", worker_group=self.trainer.actor_rollout_wg)
            if self.config.trainer.balance_batch:
                self.trainer._balance_batch(
                    actor_batch,
                    metrics=metrics,
                    logging_prefix="actor_global_seqlen",
                    worker_group=self.trainer.actor_rollout_wg,
                    role="actor",
                )
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self.trainer._update_actor(actor_batch)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
            actor_updated = True

        metrics.update(
            self._token_metrics(
                bundles,
                critic_batch=critic_batch,
                actor_batch=actor_batch,
            )
        )
        metrics.update(
            {
                "intermediate_mc/warmup": float(in_warmup),
                "intermediate_mc/bundles": len(bundles),
                "intermediate_mc/critiques": len(bundles) * self.feature.num_critiques,
                "intermediate_mc/critique_advantage_zero": float(self.feature.num_critiques == 1),
                "intermediate_mc/selected_marks": sum(len(bundle.marks) for bundle in bundles),
                "intermediate_mc/surviving_marks": sum(len(bundle.per_mark_targets) for bundle in bundles),
                "intermediate_mc/failed_continuations": sum(len(bundle.failed_continuations) for bundle in bundles),
            }
        )
        return actor_updated
