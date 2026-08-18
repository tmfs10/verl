# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Blocking VeRL trainer for self-critique and intermediate MC value labels."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoConfig

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.config import IntermediateMCValueConfig
from verl.trainer.ppo.intermediate_mc_value import (
    CRITIQUE_DELIMITER,
    SOLUTION_DELIMITER,
    CriticContext,
    VarianceCandidate,
    aggregate_mark_targets,
    build_critic_context,
    candidate_bounds,
    critique_accuracy_reward,
    critique_group_advantages,
    masked_whiten,
    select_random_marks,
    select_variance_marks,
    stable_rng,
    token_gae,
    validate_reward,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import compute_reward
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import Tracking


@dataclass
class _Bundle:
    order: int
    dataset_index: object
    rollout_id: str
    prompt_group_id: str
    source_row: int
    prompt_ids: list[int]
    solution_ids: list[int]
    terminal_reward: float
    critique_rows: list[DataProto] = field(default_factory=list)
    critique_ids: list[list[int]] = field(default_factory=list)
    contexts: list[CriticContext] = field(default_factory=list)
    critic_values: list[list[float]] = field(default_factory=list)
    critic_variances: list[list[float]] = field(default_factory=list)
    marks: list[int] = field(default_factory=list)
    per_mark_targets: dict[int, float] = field(default_factory=dict)
    dense_targets: dict[int, float] = field(default_factory=dict)


def _tokenizer_fingerprint(tokenizer) -> str:
    payload = {
        "vocab": sorted((str(token), int(index)) for token, index in tokenizer.get_vocab().items()),
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_intermediate_mc_runtime_config(config, actor_tokenizer=None, critic_tokenizer=None) -> None:
    """Fail closed before allocating workers for unsupported combinations."""

    feature = omega_conf_to_dataclass(
        config.algorithm.intermediate_mc_value,
        dataclass_type=IntermediateMCValueConfig,
    )
    if not feature.enable:
        return
    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError("intermediate MC currently supports only the legacy FSDP/FSDP2 workers")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("intermediate MC actor strategy must be fsdp or fsdp2")
    if config.critic.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("intermediate MC critic strategy must be fsdp or fsdp2")
    if config.critic.get("enable", None) is False:
        raise ValueError("intermediate MC requires critic.enable=true")
    if config.algorithm.adv_estimator not in {"gae", "GAE"}:
        raise ValueError("intermediate MC requires algorithm.adv_estimator=gae")
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    get_policy_loss_fn(feature.actor_loss_mode)
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        raise ValueError("intermediate MC does not support actor KL or KL-in-reward")
    rollout_correction = config.algorithm.get("rollout_correction", None)
    if rollout_correction is not None and (
        rollout_correction.get("rollout_is", None) is not None
        or rollout_correction.get("rollout_rs", None) is not None
        or rollout_correction.get("bypass_mode", False)
    ):
        raise ValueError("intermediate MC uses recorded behavior log probabilities and rejects rollout correction")
    if config.trainer.critic_warmup != 0:
        raise ValueError("set trainer.critic_warmup=0; intermediate_mc_value owns its critic-update warmup")
    if float(config.actor_rollout_ref.rollout.temperature) != 1.0:
        raise ValueError("all intermediate MC solution, critique, and continuation generation requires temperature=1.0")
    if float(config.actor_rollout_ref.rollout.val_kwargs.temperature) != 1.0:
        raise ValueError("intermediate MC validation generation requires temperature=1.0")
    if bool(config.actor_rollout_ref.rollout.multi_turn.enable):
        raise ValueError("intermediate MC supports only text-only single-turn rollouts")
    if bool(config.actor_rollout_ref.rollout.get("skip_rollout", False)):
        raise ValueError("intermediate MC cannot use precomputed or skipped rollouts")
    if bool(config.actor_rollout_ref.rollout.get("enable_rollout_routing_replay", False)):
        raise ValueError("intermediate MC does not support rollout routing replay")
    if bool(config.reward.reward_model.get("launch_reward_fn_async", False)):
        raise ValueError("intermediate MC reward evaluation is an iteration barrier and cannot launch asynchronously")
    if bool(config.reward.reward_model.get("enable", False)):
        raise ValueError(
            "intermediate MC currently requires a synchronous environment reward function, not a reward model"
        )
    reward_loop_keys = ("reward_loop_source", "reward_loop_module_path", "reward_loop_class_name")
    if any(config.reward.reward_model.get(key, None) is not None for key in reward_loop_keys):
        raise ValueError("intermediate MC does not support rollout-time reward loops")
    if config.data.get("use_dataset_responses", False):
        raise ValueError("intermediate MC does not support off-policy dataset responses")
    if OmegaConf.select(config, "algorithm.opsd.enable", default=False):
        raise ValueError("intermediate MC and OPSD cannot be enabled in the same trainer")
    if actor_tokenizer is not None and critic_tokenizer is not None:
        if _tokenizer_fingerprint(actor_tokenizer) != _tokenizer_fingerprint(critic_tokenizer):
            raise ValueError("actor and critic tokenizers must have identical vocabularies and special-token IDs")

    with open_dict(config):
        config.critic.enable = True
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.actor.use_rollout_log_probs = True
        config.actor_rollout_ref.actor.policy_loss.loss_mode = feature.actor_loss_mode


class IntermediateMCRayPPOTrainer(RayPPOTrainer):
    """A strict iteration-barrier implementation; no payload overlaps another."""

    STATE_FILENAME = "intermediate_mc_value_state.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.intermediate_mc_value,
            dataclass_type=IntermediateMCValueConfig,
        )
        if not self.feature.enable:
            raise ValueError("IntermediateMCRayPPOTrainer requires intermediate_mc_value.enable=true")
        if self.processor is not None:
            raise ValueError("intermediate MC currently supports text-only models and datasets")
        if self.reward_fn is None:
            raise ValueError("intermediate MC requires a synchronous environment reward function")
        self.critic_update_count = 0
        self._critic_delimiter_ids = self._encode_boundary(CRITIQUE_DELIMITER)
        self._solution_delimiter_ids = self._encode_boundary(SOLUTION_DELIMITER)
        self._critique_instruction_ids = self._encode_boundary("\n\n" + self.feature.critique_prompt)
        critic_path = os.path.expanduser(self.config.critic.model.path)
        critic_hf_config = AutoConfig.from_pretrained(
            critic_path,
            trust_remote_code=self.config.critic.model.get("trust_remote_code", False),
        )
        self._critic_context_limit = int(getattr(critic_hf_config, "max_position_embeddings", 0) or 0)
        if self._critic_context_limit <= 0:
            raise ValueError("critic model must declare a positive max_position_embeddings")
        self._tokenizer_fingerprint = _tokenizer_fingerprint(self.tokenizer)
        self._audit_path = None
        if self.feature.audit_output_dir:
            audit_dir = os.path.abspath(os.path.expanduser(self.feature.audit_output_dir))
            os.makedirs(audit_dir, exist_ok=True)
            self._audit_path = os.path.join(audit_dir, "intermediate_mc_value.jsonl")

    def _encode_boundary(self, text: str) -> list[int]:
        result = self.tokenizer.encode(text, add_special_tokens=False)
        if not result:
            raise ValueError(f"boundary must tokenize to a non-empty sequence: {text!r}")
        return [int(token) for token in result]

    def _pad_token_id(self) -> int:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("intermediate MC requires the tokenizer to define a pad or EOS token")
        return int(pad_token_id)

    def _contract(self) -> dict[str, object]:
        feature_contract = asdict(self.feature)
        feature_contract.pop("_target_", None)
        feature_contract.pop("enable", None)
        feature_contract.pop("audit_output_dir", None)
        return {
            "version": 1,
            "feature": feature_contract,
            "gamma": float(self.config.algorithm.gamma),
            "gae_lambda": float(self.config.algorithm.lam),
            "tokenizer_fingerprint": self._tokenizer_fingerprint,
        }

    def _save_additional_trainer_state(self, checkpoint_folder: str) -> None:
        state = {"critic_update_count": self.critic_update_count, "contract": self._contract()}
        state_path = os.path.join(checkpoint_folder, self.STATE_FILENAME)
        temporary_path = f"{state_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, state_path)

    def _read_feature_state(self, checkpoint_folder: str) -> dict[str, object]:
        state_path = os.path.join(checkpoint_folder, self.STATE_FILENAME)
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"intermediate MC checkpoint is missing {state_path}")
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError(f"invalid intermediate MC checkpoint state in {state_path}")
        return state

    def _validate_additional_trainer_state(self, checkpoint_folder: str) -> None:
        state = self._read_feature_state(checkpoint_folder)
        if state.get("contract") != self._contract():
            raise RuntimeError(
                "intermediate MC checkpoint contract does not match the current recipe/tokenizer/loss configuration"
            )
        count = state.get("critic_update_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("checkpoint critic_update_count must be a non-negative integer")

    def _load_additional_trainer_state(self, checkpoint_folder: str) -> None:
        state = self._read_feature_state(checkpoint_folder)
        self.critic_update_count = int(state["critic_update_count"])

    def _audit(self, event: str, **payload: object) -> None:
        if self._audit_path is None:
            return
        record = {"event": event, "global_step": self.global_steps, **payload}
        with open(self._audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _valid_prompt_ids(batch: DataProto, row: int) -> list[int]:
        prompt_width = batch.batch["prompts"].shape[1]
        prompt_mask = batch.batch["attention_mask"][row, :prompt_width].bool()
        return [int(token) for token in batch.batch["prompts"][row][prompt_mask].tolist()]

    @staticmethod
    def _valid_response_ids(batch: DataProto, row: int, cap: int | None = None) -> list[int]:
        mask = batch.batch["response_mask"][row].bool()
        valid_count = int(mask.sum().item())
        if valid_count == 0 or not torch.all(mask[:valid_count]) or torch.any(mask[valid_count:]):
            raise ValueError("intermediate MC requires a non-empty contiguous single-turn response mask")
        tokens = [int(token) for token in batch.batch["responses"][row][mask].tolist()]
        return tokens if cap is None else tokens[:cap]

    def _secondary_request_batch(
        self,
        source: DataProto,
        source_rows: list[int],
        prompt_overrides: list[list[int]],
    ) -> DataProto:
        if len(source_rows) != len(prompt_overrides) or not source_rows:
            raise ValueError("secondary request rows and prompt overrides must be equal and non-empty")
        non_tensors = {
            key: np.take(values, source_rows, axis=0).copy() for key, values in source.non_tensor_batch.items()
        }
        overrides = np.empty(len(prompt_overrides), dtype=object)
        overrides[:] = [list(tokens) for tokens in prompt_overrides]
        non_tensors["prompt_ids_override"] = overrides
        non_tensors["agent_name"] = np.array(["single_turn_agent"] * len(overrides), dtype=object)
        request = DataProto.from_dict(non_tensors=non_tensors, meta_info={"global_steps": self.global_steps})
        return request

    def _generate_rows_with_isolation(
        self,
        request: DataProto,
        *,
        allow_all_failures: bool = False,
    ) -> list[DataProto | None]:
        try:
            output = self.async_rollout_manager.generate_sequences(request)
            if len(output) != len(request):
                raise RuntimeError(f"secondary generation returned {len(output)} rows for a request of {len(request)}")
            return [output[index : index + 1] for index in range(len(output))]
        except Exception as batch_error:
            self._audit("secondary_generation_batch_failure", error=repr(batch_error), rows=len(request))
            rows: list[DataProto | None] = []
            for index in range(len(request)):
                try:
                    output = self.async_rollout_manager.generate_sequences(request[index : index + 1])
                    if len(output) != 1:
                        raise RuntimeError(f"isolated secondary generation returned {len(output)} rows")
                    rows.append(output[0:1])
                except Exception as row_error:
                    self._audit("secondary_generation_row_failure", row=index, error=repr(row_error))
                    rows.append(None)
            if all(row is None for row in rows) and not allow_all_failures:
                raise RuntimeError("all secondary generation rows failed during isolated retry") from batch_error
            return rows

    def _truncate_generated_row(self, row: DataProto | None, cap: int) -> DataProto | None:
        if row is None or cap <= 0:
            return None
        if "rollout_log_probs" not in row.batch:
            raise RuntimeError("intermediate MC requires sampling-time rollout_log_probs")
        if not torch.any(row.batch["response_mask"][0].bool()):
            return None
        valid_tokens = self._valid_response_ids(row, 0, cap=cap)
        response_width = row.batch["responses"].shape[1]
        prompt_width = row.batch["prompts"].shape[1]
        keep = min(len(valid_tokens), cap, response_width)
        pad_token_id = self._pad_token_id()
        row.batch["responses"][0, keep:] = pad_token_id
        row.batch["response_mask"][0, keep:] = 0
        row.batch["rollout_log_probs"][0, keep:] = 0.0
        response_attention = torch.zeros(response_width, dtype=row.batch["attention_mask"].dtype)
        response_attention[:keep] = 1
        row.batch["attention_mask"][0, prompt_width:] = response_attention
        row.batch["input_ids"] = torch.cat([row.batch["prompts"], row.batch["responses"]], dim=1)
        row.batch["position_ids"] = compute_position_id_with_mask(row.batch["attention_mask"])
        return row

    def _solution_rewards(self, batch: DataProto) -> list[float]:
        reward_tensor, _ = compute_reward(batch, self.reward_fn, actor_wg=self.actor_rollout_wg)
        raw = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        if len(raw) != len(batch):
            raise RuntimeError(f"reward function returned {len(raw)} rows for a solution batch of {len(batch)}")
        return [validate_reward(value, self.feature.max_reward) for value in raw]

    def _make_reward_batch(
        self,
        source: DataProto,
        source_rows: list[int],
        prompt_ids: list[list[int]],
        response_ids: list[list[int]],
    ) -> DataProto:
        prompt_width = int(self.config.actor_rollout_ref.rollout.prompt_length)
        response_width = int(self.config.actor_rollout_ref.rollout.response_length)
        if any(len(tokens) > prompt_width for tokens in prompt_ids):
            raise ValueError("continuation reward prompt exceeds rollout.prompt_length")
        if any(not tokens or len(tokens) > response_width for tokens in response_ids):
            raise ValueError("full continuation response must fit rollout.response_length")
        batch_size = len(source_rows)
        pad_id = self._pad_token_id()
        prompts = torch.full((batch_size, prompt_width), pad_id, dtype=torch.long)
        responses = torch.full((batch_size, response_width), pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long)
        response_mask = torch.zeros((batch_size, response_width), dtype=torch.long)
        for row, (prompt, response) in enumerate(zip(prompt_ids, response_ids, strict=True)):
            prompts[row, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            prompt_mask[row, -len(prompt) :] = 1
            responses[row, : len(response)] = torch.tensor(response, dtype=torch.long)
            response_mask[row, : len(response)] = 1
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        tensors = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }
        non_tensors = {
            key: np.take(values, source_rows, axis=0).copy() for key, values in source.non_tensor_batch.items()
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    def _continuation_rewards_with_isolation(
        self,
        reward_batch: DataProto,
    ) -> list[float | None]:
        try:
            reward_tensor, _ = compute_reward(reward_batch, self.reward_fn, actor_wg=self.actor_rollout_wg)
            raw = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        except Exception as batch_error:
            self._audit("continuation_reward_batch_failure", error=repr(batch_error), rows=len(reward_batch))
            results: list[float | None] = []
            for index in range(len(reward_batch)):
                try:
                    reward_tensor, _ = compute_reward(
                        reward_batch[index : index + 1],
                        self.reward_fn,
                        actor_wg=self.actor_rollout_wg,
                    )
                except Exception as row_error:
                    self._audit("continuation_reward_row_failure", row=index, error=repr(row_error))
                    results.append(None)
                else:
                    results.append(validate_reward(reward_tensor.sum().item(), self.feature.max_reward))
            return results
        if len(raw) != len(reward_batch):
            raise RuntimeError(
                f"continuation reward function returned {len(raw)} rows for a batch of {len(reward_batch)}"
            )
        return [validate_reward(value, self.feature.max_reward) for value in raw]

    def _make_critic_batch(self, bundles: list[_Bundle]) -> tuple[DataProto, list[tuple[int, int]]]:
        contexts = [context for bundle in bundles for context in bundle.contexts]
        mapping = [
            (bundle_index, critique_index)
            for bundle_index, bundle in enumerate(bundles)
            for critique_index in range(len(bundle.contexts))
        ]
        max_sequence = max(len(context.token_ids) for context in contexts)
        max_positions = max(len(context.value_positions) for context in contexts)
        pad_id = self._pad_token_id()
        input_ids = torch.full((len(contexts), max_sequence), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(contexts), max_sequence), dtype=torch.long)
        positions = torch.zeros((len(contexts), max_positions), dtype=torch.long)
        position_mask = torch.zeros((len(contexts), max_positions), dtype=torch.float32)
        for row, context in enumerate(contexts):
            length = len(context.token_ids)
            value_count = len(context.value_positions)
            input_ids[row, :length] = torch.tensor(context.token_ids, dtype=torch.long)
            attention_mask[row, :length] = 1
            positions[row, :value_count] = torch.tensor(context.value_positions, dtype=torch.long)
            position_mask[row, :value_count] = 1.0
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
        batch = DataProto.from_dict(tensors=tensors)
        batch.meta_info.update(
            {
                "global_token_num": attention_mask.sum(dim=-1).tolist(),
                "micro_batch_size": self.config.critic.forward_micro_batch_size_per_gpu,
                "max_token_len": self.config.critic.forward_max_token_len_per_gpu,
                "use_dynamic_bsz": self.config.critic.use_dynamic_bsz,
            }
        )
        return batch, mapping

    def _score_contexts(
        self,
        critic_batch: DataProto,
        mapping: list[tuple[int, int]],
        bundles: list[_Bundle],
    ) -> None:
        inference_batch, pad_size = pad_dataproto_to_divisor(critic_batch, self.critic_wg.world_size)
        output = self.critic_wg.compute_values(inference_batch)
        values = output.batch["values"].detach().cpu()
        variances = output.batch.get("variances")
        if variances is not None:
            variances = variances.detach().cpu()
        if pad_size:
            values = values[:-pad_size]
            if variances is not None:
                variances = variances[:-pad_size]
        expected_shape = critic_batch.batch["critic_position_mask"].shape
        if tuple(values.shape) != tuple(expected_shape) or not torch.isfinite(values).all():
            raise RuntimeError(
                f"critic returned invalid values: expected finite {tuple(expected_shape)}, got {tuple(values.shape)}"
            )
        if self.feature.recipe == "beta_variance":
            if variances is None or tuple(variances.shape) != tuple(expected_shape):
                raise RuntimeError("beta_variance critic must return one variance for every requested value")
            if not torch.isfinite(variances).all() or torch.any(variances < 0):
                raise RuntimeError("beta_variance critic returned invalid variances")
        elif variances is not None:
            raise RuntimeError("scalar_random critic unexpectedly returned variances")
        critic_batch.batch["critic_old_values"] = values.clone()
        for row, (bundle_index, critique_index) in enumerate(mapping):
            position_count = len(bundles[bundle_index].contexts[critique_index].value_positions)
            bundles[bundle_index].critic_values.append(values[row, :position_count].tolist())
            if variances is not None:
                bundles[bundle_index].critic_variances.append(variances[row, :position_count].tolist())

    def _set_warmup_targets(
        self,
        critic_batch: DataProto,
        mapping: list[tuple[int, int]],
        bundles: list[_Bundle],
    ) -> None:
        for row, (bundle_index, _) in enumerate(mapping):
            terminal = len(bundles[bundle_index].solution_ids)
            critic_batch.batch["critic_targets"][row, terminal] = bundles[bundle_index].terminal_reward
            critic_batch.batch["critic_target_mask"][row, terminal] = 1.0

    def _select_marks(self, bundles: list[_Bundle]) -> None:
        if self.feature.mark_selector == "random":
            for bundle in bundles:
                rng = stable_rng(
                    self.feature.selection_seed,
                    self.global_steps,
                    bundle.dataset_index,
                    bundle.order,
                )
                bundle.marks = select_random_marks(
                    len(bundle.solution_ids),
                    k=self.feature.max_marks,
                    min_gap=self.feature.min_mark_gap,
                    start_fraction=self.feature.mark_start_fraction,
                    end_fraction=self.feature.mark_end_fraction,
                    rng=rng,
                )
                for token in bundle.marks:
                    self._audit(
                        "mark_selection",
                        rollout_id=bundle.rollout_id,
                        token=token,
                        reason="random",
                        scope=bundle.rollout_id,
                    )
            return

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
                if len(bundle.critic_variances) != self.feature.num_critiques:
                    raise RuntimeError("beta_variance selection requires one variance stream per critique")
                averaged = [sum(values) / len(values) for values in zip(*bundle.critic_variances, strict=True)]
                low, high = candidate_bounds(
                    len(bundle.solution_ids),
                    self.feature.mark_start_fraction,
                    self.feature.mark_end_fraction,
                )
                for token in range(low, high + 1):
                    candidates.append(VarianceCandidate(bundle.order, bundle.rollout_id, token, averaged[token]))
            rng = stable_rng(self.feature.selection_seed, self.global_steps, scope)
            selections = select_variance_marks(
                candidates,
                k=self.feature.max_marks,
                min_gap=self.feature.min_mark_gap,
                random_probability=self.feature.variance_random_probability,
                rng=rng,
            )
            for selection in selections:
                by_rollout[selection.candidate.rollout_id].marks.append(selection.candidate.token)
                self._audit(
                    "mark_selection",
                    rollout_id=selection.candidate.rollout_id,
                    token=selection.candidate.token,
                    variance=selection.candidate.variance,
                    reason=selection.reason,
                    draw=selection.draw,
                    scope=scope,
                )
        for bundle in bundles:
            bundle.marks.sort()

    def _run_continuations(self, source: DataProto, bundles: list[_Bundle]) -> None:
        requests: list[tuple[int, int, int]] = []
        source_rows: list[int] = []
        prompt_overrides: list[list[int]] = []
        for bundle_index, bundle in enumerate(bundles):
            for mark in bundle.marks:
                for sample_index in range(self.feature.continuations_per_mark):
                    requests.append((bundle_index, mark, sample_index))
                    source_rows.append(bundle.source_row)
                    prompt_overrides.append([*bundle.prompt_ids, *bundle.solution_ids[:mark]])
        if not requests:
            self.checkpoint_manager.sleep_replicas()
            return
        try:
            request_batch = self._secondary_request_batch(source, source_rows, prompt_overrides)
            generated_rows = self._generate_rows_with_isolation(request_batch, allow_all_failures=True)
        finally:
            self.checkpoint_manager.sleep_replicas()

        reward_source_rows: list[int] = []
        reward_prompts: list[list[int]] = []
        reward_responses: list[list[int]] = []
        successful_requests: list[tuple[int, int, int]] = []
        for request, generated in zip(requests, generated_rows, strict=True):
            bundle_index, mark, sample_index = request
            remaining = int(self.config.actor_rollout_ref.rollout.response_length) - mark
            generated = self._truncate_generated_row(generated, remaining)
            if generated is None:
                self._audit(
                    "continuation_failure",
                    rollout_id=bundles[bundle_index].rollout_id,
                    mark=mark,
                    sample=sample_index,
                    reason="generation",
                )
                continue
            suffix = self._valid_response_ids(generated, 0, cap=remaining)
            full_response = [*bundles[bundle_index].solution_ids[:mark], *suffix]
            reward_source_rows.append(bundles[bundle_index].source_row)
            reward_prompts.append(bundles[bundle_index].prompt_ids)
            reward_responses.append(full_response)
            successful_requests.append(request)
        if not successful_requests:
            return
        reward_batch = self._make_reward_batch(
            source,
            reward_source_rows,
            reward_prompts,
            reward_responses,
        )
        rewards = self._continuation_rewards_with_isolation(reward_batch)
        by_bundle_mark: dict[tuple[int, int], list[float]] = {}
        for request, reward in zip(successful_requests, rewards, strict=True):
            bundle_index, mark, sample_index = request
            if reward is None:
                self._audit(
                    "continuation_failure",
                    rollout_id=bundles[bundle_index].rollout_id,
                    mark=mark,
                    sample=sample_index,
                    reason="reward",
                )
                continue
            by_bundle_mark.setdefault((bundle_index, mark), []).append(reward)
            self._audit(
                "continuation",
                rollout_id=bundles[bundle_index].rollout_id,
                mark=mark,
                sample=sample_index,
                reward=reward,
            )
        for bundle_index, bundle in enumerate(bundles):
            mark_rewards = {
                mark: by_bundle_mark[(bundle_index, mark)]
                for mark in bundle.marks
                if (bundle_index, mark) in by_bundle_mark
            }
            bundle.per_mark_targets, bundle.dense_targets = aggregate_mark_targets(mark_rewards)

    def _set_training_targets(
        self,
        critic_batch: DataProto,
        mapping: list[tuple[int, int]],
        bundles: list[_Bundle],
    ) -> None:
        for row, (bundle_index, _) in enumerate(mapping):
            bundle = bundles[bundle_index]
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
                terminal_token=len(bundle.solution_ids),
            )

    def _solution_advantages(self, bundles: list[_Bundle], response_width: int) -> torch.Tensor:
        advantages = torch.zeros((len(bundles), response_width), dtype=torch.float32)
        mask = torch.zeros_like(advantages)
        for row, bundle in enumerate(bundles):
            averaged_values = [sum(values) / len(values) for values in zip(*bundle.critic_values, strict=True)]
            action_advantages = token_gae(
                averaged_values,
                bundle.terminal_reward,
                gamma=float(self.config.algorithm.gamma),
                gae_lambda=float(self.config.algorithm.lam),
            )
            length = len(action_advantages)
            advantages[row, :length] = torch.tensor(action_advantages)
            mask[row, :length] = 1.0
        return masked_whiten(advantages, mask)

    def _critique_advantages(self, bundles: list[_Bundle], response_width: int) -> torch.Tensor:
        result = torch.zeros((len(bundles) * self.feature.num_critiques, response_width), dtype=torch.float32)
        row = 0
        for bundle in bundles:
            rewards: list[float] = []
            points = [len(bundle.solution_ids), *sorted(bundle.per_mark_targets)]
            targets = [
                bundle.terminal_reward,
                *(bundle.per_mark_targets[mark] for mark in sorted(bundle.per_mark_targets)),
            ]
            for critique_values in bundle.critic_values:
                predictions = [critique_values[point] for point in points]
                rewards.append(critique_accuracy_reward(predictions, targets, max_reward=self.feature.max_reward))
            normalized = critique_group_advantages(
                rewards,
                self.feature.critique_normalization_epsilon,
            )
            for critique_row, advantage in zip(bundle.critique_rows, normalized, strict=True):
                mask = critique_row.batch["response_mask"][0].float()
                result[row] = advantage * mask
                row += 1
            self._audit(
                "critique_credit",
                rollout_id=bundle.rollout_id,
                rewards=rewards,
                advantages=normalized,
                points=points,
                targets=targets,
            )
        return result

    @staticmethod
    def _actor_keys() -> list[str]:
        return [
            "prompts",
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "rollout_log_probs",
        ]

    def _make_actor_batch(self, source: DataProto, bundles: list[_Bundle]) -> DataProto:
        solution_rows = source.select_idxs([bundle.source_row for bundle in bundles]).select(
            batch_keys=self._actor_keys(), non_tensor_batch_keys=[], meta_info_keys=[]
        )
        critique_batch = DataProto.concat(
            [
                row.select(batch_keys=self._actor_keys(), non_tensor_batch_keys=[], meta_info_keys=[])
                for bundle in bundles
                for row in bundle.critique_rows
            ]
        )
        response_width = solution_rows.batch["responses"].shape[1]
        solution_rows.batch["advantages"] = self._solution_advantages(bundles, response_width)
        critique_batch.batch["advantages"] = self._critique_advantages(bundles, response_width)
        solution_rows.batch["old_log_probs"] = solution_rows.batch["rollout_log_probs"].clone()
        critique_batch.batch["old_log_probs"] = critique_batch.batch["rollout_log_probs"].clone()
        actor_batch = DataProto.concat([solution_rows, critique_batch])
        actor_batch.meta_info.update(
            {
                "temperature": 1.0,
                "global_token_num": actor_batch.batch["attention_mask"].sum(dim=-1).tolist(),
            }
        )
        global_minibatch = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(
            self.config.actor_rollout_ref.rollout.n
        )
        actor_batch, pad_size = pad_dataproto_to_divisor(actor_batch, global_minibatch)
        if pad_size:
            actor_batch.batch["response_mask"][-pad_size:] = 0
            actor_batch.batch["advantages"][-pad_size:] = 0
        self._audit(
            "actor_batch",
            solutions=len(bundles),
            critiques=len(bundles) * self.feature.num_critiques,
            continuations=0,
            padding=pad_size,
        )
        return actor_batch

    def _pad_critic_batch(self, critic_batch: DataProto) -> DataProto:
        global_minibatch = int(self.config.critic.ppo_mini_batch_size) * int(self.config.actor_rollout_ref.rollout.n)
        critic_batch, pad_size = pad_dataproto_to_divisor(critic_batch, global_minibatch)
        if pad_size:
            critic_batch.batch["critic_target_mask"][-pad_size:] = 0
        critic_batch.meta_info["global_token_num"] = critic_batch.batch["attention_mask"].sum(dim=-1).tolist()
        return critic_batch

    def _build_bundles_and_critiques(self, source: DataProto, rewards: list[float]) -> list[_Bundle]:
        request_rows: list[int] = []
        prompt_overrides: list[list[int]] = []
        provisional: list[_Bundle] = []
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        critique_cap_config = self.feature.critique_max_response_length
        critique_cap = int(critique_cap_config or self.config.actor_rollout_ref.rollout.response_length)
        prompt_limit = int(self.config.actor_rollout_ref.rollout.prompt_length)
        for row, reward in enumerate(rewards):
            prompt_ids = self._valid_prompt_ids(source, row)
            solution_ids = self._valid_response_ids(source, row)
            critique_prompt = [*prompt_ids, *solution_ids, *self._critique_instruction_ids]
            if len(critique_prompt) > prompt_limit:
                raise ValueError(
                    "self-critique prompt exceeds actor_rollout_ref.rollout.prompt_length: "
                    f"actual={len(critique_prompt)} limit={prompt_limit}"
                )
            fixed_context = (
                len(prompt_ids)
                + len(self._critic_delimiter_ids)
                + len(self._solution_delimiter_ids)
                + len(solution_ids)
            )
            effective_cap = min(critique_cap, self._critic_context_limit - fixed_context)
            if effective_cap <= 0:
                raise ValueError("self-critique has no capacity in the critic context window")
            dataset_values = source.non_tensor_batch.get("index", np.arange(len(source), dtype=object))
            group_values = source.non_tensor_batch.get("prompt_group_id", np.arange(len(source), dtype=object))
            rollout_values = source.non_tensor_batch.get("intermediate_mc_rollout_id")
            rollout_id = (
                str(rollout_values[row]) if rollout_values is not None else f"{group_values[row]}:{row % rollout_n}"
            )
            bundle = _Bundle(
                order=row,
                dataset_index=dataset_values[row],
                rollout_id=rollout_id,
                prompt_group_id=str(group_values[row]),
                source_row=row,
                prompt_ids=prompt_ids,
                solution_ids=solution_ids,
                terminal_reward=reward,
            )
            provisional.append(bundle)
            for _ in range(self.feature.num_critiques):
                request_rows.append(row)
                prompt_overrides.append(critique_prompt)
        request_batch = self._secondary_request_batch(source, request_rows, prompt_overrides)
        generated = self._generate_rows_with_isolation(request_batch)
        bundles: list[_Bundle] = []
        offset = 0
        for bundle in provisional:
            fixed_context = (
                len(bundle.prompt_ids)
                + len(self._critic_delimiter_ids)
                + len(self._solution_delimiter_ids)
                + len(bundle.solution_ids)
            )
            cap = min(critique_cap, self._critic_context_limit - fixed_context)
            rows = [
                self._truncate_generated_row(generated[offset + index], cap)
                for index in range(self.feature.num_critiques)
            ]
            offset += self.feature.num_critiques
            if any(row is None for row in rows):
                self._audit("bundle_dropped", rollout_id=bundle.rollout_id, reason="incomplete_critiques")
                continue
            bundle.critique_rows = [row for row in rows if row is not None]
            bundle.critique_ids = [self._valid_response_ids(row, 0, cap=cap) for row in bundle.critique_rows]
            bundle.contexts = [
                build_critic_context(
                    bundle.prompt_ids,
                    critique_ids,
                    bundle.solution_ids,
                    critique_delimiter_ids=self._critic_delimiter_ids,
                    solution_delimiter_ids=self._solution_delimiter_ids,
                )
                for critique_ids in bundle.critique_ids
            ]
            self._audit(
                "bundle",
                rollout_id=bundle.rollout_id,
                solution_tokens=len(bundle.solution_ids),
                critique_tokens=[len(tokens) for tokens in bundle.critique_ids],
                value_positions=len(bundle.contexts[0].value_positions),
            )
            bundles.append(bundle)
        if not bundles:
            raise RuntimeError("no complete solution/critique bundles remain in this synchronous iteration")
        return bundles

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self.training_start_time = time.time()
        self._load_checkpoint()
        if self._completed_training_resume():
            return
        self.checkpoint_manager.update_weights(self.global_steps)
        if self.config.trainer.get("val_before_train", True):
            metrics = self._validate()
            logger.log(data=metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        current_epoch = self.global_steps // len(self.train_dataloader)
        progress = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics: dict[str, float] = {}
                batch = DataProto.from_single_dict(batch_dict)
                group_ids = np.array(
                    [f"step-{self.global_steps}:prompt-{row}" for row in range(len(batch))],
                    dtype=object,
                )
                batch.non_tensor_batch["uid"] = group_ids.copy()
                batch.non_tensor_batch["prompt_group_id"] = group_ids
                gen_batch = self._get_gen_batch(batch)
                gen_batch.non_tensor_batch["agent_name"] = np.array(
                    ["single_turn_agent"] * len(gen_batch),
                    dtype=object,
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                try:
                    source = self.async_rollout_manager.generate_sequences(gen_batch)
                    if "response_mask" not in source.batch:
                        raise RuntimeError("single-turn rollout did not return response_mask")
                    if "rollout_log_probs" not in source.batch:
                        raise RuntimeError("single-turn rollout did not return sampling-time rollout_log_probs")
                    repeated = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    self._drop_overlapping_non_tensor_keys(source, repeated)
                    source = repeated.union(source)
                    rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                    source.non_tensor_batch["intermediate_mc_rollout_id"] = np.array(
                        [
                            f"{source.non_tensor_batch['prompt_group_id'][row]}:{row % rollout_n}"
                            for row in range(len(source))
                        ],
                        dtype=object,
                    )
                    if self.config.trainer.balance_batch:
                        self._balance_batch(source, metrics=metrics)
                    rewards = self._solution_rewards(source)
                    bundles = self._build_bundles_and_critiques(source, rewards)
                finally:
                    self.checkpoint_manager.sleep_replicas()

                critic_batch, mapping = self._make_critic_batch(bundles)
                self._score_contexts(critic_batch, mapping, bundles)
                in_warmup = self.critic_update_count < self.feature.critic_warmup_updates
                if in_warmup:
                    self._set_warmup_targets(critic_batch, mapping, bundles)
                    self._audit("warmup", critic_update_count=self.critic_update_count, continuations=0)
                else:
                    self._select_marks(bundles)
                    if any(bundle.marks for bundle in bundles):
                        self.checkpoint_manager.wake_up_replicas()
                        self._run_continuations(source, bundles)
                    self._set_training_targets(critic_batch, mapping, bundles)

                critic_output = self._update_critic(self._pad_critic_batch(critic_batch))
                self.critic_update_count += 1
                metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                metrics.update(
                    {
                        "intermediate_mc/critic_update_count": self.critic_update_count,
                        "intermediate_mc/warmup": float(in_warmup),
                        "intermediate_mc/bundles": len(bundles),
                        "intermediate_mc/critiques": len(bundles) * self.feature.num_critiques,
                        "intermediate_mc/selected_marks": sum(len(bundle.marks) for bundle in bundles),
                        "intermediate_mc/surviving_marks": sum(len(bundle.per_mark_targets) for bundle in bundles),
                    }
                )

                if not in_warmup:
                    actor_batch = self._make_actor_batch(source, bundles)
                    actor_output = self._update_actor(actor_batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                save_due = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                )
                if save_due:
                    self._save_checkpoint()

                if in_warmup:
                    self.checkpoint_manager.wake_up_replicas()
                else:
                    self.checkpoint_manager.update_weights(self.global_steps)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    metrics.update(self._validate())

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)
                progress.update(1)
                self.global_steps += 1

                if is_last_step:
                    progress.close()
                    return
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=source)
