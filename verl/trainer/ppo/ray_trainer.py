# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.one_logger_integration import OneLoggerInstrumented
from verl.trainer.ppo.reward import compute_reward, compute_reward_async, extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path,
    should_save_ckpt_esi,
    should_save_ckpt_timeout,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.reward_manager.conditional import _select_low_confidence_token_indices
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _build_critique_manager_config(config):
    """Clone the actor config and give the critique rollout a distinct Ray namespace."""

    critique_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    custom = OmegaConf.select(critique_config, "actor_rollout_ref.rollout.custom", default=None)
    if custom is None:
        custom = {}
    else:
        custom = OmegaConf.to_container(custom, resolve=True)
        if not isinstance(custom, dict):
            raise TypeError("actor_rollout_ref.rollout.custom must be a mapping")
    custom["server_name_prefix"] = "critique_actor"
    OmegaConf.update(
        critique_config,
        "actor_rollout_ref.rollout.custom",
        custom,
        merge=False,
        force_add=True,
    )
    return critique_config


def _pack_reward_extra_info(values: list[Any]) -> np.ndarray:
    """Preserve ragged per-sample reward metadata as object arrays.

    Scalar-like fields stay as dense numpy arrays. Variable-length fields such as
    token index lists and top-k hit flags fall back to dtype=object.
    """
    try:
        return np.array(values)
    except ValueError:
        return np.array(values, dtype=object)


def _compute_shortest_success_reward_metrics(reward_extra_info: dict[str, Any]) -> dict[str, float]:
    """Summarize grouped shortest-success rewards without changing them."""
    required = {
        "acc",
        "shortest_success_selected",
        "shortest_success_response_tokens",
        "shortest_success_group_id",
        "shortest_success_group_has_success",
        "shortest_success_group_min_tokens",
    }
    if not required.issubset(reward_extra_info):
        return {}

    acc = np.asarray(reward_extra_info["acc"], dtype=np.float64)
    selected = np.asarray(reward_extra_info["shortest_success_selected"], dtype=np.float64)
    response_tokens = np.asarray(reward_extra_info["shortest_success_response_tokens"], dtype=np.float64)
    group_ids = [str(value) for value in reward_extra_info["shortest_success_group_id"]]
    group_has_success = list(reward_extra_info["shortest_success_group_has_success"])
    group_min_tokens = list(reward_extra_info["shortest_success_group_min_tokens"])
    if not (
        len(acc)
        == len(selected)
        == len(response_tokens)
        == len(group_ids)
        == len(group_has_success)
        == len(group_min_tokens)
    ):
        raise ValueError("Shortest-success reward metric fields must have identical lengths")

    correct_count = float((acc > 0.5).sum())
    selected_mask = selected > 0.5
    group_first_indices: dict[str, int] = {}
    for idx, group_id in enumerate(group_ids):
        group_first_indices.setdefault(group_id, idx)
    first_indices = list(group_first_indices.values())
    successful_group_indices = [idx for idx in first_indices if bool(group_has_success[idx])]
    successful_group_mins = [float(group_min_tokens[idx]) for idx in successful_group_indices]

    return {
        "reward/shortest_success/selected_fraction": float(selected.mean()) if len(selected) else 0.0,
        "reward/shortest_success/selected_per_correct": (
            float(selected.sum()) / correct_count if correct_count > 0 else 0.0
        ),
        "reward/shortest_success/groups_with_success_fraction": (
            float(len(successful_group_indices)) / float(len(first_indices)) if first_indices else 0.0
        ),
        "reward/shortest_success/mean_min_success_tokens": (
            float(np.mean(successful_group_mins)) if successful_group_mins else 0.0
        ),
        "reward/shortest_success/mean_selected_tokens": (
            float(response_tokens[selected_mask].mean()) if bool(selected_mask.any()) else 0.0
        ),
        "reward/shortest_success/raw_acc_mean": float(acc.mean()) if len(acc) else 0.0,
    }


def _compute_longest_success_penalty_reward_metrics(
    reward_extra_info: dict[str, Any],
) -> dict[str, float]:
    """Summarize longest-success penalties without changing reward values."""
    required = {
        "acc",
        "longest_success_penalty_reward",
        "longest_success_penalized",
        "longest_success_response_tokens",
        "longest_success_group_id",
        "longest_success_group_has_success",
        "longest_success_group_within_margin",
        "longest_success_group_min_tokens",
        "longest_success_group_max_tokens",
    }
    if not required.issubset(reward_extra_info):
        return {}

    acc = np.asarray(reward_extra_info["acc"], dtype=np.float64)
    reward = np.asarray(reward_extra_info["longest_success_penalty_reward"], dtype=np.float64)
    penalized = np.asarray(reward_extra_info["longest_success_penalized"], dtype=np.float64)
    response_tokens = np.asarray(reward_extra_info["longest_success_response_tokens"], dtype=np.float64)
    group_ids = [str(value) for value in reward_extra_info["longest_success_group_id"]]
    group_has_success = list(reward_extra_info["longest_success_group_has_success"])
    group_within_margin = list(reward_extra_info["longest_success_group_within_margin"])
    group_min_tokens = list(reward_extra_info["longest_success_group_min_tokens"])
    group_max_tokens = list(reward_extra_info["longest_success_group_max_tokens"])
    lengths = {
        len(acc),
        len(reward),
        len(penalized),
        len(response_tokens),
        len(group_ids),
        len(group_has_success),
        len(group_within_margin),
        len(group_min_tokens),
        len(group_max_tokens),
    }
    if len(lengths) != 1:
        raise ValueError("Longest-success-penalty reward metric fields must have identical lengths")

    correct_count = float((acc > 0.5).sum())
    rewarded_mask = reward > 0.5
    penalized_mask = penalized > 0.5
    group_first_indices: dict[str, int] = {}
    for idx, group_id in enumerate(group_ids):
        group_first_indices.setdefault(group_id, idx)
    first_indices = list(group_first_indices.values())
    successful_group_indices = [idx for idx in first_indices if bool(group_has_success[idx])]
    within_margin_count = sum(bool(group_within_margin[idx]) for idx in successful_group_indices)
    penalized_group_count = len(successful_group_indices) - within_margin_count
    successful_group_mins = [float(group_min_tokens[idx]) for idx in successful_group_indices]
    successful_group_maxes = [float(group_max_tokens[idx]) for idx in successful_group_indices]
    max_to_min_ratios = [
        float(group_max_tokens[idx]) / float(group_min_tokens[idx]) for idx in successful_group_indices
    ]

    return {
        "reward/longest_success_penalty/rewarded_fraction": float(reward.mean()) if len(reward) else 0.0,
        "reward/longest_success_penalty/rewarded_per_correct": (
            float(reward.sum()) / correct_count if correct_count > 0 else 0.0
        ),
        "reward/longest_success_penalty/penalized_per_correct": (
            float(penalized.sum()) / correct_count if correct_count > 0 else 0.0
        ),
        "reward/longest_success_penalty/groups_with_success_fraction": (
            float(len(successful_group_indices)) / float(len(first_indices)) if first_indices else 0.0
        ),
        "reward/longest_success_penalty/successful_groups_within_margin_fraction": (
            float(within_margin_count) / float(len(successful_group_indices)) if successful_group_indices else 0.0
        ),
        "reward/longest_success_penalty/successful_groups_penalized_fraction": (
            float(penalized_group_count) / float(len(successful_group_indices)) if successful_group_indices else 0.0
        ),
        "reward/longest_success_penalty/mean_min_success_tokens": (
            float(np.mean(successful_group_mins)) if successful_group_mins else 0.0
        ),
        "reward/longest_success_penalty/mean_max_success_tokens": (
            float(np.mean(successful_group_maxes)) if successful_group_maxes else 0.0
        ),
        "reward/longest_success_penalty/mean_max_to_min_ratio": (
            float(np.mean(max_to_min_ratios)) if max_to_min_ratios else 0.0
        ),
        "reward/longest_success_penalty/mean_rewarded_tokens": (
            float(response_tokens[rewarded_mask].mean()) if bool(rewarded_mask.any()) else 0.0
        ),
        "reward/longest_success_penalty/mean_penalized_tokens": (
            float(response_tokens[penalized_mask].mean()) if bool(penalized_mask.any()) else 0.0
        ),
        "reward/longest_success_penalty/raw_acc_mean": float(acc.mean()) if len(acc) else 0.0,
    }


def _validation_metric_section(var_name: str, core_var: str, metric_name: str, n_max: int) -> str | None:
    """Return the validation metric section for exported metrics.

    Validation logging is intentionally restricted to the aggregated selection
    metrics, except that a single validation rollout has no selection metric and
    therefore exports its core mean. Plain mean/std summaries for multi-rollout
    validation and aux counters are not exported.
    """
    if n_max == 1 and var_name == core_var and metric_name == "mean@1":
        return "val-core"
    if metric_name.startswith(("best@", "maj@", "worst@")):
        return "val-agg"
    if metric_name in {"max", "min"} and var_name.endswith("trade_pnl_percent"):
        return "val-agg"

    return None


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def _to_jsonable(value: Any) -> Any:
    """Recursively normalize numpy-backed rollout metadata for JSONL dumps."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_response_tokens_and_logprobs(
    tokenizer,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_logprobs: Optional[torch.Tensor] = None,
) -> tuple[list[list[str]], Optional[list[list[float]]]]:
    """Convert response ids to token strings and align optional per-token logprobs."""
    response_rows = responses.detach().cpu()
    response_mask_rows = response_mask.detach().cpu().bool()
    logprob_rows = response_logprobs.detach().cpu() if response_logprobs is not None else None

    response_tokens: list[list[str]] = []
    response_token_logprobs: Optional[list[list[float]]] = [] if logprob_rows is not None else None

    for row_idx in range(response_rows.size(0)):
        valid_mask = response_mask_rows[row_idx]
        token_ids = response_rows[row_idx][valid_mask].tolist()
        response_tokens.append(tokenizer.convert_ids_to_tokens(token_ids))

        if response_token_logprobs is not None:
            row_logprobs = logprob_rows[row_idx][valid_mask].to(torch.float32).tolist()
            response_token_logprobs.append([float(logprob) for logprob in row_logprobs])

    return response_tokens, response_token_logprobs


class RayPPOTrainer(OneLoggerInstrumented):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self._enable_rollout_logprobs_for_generation_dumps()

        self.intermediate_mc_controller = None
        if bool(OmegaConf.select(self.config, "algorithm.intermediate_mc_value.enable", default=False)):
            from verl.trainer.ppo.ray_trainer_intermediate_mc import IntermediateMCValueController

            self.intermediate_mc_controller = IntermediateMCValueController(self)
        self.branch_revision_controller = None
        if bool(OmegaConf.select(self.config, "algorithm.branch_revision_grpo.enable", default=False)):
            from verl.trainer.ppo.ray_trainer_branch_revision import BranchRevisionGRPOController

            self.branch_revision_controller = BranchRevisionGRPOController(self)
        self.random_continuation_controller = None
        if bool(OmegaConf.select(self.config, "algorithm.random_continuation_baseline.enable", default=False)):
            from verl.trainer.ppo.ray_trainer_random_continuation import RandomContinuationBaselineController

            self.random_continuation_controller = RandomContinuationBaselineController(self)
        controllers = [
            self.intermediate_mc_controller,
            self.branch_revision_controller,
            self.random_continuation_controller,
        ]
        if sum(controller is not None for controller in controllers) > 1:
            raise ValueError("intermediate MC, branch revision, and random continuation are mutually exclusive")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _enable_rollout_logprobs_for_generation_dumps(self) -> None:
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        validation_data_dir = self.config.trainer.get("validation_data_dir", None)
        if not rollout_data_dir and not validation_data_dir:
            return
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            return

        with open_dict(self.config):
            self.config.actor_rollout_ref.rollout.calculate_log_probs = True
        print("Enabled actor_rollout_ref.rollout.calculate_log_probs because generation dump JSONLs are configured.")

    def _has_custom_synchronous_actor_update(self) -> bool:
        return (
            getattr(self, "intermediate_mc_controller", None) is not None
            or getattr(self, "branch_revision_controller", None) is not None
            or getattr(self, "random_continuation_controller", None) is not None
        )

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        response_tokens=None,
        response_token_logprobs=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        if response_tokens is not None:
            base_data["response_tokens"] = response_tokens
        if response_token_logprobs is not None:
            base_data["response_token_logprobs"] = response_token_logprobs

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: _to_jsonable(v[i]) for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            response_logprobs = batch.batch.get("rollout_log_probs", None)
            if response_logprobs is None:
                response_logprobs = batch.batch.get("old_log_probs", None)
            response_tokens, response_token_logprobs = _extract_response_tokens_and_logprobs(
                self.tokenizer,
                batch.batch["responses"],
                batch.batch["response_mask"],
                response_logprobs=response_logprobs,
            )

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
                response_tokens=response_tokens,
                response_token_logprobs=response_token_logprobs,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # Keep algorithm-side supervision fields on the driver batch so post-generation
        # updates (for example OPSD teacher construction) can still access them.
        reward_keys = (
            set(
                {
                    "data_source",
                    "reward_model",
                    "extra_info",
                    "uid",
                    "prompt_group_id",
                    "ground_truth_answer",
                    "problem",
                }
            )
            & batch.non_tensor_batch.keys()
        )
        opsd_ground_truth_field = OmegaConf.select(self.config, "algorithm.opsd.ground_truth_field")
        if OmegaConf.select(self.config, "algorithm.opsd.enable") and opsd_ground_truth_field:
            ground_truth_root = str(opsd_ground_truth_field).split(".", 1)[0]
            if ground_truth_root in batch.non_tensor_batch:
                reward_keys.add(ground_truth_root)

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    @staticmethod
    def _drop_overlapping_non_tensor_keys(target: DataProto, reference: DataProto) -> None:
        overlap = set(target.non_tensor_batch.keys()) & set(reference.non_tensor_batch.keys())
        if overlap:
            target.pop(non_tensor_batch_keys=list(overlap))

    @staticmethod
    def _resolve_pad_token_id_from_tokenizer(tokenizer) -> int:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    @staticmethod
    def _left_pad_token_lists(
        token_lists: list[list[int]],
        *,
        pad_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max((len(tokens) for tokens in token_lists), default=0)
        if max_len <= 0:
            raise ValueError("Expected at least one non-empty prompt prefix token list.")

        batch_size = len(token_lists)
        prompt_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long, device=device)
        prompt_attn = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        prompt_pos = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for row_idx, tokens in enumerate(token_lists):
            token_count = len(tokens)
            if token_count <= 0:
                continue
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            prompt_ids[row_idx, -token_count:] = token_tensor
            prompt_attn[row_idx, -token_count:] = 1
            prompt_pos[row_idx, -token_count:] = torch.arange(token_count, dtype=torch.long, device=device)

        return prompt_ids, prompt_attn, prompt_pos

    @staticmethod
    def _pad_right_token_lists(
        token_lists: list[list[int]],
        *,
        pad_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max((len(tokens) for tokens in token_lists), default=0)
        if max_len <= 0:
            raise ValueError("Expected at least one non-empty ground-truth token list.")

        batch_size = len(token_lists)
        token_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long, device=device)
        token_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for row_idx, tokens in enumerate(token_lists):
            token_count = len(tokens)
            if token_count <= 0:
                continue
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            token_ids[row_idx, :token_count] = token_tensor
            token_mask[row_idx, :token_count] = 1

        return token_ids, token_mask

    def _maybe_apply_reward_focus_tail_mask(self, batch: DataProto) -> None:
        if self.config.data.get("masked_solution_selection_mode", "random_fraction") != "reward_focus_tail":
            return

        dataset = self.train_dataset
        required_methods = (
            "_has_masked_solution_placeholders",
            "_build_prompt_template_with_sentinel",
            "_masked_solution_sentinel",
            "materialize_masked_solution_prompt",
            "_build_messages",
        )
        if not all(hasattr(dataset, method_name) for method_name in required_methods):
            raise ValueError(
                "data.masked_solution_selection_mode='reward_focus_tail' requires the default RLHFDataset "
                "masked-solution helpers on the training dataset."
            )

        prompt_key = self.config.data.get("prompt_key", "prompt")
        solution_key = self.config.data.get("solution_key", "ground_truth_answer")
        reward_tail_percent = float(self.config.reward.reward_kwargs.low_confidence_tail_percent)
        reward_min_tokens = int(
            OmegaConf.select(self.config, "reward.reward_kwargs.low_confidence_min_tokens", default=1)
        )

        prompt_messages = list(batch.non_tensor_batch[prompt_key])
        solution_values = list(
            batch.non_tensor_batch.get(solution_key, np.array([None] * len(prompt_messages), dtype=object))
        )
        sample_indices = list(batch.non_tensor_batch.get("index", np.arange(len(prompt_messages), dtype=object)))

        selected_rows: list[int] = []
        prompt_prefix_token_lists: list[list[int]] = []
        gt_token_lists: list[list[int]] = []
        sample_examples: list[dict[str, Any]] = []
        sample_items: list[Any] = []

        for row_idx, (messages, solution_text, item_key) in enumerate(
            zip(prompt_messages, solution_values, sample_indices, strict=False)
        ):
            example = {
                prompt_key: messages,
                solution_key: solution_text,
            }
            if not dataset._has_masked_solution_placeholders(example):
                continue
            if not isinstance(solution_text, str) or not solution_text:
                continue

            prompt_template, _, _ = dataset._build_prompt_template_with_sentinel(example, item=item_key)
            sentinel = dataset._masked_solution_sentinel(item_key)
            prompt_prefix_text = prompt_template.replace(sentinel, "")
            prefix_token_ids = normalize_token_ids(
                self.tokenizer(prompt_prefix_text, add_special_tokens=False)["input_ids"]
            )
            gt_token_ids = normalize_token_ids(self.tokenizer(solution_text, add_special_tokens=False)["input_ids"])
            if not prefix_token_ids or not gt_token_ids:
                continue

            selected_rows.append(row_idx)
            prompt_prefix_token_lists.append(list(prefix_token_ids))
            gt_token_lists.append(list(gt_token_ids))
            sample_examples.append(example)
            sample_items.append(item_key)

        if not selected_rows:
            return

        device = torch.device("cpu")
        pad_token_id = self._resolve_pad_token_id_from_tokenizer(self.tokenizer)
        prompt_ids, prompt_attn, prompt_pos = self._left_pad_token_lists(
            prompt_prefix_token_lists,
            pad_token_id=pad_token_id,
            device=device,
        )
        gt_ids, gt_mask = self._pad_right_token_lists(
            gt_token_lists,
            pad_token_id=pad_token_id,
            device=device,
        )

        gt_len = gt_ids.size(1)
        seq_concat = torch.cat([prompt_ids, gt_ids], dim=-1)
        delta_pos = (
            torch.arange(1, gt_len + 1, device=device, dtype=torch.long).unsqueeze(0).expand(prompt_ids.size(0), -1)
        )
        gt_position_ids = prompt_pos[:, -1:] + delta_pos
        position_ids = torch.cat([prompt_pos, gt_position_ids], dim=-1)
        attention_mask = torch.cat([prompt_attn, gt_mask], dim=-1)

        prompt_only_batch = DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": gt_ids,
                "input_ids": seq_concat,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
        )
        prompt_only_batch_padded, pad_size = pad_dataproto_to_divisor(
            prompt_only_batch, self.actor_rollout_wg.world_size
        )
        prompt_only_output = self.actor_rollout_wg.compute_log_prob(prompt_only_batch_padded)
        prompt_only_log_probs = unpad_dataproto(prompt_only_output, pad_size=pad_size).batch["old_log_probs"]

        updated_prompt_messages = np.empty(len(prompt_messages), dtype=object)
        updated_prompt_messages[:] = list(prompt_messages)
        updated_raw_prompts = np.empty(len(prompt_messages), dtype=object)
        updated_raw_prompts[:] = list(batch.non_tensor_batch["raw_prompt"])
        updated_prompt_overrides = np.empty(len(prompt_messages), dtype=object)
        updated_prompt_overrides[:] = None
        updated_focus_indices = np.empty(len(prompt_messages), dtype=object)
        updated_focus_indices[:] = None

        for local_idx, row_idx in enumerate(selected_rows):
            focus_indices = _select_low_confidence_token_indices(
                prompt_only_log_probs[local_idx],
                len(gt_token_lists[local_idx]),
                tail_percent=reward_tail_percent,
                min_tokens=reward_min_tokens,
            )
            prepared_messages, prompt_ids_override = dataset.materialize_masked_solution_prompt(
                sample_examples[local_idx],
                masked_positions=set(focus_indices),
                item=sample_items[local_idx],
            )
            prompt_example = dict(sample_examples[local_idx])
            prompt_example[prompt_key] = prepared_messages

            updated_prompt_messages[row_idx] = prepared_messages
            updated_raw_prompts[row_idx] = dataset._build_messages(prompt_example)
            updated_prompt_overrides[row_idx] = prompt_ids_override
            updated_focus_indices[row_idx] = list(focus_indices)

        batch.non_tensor_batch[prompt_key] = updated_prompt_messages
        batch.non_tensor_batch["raw_prompt"] = updated_raw_prompts
        batch.non_tensor_batch["prompt_ids_override"] = updated_prompt_overrides
        batch.non_tensor_batch["masked_solution_focus_token_indices"] = updated_focus_indices

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_response_tokens = []
        sample_response_token_logprobs = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            val_response_length = self.config.actor_rollout_ref.rollout.val_kwargs.response_length
            if val_response_length is not None:
                test_gen_batch.meta_info["response_length"] = val_response_length
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            response_logprobs = test_output_gen_batch.batch.get("rollout_log_probs", None)
            if response_logprobs is None:
                response_logprobs = test_output_gen_batch.batch.get("old_log_probs", None)
            response_tokens, response_token_logprobs = _extract_response_tokens_and_logprobs(
                self.tokenizer,
                test_output_gen_batch.batch["responses"],
                test_output_gen_batch.batch["response_mask"],
                response_logprobs=response_logprobs,
            )
            sample_response_tokens.extend(response_tokens)
            if response_token_logprobs is None:
                sample_response_token_logprobs.extend([None] * len(response_tokens))
            else:
                sample_response_token_logprobs.extend(response_token_logprobs)

            self._drop_overlapping_non_tensor_keys(test_output_gen_batch, test_batch)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Validation must use the dedicated val_reward_fn rather than any inline reward
            # scores attached during generation. This keeps validation behavior consistent
            # across reward-manager variants.
            if "rm_scores" in test_batch.batch.keys():
                test_batch.pop(batch_keys=["rm_scores"])
            reward_extra_keys = list(test_batch.meta_info.get("reward_extra_keys", []))
            reward_extra_keys = [key for key in reward_extra_keys if key in test_batch.non_tensor_batch]
            if reward_extra_keys:
                test_batch.pop(non_tensor_batch_keys=reward_extra_keys)
            if "reward_extra_keys" in test_batch.meta_info:
                test_batch.pop(meta_info_keys=["reward_extra_keys"])

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            val_reward_fn = getattr(self, "val_reward_fn", None)
            if val_reward_fn is not None:
                reward_tensor, reward_extra_info = compute_reward(
                    test_batch, val_reward_fn, actor_wg=self.actor_rollout_wg
                )
            else:
                reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key == "reward":
                    continue
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, torch.Tensor):
                    reward_extra_infos_dict[key].extend(values.detach().cpu().tolist())
                elif isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                response_tokens=sample_response_tokens,
                response_token_logprobs=sample_response_token_logprobs,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys() if "@" in name], default=1
                )
                for metric_name, metric_val in metric2val.items():
                    metric_sec = _validation_metric_section(
                        var_name=var_name, core_var=core_var, metric_name=metric_name, n_max=n_max
                    )
                    if metric_sec is None:
                        continue
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _should_enable_agent_reward_loop(self) -> bool:
        """Return whether rollout workers may stream reward computation.

        Synchronous custom actor modes evaluate child continuations through the
        blocking driver-side reward path. Keeping the streaming reward loop
        disabled also preserves rollout identity columns in AgentLoop output.
        """

        if self._has_custom_synchronous_actor_update():
            return False
        disable_agent_reward_loop = bool(getattr(self.reward_fn, "disable_async_reward_loop", False))
        return (
            not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        ) and not disable_agent_reward_loop

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        critique_actor_resource_pool = None
        if Role.CritiqueActorRollout in self.role_worker_mapping:
            critique_actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.CritiqueActorRollout)
            self.critique_manager_config = _build_critique_manager_config(self.config)
            critique_actor_config = self.critique_manager_config.actor_rollout_ref
            critique_actor_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.CritiqueActorRollout],
                config=critique_actor_config,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[critique_actor_resource_pool][str(Role.CritiqueActorRollout)] = critique_actor_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # Native streaming is used when (1) no reward model, or (2) a reward model
        # has an extra resource pool. Intermediate MC deliberately disables it.
        enable_agent_reward_loop = self._should_enable_agent_reward_loop()

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

        if Role.CritiqueActorRollout in self.role_worker_mapping:
            if critique_actor_resource_pool is None:
                raise RuntimeError("separate critique policy resource pool was not initialized")
            self.critique_actor_rollout_wg = all_wg[str(Role.CritiqueActorRollout)]
            self.critique_actor_rollout_wg.init_model()
            self.critique_async_rollout_manager = AgentLoopManager.create(
                config=self.critique_manager_config,
                worker_group=self.critique_actor_rollout_wg,
                rollout_resource_pool=critique_actor_resource_pool,
                reward_loop_worker_handles=None,
            )
            self.critique_checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=self.critique_actor_rollout_wg,
                replicas=self.critique_async_rollout_manager.rollout_replicas,
            )
            self.critique_checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if hasattr(self, "critique_actor_rollout_wg"):
            critique_actor_local_path = os.path.join(local_global_step_folder, str(Role.CritiqueActorRollout))
            critique_actor_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    str(Role.CritiqueActorRollout),
                )
            )
            self.critique_actor_rollout_wg.save_checkpoint(
                critique_actor_local_path,
                critique_actor_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate_expected_resume_step(self, actual_step: int) -> None:
        """Fail closed when a chained job discovers an unexpected checkpoint."""
        expected_step = OmegaConf.select(self.config, "trainer.expected_resume_step", default=None)
        if expected_step is None:
            return
        if isinstance(expected_step, bool):
            raise ValueError("trainer.expected_resume_step must be a non-negative integer or null")
        try:
            expected_step = int(expected_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.expected_resume_step must be a non-negative integer or null") from exc
        if expected_step < 0:
            raise ValueError("trainer.expected_resume_step must be a non-negative integer or null")
        if actual_step != expected_step:
            raise RuntimeError(
                "Resume checkpoint guard failed: "
                f"expected global step {expected_step}, discovered {actual_step}. "
                "Refusing to load a stale, missing, or unexpectedly advanced checkpoint."
            )
        print(f"Resume checkpoint guard passed: expected={expected_step} actual={actual_step}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            self._validate_expected_resume_step(actual_step=0)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self._validate_expected_resume_step(actual_step=0)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        self._validate_expected_resume_step(actual_step=self.global_steps)

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        # Validate the requested dataloader-resume contract before mutating any
        # model state. A segmented production run must never silently restart
        # its data cursor when the caller explicitly requested restoration.
        load_dataloader_state = OmegaConf.select(self.config, "trainer.load_dataloader_state_on_resume", default=True)
        if not isinstance(load_dataloader_state, bool):
            raise ValueError("trainer.load_dataloader_state_on_resume must be true or false")
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if load_dataloader_state and not os.path.exists(dataloader_local_path):
            raise FileNotFoundError(
                "trainer.load_dataloader_state_on_resume=true but the requested "
                f"checkpoint has no dataloader state: {dataloader_local_path}"
            )

        actor_path = os.path.join(global_step_folder, "actor")
        critique_actor_path = os.path.join(global_step_folder, str(Role.CritiqueActorRollout))
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        if hasattr(self, "critique_actor_rollout_wg") and not os.path.isdir(critique_actor_path):
            raise FileNotFoundError(
                f"separate critique policy resume requires its native checkpoint: {critique_actor_path}"
            )
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if hasattr(self, "critique_actor_rollout_wg"):
            self.critique_actor_rollout_wg.load_checkpoint(
                critique_actor_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        if not load_dataloader_state:
            print(
                "Resume dataloader state intentionally reset: "
                f"global_step={self.global_steps} ignored_state={dataloader_local_path}"
            )
        else:
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"Resume dataloader state restored: global_step={self.global_steps} state={dataloader_local_path}")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if hasattr(self, "critique_actor_rollout_wg"):
                self.critique_actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if hasattr(self, "critique_actor_rollout_wg"):
                self.critique_actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(
        self,
        batch: DataProto,
        metrics,
        logging_prefix="global_seqlen",
        keep_minibatch=False,
        worker_group=None,
        role="actor",
    ):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        worker_group = self.actor_rollout_wg if worker_group is None else worker_group
        dp_size = self._get_dp_size(worker_group, role)

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            if role == "critic":
                minibatch_size = self.config.critic.get("ppo_mini_batch_size")
            else:
                minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
                )
            else:
                old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto, *, worker_group=None) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        worker_group = self.actor_rollout_wg if worker_group is None else worker_group
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = worker_group.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = worker_group.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def _completed_training_resume(self) -> bool:
        """Return whether training should exit before any validation or update."""
        if self.config.trainer.get("val_only", False):
            return False
        return self.global_steps >= self.total_training_steps

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.training_start_time = time.time()

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        if self._completed_training_resume():
            pprint(
                "Training target already reached by resumed checkpoint: "
                f"global_step={self.global_steps} total_training_steps={self.total_training_steps}. "
                "Exiting without validation or an optimizer update."
            )
            return
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                stop_after_timeout_checkpoint = False

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                prompt_group_ids = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch.non_tensor_batch["uid"] = prompt_group_ids.copy()
                batch.non_tensor_batch["prompt_group_id"] = prompt_group_ids

                self._maybe_apply_reward_focus_tail_mask(batch)
                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                intermediate_mc_records = None
                if self.intermediate_mc_controller is not None:
                    self.intermediate_mc_controller.prepare_generation_batch(gen_batch_output)
                elif self.branch_revision_controller is not None:
                    self.branch_revision_controller.prepare_original_generation_batch(gen_batch_output)
                elif self.random_continuation_controller is not None:
                    self.random_continuation_controller.prepare_generation_batch(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.intermediate_mc_controller is not None:
                            # The controller uses the same blocking lifecycle for
                            # this first stage and the critic-dependent continuation stage.
                            # It independently attempts sleep/profile cleanup
                            # after partial lifecycle failures.
                            gen_batch_output = self.intermediate_mc_controller._generate_sequences_with_lifecycle(
                                gen_batch_output,
                                profile_rollout=curr_step_profile,
                                restore_rollout=False,
                            )
                            intermediate_mc_records = self.intermediate_mc_controller.extract_generation_records(
                                gen_batch_output
                            )
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            self._drop_overlapping_non_tensor_keys(gen_baseline_output, batch)
                            batch = batch.union(gen_baseline_output)
                            reward_fn = getattr(self, "reward_fn", None)
                            rm_scores = None
                            if reward_fn is not None:
                                reward_baseline_tensor, _ = compute_reward(
                                    batch, reward_fn, actor_wg=self.actor_rollout_wg
                                )
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            else:
                                # compute reward model score on batch
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    rm_scores = self._compute_reward_colocate(batch)
                                    batch = batch.union(rm_scores)

                                # Compute or extract reward for REMAX baseline
                                reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    self._drop_overlapping_non_tensor_keys(gen_batch_output, batch)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    future_reward = None
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        reward_fn = getattr(self, "reward_fn", None)
                        launch_reward_fn_async = OmegaConf.select(
                            self.config,
                            "reward.reward_model.launch_reward_fn_async",
                            default=False,
                        )
                        if reward_fn is not None and launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, reward_fn=reward_fn, actor_wg=self.actor_rollout_wg
                            )
                        elif reward_fn is not None:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, reward_fn, actor_wg=self.actor_rollout_wg
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if self._has_custom_synchronous_actor_update():
                        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"].clone()
                    elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy and not self._has_custom_synchronous_actor_update():
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic and not self._has_custom_synchronous_actor_update():
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if future_reward is not None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            metrics.update(_compute_shortest_success_reward_metrics(reward_extra_infos_dict))
                            metrics.update(_compute_longest_success_penalty_reward_metrics(reward_extra_infos_dict))
                            batch.non_tensor_batch.update(
                                {k: _pack_reward_extra_info(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.intermediate_mc_controller is not None:
                            if intermediate_mc_records is None:
                                raise RuntimeError("intermediate MC generation records were not captured")
                            actor_updated = self.intermediate_mc_controller.run_update(
                                batch,
                                intermediate_mc_records,
                                reward_tensor,
                                metrics,
                                timing_raw,
                                profile_rollout=curr_step_profile,
                            )
                            metrics["intermediate_mc/actor_updated"] = float(actor_updated)
                        elif self.branch_revision_controller is not None:
                            actor_updated = self.branch_revision_controller.run_update(
                                batch,
                                reward_tensor,
                                metrics,
                                timing_raw,
                                profile_rollout=curr_step_profile,
                            )
                            metrics["branch_revision/actor_updated"] = float(actor_updated)
                        elif self.random_continuation_controller is not None:
                            actor_updated = self.random_continuation_controller.run_evaluation(
                                batch,
                                reward_tensor,
                                metrics,
                            )
                            metrics["random_continuation/actor_updated"] = float(actor_updated)
                        else:
                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                )

                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic and not self._has_custom_synchronous_actor_update():
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if (
                        not self._has_custom_synchronous_actor_update()
                        and self.config.trainer.critic_warmup <= self.global_steps
                    ):
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            save_ckpt_duration=self.config.trainer.get("checkpoint_save_duration", 60),
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        timeout_close_to_expiration = should_save_ckpt_timeout(
                            max_steps_duration=self.max_steps_duration,
                            save_ckpt_duration=self.config.trainer.get("checkpoint_save_duration", 60),
                            redundant_time=self.config.trainer.esi_redundant_time,
                            checkpoint_must_save_by=self.config.trainer.get("checkpoint_must_save_by", None),
                            start_time=self.training_start_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        save_due_to_schedule = self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        )
                        if save_due_to_schedule or esi_close_to_expiration or timeout_close_to_expiration:
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            if timeout_close_to_expiration:
                                print("Force saving checkpoint: job timeout approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                            if timeout_close_to_expiration:
                                stop_after_timeout_checkpoint = True

                        # update weights from trainer to rollout
                        if not stop_after_timeout_checkpoint:
                            with marked_timer("update_weights", timing_raw, color="red"):
                                self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self._has_custom_synchronous_actor_update() and self.random_continuation_controller is None:
                        # Keep VeRL's native checkpoint cadence and timeout handling during both
                        # critic-only warmup and joint actor/critic updates.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            save_ckpt_duration=self.config.trainer.get("checkpoint_save_duration", 60),
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        timeout_close_to_expiration = should_save_ckpt_timeout(
                            max_steps_duration=self.max_steps_duration,
                            save_ckpt_duration=self.config.trainer.get("checkpoint_save_duration", 60),
                            redundant_time=self.config.trainer.esi_redundant_time,
                            checkpoint_must_save_by=self.config.trainer.get("checkpoint_must_save_by", None),
                            start_time=self.training_start_time,
                        )
                        save_due_to_schedule = self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        )
                        if save_due_to_schedule or esi_close_to_expiration or timeout_close_to_expiration:
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            if timeout_close_to_expiration:
                                print("Force saving checkpoint: job timeout approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                            if timeout_close_to_expiration:
                                stop_after_timeout_checkpoint = True
                        if not stop_after_timeout_checkpoint:
                            with marked_timer("update_weights", timing_raw, color="red"):
                                # This also wakes the unchanged rollout actor after critic-only warmup.
                                self.checkpoint_manager.update_weights(self.global_steps)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir and not stop_after_timeout_checkpoint:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    (not stop_after_timeout_checkpoint)
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if stop_after_timeout_checkpoint:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    print("Timeout-triggered checkpoint saved, stopping training early.")
                    progress_bar.close()
                    return

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
