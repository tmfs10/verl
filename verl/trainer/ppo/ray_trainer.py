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
import uuid
from collections import defaultdict
from copy import deepcopy
import numpy as _np
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    compute_rmauc,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    compute_group_loss_weights,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.one_logger_integration import OneLoggerInstrumented
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


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
        lam_input = lam
        # Length-adaptive GAE: per-sequence lambda: 1 - 1/(alpha * L)
        if config is not None and hasattr(config, "length_adaptive_gae") and config.length_adaptive_gae["enable"]:
            alpha = float(config.length_adaptive_gae["alpha"])
            assert alpha > 0.0, "algorithm.length_adaptive_gae.alpha must be > 0"
            response_mask = data.batch["response_mask"]
            resp_len = response_mask.sum(dim=-1).to(dtype=torch.float32)
            # ensure there is at least one response token
            assert torch.all(resp_len > 0), "Found sequence with zero valid response length when computing length-adaptive GAE"
            lam_input = 1.0 - 1.0 / (alpha * resp_len)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam_input,
            index=data.non_tensor_batch.get("uid", None),
            config=config,
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

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


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
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
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
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
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
        # Critic-only mode: do rollouts and update only the critic. Actor is not loaded/updated.
        self.critic_only = config.trainer.get("critic_only", False)

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        # Default to None to avoid attribute errors when actor/rollout is not created
        self.actor_rollout_wg = None

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # Enforce incompatibility: probability reweighting disables rmAUC weighting
        prob_rw_enable = self.config.algorithm.get("critic_prob_reweighting", {}).get("enable", False)
        rmauc_enable = self.config.algorithm.get("critic_rmauc", {}).get("enable", False)
        assert not (prob_rw_enable and rmauc_enable), (
            "algorithm.critic_prob_reweighting.enable=True is incompatible with algorithm.critic_rmauc.enable=True."
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _maybe_build_critic_batch_with_solution(self, batch: DataProto) -> DataProto:
        """Optionally append a sample solution to the prompt for critic-only computation.

        - Uses `config.critic.append_solution_to_prompt` to gate behavior.
        - Expects a `solution` field in non-tensor batch (np.ndarray of strings/None).
        - Constructs new input_ids/attention_mask/position_ids with suffix appended to the prompt only.
        - Keeps responses/response_mask unchanged. Returns a new DataProto used only for critic compute.
        """
        try:
            use_suffix = self.config.critic.get("append_solution_to_prompt", False)
        except Exception:
            use_suffix = False
        if not use_suffix:
            return batch

        solution_field_name = self.config.data.get("solution_field_name", "ground_truth_answer")

        # Enforce presence of solution field when requested
        assert solution_field_name in batch.non_tensor_batch, (
            f"append_solution_to_prompt=True requires '{solution_field_name}' in non_tensor_batch"
        )
        solution_arr = batch.non_tensor_batch[solution_field_name]

        # Ensure required fields exist
        required_keys = ["input_ids", "attention_mask", "responses", "response_mask"]
        missing = [k for k in required_keys if k not in batch.batch]
        assert len(missing) == 0, f"Missing required keys for append_solution_to_prompt: {missing}"

        input_ids = batch.batch["input_ids"]  # (B, T)
        attention_mask = batch.batch["attention_mask"]  # (B, T)
        responses = batch.batch["responses"]  # (B, R)
        response_mask = batch.batch["response_mask"]  # (B, R)

        B, T = input_ids.shape
        R = responses.size(1)
        device = input_ids.device

        # Validate solution array shape matches batch size
        try:
            sol_len = len(solution_arr)
        except Exception:
            sol_len = -1
        assert sol_len == B, (
            f"Length of '{solution_field_name}' ({sol_len}) must match batch size ({B}) when appending solution"
        )

        prompt_ids = input_ids[:, :-R]
        prompt_mask = attention_mask[:, :-R]

        # Build per-sample suffix tokens (handle repeated prompts correctly)
        suffix_token_lists: list[list[int]] = []
        for i in range(B):
            sol_i = solution_arr[i]
            # Enforce solution presence and non-empty string per-sample
            assert isinstance(sol_i, str) and len(sol_i) > 0, (
                f"append_solution_to_prompt=True but solution missing or empty at index {i} for field "
                f"'{solution_field_name}'"
            )
            # Build suffix string for this sample
            suffix_str = f"\n\nHere is a sample solution:\n```python\n{sol_i}\n```\n"
            token_ids = self.tokenizer.encode(suffix_str, add_special_tokens=False)
            suffix_token_lists.append(token_ids)

        # Rebuild prompt with suffixes: enforce right-truncation to max prompt length, then left-pad to batch max
        new_prompt_unpadded: list[torch.Tensor] = []
        new_prompt_lens: list[int] = []
        max_prompt_len_cfg = self.config.data.get("max_prompt_length", None)
        for i in range(B):
            pm = prompt_mask[i]
            plen = int(pm.sum().item())
            unpadded = prompt_ids[i, -plen:]
            suffix_tokens = suffix_token_lists[i]
            if len(suffix_tokens) > 0:
                unpadded = torch.cat([unpadded, torch.tensor(suffix_tokens, dtype=unpadded.dtype)], dim=0)
            # If prompt+suffix exceeds limit, truncate from the right (keep left-most tokens)
            if max_prompt_len_cfg is not None and unpadded.size(0) > max_prompt_len_cfg:
                unpadded = unpadded[: max_prompt_len_cfg]
            new_prompt_unpadded.append(unpadded)
            new_prompt_lens.append(int(unpadded.size(0)))

        pad_token_id = int(self.tokenizer.pad_token_id)
        new_prompt_ids = []
        new_prompt_masks = []
        for i in range(B):
            unpadded = new_prompt_unpadded[i]
            np_len = new_prompt_lens[i]
            pad_len = max_prompt_len_cfg - np_len
            if pad_len > 0:
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=unpadded.dtype)
                padded = torch.cat([pad_ids, unpadded], dim=0)
                pad_mask = torch.zeros((pad_len,), dtype=prompt_mask.dtype)
                ones = torch.ones((np_len,), dtype=prompt_mask.dtype)
                pmask = torch.cat([pad_mask, ones], dim=0)
            else:
                padded = unpadded
                pmask = torch.ones((np_len,), dtype=prompt_mask.dtype)
            new_prompt_ids.append(padded)
            new_prompt_masks.append(pmask)

        new_prompt_ids = torch.stack(new_prompt_ids, dim=0).to(device)
        new_prompt_mask = torch.stack(new_prompt_masks, dim=0).to(device)

        # Build new full sequence
        new_input_ids = torch.cat([new_prompt_ids, responses], dim=1)
        new_attention_mask = torch.cat([new_prompt_mask, response_mask.to(new_prompt_mask.dtype)], dim=1)
        new_position_ids = compute_position_id_with_mask(new_attention_mask)

        # Construct a critic-only batch (do not mutate original batch)
        from copy import deepcopy
        critic_batch = deepcopy(batch)
        critic_batch.batch["input_ids"] = new_input_ids
        critic_batch.batch["attention_mask"] = new_attention_mask
        critic_batch.batch["position_ids"] = new_position_ids

        return critic_batch

    def _build_gen_output_from_dataset_responses(self, batch: DataProto) -> DataProto:
        """Build a generation-like DataProto using dataset-provided responses.

        Expects batch.non_tensor_batch["response_strs"] to provide response text(s) per sample.
        - If a sample provides a list[str], build one entry per string (expands the batch).
        - If a sample provides a str, it is used directly.
        Returns a DataProto with fields similar to rollout output: input_ids, attention_mask,
        position_ids, and responses. The response_mask can be derived downstream.
        """
        use_ds_resp = self.config.data.get("use_dataset_responses", False)
        response_strs_field = self.config.data.get("response_strs_field", "response_strs")
        if not use_ds_resp:
            return batch

        responses_text = batch.non_tensor_batch[response_strs_field]
        # Build per-sample repeat counts and flatten responses
        B = len(responses_text)
        repeat_counts = []
        flat_responses: list[str] = []
        for i in range(B):
            item = responses_text[i]
            if item is None:
                raise ValueError("Found None in dataset 'response_strs' while use_dataset_responses=True")
            if isinstance(item, str):
                repeat_counts.append(1)
                flat_responses.append(item)
            elif isinstance(item, (list, tuple, np.ndarray)):
                if len(item) == 0:
                    raise ValueError("Encountered empty list for 'response_strs' entry while use_dataset_responses=True")
                repeat_counts.append(len(item))
                for s in item:
                    if s is None or not isinstance(s, str):
                        raise ValueError("All elements of 'response_strs' must be str when use_dataset_responses=True")
                    flat_responses.append(s)
            else:
                raise ValueError(
                    f"Each 'response_strs' entry must be a str or list[str] when use_dataset_responses=True, got {type(item)}"
                )

        # Repeat prompts per-sample to align with flattened responses
        expanded_prompts = batch.sample_level_repeat(repeat_counts)

        # Tokenize flattened responses and pad/truncate to configured response length
        resp_ids = []
        pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        resp_max_len = int(self.config.actor_rollout_ref.rollout.response_length)
        for text in flat_responses:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) == 0:
                raise ValueError("Encountered empty tokenized response in dataset while use_dataset_responses=True")
            t = torch.tensor(tokens[:resp_max_len], dtype=torch.long)
            if t.size(0) < resp_max_len:
                pad = torch.full((resp_max_len - t.size(0),), pad_id, dtype=torch.long)
                t = torch.cat([t, pad], dim=0)
            resp_ids.append(t)

        device = expanded_prompts.batch["input_ids"].device
        dtype = expanded_prompts.batch["attention_mask"].dtype
        prompts = expanded_prompts.batch["input_ids"]
        prompt_attention = expanded_prompts.batch["attention_mask"]
        prompt_position = expanded_prompts.batch.get(
            "position_ids", compute_position_id_with_mask(prompt_attention)
        )

        responses = torch.stack(resp_ids, dim=0).to(device)
        seq = torch.cat([prompts, responses], dim=-1)

        resp_len = responses.size(1)
        delta_position_id = torch.arange(1, resp_len + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(seq.size(0), -1)
        if prompt_position.dim() == 3:  # mrope (e.g., qwen2vl)
            delta_position_id = delta_position_id.view(seq.size(0), 1, -1).expand(seq.size(0), 3, -1)
        response_position_ids = prompt_position[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=self.tokenizer.eos_token_id, dtype=dtype
        )
        attention_mask = torch.cat([prompt_attention, response_attention_mask], dim=-1)

        tensors = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_attention_mask,
        }
        meta = {"timing": {}, "repeat_counts": np.asarray(repeat_counts, dtype=np.int32)}
        return DataProto.from_dict(tensors=tensors, meta_info=meta)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
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

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
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

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
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
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        # Preserve solution field (for critic prompt suffixing) in the original batch
        solution_field_name = self.config.data.get("solution_field_name", "ground_truth_answer")
        if solution_field_name in batch.non_tensor_batch:
            reward_model_keys.add(solution_field_name)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        # Accumulators for validation critic diagnostics
        val_token_mse_num_total = 0.0
        val_token_mse_den_total = 0.0
        val_final_mse_num_total = 0.0
        val_final_mse_count = 0
        val_rmauc_sum_total = 0.0
        val_rmauc_count_total = 0

        # Optional debug flag to print reward extras lengths
        debug_reward_extras = self.config.trainer.get("debug_reward_extras", False)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch only when generating rollouts
            if not self.config.data.get("use_dataset_responses", False):
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs (and repeat per dataset responses if used)
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]

            if not self.config.data.get("use_dataset_responses", False):
                sample_inputs.extend(input_texts)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])
                sample_gts.extend(ground_truths)
            else:
                response_strs_field = self.config.data.get("response_strs_field", "response_strs")
                resp_strs = test_batch.non_tensor_batch[response_strs_field]
                repeat_counts = []
                for item in resp_strs:
                    if isinstance(item, str):
                        repeat_counts.append(1)
                    elif isinstance(item, (list, tuple, np.ndarray)):
                        if len(item) == 0:
                            raise ValueError(
                                "Encountered empty list for 'response_strs' entry while use_dataset_responses=True"
                            )
                        repeat_counts.append(len(item))
                    else:
                        raise ValueError(f"'response_strs' must be str or list[str], got {type(item)}")

                # Repeat inputs/uids/gts to align with dataset responses
                for i, r in enumerate(repeat_counts):
                    sample_inputs.extend([input_texts[i]] * r)
                    sample_uids.extend([test_batch.non_tensor_batch["uid"][i]] * r)
                    sample_gts.extend([ground_truths[i]] * r)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # If using dataset responses, build outputs from dataset; otherwise generate
            if self.config.data.get("use_dataset_responses", False):
                test_output_gen_batch = self._build_gen_output_from_dataset_responses(test_gen_batch)
            else:
                size_divisor = (
                    getattr(self.actor_rollout_wg, "world_size", 1)
                    if not self.async_rollout_mode
                    else self.config.actor_rollout_ref.rollout.agent.num_workers
                )
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                        test_gen_batch_padded
                    )
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Align test_batch size with outputs when using dataset responses
            if self.config.data.get("use_dataset_responses", False):
                response_strs_field = self.config.data.get("response_strs_field", "response_strs")
                resp_strs = test_gen_batch.non_tensor_batch[response_strs_field]
                if resp_strs is None:
                    raise ValueError(
                        "use_dataset_responses=True but 'response_strs' missing in test_gen_batch.non_tensor_batch"
                    )
                repeat_counts = []
                for item in resp_strs:
                    if isinstance(item, str):
                        repeat_counts.append(1)
                    elif isinstance(item, (list, tuple, np.ndarray)):
                        if len(item) == 0:
                            raise ValueError(
                                "Encountered empty list for 'response_strs' entry while use_dataset_responses=True"
                            )
                        repeat_counts.append(len(item))
                    else:
                        raise ValueError(f"'response_strs' must be str or list[str], got {type(item)}")
                test_batch = test_batch.sample_level_repeat(repeat_counts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            # Route through compute_reward to pass optional kwargs like actor_wg
            # Allow reward to omit actor_wg if actor/rollout is absent
            reward_tensor, reward_extras = compute_reward(
                test_batch, self.val_reward_fn, actor_wg=self.actor_rollout_wg
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if debug_reward_extras:
                print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if reward_extras:
                for key, lst in reward_extras.items():
                    if key == "reward":
                        continue
                    # Ensure list type
                    if isinstance(lst, torch.Tensor):
                        lst = lst.detach().cpu().tolist()
                    reward_extra_infos_dict[key].extend(lst)
                    if debug_reward_extras:
                        print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # Prepare batch for critic diagnostics: compute values and compare to accuracy
            test_batch.batch["token_level_scores"] = reward_tensor
            test_batch.batch["token_level_rewards"] = reward_tensor
            if "response_mask" not in test_batch.batch:
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)
            if self.use_critic:
                # Skip computing critic values in validation if any element lacks solution field
                sol_field = self.config.data.get("solution_field_name", "ground_truth_answer")
                has_sol_field = sol_field in test_batch.non_tensor_batch
                missing_any = False
                if has_sol_field:
                    sol_arr = test_batch.non_tensor_batch[sol_field]
                    try:
                        import numpy as _np

                        sol_np = _np.asarray(sol_arr, dtype=object)
                        # consider None as missing
                        missing_any = _np.any(sol_np == None)  # noqa: E711
                    except Exception:
                        # Fallback: iterate
                        missing_any = any(x is None for x in sol_arr)

                if has_sol_field and not missing_any:
                    critic_view = self._maybe_build_critic_batch_with_solution(test_batch)
                    size_divisor = getattr(self.critic_wg, "world_size", 1)
                    critic_view_padded, pad_size = pad_dataproto_to_divisor(critic_view, size_divisor)
                    values_padded = self.critic_wg.compute_values(critic_view_padded)
                    values = unpad_dataproto(values_padded, pad_size=pad_size)
                    test_batch = test_batch.union(values)
                    # Token-wise MSE vs. accuracy (mean over response tokens)
                    if reward_extras and ("acc" in reward_extras):
                        acc_vec = reward_extras["acc"]
                        acc_tensor = torch.tensor(
                            acc_vec, dtype=test_batch.batch["values"].dtype, device=test_batch.batch["values"].device
                        )
                        resp_mask_f = test_batch.batch["response_mask"].to(dtype=test_batch.batch["values"].dtype)
                        acc_expanded = acc_tensor.unsqueeze(1).expand_as(test_batch.batch["values"]) 
                        sq_err = ((test_batch.batch["values"] - acc_expanded) ** 2) * resp_mask_f
                        val_token_mse_num_total += float(torch.sum(sq_err).detach().item())
                        val_token_mse_den_total += float(torch.sum(resp_mask_f).detach().item())
                        # Final-token MSE vs. acc
                        resp_len = torch.sum(test_batch.batch["response_mask"], dim=-1).long()
                        valid_mask = resp_len > 0
                        if torch.any(valid_mask):
                            last_idx = (resp_len - 1).clamp(min=0)
                            arange = torch.arange(test_batch.batch["values"].size(0), device=test_batch.batch["values"].device)
                            final_v = test_batch.batch["values"][arange, last_idx]
                            diff2 = (final_v[valid_mask] - acc_tensor[valid_mask]) ** 2
                            val_final_mse_num_total += float(torch.sum(diff2).detach().item())
                            val_final_mse_count += int(valid_mask.sum().item())
                        # RMAUC per batch (weighted by valid samples)
                        rmauc_batch = compute_rmauc(
                            values=test_batch.batch["values"], acc_vec=acc_tensor, response_mask=test_batch.batch["response_mask"]
                        )
                        val_rmauc_sum_total += float(rmauc_batch) * int(valid_mask.sum().item())
                        val_rmauc_count_total += int(valid_mask.sum().item())

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # When using dataset-provided generations, skip dumping validation generations to disk
        if val_data_dir and not self.config.data.get("use_dataset_responses", False):
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Add critic diagnostics for validation
        if val_token_mse_den_total > 0:
            metric_dict["val-aux/critic/mse/token/mean"] = val_token_mse_num_total / val_token_mse_den_total
        if val_final_mse_count > 0:
            metric_dict["val-aux/critic/mse/final_vs_acc"] = val_final_mse_num_total / val_final_mse_count
        if val_rmauc_count_total > 0:
            metric_dict["val-aux/critic/rmauc"] = val_rmauc_sum_total / val_rmauc_count_total

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and/or rollout as needed
        if self.hybrid_engine:
            use_ds_resp = self.config.data.get("use_dataset_responses", False)
            prob_rw_enable = self.config.algorithm.get("critic_prob_reweighting", {}).get("enable", False)
            # If critic-only + dataset responses, we normally skip actor/rollout.
            # Exception: when probability reweighting is enabled, we need the actor to compute log-probs.
            if not (self.critic_only and use_ds_resp and not prob_rw_enable):
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                # Role selection:
                # - critic_only + dataset responses + prob_rw_enable: actor only
                # - critic_only + dataset responses + prob_rw_disable: no actor/rollout (handled above)
                # - critic_only + no dataset responses + prob_rw_enable: actor + rollout (need both)
                # - critic_only + no dataset responses + prob_rw_disable: rollout only
                # - non-critic_only + dataset responses: actor only
                # - non-critic_only + no dataset responses: actor + rollout
                if self.critic_only:
                    if use_ds_resp:
                        role_name = "actor"  # only to compute log-probs when prob_rw_enable, otherwise skipped
                    else:
                        role_name = "actor_rollout" if prob_rw_enable else "rollout"
                else:
                    role_name = "actor" if use_ds_resp else "actor_rollout"

                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=self.config.actor_rollout_ref,
                    role=role_name,
                    critic_only=self.critic_only,
                )
                self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed (skip in critic-only mode)
        if self.use_reference_policy and not self.critic_only:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        ray_wait_register_center_timeout = self.config.trainer.get("ray_wait_register_center_timeout", None)
        if ray_wait_register_center_timeout is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = ray_wait_register_center_timeout
        profile_steps = self.config.global_profiler.get("steps", None)
        if profile_steps is not None:
            wg_kwargs["profile_steps"] = profile_steps
            # Only require nsight worker options when tool is nsys
            if self.config.global_profiler.get("tool", None) == "nsys":
                worker_nsight_options = (
                    self.config.global_profiler.get("global_tool_config", {})
                    .get("nsys", {})
                    .get("worker_nsight_options", None)
                )
                assert worker_nsight_options is not None, (
                    "worker_nsight_options must be set when using nsys with profile_steps"
                )
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(worker_nsight_options)
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor and not self.critic_only:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        if "actor_rollout" in all_wg:
            self.actor_rollout_wg = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.actor_rollout_wg is not None and self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

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

        if not self.critic_only:
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
            )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
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
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
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

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        if not self.critic_only:
            self.actor_rollout_wg.load_checkpoint(
                actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            if not self.critic_only:
                self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy and not self.critic_only:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            if not self.critic_only:
                self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy and not self.critic_only:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = getattr(self.actor_rollout_wg, "world_size", None)
        if world_size is None:
            # Fallback to critic world size or 1 if actor/rollout is absent
            world_size = getattr(self, "critic_wg", None)
            world_size = world_size.world_size if world_size is not None else 1
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
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

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                if not self.config.data.get("use_dataset_responses", False):
                    gen_batch = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.config.data.get("use_dataset_responses", False):
                            gen_batch_output = self._build_gen_output_from_dataset_responses(gen_batch)
                        else:
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                        if self.config.data.get("use_dataset_responses", False):
                            raise ValueError(
                                "use_dataset_responses=True not compatible with REMAX baseline generation"
                            )
                        else:
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                    gen_baseline_batch
                                )
                            batch = batch.union(gen_baseline_output)
                            # Compute baseline reward with optional kwargs (e.g., actor_wg)
                            reward_baseline_tensor, _ = compute_reward(
                                batch, self.reward_fn, actor_wg=self.actor_rollout_wg
                            )
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    if not self.config.data.get("use_dataset_responses", False):
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        batch = batch.union(gen_batch_output)
                    else:
                        # Expand batch per-sample according to dataset-provided response counts
                        response_strs_field = self.config.data.get("response_strs_field", "response_strs")
                        resp_strs = gen_batch.non_tensor_batch[response_strs_field]
                        repeat_counts = []
                        for item in resp_strs:
                            if isinstance(item, str):
                                repeat_counts.append(1)
                            elif isinstance(item, (list, tuple, np.ndarray)):
                                if len(item) == 0:
                                    raise ValueError(
                                        "Encountered empty list for 'response_strs' entry while use_dataset_responses=True"
                                    )
                                repeat_counts.append(len(item))
                            else:
                                raise ValueError(
                                    f"'response_strs' must be str or list[str], got {type(item)}"
                                )
                        batch = batch.sample_level_repeat(repeat_counts)
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, reward_fn=self.reward_fn, actor_wg=self.actor_rollout_wg
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn, actor_wg=self.actor_rollout_wg
                            )

                    # recompute old_log_probs
                    need_prob_rw = self.config.algorithm.get("critic_prob_reweighting", {}).get("enable", False)
                    if (not self.critic_only) or need_prob_rw:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            if not self.critic_only:
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy and not self.critic_only:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values (optionally with solution-suffixed prompts only for critic)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            critic_view = self._maybe_build_critic_batch_with_solution(batch)
                            values = self.critic_wg.compute_values(critic_view)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward and not self.critic_only:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute returns (and advantages if needed by actor). For critic-only, discard advantages.
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        # Convex-average GAE with final reward using cumulative mean negative log-prob (optional)
                        convex_cfg = self.config.algorithm.get("convex_average_with_final_reward", {})
                        if (
                            convex_cfg.get("enable", False)
                            and not self.critic_only
                            and self.config.algorithm.adv_estimator == AdvantageEstimator.GAE
                            and ("advantages" in batch.batch)
                            and ("old_log_probs" in batch.batch)
                        ):
                            # p_gt: cumulative mean negative log-prob per token
                            old_lp = -batch.batch["old_log_probs"].to(torch.float32)
                            resp_mask = batch.batch["response_mask"].to(dtype=old_lp.dtype)
                            cum_sum = torch.cumsum(old_lp * resp_mask, dim=-1)
                            cum_cnt = torch.cumsum(resp_mask, dim=-1)
                            eps = 1e-6
                            p_gt = torch.where(cum_cnt > 0, cum_sum / cum_cnt.clamp_min(eps), torch.zeros_like(old_lp))

                            # M: mean over all such cumulative means in the batch (valid tokens only)
                            M = masked_mean(p_gt, mask=resp_mask)

                            # convex_coeff_gt = min(0.5 * M / p_gt, 1)
                            convex_coeff = (0.5 * M) / p_gt.clamp_min(eps)
                            convex_coeff = torch.minimum(convex_coeff, torch.ones_like(convex_coeff))

                            # final reward for generation g: sum of token-level rewards over response tokens (broadcast to tokens)
                            tok_rewards = batch.batch["token_level_rewards"].to(dtype=old_lp.dtype)
                            final_reward_g = masked_sum(tok_rewards, mask=resp_mask, axis=-1).unsqueeze(-1)

                            gae = batch.batch["advantages"].to(dtype=old_lp.dtype)
                            mixed_adv = convex_coeff * gae + (1.0 - convex_coeff) * final_reward_g
                            # Only replace within response mask to avoid touching non-response tokens
                            batch.batch["advantages"] = torch.where(resp_mask > 0, mixed_adv, gae)

                        # Critic-prediction-based weighting between GAE and GRPO advantages (optional)
                        cpw_cfg = self.config.algorithm.get("critic_pred_weighting", {})
                        if (
                            self.use_critic
                            and cpw_cfg.get("enable", False)
                            and not self.critic_only
                            and ("values" in batch.batch)
                        ):
                            # Prepare masks and types
                            values = batch.batch["values"].to(torch.float32)
                            resp_mask = batch.batch["response_mask"].to(dtype=values.dtype)
                            eps = 1e-6

                            if cpw_cfg.get("cummean", True):
                                # Compute cumulative mean of critic predictions per token: V
                                cum_sum_v = torch.cumsum(values * resp_mask, dim=-1)
                                cum_cnt_v = torch.cumsum(resp_mask, dim=-1)
                                V = torch.where(
                                    cum_cnt_v > 0, cum_sum_v / cum_cnt_v.clamp_min(eps), torch.zeros_like(values)
                                )
                            else:
                                V = values.mean(dim=-1, keepdim=True)
                                V = V.expand_as(values)

                            # Compute GRPO advantages separately
                            grpo_adv, _ = core_algos.compute_grpo_outcome_advantage(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                response_mask=batch.batch["response_mask"],
                                index=batch.non_tensor_batch["uid"],
                                norm_adv_by_std_in_grpo=self.config.algorithm.get(
                                    "norm_adv_by_std_in_grpo", True
                                ),
                            )

                            # Compute GAE advantages separately to avoid interference from other transforms
                            # Handle length-adaptive GAE if enabled
                            lam_input = self.config.algorithm.lam
                            try:
                                if (
                                    hasattr(self.config.algorithm, "length_adaptive_gae")
                                    and self.config.algorithm.length_adaptive_gae.get("enable", False)
                                ):
                                    alpha = float(self.config.algorithm.length_adaptive_gae.get("alpha", 1.0))
                                    resp_len = resp_mask.sum(dim=-1).to(dtype=torch.float32)
                                    lam_input = 1.0 - 1.0 / (alpha * resp_len)
                            except Exception:
                                lam_input = self.config.algorithm.lam

                            gae_adv, _ = core_algos.compute_gae_advantage_return(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                values=values,
                                response_mask=batch.batch["response_mask"],
                                gamma=self.config.algorithm.gamma,
                                lam=lam_input,
                                index=batch.non_tensor_batch.get("uid", None),
                                config=self.config.algorithm,
                            )

                            # Fetch A (sequence-level accuracy) as a tensor
                            A_tensor = None
                            if "acc" in batch.non_tensor_batch:
                                try:
                                    import numpy as _np

                                    acc_np = _np.asarray(batch.non_tensor_batch["acc"])  # shape (B,)
                                    A_tensor = torch.tensor(acc_np, dtype=values.dtype, device=values.device)
                                except Exception:
                                    pass
                            if A_tensor is None and "acc" in batch.batch:
                                try:
                                    A_tensor = batch.batch["acc"].to(dtype=values.dtype, device=values.device).view(-1)
                                except Exception:
                                    A_tensor = None

                            if A_tensor is not None:
                                A_expand = A_tensor.unsqueeze(-1).expand_as(V)
                                V = torch.clip(V, min=0.0, max=1.0)
                                w = torch.abs(V - A_expand)
                                w = w * resp_mask  # ensure non-response tokens stay zero
                                final_adv = (1.0 - w) * gae_adv.to(values.dtype) + w * grpo_adv.to(values.dtype)
                                # Keep only response tokens mixed; preserve others
                                batch.batch["advantages"] = torch.where(resp_mask > 0, final_adv, gae_adv)
                            else:
                                # If accuracy is unavailable, skip mixing silently
                                pass
                        if self.critic_only and "advantages" in batch.batch:
                            # reduce overhead/metrics footprint in critic-only mode
                            batch.batch.pop("advantages", None)

                    # update critic (use the same critic-only view)
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_view = self._maybe_build_critic_batch_with_solution(batch)
                            # ensure values/returns exist in critic_view
                            for key in ("values", "returns"):
                                if key in batch.batch and key not in critic_view.batch:
                                    critic_view.batch[key] = batch.batch[key]

                            # If using GAE for advantage estimation, allow a different lambda for critic returns
                            try:
                                adv_estimator = self.config.algorithm.adv_estimator
                            except Exception:
                                adv_estimator = None

                            if adv_estimator == AdvantageEstimator.GAE:
                                critic_lam = self.config.algorithm.get("critic_lam", self.config.algorithm.lam)
                                # Only recompute if critic lambda differs or if returns missing
                                if ("returns" not in critic_view.batch) or (critic_lam != self.config.algorithm.lam):
                                    returns_values = critic_view.batch.get("values", batch.batch.get("values", None))
                                    if returns_values is None:
                                        raise ValueError("Values required to compute critic returns with GAE are missing.")
                                    critic_returns_adv, critic_returns = core_algos.compute_gae_advantage_return(
                                        token_level_rewards=batch.batch["token_level_rewards"],
                                        values=returns_values,
                                        response_mask=critic_view.batch["response_mask"],
                                        gamma=self.config.algorithm.gamma,
                                        lam=critic_lam,
                                    )
                                    # Overwrite returns for critic update only
                                    critic_view.batch["returns"] = critic_returns

                            # Compute critic logprob reweighting first (to be applied after other weightings)
                            prob_cfg = self.config.algorithm.get("critic_prob_reweighting", {})
                            if prob_cfg.get("enable", False):
                                assert "old_log_probs" in batch.batch, (
                                    "critic_prob_reweighting requires actor log-probs; ensure actor is initialized when enabled"
                                )
                                old_lp = -batch.batch["old_log_probs"].to(torch.float32)
                                resp_mask = critic_view.batch["response_mask"].to(old_lp.dtype)
                                T = resp_mask.sum(dim=-1, keepdim=True)
                                # Optional: cumulative mean across tokens before weighting
                                if prob_cfg.get("cummean", False):
                                    cum_sum = torch.cumsum(old_lp * resp_mask, dim=-1)
                                    cum_cnt = torch.cumsum(resp_mask, dim=-1)
                                    old_lp_used = torch.where(
                                        cum_cnt > 0, cum_sum / cum_cnt.clamp_min(1e-6), torch.zeros_like(old_lp)
                                    )
                                else:
                                    old_lp_used = old_lp
                                S = (old_lp_used * resp_mask).sum(dim=-1, keepdim=True) + T
                                eps = 1e-6
                                S = torch.where(torch.abs(S) < eps, torch.full_like(S, eps), S)
                                prob_w = ((old_lp_used + 1.0) / S) * T
                                prob_w = prob_w * resp_mask
                                critic_view.batch["critic_prob_weight"] = prob_w

                            # Optional: in-group normalization for critic loss between acc==0 and acc==1
                            try:
                                apply_balancing = self.config.algorithm.get(
                                    "critic_in_group_normalization", {}
                                ).get("enable", False)
                                apply_skip_zero = self.config.algorithm.get(
                                    "critic_skip_zero_advantage", False
                                )
                                token_mean = self.config.algorithm.get("critic_group_normalization", {}).get("token_mean", False)
                                if (apply_balancing or apply_skip_zero):
                                    assert "uid" in critic_view.non_tensor_batch, "uid field required for critic in group normalization"
                                    acc_arr = critic_view.non_tensor_batch["acc"]
                                    uids = critic_view.non_tensor_batch["uid"]
                                    w_tensor = compute_group_loss_weights(
                                        uids=uids,
                                        acc_arr=acc_arr,
                                        response_mask=critic_view.batch["response_mask"],
                                        token_mean=bool(token_mean),
                                        skip_zero=bool(apply_skip_zero),
                                    )
                                    critic_view.batch["critic_loss_weight"] = w_tensor
                            except Exception as _e:
                                print(f"critic_in_group_normalization skipped due to error: {_e}")

                            # If prob reweighting is enabled, multiply it with existing critic_loss_weight now
                            if prob_cfg.get("enable", False):
                                resp_mask = critic_view.batch["response_mask"].to(torch.float32)
                                T = resp_mask.sum(dim=-1, keepdim=True)
                                prob_w = critic_view.batch.pop("critic_prob_weight")
                                if "critic_loss_weight" in critic_view.batch:
                                    w_exist = critic_view.batch["critic_loss_weight"]
                                    if w_exist.dim() == 1:
                                        w_exist = w_exist.unsqueeze(-1).expand_as(prob_w)
                                    else:
                                        assert (
                                            w_exist.shape == prob_w.shape
                                        ), f"critic_loss_weight shape {w_exist.shape} must match prob weighting {prob_w.shape}"
                                    final_w = prob_w * w_exist.to(prob_w.dtype)
                                else:
                                    final_w = prob_w
                                # Renormalize so per-seq sum of weights equals T
                                eps = 1e-6
                                denom = final_w.sum(dim=-1, keepdim=True).clamp_min(eps)
                                final_w = final_w * (T / denom)
                                critic_view.batch["critic_loss_weight"] = final_w

                            critic_output = self.critic_wg.update_critic(critic_view)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup; do not update actor in critic-only mode
                    if (not self.critic_only) and self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            # Optional: in-group normalization for actor loss between acc==0 and acc==1
                            actor_cfg = self.config.algorithm.get("actor_group_normalization", {})
                            if actor_cfg.get("enable", False):
                                assert "uid" in batch.non_tensor_batch, "uid field required for actor group normalization"
                                assert "acc" in batch.non_tensor_batch, "acc field required for actor group normalization"
                                w_tensor = compute_group_loss_weights(
                                    uids=batch.non_tensor_batch["uid"],
                                    acc_arr=batch.non_tensor_batch["acc"],
                                    response_mask=batch.batch["response_mask"],
                                    token_mean=bool(actor_cfg.get("token_mean", False)),
                                    skip_zero=bool(actor_cfg.get("skip_zero", False)),
                                )
                                batch.batch["actor_loss_weight"] = w_tensor

                            # Optional: SFT CE loss only for correct generations (acc==1)
                            sft_cfg = self.config.algorithm.get("sft_objective", {})
                            if sft_cfg.get("enable", False):
                                acc_arr = None
                                if "acc" in batch.non_tensor_batch:
                                    acc_arr = batch.non_tensor_batch["acc"]
                                elif "acc" in batch.batch:
                                    try:
                                        acc_arr = batch.batch["acc"].detach().cpu().numpy()
                                    except Exception:
                                        acc_arr = None
                                if acc_arr is not None:
                                    import numpy as _np

                                    acc_np = _np.asarray(acc_arr)
                                    sft_sample_mask = torch.tensor(acc_np == 1, dtype=torch.bool)
                                    batch.batch["sft_sample_mask"] = sft_sample_mask

                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # When using dataset-provided generations, skip dumping training generations to disk
                    if rollout_data_dir and not self.config.data.get("use_dataset_responses", False):
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

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
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
