# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.rmauc_state = {"vf_ema": None, "rmauc_ema": None}
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1 : -1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Optionally drop samples with zero critic weight (skip-zero-advantage)
        if "critic_loss_weight" in data.batch.keys():
            w = data.batch["critic_loss_weight"]
            try:
                if w.dim() == 1:
                    keep = w > 0
                else:
                    keep = (w > 0).any(dim=-1)
                if keep.dtype != torch.bool:
                    keep = keep.bool()
                if keep.numel() > 0 and keep.any():
                    data = data.select_idxs(keep)
                else:
                    # No samples contribute; keep empty batch and let downstream no-op
                    pass
            except Exception:
                # If anything goes wrong, fall back to keeping all
                pass

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
            response_mask = response_mask.to(values.device)
            values = values * response_mask  # Only action tokens have values
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}
        diff_cfg = data.meta_info.get("critic_diff_penalty", {}) or {}
        diff_coeff = float(diff_cfg.get("coeff", 0.0) or 0.0)

        select_keys = [
            "input_ids",
            "responses",
            "response_mask",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
        ]
        use_rmauc_loss = data.meta_info.get("use_rmauc_loss", False)
        rmauc_cfg = data.meta_info.get("rmauc_cfg", {}) or {}
        rmauc_normalize = bool(rmauc_cfg.get("normalize", False))
        rmauc_beta = float(rmauc_cfg.get("ema_beta", 0.9))
        if use_rmauc_loss:
            select_keys.append("acc")
        if "critic_loss_weight" in data.batch.keys():
            select_keys.append("critic_loss_weight")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    vpreds = self._forward_micro_batch(model_inputs)
                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                        loss_weights=model_inputs.get("critic_loss_weight", None),
                    )
                    rmauc_loss = None
                    rmauc_scale = 1.0
                    if use_rmauc_loss:
                        acc_tensor = model_inputs.get("acc", None)
                        assert acc_tensor is not None, "critic_rmauc enabled but 'acc' missing in batch"
                        acc_tensor = acc_tensor.to(vpreds.device, vpreds.dtype).view(-1)
                        rmauc_loss = core_algos.compute_rmauc_loss(
                            values=vpreds, acc=acc_tensor, response_mask=response_mask
                        )
                        if rmauc_normalize:
                            rmauc_loss_norm, self.rmauc_state, rmauc_scale = core_algos.normalize_rmauc_loss(
                                rmauc_loss=rmauc_loss, vf_loss=vf_loss, state=self.rmauc_state, beta=rmauc_beta
                            )
                        else:
                            rmauc_loss_norm = rmauc_loss
                            rmauc_scale = 1.0
                        base_loss = 2.0 - rmauc_loss_norm
                    else:
                        base_loss = vf_loss

                    diff_loss, num_pairs = None, 0
                    if diff_coeff > 0:
                        diff_loss, num_pairs, _ = core_algos.compute_critic_diff_penalty(
                            vpreds=vpreds, response_mask=response_mask
                        )
                        total_loss = (1.0 - diff_coeff) * base_loss + diff_coeff * diff_loss
                    else:
                        total_loss = base_loss
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = total_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = total_loss * loss_scale_factor

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                        }
                    )
                    if use_rmauc_loss:
                        micro_batch_metrics.update(
                            {
                                "critic/rmauc": rmauc_loss.detach().item(),
                                "critic/rmauc_scale": rmauc_scale,
                            }
                        )
                    elif diff_coeff > 0:
                        micro_batch_metrics.update(
                            {
                                "critic/value_diff_mse": diff_loss.detach().item(),
                                "critic/value_diff_penalty": (diff_coeff * diff_loss * loss_scale_factor)
                                .detach()
                                .item(),
                                "critic/value_diff_pairs": num_pairs,
                            }
                        )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
