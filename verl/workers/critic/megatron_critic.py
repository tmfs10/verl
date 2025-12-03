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

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor, masked_mean
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronPPOCritic(BasePPOCritic):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        critic_module: nn.ModuleList,
        critic_optimizer: DistributedOptimizer,
        critic_optimizer_config: OptimizerConfig,
    ):
        super().__init__(config=config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config  # huggingface config
        self.tf_config = tf_config  # mcore transformer config

        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.critic_optimizer_config = critic_optimizer_config
        self.rmauc_state = {"vf_ema": None, "rmauc_ema": None}

        # we create a separate nametuple for optimizer step so that global args won't affect it.
        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.shuffle:
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        self.config = config

    @GPUMemoryLogger("megatron critic", logger=logger)
    def compute_values(self, data: DataProto) -> DataProto:
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        assert micro_batch_size is not None, "micro batch size is needed for forward compute"
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        response_length = responses.size(1)
        with torch.no_grad():
            output = self.forward_backward_batch(
                data=data,
                forward_only=True,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=None,
            )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                values = [o["vpreds"] for o in output["output"]]  # (bs, seq_size, vocal_size)
                values = torch.cat(values, dim=0).to(torch.float32)
                if use_dynamic_bsz:
                    indices = output["indices"]
                    indices = list(itertools.chain.from_iterable(indices))
                    assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
                    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                    values = values[revert_indices]
            else:
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            # each tp ranks should contain the same value
            values = values[
                :, -response_length - 1 : -1
            ]  # Values are predicted at the ends of prefixes, e.g., the last prompt token
            response_mask = attention_mask[:, -response_length:]
            values = values * response_mask  # Only action tokens have values
            values = values.contiguous()

            # sync among pp ranks
            values = values.to(get_device_id())
            torch.distributed.broadcast(
                tensor=values,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
            values = values.to("cpu")

        # add empty cache after each compute
        get_torch_device().empty_cache()

        return values

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = [
            "input_ids",
            "responses",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
        ]
        if data.meta_info.get("use_rmauc_loss", False):
            select_keys.append("acc")
        if "critic_loss_weight" in data.batch.keys():
            select_keys.append("critic_loss_weight")
        data = data.select(batch_keys=select_keys)
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
                    # No samples contribute; leave empty so iterator yields nothing
                    pass
            except Exception:
                # If anything goes wrong, fall back to keeping all
                pass
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        # broadcast from last pp rank to all other pp ranks
        data.to(get_device_id())
        mini_batch = data
        mini_batch.batch = mini_batch.batch.contiguous()
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)

        indices = None
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
            micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        n_micro_batch = len(micro_batches)
        diff_cfg = data.meta_info.get("critic_diff_penalty", {}) or {}
        diff_coeff = float(diff_cfg.get("coeff", 0.0) or 0.0)
        use_rmauc_loss = data.meta_info.get("use_rmauc_loss", False)
        rmauc_cfg = data.meta_info.get("rmauc_cfg", {}) or {}
        rmauc_normalize = bool(rmauc_cfg.get("normalize", False))
        rmauc_beta = float(rmauc_cfg.get("ema_beta", 0.9))

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            nonlocal use_dynamic_bsz

            if forward_only:
                return torch.tensor(1.0, device=output.device), {"vpreds": output}

            responses = data["responses"]
            attention_mask = data["attention_mask"]
            values = data["values"]
            returns = data["returns"]
            response_length = responses.size(1)
            acc_tensor = data.get("acc", None)

            response_mask = attention_mask[:, -response_length:]

            # Optional: apply per-sample/group weighting by scaling the mask
            if "critic_loss_weight" in data:
                w = data["critic_loss_weight"]
                if w.dim() == 1:
                    w = w.unsqueeze(-1).expand_as(response_mask)
                else:
                    assert w.shape == response_mask.shape, (
                        f"critic_loss_weight shape {w.shape} must match response_mask {response_mask.shape}"
                    )
                response_mask = response_mask * w.to(response_mask.dtype)

            cliprange_value = self.config.cliprange_value

            vpreds = output  # (bs, sequence_length)
            vpreds = vpreds[:, -response_length - 1 : -1]

            vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=cliprange_value,
                loss_agg_mode=self.config.loss_agg_mode,
            )
            rmauc_loss = None
            rmauc_scale = 1.0
            if use_rmauc_loss:
                assert acc_tensor is not None, "critic_rmauc enabled but 'acc' missing in batch"
                acc_tensor = acc_tensor.to(vpreds.device, vpreds.dtype).view(-1)
                rmauc_loss = core_algos.compute_rmauc_loss(values=vpreds, acc=acc_tensor, response_mask=response_mask)
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

            stats = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
            }
            if use_rmauc_loss:
                stats.update({"critic/rmauc": rmauc_loss.detach().item(), "critic/rmauc_scale": rmauc_scale})
            elif diff_coeff > 0:
                stats.update(
                    {
                        "critic/value_diff_mse": diff_loss.detach().item(),
                        "critic/value_diff_penalty": (diff_coeff * diff_loss).detach().item(),
                        "critic/value_diff_pairs": num_pairs,
                    }
                )

            return total_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                value_model=True,
            )

            return output, partial(loss_func, data=batch, meta_info={})

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.critic_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger("megatron critic", logger=logger)
    def update_critic(self, dataloader: Iterable[DataProto]):
        metrics = {}

        for data in dataloader:
            self.critic_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.critic_module:
                chunk.zero_grad_buffer()

            micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                forward_only=False,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            metric_micro_batch = metric_micro_batch["output"]
            update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step()
            learning_rate = self.critic_optimizer.param_groups[-1]["lr"]
            data = {"critic/grad_norm": grad_norm, "critic/lr": learning_rate}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        # add empty cache after each compute
        get_torch_device().empty_cache()
        return metrics
