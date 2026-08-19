# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Conditioned scalar/Beta token critic for synchronous intermediate MC PPO."""

from __future__ import annotations

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.intermediate_mc_value import (
    FP32_EPSILON,
    beta_value_loss_components,
    scalar_value_loss_components,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic.dp_critic import DataParallelPPOCritic


class DataParallelIntermediateMCCritic(DataParallelPPOCritic):
    """A critic whose labels refer to explicit positions in a conditioned context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        feature = self.config.intermediate_mc_value
        self.critic_head = str(feature["critic_head"])
        self.max_reward = float(feature["max_reward"])
        self.scalar_loss = str(feature["scalar_loss"])
        self.cliprange_value = float(self.config.cliprange_value)
        self.beta_target_epsilon = float(feature["beta_target_epsilon"])

    def _forward_context_micro_batch(self, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return raw critic logits at ``critic_positions`` without causal shifting."""

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, sequence_length = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                if hasattr(self.critic_module, "v_head"):
                    logits_rmpad = output[2]
                    if logits_rmpad.ndim == 2:
                        logits_rmpad = logits_rmpad.unsqueeze(-1)
                    logits_rmpad = logits_rmpad.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                if self.ulysses_sequence_parallel_size > 1:
                    logits_rmpad = gather_outputs_and_unpad(
                        logits_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                logits = pad_input(
                    logits_rmpad,
                    indices=indices,
                    batch=batch_size,
                    seqlen=sequence_length,
                )
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                if hasattr(self.critic_module, "v_head"):
                    logits = output[2]
                    if logits.ndim == 2:
                        logits = logits.unsqueeze(-1)
                else:
                    logits = output.logits

        positions = micro_batch["critic_positions"].long().clamp(min=0, max=sequence_length - 1)
        gather_index = positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        return logits.gather(dim=1, index=gather_index).float()

    def _distribution(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.critic_head == "scalar":
            if logits.shape[-1] != 1:
                raise RuntimeError(f"scalar critic expected one logit, got {tuple(logits.shape)}")
            return self.max_reward * torch.sigmoid(logits[..., 0].float()), None
        if self.critic_head != "beta" or logits.shape[-1] != 2:
            raise RuntimeError(
                f"Beta critic expected two logits, got critic_head={self.critic_head} shape={tuple(logits.shape)}"
            )
        normalized_mean = torch.sigmoid(logits[..., 0].float()).clamp(
            FP32_EPSILON,
            1.0 - FP32_EPSILON,
        )
        mean = self.max_reward * normalized_mean
        q = torch.sigmoid(logits[..., 1].float())
        return mean, q * mean * (self.max_reward - mean)

    def compute_values(
        self,
        data: DataProto,
        dp_group=None,
        same_micro_num_in_dp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.critic_module.eval()
        keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "critic_positions",
            "critic_position_mask",
        ]
        data = data.select(batch_keys=keys)
        if data.meta_info["use_dynamic_bsz"]:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_indices = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=dp_group,
                same_micro_num_in_dp=same_micro_num_in_dp,
            )
        else:
            micro_batches = data.split(data.meta_info["micro_batch_size"])
            batch_indices = None

        value_parts: list[torch.Tensor] = []
        variance_parts: list[torch.Tensor] = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                logits = self._forward_context_micro_batch(inputs)
                values, variances = self._distribution(logits)
            position_mask = inputs["critic_position_mask"].to(values.dtype)
            value_parts.append(values * position_mask)
            if variances is not None:
                variance_parts.append(variances * position_mask)

        values = torch.cat(value_parts, dim=0)
        variances = torch.cat(variance_parts, dim=0) if variance_parts else None
        if batch_indices is not None:
            values = restore_dynamic_batch(values, batch_indices)
            if variances is not None:
                variances = restore_dynamic_batch(variances, batch_indices)
        return values, variances

    def update_critic(
        self,
        data: DataProto,
        dp_group=None,
        same_micro_num_in_dp: bool = True,
    ) -> dict[str, object]:
        self.critic_module.train()
        keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "critic_positions",
            "critic_position_mask",
            "critic_targets",
            "critic_target_mask",
            "critic_old_values",
        ]
        data = data.select(batch_keys=keys)
        metrics: dict[str, object] = {"critic/vf_loss": 0.0}
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                        dp_group=dp_group,
                        same_micro_num_in_dp=same_micro_num_in_dp,
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    logits = self._forward_context_micro_batch(inputs)
                    targets = inputs["critic_targets"].float()
                    old_values = inputs["critic_old_values"].float()
                    target_mask = inputs["critic_target_mask"].float()
                    if not torch.any(target_mask):
                        raise RuntimeError("intermediate MC critic microbatch contains no supervised positions")
                    if self.critic_head == "scalar":
                        components = scalar_value_loss_components(
                            logits[..., 0],
                            targets,
                            old_values,
                            max_reward=self.max_reward,
                            cliprange_value=self.cliprange_value,
                            target_loss=self.scalar_loss,
                        )
                        predictions = components.values
                    else:
                        components = beta_value_loss_components(
                            logits,
                            targets,
                            old_values,
                            max_reward=self.max_reward,
                            cliprange_value=self.cliprange_value,
                            beta_target_epsilon=self.beta_target_epsilon,
                        )
                        predictions = components.mean
                    loss_matrix = torch.maximum(components.current_loss, components.clipped_loss)
                    critic_loss = agg_loss(
                        loss_mat=loss_matrix,
                        loss_mask=target_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    clip_fraction = masked_mean(
                        (components.clipped_loss > components.current_loss).float(),
                        target_mask,
                    )
                    prediction_mean = masked_mean(predictions, target_mask)
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = target_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1.0 / self.gradient_accumulation
                    loss = critic_loss * loss_scale_factor
                    if not torch.isfinite(loss):
                        raise FloatingPointError("intermediate MC critic produced a non-finite clipped value loss")
                    loss.backward()
                    append_to_dict(
                        metrics,
                        {
                            "critic/vf_clipfrac": clip_fraction.detach().item(),
                            "critic/vpred_mean": prediction_mean.detach().item(),
                        },
                    )
                    metrics["critic/vf_loss"] += critic_loss.detach().item() * loss_scale_factor

                grad_norm = self._optimizer_step()
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("intermediate MC critic produced a non-finite gradient norm")
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})
        self.critic_optimizer.zero_grad()
        return metrics
