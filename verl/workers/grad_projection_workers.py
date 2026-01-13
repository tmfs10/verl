# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _int_to_signed_int64(x: int) -> int:
    mask64 = (1 << 64) - 1
    signbit = 1 << 63
    x = x & mask64
    if x >= signbit:
        x -= (1 << 64)
    return int(x)


_MIX64_MULT1 = _int_to_signed_int64(0xBF58476D1CE4E5B9)
_MIX64_MULT2 = _int_to_signed_int64(0x94D049BB133111EB)
_SEED_XOR = _int_to_signed_int64(0x9E3779B97F4A7C15)
_J_CONST = _int_to_signed_int64(0xD1342543DE82EF95)


def _mix64(x: torch.Tensor) -> torch.Tensor:
    x = (x ^ (x >> 30)) * _MIX64_MULT1
    x = (x ^ (x >> 27)) * _MIX64_MULT2
    x = x ^ (x >> 31)
    return x


def _project_rademacher(params, k: int, seed: int, chunk_size: int, scale: bool = True) -> torch.Tensor:
    """Dense Rademacher projection (full ±1 matrix) computed in a streaming, low-memory way."""
    proj_cpu = torch.zeros(k, dtype=torch.float32, device="cpu")
    offset = 0

    seed_mix = _int_to_signed_int64(seed) ^ _SEED_XOR

    proj_device = None
    device = None
    dtype = None

    for p in params:
        grad = p.grad
        n = p.numel()
        if grad is None:
            offset += n
            continue

        g_flat = grad.detach().reshape(-1)
        if proj_device is None or g_flat.device != device or g_flat.dtype != dtype:
            if proj_device is not None:
                proj_cpu += proj_device.detach().cpu().to(torch.float32)
            device = g_flat.device
            dtype = g_flat.dtype
            proj_device = torch.zeros(k, device=device, dtype=dtype)

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            m = end - start
            g_chunk = g_flat[start:end]

            idx = torch.arange(m, device=device, dtype=torch.int64) + (offset + start)
            idx ^= seed_mix

            bytes_per_grad = g_chunk.element_size()
            per_elem_bytes = 8 + bytes_per_grad
            target_bytes = 64 * 1024 * 1024
            block_k = int(target_bytes // (per_elem_bytes * m))
            if block_k < 1:
                block_k = 1
            if block_k > k:
                block_k = k

            idx_row = idx.unsqueeze(0)
            for j0 in range(0, k, block_k):
                j1 = min(k, j0 + block_k)
                j = torch.arange(j0, j1, device=device, dtype=torch.int64).unsqueeze(1)
                x = idx_row ^ (j * _J_CONST)
                x = _mix64(x)
                sign = torch.where((x & 1) == 0, 1.0, -1.0).to(dtype)
                proj_device[j0:j1] += (sign * g_chunk).sum(dim=1)

        offset += n

    if proj_device is not None:
        proj_cpu += proj_device.detach().cpu().to(torch.float32)

    if scale:
        proj_cpu /= k**0.5
    return proj_cpu


def _grad_l2_norm(params) -> float:
    total_sq = 0.0
    for p in params:
        grad = p.grad
        if grad is None:
            continue
        total_sq += grad.detach().float().pow(2).sum().item()
    return total_sq**0.5


class FSDPGradProjectionWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and self.actor_optimizer is not None:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            self.actor_module_fsdp.train()

            temperature = data.meta_info.get("temperature", 1.0)
            use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)

            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(data, max_token_len=max_token_len)
            else:
                micro_batch_size = data.meta_info.get("micro_batch_size", None)
                if micro_batch_size is None:
                    micro_batch_size = data.batch.batch_size[0]
                micro_batches = data.split(micro_batch_size)

            for p in self.actor_module_fsdp.parameters():
                p.grad = None

            total_response_tokens = float(data.batch["response_mask"].sum().item())
            if total_response_tokens <= 0:
                total_response_tokens = 1.0

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_probs = self.actor._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False
                )
                response_mask = model_inputs["response_mask"]
                assert response_mask.shape == log_probs.shape, (
                    f"response_mask shape {tuple(response_mask.shape)} != log_probs shape {tuple(log_probs.shape)}"
                )
                loss = -(log_probs * response_mask).sum() / total_response_tokens
                loss.backward()

            k = data.meta_info.get("rademacher_k", None)
            if k is None:
                raise ValueError("rademacher_k must be provided in DataProto.meta_info")
            seed = int(data.meta_info.get("rademacher_seed", 0))
            chunk_size = int(data.meta_info.get("rademacher_chunk_size", 1_000_000))
            proj = _project_rademacher(
                params=self.actor_module_fsdp.parameters(),
                k=int(k),
                seed=seed,
                chunk_size=chunk_size,
                scale=True,
            )
            grad_norm = _grad_l2_norm(self.actor_module_fsdp.parameters())
            if grad_norm > 0.0:
                proj_normalized = proj / grad_norm
            else:
                proj_normalized = torch.zeros_like(proj)

            output = DataProto.from_dict(
                tensors={
                    "projection": proj.unsqueeze(0),
                    "projection_normalized": proj_normalized.unsqueeze(0),
                }
            )
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and self.actor_optimizer is not None:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output
