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


def _mix32(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 16)
    x = x * torch.tensor(_to_signed_int32(0x7FEB352D), dtype=torch.int32, device=x.device)
    x = x ^ (x >> 15)
    x = x * torch.tensor(_to_signed_int32(0x846CA68B), dtype=torch.int32, device=x.device)
    x = x ^ (x >> 16)
    return x


def _to_signed_int32(x: int) -> int:
    x = x & 0xFFFFFFFF
    if x >= 0x80000000:
        x -= 0x100000000
    return int(x)


def _project_countsketch(
    params,
    k: int,
    seed: int,
    chunk_size: int,
    t: int = 2,
    scale: bool = True,
) -> torch.Tensor:
    """CountSketch projection (sparse JL): y[b(i)] += s(i) * x[i], with t hashes per coordinate."""
    if k & (k - 1) != 0:
        raise ValueError("For best performance, k must be a power of two (e.g., 1024).")
    if t < 1 or t > 4:
        raise ValueError("In practice t in {1,2,4} is recommended for pure PyTorch performance.")

    proj_by_device: dict[torch.device, torch.Tensor] = {}
    base_by_device: dict[torch.device, torch.Tensor] = {}

    seed32 = torch.tensor(_to_signed_int32(seed), dtype=torch.int32)
    j_cpu = torch.tensor(
        [
            _to_signed_int32(0x9E3779B1),
            _to_signed_int32(0x85EBCA6B),
            _to_signed_int32(0xC2B2AE35),
            _to_signed_int32(0x27D4EB2F),
        ],
        dtype=torch.int32,
        device="cpu",
    )

    offset = 0

    with torch.no_grad():
        for p in params:
            grad = p.grad
            n = p.numel()
            if grad is None:
                offset += n
                continue

            g_flat = grad.detach().reshape(-1)
            device = g_flat.device

            if device not in proj_by_device:
                proj_by_device[device] = torch.zeros(k, device=device, dtype=torch.float32)
                base_by_device[device] = torch.arange(chunk_size, device=device, dtype=torch.int32)

            proj_device = proj_by_device[device]
            base = base_by_device[device]
            j = j_cpu.to(device=device)

            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                m = end - start
                g_chunk = g_flat[start:end].to(torch.float32)

                off32 = torch.tensor(_to_signed_int32(offset + start), dtype=torch.int32, device=device)
                idx = base[:m] + off32
                idx = idx ^ seed32

                for r in range(t):
                    h = _mix32(idx ^ j[r])
                    k_mask = torch.tensor(_to_signed_int32(k - 1), dtype=torch.int32, device=device)
                    bucket = (h & k_mask).to(torch.int64)
                    sign = (1.0 - 2.0 * ((h >> 31) & 1).to(torch.float32))
                    proj_device.index_add_(0, bucket, sign * g_chunk)

            offset += n

    proj_cpu = torch.zeros(k, dtype=torch.float32, device="cpu")
    for proj in proj_by_device.values():
        proj_cpu += proj.detach().cpu()

    if scale:
        proj_cpu /= (k**0.5) * (t**0.5)
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
            t = int(data.meta_info.get("countsketch_t", 2))
            proj = _project_countsketch(
                params=self.actor_module_fsdp.parameters(),
                k=int(k),
                seed=seed,
                chunk_size=chunk_size,
                t=t,
                scale=True,
            )
            grad_norm = _grad_l2_norm(self.actor_module_fsdp.parameters())
            if grad_norm > 0.0:
                inv_norm = 1.0 / grad_norm
                for p in self.actor_module_fsdp.parameters():
                    if p.grad is not None:
                        p.grad.detach().mul_(inv_norm)
                proj_normalized = _project_countsketch(
                    params=self.actor_module_fsdp.parameters(),
                    k=int(k),
                    seed=seed,
                    chunk_size=chunk_size,
                    t=t,
                    scale=True,
                )
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
