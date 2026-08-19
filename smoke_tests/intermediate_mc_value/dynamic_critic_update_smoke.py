#!/usr/bin/env python3
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

"""Two-GPU parity smoke for dynamic intermediate-MC critic updates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from verl import DataProto
from verl.utils.fsdp_utils import fully_shard
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.critic.intermediate_mc_critic import DataParallelIntermediateMCCritic


class TinyTokenCritic(nn.Module):
    """Small deterministic token head that still exercises real FSDP collectives."""

    def __init__(self, output_width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, output_width)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        del attention_mask, position_ids, use_cache
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("fsdp", "fsdp2"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def critic_config(*, critic_head: str, use_dynamic_bsz: bool):
    return OmegaConf.create(
        {
            "model": {"use_remove_padding": True},
            "intermediate_mc_value": {
                "critic_head": critic_head,
                "max_reward": 1.0,
                "scalar_loss": "mse",
                "beta_target_epsilon": 1.0e-4,
            },
            "cliprange_value": 0.2,
            "grad_clip": 10.0,
            "loss_agg_mode": "token-mean",
            "ppo_epochs": 1,
            "ppo_mini_batch_size": 5,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_max_token_len_per_gpu": 24,
            "ulysses_sequence_parallel_size": 1,
            "use_dynamic_bsz": use_dynamic_bsz,
        }
    )


def make_batch(rank: int) -> DataProto:
    sequence_length = 8
    lengths = [8, 8, 8, 8, 8] if rank == 0 else [2, 2, 2, 2, 2]
    input_ids = torch.zeros((5, sequence_length), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = (torch.arange(length) + 1 + rank * 7 + row * 3) % 31 + 1
        attention_mask[row, :length] = 1
    targets = torch.tensor([[0.10, 0.90], [0.25, 0.75], [0.40, 0.60], [0.55, 0.45], [0.70, 0.30]])
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(sequence_length).repeat(5, 1),
            "critic_positions": torch.tensor([[0, 1]] * 5),
            "critic_position_mask": torch.ones((5, 2)),
            "critic_targets": targets,
            # Actor-native dynamic loss scaling is row-weighted. Equal target
            # cardinality makes fixed/dynamic parameter equality a valid
            # invariant while unequal post-sync row counts test that scaling.
            "critic_target_mask": torch.ones((5, 2)),
            "critic_old_values": torch.full((5, 2), 0.5),
        }
    )


def wrap_model(model: nn.Module, *, strategy: str, mesh, device: torch.device) -> nn.Module:
    model = model.to(device)
    if strategy == "fsdp":
        return FSDP(
            model,
            device_id=device,
            device_mesh=mesh,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
        )
    if fully_shard is None:
        raise RuntimeError("FSDP2 requires PyTorch 2.4 or newer")
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    return model


def local_parameter_vector(model: nn.Module) -> torch.Tensor:
    names_and_parameters = list(model.named_parameters())
    if not names_and_parameters:
        raise AssertionError("wrapped critic exposes no parameters")
    shards = []
    for _, parameter in names_and_parameters:
        value = parameter.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        shards.append(value.float().reshape(-1).clone())
    return torch.cat(shards)


def probe_logits(model: nn.Module, device: torch.device) -> torch.Tensor:
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=torch.arange(4, device=device).unsqueeze(0),
            use_cache=False,
        ).logits.float()


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    return float((left - right).abs().max().item())


def run_head(*, critic_head: str, strategy: str, mesh, rank: int, device: torch.device) -> dict[str, object]:
    output_width = 1 if critic_head == "scalar" else 2
    torch.manual_seed(20260818)
    fixed_model = wrap_model(TinyTokenCritic(output_width), strategy=strategy, mesh=mesh, device=device)
    torch.manual_seed(20260818)
    dynamic_model = wrap_model(TinyTokenCritic(output_width), strategy=strategy, mesh=mesh, device=device)

    fixed_optimizer = torch.optim.SGD(fixed_model.parameters(), lr=0.05)
    dynamic_optimizer = torch.optim.SGD(dynamic_model.parameters(), lr=0.05)
    fixed_critic = DataParallelIntermediateMCCritic(
        config=critic_config(critic_head=critic_head, use_dynamic_bsz=False),
        critic_module=fixed_model,
        critic_optimizer=fixed_optimizer,
    )
    dynamic_critic = DataParallelIntermediateMCCritic(
        config=critic_config(critic_head=critic_head, use_dynamic_bsz=True),
        critic_module=dynamic_model,
        critic_optimizer=dynamic_optimizer,
    )

    unsynchronized, _ = prepare_dynamic_batch(
        make_batch(rank),
        max_token_len=24,
        dp_group=None,
        same_micro_num_in_dp=False,
    )
    count = torch.tensor([len(unsynchronized)], device=device)
    gathered_counts = [torch.zeros_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_counts, count)
    pre_sync_counts = [int(item.item()) for item in gathered_counts]
    synchronized, _ = prepare_dynamic_batch(
        make_batch(rank),
        max_token_len=24,
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=True,
    )
    post_sync_rows = sorted(len(micro_batch) for micro_batch in synchronized)

    initial_fixed_parameters = local_parameter_vector(fixed_model)
    initial_dynamic_parameters = local_parameter_vector(dynamic_model)
    initial_fixed_probe = probe_logits(fixed_model, device)
    initial_dynamic_probe = probe_logits(dynamic_model, device)

    fixed_metrics = fixed_critic.update_critic(
        make_batch(rank),
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=True,
    )
    dynamic_metrics = dynamic_critic.update_critic(
        make_batch(rank),
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=True,
    )
    torch.cuda.synchronize(device)

    final_fixed_parameters = local_parameter_vector(fixed_model)
    final_dynamic_parameters = local_parameter_vector(dynamic_model)
    final_fixed_probe = probe_logits(fixed_model, device)
    final_dynamic_probe = probe_logits(dynamic_model, device)

    result = {
        "rank": rank,
        "pre_sync_microbatches": pre_sync_counts,
        "post_sync_microbatch_rows": post_sync_rows,
        "fixed_loss": float(fixed_metrics["critic/vf_loss"]),
        "dynamic_loss": float(dynamic_metrics["critic/vf_loss"]),
        "initial_parameter_max_diff": maximum_difference(initial_fixed_parameters, initial_dynamic_parameters),
        "final_parameter_max_diff": maximum_difference(final_fixed_parameters, final_dynamic_parameters),
        "fixed_parameter_update": maximum_difference(initial_fixed_parameters, final_fixed_parameters),
        "dynamic_parameter_update": maximum_difference(initial_dynamic_parameters, final_dynamic_parameters),
        "initial_probe_max_diff": maximum_difference(initial_fixed_probe, initial_dynamic_probe),
        "final_probe_max_diff": maximum_difference(final_fixed_probe, final_dynamic_probe),
    }
    result["passed"] = bool(
        pre_sync_counts == [2, 1]
        and post_sync_rows == [2, 3]
        and math.isfinite(result["fixed_loss"])
        and math.isfinite(result["dynamic_loss"])
        and abs(result["fixed_loss"] - result["dynamic_loss"]) <= 2.0e-5
        and result["initial_parameter_max_diff"] == 0.0
        and result["final_parameter_max_diff"] <= 2.0e-5
        and result["fixed_parameter_update"] > 1.0e-8
        and result["dynamic_parameter_update"] > 1.0e-8
        and result["initial_probe_max_diff"] <= 2.0e-5
        and result["final_probe_max_diff"] <= 2.0e-5
    )

    all_results: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(all_results, result)
    failures = [item for item in all_results if not item or not item["passed"]]
    if failures:
        raise AssertionError(f"{strategy}/{critic_head} dynamic critic parity failed: {all_results}")

    del fixed_critic, dynamic_critic, fixed_optimizer, dynamic_optimizer, fixed_model, dynamic_model
    torch.cuda.empty_cache()
    dist.barrier()
    return {"critic_head": critic_head, "ranks": all_results}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("dynamic critic update smoke requires at least two CUDA devices")

    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"dynamic critic update smoke requires exactly two ranks, got {world_size}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        head_results = []
        for critic_head in ("scalar", "beta"):
            if rank == 0:
                print(f"[{args.strategy}/{critic_head}] comparing fixed and dynamic critic updates", flush=True)
            head_results.append(
                run_head(
                    critic_head=critic_head,
                    strategy=args.strategy,
                    mesh=mesh,
                    rank=rank,
                    device=device,
                )
            )
        if rank == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps({"strategy": args.strategy, "heads": head_results}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{args.strategy}] dynamic critic update parity verified: {args.output_json}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
