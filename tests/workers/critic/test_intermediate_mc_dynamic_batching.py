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

from __future__ import annotations

import os
from types import MethodType

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.workers.critic.intermediate_mc_critic import DataParallelIntermediateMCCritic


class _EvalOnlyModule:
    def eval(self) -> None:
        return None


def _run_rank(rank: int, world_size: int, init_file: str, queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        import verl.utils.seqlen_balancing as seqlen_balancing
        import verl.workers.critic.intermediate_mc_critic as critic_module

        seqlen_balancing.get_device_name = lambda: "cpu"
        critic_module.get_device_id = lambda: "cpu"
        critic = critic_module.DataParallelIntermediateMCCritic.__new__(critic_module.DataParallelIntermediateMCCritic)
        critic.critic_module = _EvalOnlyModule()
        critic.ulysses_sequence_parallel_size = 1
        critic.critic_head = "scalar"
        critic.max_reward = 1.0
        calls: list[int] = []

        def forward(_self, micro_batch):
            calls.append(int(micro_batch["input_ids"].shape[0]))
            batch_size = micro_batch["input_ids"].shape[0]
            positions = micro_batch["critic_positions"].shape[1]
            return torch.zeros((batch_size, positions, 1), dtype=torch.float32)

        critic._forward_context_micro_batch = MethodType(forward, critic)
        lengths = [4, 4, 4, 4] if rank == 0 else [1, 1, 1, 1]
        attention_mask = torch.zeros((4, 4), dtype=torch.long)
        for row, length in enumerate(lengths):
            attention_mask[row, :length] = 1
        data = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((4, 4), dtype=torch.long),
                "attention_mask": attention_mask,
                "position_ids": torch.arange(4).repeat(4, 1),
                "critic_positions": torch.tensor([[0, 1]] * 4),
                "critic_position_mask": torch.ones((4, 2)),
            },
            non_tensors={"row": np.arange(4)},
            meta_info={"use_dynamic_bsz": True, "max_token_len": 8, "micro_batch_size": 1},
        )
        values, variances = critic.compute_values(
            data,
            dp_group=dist.group.WORLD,
            same_micro_num_in_dp=True,
        )
        queue.put((rank, len(calls), tuple(values.shape), variances is None))
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_critic_synchronizes_dynamic_microbatch_count(tmp_path) -> None:
    init_file = tmp_path / "gloo-init"
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(
        _run_rank,
        args=(2, os.fspath(init_file), queue),
        nprocs=2,
        join=True,
    )
    results = sorted(queue.get() for _ in range(2))
    assert results == [
        (0, 2, (4, 2), True),
        (1, 2, (4, 2), True),
    ]


def test_critic_positions_normalize_only_inactive_padding() -> None:
    positions = DataParallelIntermediateMCCritic._validated_gather_positions(
        {
            "critic_positions": torch.tensor([[0, 2, 999], [1, -9, -9]]),
            "critic_position_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        },
        sequence_length=3,
    )
    assert positions.tolist() == [[0, 2, 0], [1, 0, 0]]


@pytest.mark.parametrize(
    ("position", "attention_mask", "message"),
    [
        (-1, [1, 1, 1], "outside"),
        (3, [1, 1, 1], "outside"),
        (2, [1, 1, 0], "padding"),
    ],
)
def test_active_critic_positions_fail_closed(position, attention_mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        DataParallelIntermediateMCCritic._validated_gather_positions(
            {
                "critic_positions": torch.tensor([[position]]),
                "critic_position_mask": torch.ones((1, 1)),
                "attention_mask": torch.tensor([attention_mask]),
            },
            sequence_length=3,
        )
