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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.fsdp_workers import CriticWorker, _configure_critic_model_config


class _NativeCritic:
    def __init__(self):
        self.compute_calls = []
        self.update_calls = []

    def compute_values(self, *, data):
        self.compute_calls.append(data)
        return torch.zeros((len(data), 2))

    def update_critic(self, *, data):
        self.update_calls.append(data)
        return {"critic/vf_loss": 0.0}


def _feature_off_worker() -> CriticWorker:
    def reject_dp_group():
        raise AssertionError("feature-off critic must not request a DP group")

    worker = CriticWorker.__new__(CriticWorker)
    worker.config = OmegaConf.create(
        {
            "forward_micro_batch_size_per_gpu": 1,
            "forward_max_token_len_per_gpu": 32,
            "use_dynamic_bsz": False,
            "ppo_epochs": 1,
            "intermediate_mc_value": {"enable": False},
        }
    )
    worker._is_offload_param = False
    worker._is_offload_optimizer = False
    worker.ulysses_sharding_manager = nullcontext()
    worker.critic = _NativeCritic()
    worker.flops_counter = SimpleNamespace(estimate_flops=lambda _tokens, _time: (0.0, 1.0))
    worker.critic_lr_scheduler = SimpleNamespace(get_last_lr=lambda: [0.0], step=lambda: None)
    worker._world_size = 1
    worker._get_data_parallel_group = reject_dp_group
    return worker


def test_feature_off_worker_preserves_native_critic_call_signatures() -> None:
    worker = _feature_off_worker()
    data = DataProto.from_dict(
        tensors={"input_ids": torch.ones((2, 2), dtype=torch.long)},
        meta_info={"global_token_num": [2, 2]},
    )
    values = worker.compute_values(data)
    update = worker.update_critic(data)
    assert values.batch["values"].shape == (2, 2)
    assert update.meta_info["metrics"]["critic/vf_loss"] == 0.0
    assert len(worker.critic.compute_calls) == 1
    assert len(worker.critic.update_calls) == 1


@pytest.mark.parametrize(("head", "labels"), [("scalar", 1), ("beta", 2)])
def test_feature_critic_model_config_matches_actor_override_precedence_and_forces_head_width(head, labels) -> None:
    model_config = SimpleNamespace(
        bos_token_id=99,
        pad_token_id=99,
        max_position_embeddings=1024,
        num_labels=99,
    )
    enabled, configured_head = _configure_critic_model_config(
        model_config,
        {"bos_token_id": 1, "pad_token_id": 2, "max_position_embeddings": 2048, "num_labels": 17},
        {"enable": True, "critic_head": head},
    )
    assert enabled is True
    assert configured_head == head
    assert model_config.bos_token_id == 1
    assert model_config.pad_token_id == 2
    assert model_config.max_position_embeddings == 2048
    assert model_config.num_labels == labels


def test_feature_off_critic_model_config_keeps_native_override_behavior() -> None:
    model_config = SimpleNamespace(bos_token_id=99, num_labels=99)
    enabled, configured_head = _configure_critic_model_config(
        model_config,
        {"bos_token_id": 1, "num_labels": 17},
        {"enable": False},
    )
    assert enabled is False
    assert configured_head is None
    assert model_config.bos_token_id == 99
    assert model_config.num_labels == 1


def test_feature_critic_model_config_rejects_unknown_head() -> None:
    with pytest.raises(ValueError, match="critic head"):
        _configure_critic_model_config(
            SimpleNamespace(),
            {},
            {"enable": True, "critic_head": "categorical"},
        )
