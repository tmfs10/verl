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

import hashlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from smoke_tests.intermediate_mc_value.topology.main_ppo_topology import _validate_contract, _validate_inputs


def _contract_config():
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 64, "max_prompt_length": 4096, "max_response_length": 8192},
            "actor_rollout_ref": {
                "model": {"path": "/model", "enable_gradient_checkpointing": True},
                "actor": {
                    "strategy": "fsdp2",
                    "ulysses_sequence_parallel_size": 1,
                    "use_dynamic_bsz": True,
                    "ppo_max_token_len_per_gpu": 24576,
                    "ppo_mini_batch_size": 64,
                    "ppo_epochs": 1,
                    "fsdp_config": {"fsdp_size": 8, "reshard_after_forward": True},
                },
                "rollout": {
                    "n": 8,
                    "temperature": 1.0,
                    "val_kwargs": {"temperature": 1.0},
                    "tensor_model_parallel_size": 1,
                    "max_model_len": 24576,
                    "max_num_batched_tokens": 32768,
                    "max_num_seqs": 256,
                    "gpu_memory_utilization": 0.6,
                    "enforce_eager": True,
                },
            },
            "algorithm": {
                "intermediate_mc_value": {
                    "enable": True,
                    "critic_head": "scalar",
                    "mark_selector": "random",
                    "num_critiques": 0,
                    "continuations_per_mark": 1,
                    "max_marks": 1,
                    "critique_max_response_length": 8192,
                    "audit_output_dir": None,
                }
            },
            "critic": {
                "strategy": "fsdp2",
                "ulysses_sequence_parallel_size": 1,
                "use_dynamic_bsz": True,
                "ppo_max_token_len_per_gpu": 24576,
                "forward_max_token_len_per_gpu": 24576,
                "ppo_mini_batch_size": 64,
                "ppo_epochs": 1,
                "model": {
                    "path": "/model",
                    "enable_gradient_checkpointing": True,
                    "fsdp_config": {"fsdp_size": 8, "reshard_after_forward": True},
                },
            },
            "trainer": {
                "critic_warmup": 0,
                "total_training_steps": 3,
                "nnodes": 2,
                "n_gpus_per_node": 8,
                "val_before_train": False,
                "save_freq": -1,
                "test_freq": -1,
                "resume_mode": "disable",
                "rollout_data_dir": None,
                "validation_data_dir": None,
                "logger": ["file"],
            },
        }
    )


def _benchmark_contract() -> dict[str, object]:
    return {
        "stabilization_steps": 1,
        "measured_steps": 2,
        "expected_model_path": "/model",
        "expected_critic_head": "scalar",
        "expected_mark_selector": "random",
        "expected_num_critiques": 0,
        "expected_nodes": 2,
        "expected_strategy": "fsdp2",
        "expected_actor_fsdp_size": 8,
        "expected_critic_fsdp_size": 8,
        "expected_rollout_tp": 1,
        "expected_sequence_parallel_size": 1,
        "expected_actor_dynamic": True,
        "expected_critic_dynamic": True,
        "expected_actor_token_cap": 24576,
        "expected_critic_token_cap": 24576,
        "expected_rollout_batched_tokens": 32768,
        "expected_rollout_max_num_seqs": 256,
        "expected_rollout_gpu_memory_utilization": 0.6,
        "expected_rollout_enforce_eager": True,
        "expected_gradient_checkpointing": True,
        "expected_reshard_after_forward": True,
    }


def test_contract_accepts_feature_enabled_m0_and_rejects_detailed_audit() -> None:
    config = _contract_config()
    benchmark = _benchmark_contract()
    _validate_contract(config, benchmark)
    config.algorithm.intermediate_mc_value.audit_output_dir = "/output/audit"
    with pytest.raises(ValueError, match="detailed intermediate-MC audit"):
        _validate_contract(config, benchmark)


def test_contract_rejects_obsolete_launcher_critic_key() -> None:
    config = _contract_config()
    config.critic.append_solution_to_prompt = False
    with pytest.raises(ValueError, match="obsolete critic.append_solution_to_prompt"):
        _validate_contract(config, _benchmark_contract())


def test_contract_rejects_manifest_topology_drift_before_training() -> None:
    config = _contract_config()
    config.actor_rollout_ref.actor.fsdp_config.fsdp_size = 16
    with pytest.raises(ValueError, match="manifest/config mismatch"):
        _validate_contract(config, _benchmark_contract())


@pytest.mark.parametrize("as_list", [False, True])
def test_input_validation_accepts_scalar_or_list_train_file(tmp_path: Path, as_list: bool) -> None:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"row": 1}\n{"row": 2}\n', encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "qwen3"}\n', encoding="utf-8")
    train_files = [str(train_file)] if as_list else str(train_file)
    config = OmegaConf.create(
        {
            "data": {"train_files": train_files},
            "actor_rollout_ref": {"model": {"path": str(model_path)}},
            "critic": {"model": {"path": str(model_path)}},
        }
    )
    benchmark = {
        "expected_train_rows": 2,
        "expected_train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
    }
    result = _validate_inputs(config, benchmark)
    assert result["train_rows"] == 2
    assert result["model_path"] == str(model_path)
