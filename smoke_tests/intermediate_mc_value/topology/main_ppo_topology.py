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

"""Measured PPO entrypoint with fail-closed benchmark provenance capture."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import _apply_reward_focus_mask_alignment, run_ppo
from verl.utils.device import auto_set_device


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except Exception as error:  # noqa: BLE001 - provenance capture must preserve the original training error
        return {"command": command, "error": repr(error)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _local_hardware_snapshot() -> dict[str, object]:
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_NNODES",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_PARTITION",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "hostname": socket.gethostname(),
        "slurm": {key: os.environ.get(key) for key in slurm_keys},
        "nvidia_smi_list": _command_output(["nvidia-smi", "-L"]),
        "nvidia_smi_query": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def _ray_hardware_snapshots() -> list[dict[str, object]]:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0.001)
    def snapshot() -> dict[str, object]:
        return _local_hardware_snapshot()

    futures = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        futures.append(
            snapshot.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote()
        )
    return ray.get(futures, timeout=120)


def _validate_contract(config, benchmark: dict[str, object]) -> None:
    required = {
        "data.train_batch_size": 64,
        "actor_rollout_ref.rollout.n": 8,
        "data.max_prompt_length": 4096,
        "data.max_response_length": 8192,
        "algorithm.intermediate_mc_value.continuations_per_mark": 1,
        "algorithm.intermediate_mc_value.max_marks": 1,
        "algorithm.intermediate_mc_value.critique_max_response_length": 8192,
        "actor_rollout_ref.rollout.max_model_len": 24576,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "critic.ppo_mini_batch_size": 64,
        "critic.ppo_epochs": 1,
        "trainer.critic_warmup": 0,
        "trainer.val_before_train": False,
        "trainer.save_freq": -1,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
        "trainer.rollout_data_dir": None,
        "trainer.validation_data_dir": None,
    }
    for key, expected in required.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(f"topology benchmark contract requires {key}={expected}, got {actual!r}")
    if not bool(OmegaConf.select(config, "algorithm.intermediate_mc_value.enable")):
        raise ValueError("topology benchmark requires intermediate MC to remain enabled")
    if int(OmegaConf.select(config, "algorithm.intermediate_mc_value.num_critiques")) not in {0, 4}:
        raise ValueError("topology benchmark supports only matched M0 and M4 workloads")
    if float(OmegaConf.select(config, "actor_rollout_ref.rollout.temperature")) != 1.0:
        raise ValueError("topology benchmark generation temperature must be 1.0")
    if float(OmegaConf.select(config, "actor_rollout_ref.rollout.val_kwargs.temperature")) != 1.0:
        raise ValueError("topology benchmark validation temperature must be 1.0")
    if OmegaConf.select(config, "algorithm.intermediate_mc_value.audit_output_dir") is not None:
        raise ValueError("detailed intermediate-MC audit must be disabled during throughput measurement")
    if list(OmegaConf.select(config, "trainer.logger")) != ["file"]:
        raise ValueError("topology benchmark requires only VeRL's local file logger")
    expected_steps = int(benchmark["stabilization_steps"]) + int(benchmark["measured_steps"])
    if int(OmegaConf.select(config, "trainer.total_training_steps")) != expected_steps:
        raise ValueError(f"topology benchmark requires exactly {expected_steps} total steps")

    matched = {
        "actor_rollout_ref.model.path": benchmark["expected_model_path"],
        "critic.model.path": benchmark["expected_model_path"],
        "algorithm.intermediate_mc_value.critic_head": benchmark["expected_critic_head"],
        "algorithm.intermediate_mc_value.mark_selector": benchmark["expected_mark_selector"],
        "algorithm.intermediate_mc_value.num_critiques": benchmark["expected_num_critiques"],
        "trainer.nnodes": benchmark["expected_nodes"],
        "trainer.n_gpus_per_node": 8,
        "actor_rollout_ref.actor.strategy": benchmark["expected_strategy"],
        "critic.strategy": benchmark["expected_strategy"],
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": benchmark["expected_actor_fsdp_size"],
        "critic.model.fsdp_config.fsdp_size": benchmark["expected_critic_fsdp_size"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": benchmark["expected_rollout_tp"],
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": benchmark["expected_sequence_parallel_size"],
        "critic.ulysses_sequence_parallel_size": benchmark["expected_sequence_parallel_size"],
        "actor_rollout_ref.actor.use_dynamic_bsz": benchmark["expected_actor_dynamic"],
        "critic.use_dynamic_bsz": benchmark["expected_critic_dynamic"],
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": benchmark["expected_actor_token_cap"],
        "critic.ppo_max_token_len_per_gpu": benchmark["expected_critic_token_cap"],
        "critic.forward_max_token_len_per_gpu": benchmark["expected_critic_token_cap"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": benchmark["expected_rollout_batched_tokens"],
        "actor_rollout_ref.rollout.max_num_seqs": benchmark["expected_rollout_max_num_seqs"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": benchmark["expected_rollout_gpu_memory_utilization"],
        "actor_rollout_ref.rollout.enforce_eager": benchmark["expected_rollout_enforce_eager"],
        "actor_rollout_ref.model.enable_gradient_checkpointing": benchmark["expected_gradient_checkpointing"],
        "critic.model.enable_gradient_checkpointing": benchmark["expected_gradient_checkpointing"],
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": benchmark["expected_reshard_after_forward"],
        "critic.model.fsdp_config.reshard_after_forward": benchmark["expected_reshard_after_forward"],
    }
    for key, expected in matched.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(f"topology benchmark manifest/config mismatch for {key}: {actual!r} != {expected!r}")


def _validate_inputs(config, benchmark: dict[str, object]) -> dict[str, object]:
    raw_train_files = config.data.train_files
    train_files = (
        OmegaConf.to_container(raw_train_files, resolve=True)
        if OmegaConf.is_config(raw_train_files)
        else raw_train_files
    )
    if isinstance(train_files, str):
        train_paths = [train_files]
    else:
        train_paths = list(train_files)
    if len(train_paths) != 1:
        raise ValueError(f"topology benchmark requires exactly one training file, got {train_paths!r}")
    train_path = Path(train_paths[0])
    if not train_path.is_file():
        raise FileNotFoundError(f"missing topology benchmark training file: {train_path}")
    expected_sha = str(benchmark["expected_train_sha256"])
    actual_sha = _sha256(train_path)
    if actual_sha != expected_sha:
        raise ValueError(f"training dataset SHA-256 mismatch: {actual_sha} != {expected_sha}")
    rows = sum(1 for _ in train_path.open("rb"))
    expected_rows = int(benchmark["expected_train_rows"])
    if rows != expected_rows:
        raise ValueError(f"training dataset row count mismatch: {rows} != {expected_rows}")

    model_path = Path(str(config.actor_rollout_ref.model.path))
    critic_path = Path(str(config.critic.model.path))
    for role, path in (("actor", model_path), ("critic", critic_path)):
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"missing {role} model/config.json under {path}")
    if model_path.resolve() != critic_path.resolve():
        raise ValueError("matched topology benchmark requires actor and critic to start from the same model path")
    return {
        "train_file": str(train_path),
        "train_rows": rows,
        "train_sha256": actual_sha,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
    }


@hydra.main(config_path="../../../verl/trainer/config", config_name="intermediate_mc_ppo_trainer", version_base=None)
def main(config) -> None:
    benchmark_config = OmegaConf.select(config, "topology_benchmark")
    if benchmark_config is None:
        raise ValueError("missing +topology_benchmark configuration")
    benchmark = OmegaConf.to_container(benchmark_config, resolve=True)
    output_dir = Path(str(benchmark["output_dir"]))
    if not output_dir.is_absolute() or str(output_dir).startswith("/opt/verl"):
        raise ValueError("topology benchmark output_dir must be an absolute mounted data/output path")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VERL_FILE_LOGGER_PATH"] = str(output_dir / "metrics.jsonl")

    _write_json(output_dir / "resolved_config.json", OmegaConf.to_container(config, resolve=True))
    _write_json(output_dir / "driver_hardware_before.json", _local_hardware_snapshot())
    _validate_contract(config, benchmark)
    _write_json(output_dir / "input_manifest.json", _validate_inputs(config, benchmark))

    with open_dict(config):
        del config.topology_benchmark

    started = time.time()
    try:
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        config = _apply_reward_focus_mask_alignment(config)
        run_ppo(config)
        snapshots = _ray_hardware_snapshots()
        _write_json(output_dir / "ray_hardware_after.json", snapshots)
        _write_json(
            output_dir / "completed.json",
            {"status": "completed", "wall_seconds": time.time() - started, "ray_nodes": len(snapshots)},
        )
    except BaseException as error:
        _write_json(
            output_dir / "failed.json",
            {
                "status": "failed",
                "wall_seconds": time.time() - started,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
