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
"""Fail-closed entrypoint for the random-continuation baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import _apply_reward_focus_mask_alignment, run_ppo
from verl.trainer.ppo.ray_trainer_random_continuation import validate_random_continuation_runtime_config
from verl.utils.device import auto_set_device


def _validate_contract(config) -> Path:
    contract = OmegaConf.to_container(config.random_continuation_run, resolve=True)
    expected = {
        "output_dir",
        "model_path",
        "n_prompts",
        "rollouts_per_prompt",
        "points_per_rollout",
        "continuations_per_mark",
        "seed",
        "nodes",
        "max_prompt_length",
        "max_response_length",
        "max_model_len",
    }
    if set(contract) != expected:
        raise ValueError(f"random-continuation run contract must contain exactly {sorted(expected)!r}")
    checks = {
        "actor_rollout_ref.model.path": contract["model_path"],
        "critic.model.path": contract["model_path"],
        "data.train_batch_size": contract["n_prompts"],
        "data.gen_batch_size": contract["n_prompts"],
        "actor_rollout_ref.rollout.n": contract["rollouts_per_prompt"],
        "algorithm.random_continuation_baseline.points_per_rollout": contract["points_per_rollout"],
        "algorithm.random_continuation_baseline.continuations_per_mark": contract["continuations_per_mark"],
        "algorithm.random_continuation_baseline.selection_seed": contract["seed"],
        "trainer.nnodes": contract["nodes"],
        "data.max_prompt_length": contract["max_prompt_length"],
        "data.max_response_length": contract["max_response_length"],
        "actor_rollout_ref.rollout.max_model_len": contract["max_model_len"],
    }
    for path, expected_value in checks.items():
        actual = OmegaConf.select(config, path)
        if actual != expected_value:
            raise ValueError(f"{path}={actual!r}, expected {expected_value!r}")
    if list(config.trainer.logger) != ["file"]:
        raise ValueError("random-continuation evaluation must keep W&B disabled")
    output_dir = Path(str(contract["output_dir"])).resolve()
    audit_dir = Path(str(config.algorithm.random_continuation_baseline.audit_output_dir)).resolve()
    if audit_dir != output_dir / "audit":
        raise ValueError("random-continuation audit directory drifted from the run output")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VERL_FILE_LOGGER_PATH"] = str(output_dir / "metrics.jsonl")
    OmegaConf.update(
        config,
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH",
        str(output_dir / "metrics.jsonl"),
        force_add=True,
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(OmegaConf.to_container(config, resolve=True), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output_dir


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    config = _apply_reward_focus_mask_alignment(config)
    _validate_contract(config)
    validate_random_continuation_runtime_config(config)
    run_ppo(config)


if __name__ == "__main__":
    main()
