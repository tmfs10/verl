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
"""One-step branch-revision GRPO smoke with fail-closed evidence capture."""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf, open_dict

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import _apply_reward_focus_mask_alignment, run_ppo
from verl.utils.device import auto_set_device

EXPECTED = {
    "data.train_batch_size": 8,
    "data.max_prompt_length": 1024,
    "data.max_response_length": 1024,
    "actor_rollout_ref.rollout.n": 2,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
    "actor_rollout_ref.actor.ppo_epochs": 1,
    "actor_rollout_ref.rollout.max_model_len": 5120,
    "algorithm.branch_revision_grpo.num_critiques": 2,
    "algorithm.branch_revision_grpo.critique_max_response_length": 2560,
    "trainer.nnodes": 1,
    "trainer.n_gpus_per_node": 8,
    "trainer.total_training_steps": 1,
    "trainer.val_before_train": False,
    "trainer.save_freq": -1,
    "trainer.test_freq": -1,
    "trainer.resume_mode": "disable",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _configure_file_logger(config, path: Path) -> None:
    value = str(path)
    os.environ["VERL_FILE_LOGGER_PATH"] = value
    OmegaConf.update(
        config,
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH",
        value,
        force_add=True,
    )


def _validate_contract(config, output_dir: Path) -> None:
    for key, expected in EXPECTED.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(f"branch-revision smoke requires {key}={expected!r}, got {actual!r}")
    if not bool(OmegaConf.select(config, "algorithm.branch_revision_grpo.enable")):
        raise ValueError("branch-revision smoke requires the feature to be enabled")
    if bool(OmegaConf.select(config, "algorithm.intermediate_mc_value.enable")):
        raise ValueError("branch-revision smoke must not enable intermediate MC")
    if bool(OmegaConf.select(config, "critic.enable")):
        raise ValueError("branch-revision smoke is actor-only")
    if str(OmegaConf.select(config, "algorithm.adv_estimator")) != "grpo":
        raise ValueError("branch-revision smoke requires GRPO")
    if str(OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode")) != "dppo_tv":
        raise ValueError("live smoke must exercise the default dppo_tv policy loss")
    if float(OmegaConf.select(config, "actor_rollout_ref.rollout.temperature")) != 1.0:
        raise ValueError("training generation temperature must be 1.0")
    if float(OmegaConf.select(config, "actor_rollout_ref.rollout.val_kwargs.temperature")) != 1.0:
        raise ValueError("validation generation temperature must be 1.0")
    if list(OmegaConf.select(config, "trainer.logger")) != ["file"]:
        raise ValueError("smoke must use only the local file logger (W&B disabled)")
    if OmegaConf.select(config, "trainer.rollout_data_dir") is not None:
        raise ValueError("smoke child evidence belongs in the feature audit, not native rollout dumps")
    expected_audit = str(output_dir / "audit")
    actual_audit = str(OmegaConf.select(config, "algorithm.branch_revision_grpo.audit_output_dir"))
    if actual_audit != expected_audit:
        raise ValueError(f"smoke audit path mismatch: {actual_audit!r} != {expected_audit!r}")

    model_path = Path(str(OmegaConf.select(config, "actor_rollout_ref.model.path")))
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"missing actor model/config.json below {model_path}")
    raw_train_files = OmegaConf.select(config, "data.train_files")
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
        raise ValueError(f"smoke requires exactly one OPSD training file, got {train_paths!r}")
    train_file = Path(str(train_paths[0]))
    if not train_file.is_file():
        raise FileNotFoundError(f"missing OPSD Math 30K training file: {train_file}")


@hydra.main(config_path="../../verl/trainer/config", config_name="branch_revision_grpo_trainer", version_base=None)
def main(config) -> None:
    smoke_config = OmegaConf.select(config, "branch_revision_smoke")
    if smoke_config is None:
        raise ValueError("missing +branch_revision_smoke.output_dir")
    output_dir = Path(str(smoke_config.output_dir))
    if not output_dir.is_absolute() or str(output_dir).startswith("/opt/verl"):
        raise ValueError("branch-revision smoke output must be an absolute mounted data/output path")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(config):
        del config.branch_revision_smoke
    config = migrate_legacy_reward_impl(config)
    config = _apply_reward_focus_mask_alignment(config)
    _configure_file_logger(config, output_dir / "metrics.jsonl")
    OmegaConf.resolve(config)
    _validate_contract(config, output_dir)
    _write_json(output_dir / "resolved_config.json", OmegaConf.to_container(config, resolve=True))
    _write_json(
        output_dir / "environment.json",
        {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wandb_mode": os.environ.get("WANDB_MODE"),
        },
    )

    started = time.time()
    try:
        auto_set_device(config)
        run_ppo(config)
        _write_json(
            output_dir / "completed.json",
            {"status": "completed", "wall_seconds": time.time() - started},
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
