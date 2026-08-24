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
"""Multi-step branch-revision GRPO stress smoke with fail-closed evidence capture."""

from __future__ import annotations

import json
import math
import os
import socket
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf, open_dict

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import _apply_reward_focus_mask_alignment, run_ppo
from verl.trainer.ppo.ray_trainer_branch_revision import validate_branch_revision_runtime_config
from verl.utils.device import auto_set_device

EXPECTED = {
    "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
    "actor_rollout_ref.actor.ppo_epochs": 1,
    "algorithm.branch_revision_grpo.min_continuation_tokens": 128,
    "trainer.n_gpus_per_node": 8,
    "trainer.val_before_train": False,
    "trainer.test_freq": -1,
    "trainer.resume_mode": "disable",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _configure_file_logger(config, path: Path) -> None:
    value = str(path)
    os.environ["VERL_FILE_LOGGER_PATH"] = value
    OmegaConf.update(
        config,
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH",
        value,
        force_add=True,
    )


def _validate_contract(config, output_dir: Path, smoke_contract: dict[str, Any]) -> None:
    required_contract = {
        "model_path",
        "n_prompts",
        "n_samples",
        "num_critiques",
        "critique_grpo_grouping",
        "enable_positive_compression",
        "loss_mode",
        "learnability_logprob_statistic",
        "learnability_threshold_mode",
        "max_seed_window_stddevs",
        "nodes",
        "max_prompt_length",
        "max_response_length",
        "max_model_len",
        "critique_max_response_length",
        "max_tokens_per_gpu",
        "prompt_logprob_max_inflight_tokens",
        "gpu_memory_utilization",
        "training_steps",
        "separate_critique_model",
        "critique_warmup_steps",
        "critique_model_nnodes",
        "critique_model_n_gpus_per_node",
    }
    if set(smoke_contract) != required_contract:
        raise ValueError(
            f"branch-revision smoke contract must contain exactly {sorted(required_contract)!r}, "
            f"got {sorted(smoke_contract)!r}"
        )
    dynamic_expected = {
        "data.train_batch_size": smoke_contract["n_prompts"],
        "data.gen_batch_size": smoke_contract["n_prompts"],
        "actor_rollout_ref.rollout.n": smoke_contract["n_samples"],
        "algorithm.branch_revision_grpo.num_critiques": smoke_contract["num_critiques"],
        "algorithm.branch_revision_grpo.num_positive_critiques": smoke_contract["num_critiques"],
        "algorithm.branch_revision_grpo.critique_grpo_grouping": smoke_contract["critique_grpo_grouping"],
        "algorithm.branch_revision_grpo.enable_positive_compression": smoke_contract["enable_positive_compression"],
        "actor_rollout_ref.model.path": smoke_contract["model_path"],
        "critic.model.path": smoke_contract["model_path"],
        "actor_rollout_ref.actor.policy_loss.loss_mode": smoke_contract["loss_mode"],
        "algorithm.branch_revision_grpo.learnability_logprob_statistic": smoke_contract[
            "learnability_logprob_statistic"
        ],
        "algorithm.branch_revision_grpo.learnability_threshold_mode": smoke_contract["learnability_threshold_mode"],
        "algorithm.branch_revision_grpo.max_seed_window_stddevs": smoke_contract["max_seed_window_stddevs"],
        "trainer.nnodes": (
            smoke_contract["nodes"] - smoke_contract["critique_model_nnodes"]
            if smoke_contract["separate_critique_model"]
            else smoke_contract["nodes"]
        ),
        "algorithm.branch_revision_grpo.separate_critique_model": smoke_contract["separate_critique_model"],
        "algorithm.branch_revision_grpo.critique_warmup_steps": smoke_contract["critique_warmup_steps"],
        "algorithm.branch_revision_grpo.critique_model_nnodes": smoke_contract["critique_model_nnodes"],
        "algorithm.branch_revision_grpo.critique_model_n_gpus_per_node": smoke_contract[
            "critique_model_n_gpus_per_node"
        ],
        "data.max_prompt_length": smoke_contract["max_prompt_length"],
        "data.max_response_length": smoke_contract["max_response_length"],
        "actor_rollout_ref.rollout.max_model_len": smoke_contract["max_model_len"],
        "algorithm.branch_revision_grpo.critique_max_response_length": smoke_contract["critique_max_response_length"],
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": smoke_contract["max_tokens_per_gpu"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": smoke_contract["max_tokens_per_gpu"],
        "actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens": smoke_contract[
            "prompt_logprob_max_inflight_tokens"
        ],
        "actor_rollout_ref.rollout.gpu_memory_utilization": smoke_contract["gpu_memory_utilization"],
        "trainer.total_training_steps": smoke_contract["training_steps"],
        "trainer.save_freq": (smoke_contract["training_steps"] if smoke_contract["separate_critique_model"] else -1),
    }
    for name in (
        "n_prompts",
        "n_samples",
        "num_critiques",
        "nodes",
        "max_prompt_length",
        "max_response_length",
        "max_model_len",
        "critique_max_response_length",
        "max_tokens_per_gpu",
        "training_steps",
        "critique_model_nnodes",
        "critique_model_n_gpus_per_node",
    ):
        value = smoke_contract[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"branch-revision smoke {name} must be a positive integer")
    if not isinstance(smoke_contract["separate_critique_model"], bool):
        raise ValueError("branch-revision smoke separate_critique_model must be boolean")
    if smoke_contract["critique_grpo_grouping"] not in {"per_original", "batch"}:
        raise ValueError("branch-revision smoke critique_grpo_grouping must be per_original or batch")
    if not isinstance(smoke_contract["enable_positive_compression"], bool):
        raise ValueError("branch-revision smoke enable_positive_compression must be boolean")
    critique_warmup_steps = smoke_contract["critique_warmup_steps"]
    if (
        not isinstance(critique_warmup_steps, int)
        or isinstance(critique_warmup_steps, bool)
        or critique_warmup_steps < 0
    ):
        raise ValueError("branch-revision smoke critique_warmup_steps must be a nonnegative integer")
    if smoke_contract["separate_critique_model"] and smoke_contract["critique_model_nnodes"] >= smoke_contract["nodes"]:
        raise ValueError("branch-revision smoke needs at least one actor node and one critique node")
    if (
        smoke_contract["separate_critique_model"]
        and smoke_contract["training_steps"] <= smoke_contract["critique_warmup_steps"]
    ):
        raise ValueError("branch-revision smoke must include a post-warmup training step")
    prompt_logprob_capacity = smoke_contract["prompt_logprob_max_inflight_tokens"]
    if prompt_logprob_capacity is not None and (
        isinstance(prompt_logprob_capacity, bool)
        or not isinstance(prompt_logprob_capacity, int)
        or prompt_logprob_capacity <= 0
    ):
        raise ValueError("branch-revision smoke prompt_logprob_max_inflight_tokens must be null or a positive integer")
    if not isinstance(smoke_contract["model_path"], str) or not smoke_contract["model_path"].startswith("/"):
        raise ValueError("branch-revision smoke model_path must be an absolute string path")
    if smoke_contract["loss_mode"] not in {"dppo_tv", "vanilla"}:
        raise ValueError("branch-revision smoke loss_mode must be dppo_tv or vanilla")
    if smoke_contract["learnability_logprob_statistic"] not in {"mean", "min"}:
        raise ValueError("branch-revision smoke learnability_logprob_statistic must be mean or min")
    if smoke_contract["learnability_threshold_mode"] not in {"stddev", "percentile"}:
        raise ValueError("branch-revision smoke learnability_threshold_mode must be stddev or percentile")
    max_seed_window_stddevs = smoke_contract["max_seed_window_stddevs"]
    if (
        not isinstance(max_seed_window_stddevs, int | float)
        or isinstance(max_seed_window_stddevs, bool)
        or not math.isfinite(float(max_seed_window_stddevs))
        or float(max_seed_window_stddevs) < 0.0
    ):
        raise ValueError("branch-revision smoke max_seed_window_stddevs must be finite and nonnegative")
    gpu_memory_utilization = smoke_contract["gpu_memory_utilization"]
    if (
        not isinstance(gpu_memory_utilization, int | float)
        or isinstance(gpu_memory_utilization, bool)
        or not math.isfinite(float(gpu_memory_utilization))
        or not 0.0 < float(gpu_memory_utilization) < 1.0
    ):
        raise ValueError("branch-revision smoke gpu_memory_utilization must be finite and inside (0, 1)")
    if smoke_contract["n_samples"] < 2:
        raise ValueError("branch-revision smoke n_samples must be at least 2 for a GRPO acceptance group")
    if smoke_contract["num_critiques"] < 2:
        raise ValueError("branch-revision smoke num_critiques must be at least 2 for GRPO")
    if smoke_contract["nodes"] > 4:
        raise ValueError("branch-revision smoke supports at most four nodes")
    if smoke_contract["max_prompt_length"] + smoke_contract["max_response_length"] >= smoke_contract["max_model_len"]:
        raise ValueError("branch-revision smoke prompt plus response must be smaller than model context")
    if smoke_contract["max_tokens_per_gpu"] < (
        smoke_contract["max_prompt_length"] + smoke_contract["max_response_length"]
    ):
        raise ValueError("branch-revision smoke token budget must fit one maximum-length original")
    for key, expected in dynamic_expected.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(f"branch-revision smoke requires {key}={expected!r}, got {actual!r}")
    for key, expected in EXPECTED.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(f"branch-revision smoke requires {key}={expected!r}, got {actual!r}")
    if not bool(OmegaConf.select(config, "algorithm.branch_revision_grpo.enable")):
        raise ValueError("branch-revision smoke requires the feature to be enabled")
    if bool(OmegaConf.select(config, "algorithm.branch_revision_grpo.enable_positive_compression")) != bool(
        smoke_contract["enable_positive_compression"]
    ):
        raise ValueError("branch-revision smoke positive-compression mode differs from its launch contract")
    if bool(OmegaConf.select(config, "algorithm.intermediate_mc_value.enable")):
        raise ValueError("branch-revision smoke must not enable intermediate MC")
    if bool(OmegaConf.select(config, "critic.enable")):
        raise ValueError("branch-revision smoke is actor-only")
    if str(OmegaConf.select(config, "algorithm.adv_estimator")) != "grpo":
        raise ValueError("branch-revision smoke requires GRPO")
    if float(OmegaConf.select(config, "actor_rollout_ref.rollout.temperature")) != 1.0:
        raise ValueError("training generation temperature must be 1.0")
    if (
        float(OmegaConf.select(config, "actor_rollout_ref.rollout.top_p")) != 1.0
        or int(OmegaConf.select(config, "actor_rollout_ref.rollout.top_k")) != -1
        or float(OmegaConf.select(config, "actor_rollout_ref.rollout.repetition_penalty")) != 1.0
    ):
        raise ValueError("training generation must use untruncated learnability-comparable sampling")
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
    smoke_contract = OmegaConf.to_container(smoke_config, resolve=True)
    if not isinstance(smoke_contract, dict):
        raise ValueError("branch_revision_smoke must be a mapping")
    smoke_contract.pop("output_dir", None)
    if not output_dir.is_absolute() or str(output_dir).startswith("/opt/verl"):
        raise ValueError("branch-revision smoke output must be an absolute mounted data/output path")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(config):
        del config.branch_revision_smoke
    config = migrate_legacy_reward_impl(config)
    config = _apply_reward_focus_mask_alignment(config)
    validate_branch_revision_runtime_config(config)
    _configure_file_logger(config, output_dir / "metrics.jsonl")
    OmegaConf.resolve(config)
    _validate_contract(config, output_dir, smoke_contract)
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
    invocation_id = uuid.uuid4().hex
    audit_root = output_dir / "audit"
    prior_attempts = {path.name for path in audit_root.glob("attempt_*") if path.is_dir()}
    _write_json(
        output_dir / "status.json",
        {"status": "running", "invocation_id": invocation_id, "started_at": started},
    )
    try:
        auto_set_device(config)
        run_ppo(config)
        new_attempts = sorted(
            path.name for path in audit_root.glob("attempt_*") if path.is_dir() and path.name not in prior_attempts
        )
        if len(new_attempts) != 1:
            raise RuntimeError(f"smoke invocation must create exactly one audit attempt, got {new_attempts!r}")
        attempt_id = new_attempts[0].removeprefix("attempt_")
        checkpoint_manifest = None
        if bool(smoke_contract["separate_critique_model"]):
            checkpoint_root = Path(str(config.trainer.default_local_dir)).resolve()
            global_step = int(smoke_contract["training_steps"])
            step_root = checkpoint_root / f"global_step_{global_step}"
            required_paths = {
                "actor": step_root / "actor",
                "critique_actor": step_root / "critique_actor_rollout",
                "dataloader": step_root / "data.pt",
                "latest_step": checkpoint_root / "latest_checkpointed_iteration.txt",
            }
            missing = [name for name, path in required_paths.items() if not path.exists()]
            if missing:
                raise RuntimeError(f"separate critique-policy checkpoint is incomplete: {missing!r}")
            latest_step = int(required_paths["latest_step"].read_text(encoding="utf-8").strip())
            if latest_step != global_step:
                raise RuntimeError(
                    f"separate critique-policy checkpoint pointer mismatch: {latest_step} != {global_step}"
                )
            checkpoint_manifest = {
                "global_step": global_step,
                "checkpoint_root": str(checkpoint_root),
                "latest_step": latest_step,
                "actor_files": sum(path.is_file() for path in required_paths["actor"].rglob("*")),
                "critique_actor_files": sum(path.is_file() for path in required_paths["critique_actor"].rglob("*")),
                "dataloader_bytes": required_paths["dataloader"].stat().st_size,
            }
            if checkpoint_manifest["actor_files"] <= 0 or checkpoint_manifest["critique_actor_files"] <= 0:
                raise RuntimeError("separate critique-policy checkpoint contains an empty policy directory")
            _write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
        completion = {
            "status": "completed",
            "invocation_id": invocation_id,
            "audit_attempt_id": attempt_id,
            "wall_seconds": time.time() - started,
            "checkpoint_manifest": checkpoint_manifest,
        }
        _write_json(
            output_dir / "completed.json",
            completion,
        )
        _write_json(output_dir / "status.json", completion)
    except BaseException as error:
        new_attempts = sorted(
            path.name.removeprefix("attempt_")
            for path in audit_root.glob("attempt_*")
            if path.is_dir() and path.name not in prior_attempts
        )
        failure = {
            "status": "failed",
            "invocation_id": invocation_id,
            "audit_attempt_ids": new_attempts,
            "wall_seconds": time.time() - started,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(
            output_dir / "failed.json",
            failure,
        )
        _write_json(output_dir / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
