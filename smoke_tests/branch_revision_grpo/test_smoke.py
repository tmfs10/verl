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

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from smoke_tests.branch_revision_grpo.main_ppo_smoke import _validate_contract
from smoke_tests.branch_revision_grpo.submit_oci_iad import _extra_args, build_command
from smoke_tests.branch_revision_grpo.verify_smoke import _aggregate, _canonical_sha256, verify


def test_rendered_smoke_contract_is_synchronous_temperature_one_and_wandb_free(tmp_path: Path) -> None:
    command, remote_output = build_command(
        run_tag="unit",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
    )
    rendered = " ".join(command)
    assert remote_output == "/output/smoke_tests/branch_revision_grpo/unit"
    assert "algorithm.branch_revision_grpo.enable=true" in rendered
    assert "algorithm.intermediate_mc_value.enable=false" in rendered
    assert "actor_rollout_ref.rollout.temperature=1.0" in rendered
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=1.0" in rendered
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=dppo_tv" in rendered
    assert "algorithm.branch_revision_grpo.num_critiques=4" in rendered
    assert "algorithm.branch_revision_grpo.critique_grpo_grouping=per_original" in rendered
    assert "algorithm.branch_revision_grpo.enable_positive_compression=true" in rendered
    assert "algorithm.branch_revision_grpo.num_positive_critiques=4" in rendered
    assert "algorithm.branch_revision_grpo.min_seed_window_percentile=0.20" in rendered
    assert "algorithm.branch_revision_grpo.learnability_logprob_statistic=mean" in rendered
    assert "algorithm.branch_revision_grpo.learnability_threshold_mode=stddev" in rendered
    assert "algorithm.branch_revision_grpo.max_seed_window_stddevs=15.0" in rendered
    assert "actor_rollout_ref.rollout.n=4" in rendered
    assert "algorithm.branch_revision_grpo.min_continuation_tokens=128" in rendered
    assert "data.max_response_length=2048" in rendered
    assert "actor_rollout_ref.rollout.max_model_len=8192" in rendered
    assert "actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens=null" in rendered
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.6" in rendered
    assert "reward.reward_model.launch_reward_fn_async=false" in rendered
    assert "--enable_wandb" not in command
    assert "--no_requeue" in command
    assert "--add_interactive" in command


def test_rendered_smoke_scales_dataset_batch_rollouts_and_critiques_together(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="scaled",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        n_prompts=32,
        n_samples=4,
        num_critiques=6,
        seed=47,
        model_path="/hf_models/Qwen3-4B",
    )
    rendered = " ".join(command)
    assert "--n_prompts 32" in rendered
    assert "--n_samples 4" in rendered
    assert "--seed 47" in rendered
    assert "--actor_model /hf_models/Qwen3-4B" in rendered
    assert "--critic_model /hf_models/Qwen3-4B" in rendered
    assert "data.train_batch_size=32" in rendered
    assert "++data.gen_batch_size=32" in rendered
    assert "actor_rollout_ref.rollout.n=4" in rendered
    assert "algorithm.branch_revision_grpo.num_critiques=6" in rendered
    assert "algorithm.branch_revision_grpo.num_positive_critiques=6" in rendered
    assert "+branch_revision_smoke.n_prompts=32" in rendered
    assert "+branch_revision_smoke.n_samples=4" in rendered
    assert "+branch_revision_smoke.num_critiques=6" in rendered
    assert "+branch_revision_smoke.model_path=/hf_models/Qwen3-4B" in rendered


def test_rendered_smoke_supports_two_node_32k_context_and_8k_answers(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="two-node-32k",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        n_prompts=64,
        n_samples=8,
        num_critiques=2,
        nodes=2,
        max_prompt_length=2048,
        max_response_length=8192,
        max_model_len=32768,
        critique_max_response_length=8192,
        max_tokens_per_gpu=32768,
        training_steps=5,
        partition="interactive",
    )
    rendered = " ".join(command)
    assert "--nodes 2" in rendered
    assert "--max_prompt_len 2048" in rendered
    assert "--max_len 10240" in rendered
    assert "--max_tokens_per_gpu 32768" in rendered
    assert "data.max_prompt_length=2048" in rendered
    assert "data.max_response_length=8192" in rendered
    assert "actor_rollout_ref.rollout.n=8" in rendered
    assert "actor_rollout_ref.rollout.max_model_len=32768" in rendered
    assert "actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens=null" in rendered
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.6" in rendered
    assert "algorithm.branch_revision_grpo.critique_max_response_length=8192" in rendered
    assert "algorithm.branch_revision_grpo.num_critiques=2" in rendered
    assert "algorithm.branch_revision_grpo.num_positive_critiques=2" in rendered
    assert "trainer.total_training_steps=5" in rendered
    assert "+branch_revision_smoke.training_steps=5" in rendered
    assert "--partition interactive" in rendered


def test_rendered_smoke_can_disable_prompt_logprob_admission_without_changing_other_oom_controls(
    tmp_path: Path,
) -> None:
    command, _ = build_command(
        run_tag="two-node-no-admission",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        n_prompts=64,
        n_samples=8,
        num_critiques=2,
        nodes=2,
        max_prompt_length=2048,
        max_response_length=8192,
        max_model_len=32768,
        critique_max_response_length=8192,
        max_tokens_per_gpu=32768,
        prompt_logprob_max_inflight_tokens=None,
        gpu_memory_utilization=0.6,
        training_steps=5,
        partition="interactive",
    )
    rendered = " ".join(command)
    assert "actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens=null" in rendered
    assert "+branch_revision_smoke.prompt_logprob_max_inflight_tokens=null" in rendered
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.6" in rendered
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in rendered
    assert "actor_rollout_ref.rollout.max_num_seqs=32" in rendered
    assert "algorithm.branch_revision_grpo.enable=true" in rendered


def test_rendered_smoke_allows_four_nodes_only_on_normal_partitions(tmp_path: Path) -> None:
    common = {
        "run_tag": "four-node-reproduction",
        "dry_run": False,
        "python": Path("/python"),
        "launcher": Path("/launcher"),
        "verl_root": Path("/verl"),
        "reward_file": Path("/reward.py"),
        "config_dir": tmp_path,
        "nodes": 4,
    }
    command, _ = build_command(**common)
    rendered = " ".join(command)
    assert "--nodes 4" in rendered
    assert "--add_interactive" not in command
    with pytest.raises(ValueError, match="interactive.*at most two nodes"):
        build_command(**common, partition="interactive")


def test_rendered_smoke_can_select_native_clipped_ppo(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="vanilla",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        loss_mode="vanilla",
    )
    rendered = " ".join(command)
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla" in rendered
    assert "+branch_revision_smoke.loss_mode=vanilla" in rendered


def test_rendered_smoke_can_select_minimum_logprob_learnability(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="minimum",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        learnability_logprob_statistic="min",
    )
    rendered = " ".join(command)
    assert "algorithm.branch_revision_grpo.learnability_logprob_statistic=min" in rendered
    assert "+branch_revision_smoke.learnability_logprob_statistic=min" in rendered


def test_rendered_smoke_can_select_percentile_learnability(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="percentile",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        learnability_threshold_mode="percentile",
        max_seed_window_stddevs=7.5,
    )
    rendered = " ".join(command)
    assert "algorithm.branch_revision_grpo.learnability_threshold_mode=percentile" in rendered
    assert "algorithm.branch_revision_grpo.max_seed_window_stddevs=7.5" in rendered


def test_rendered_smoke_can_split_actor_and_critique_policy_across_two_nodes(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="separate-critique",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        nodes=2,
        separate_critique_model=True,
        critique_warmup_steps=1,
        training_steps=2,
        critique_model_nnodes=1,
        critique_model_n_gpus_per_node=8,
        partition="interactive",
    )
    rendered = " ".join(command)
    assert "algorithm.branch_revision_grpo.separate_critique_model=true" in rendered
    assert "algorithm.branch_revision_grpo.critique_warmup_steps=1" in rendered
    assert "algorithm.branch_revision_grpo.critique_model_nnodes=1" in rendered
    assert "algorithm.branch_revision_grpo.critique_model_n_gpus_per_node=8" in rendered
    assert "trainer.nnodes=1" in rendered
    assert "--trainer_nodes 1" in rendered
    assert "trainer.save_freq=2" in rendered
    checkpoint_override = (
        "trainer.default_local_dir=/output/smoke_tests/branch_revision_grpo/separate-critique/evidence/../checkpoints"
    )
    assert checkpoint_override in rendered
    assert "+branch_revision_smoke.separate_critique_model=true" in rendered


def test_rendered_smoke_supports_batch_grouped_recovery_only_critiques(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="batch-recovery",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        critique_grpo_grouping="batch",
        enable_positive_compression=False,
    )
    rendered = " ".join(command)
    assert "algorithm.branch_revision_grpo.critique_grpo_grouping=batch" in rendered
    assert "algorithm.branch_revision_grpo.enable_positive_compression=false" in rendered
    assert "+branch_revision_smoke.critique_grpo_grouping=batch" in rendered
    assert "+branch_revision_smoke.enable_positive_compression=false" in rendered


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_prompts": 0}, "n_prompts"),
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 1}, "at least 2"),
        ({"num_critiques": 1}, "at least 2"),
        ({"seed": -1}, "nonnegative"),
        ({"model_path": "/hf_models/not-supported"}, "model_path"),
        ({"loss_mode": "not-supported"}, "loss_mode"),
        ({"learnability_logprob_statistic": "median"}, "learnability_logprob_statistic"),
        ({"learnability_threshold_mode": "rank"}, "learnability_threshold_mode"),
        ({"max_seed_window_stddevs": -1.0}, "max_seed_window_stddevs"),
        ({"prompt_logprob_max_inflight_tokens": 0}, "prompt_logprob_max_inflight_tokens"),
        ({"gpu_memory_utilization": 1.0}, "gpu_memory_utilization"),
        ({"training_steps": 0}, "training_steps"),
        ({"critique_warmup_steps": -1}, "critique_warmup_steps"),
        ({"nodes": 1, "separate_critique_model": True}, "one actor node"),
        ({"partition": "batch_block1"}, "partition must be interactive"),
        ({"nodes": 5}, "at most four nodes"),
        ({"max_model_len": 3072}, "must be smaller than max_model_len"),
        ({"max_tokens_per_gpu": 2048}, "must fit one maximum-length original"),
    ],
)
def test_rendered_smoke_rejects_invalid_scale(
    overrides: dict[str, int | float | str], match: str, tmp_path: Path
) -> None:
    kwargs = {
        "run_tag": "invalid",
        "dry_run": False,
        "python": Path("/python"),
        "launcher": Path("/launcher"),
        "verl_root": Path("/verl"),
        "reward_file": Path("/reward.py"),
        "config_dir": tmp_path,
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        build_command(**kwargs)


def _scaled_runtime_config(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    train = tmp_path / "opsd.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    return OmegaConf.create(
        {
            "data": {
                "train_batch_size": 32,
                "gen_batch_size": 32,
                "max_prompt_length": 1024,
                "max_response_length": 2048,
                "train_files": [str(train)],
            },
            "actor_rollout_ref": {
                "model": {"path": str(model)},
                "rollout": {
                    "n": 4,
                    "max_model_len": 8192,
                    "max_num_batched_tokens": 8192,
                    "prompt_logprob_max_inflight_tokens": 8192,
                    "gpu_memory_utilization": 0.6,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                    "val_kwargs": {"temperature": 1.0},
                },
                "actor": {
                    "ppo_mini_batch_size": 8,
                    "ppo_epochs": 1,
                    "ppo_max_token_len_per_gpu": 8192,
                    "policy_loss": {"loss_mode": "dppo_tv"},
                },
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "intermediate_mc_value": {"enable": False},
                "branch_revision_grpo": {
                    "enable": True,
                    "separate_critique_model": False,
                    "critique_warmup_steps": 0,
                    "critique_model_nnodes": 1,
                    "critique_model_n_gpus_per_node": 8,
                    "critique_grpo_grouping": "per_original",
                    "num_critiques": 6,
                    "enable_positive_compression": True,
                    "num_positive_critiques": 6,
                    "learnability_logprob_statistic": "mean",
                    "learnability_threshold_mode": "stddev",
                    "max_seed_window_stddevs": 15.0,
                    "critique_max_response_length": 2560,
                    "min_continuation_tokens": 128,
                    "audit_output_dir": str(tmp_path / "audit"),
                },
            },
            "critic": {"enable": False, "model": {"path": str(model)}},
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "total_training_steps": 1,
                "val_before_train": False,
                "save_freq": -1,
                "test_freq": -1,
                "resume_mode": "disable",
                "logger": ["file"],
                "rollout_data_dir": None,
            },
        }
    )


def _scaled_smoke_contract(tmp_path: Path, *, n_prompts: int = 32) -> dict:
    return {
        "model_path": str(tmp_path / "model"),
        "n_prompts": n_prompts,
        "n_samples": 4,
        "num_critiques": 6,
        "critique_grpo_grouping": "per_original",
        "enable_positive_compression": True,
        "loss_mode": "dppo_tv",
        "learnability_logprob_statistic": "mean",
        "learnability_threshold_mode": "stddev",
        "max_seed_window_stddevs": 15.0,
        "nodes": 1,
        "max_prompt_length": 1024,
        "max_response_length": 2048,
        "max_model_len": 8192,
        "critique_max_response_length": 2560,
        "max_tokens_per_gpu": 8192,
        "prompt_logprob_max_inflight_tokens": 8192,
        "gpu_memory_utilization": 0.6,
        "training_steps": 1,
        "separate_critique_model": False,
        "critique_warmup_steps": 0,
        "critique_model_nnodes": 1,
        "critique_model_n_gpus_per_node": 8,
    }


def test_scaled_runtime_contract_accepts_matching_resolved_dimensions(tmp_path: Path) -> None:
    config = _scaled_runtime_config(tmp_path)
    _validate_contract(config, tmp_path, _scaled_smoke_contract(tmp_path))


def test_scaled_runtime_contract_rejects_stale_fixed_dimension(tmp_path: Path) -> None:
    config = _scaled_runtime_config(tmp_path)
    with pytest.raises(ValueError, match="data.train_batch_size=8"):
        _validate_contract(config, tmp_path, _scaled_smoke_contract(tmp_path, n_prompts=8))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _actor_audit_row(source: dict, *, row_index: int, response_width: int) -> dict:
    full_ids = [int(token) for token in source["full_ids"]]
    behavior = [float(value) for value in source["behavior_log_probs"]]
    train_start = source["train_start"]
    response_mask = [0] * response_width
    if train_start is None:
        train_stop = None
    else:
        train_stop = int(train_start) + len(behavior)
        response_mask[int(train_start) - 1 : train_stop - 1] = [1] * len(behavior)
    return {
        "balanced_row_index": row_index,
        "actor_row_id": source["actor_row_id"],
        "kind": source["kind"],
        "group_id": source["group_id"],
        "reward": float(np.float32(source["reward"])),
        "sequence_length": len(full_ids),
        "response_width": response_width,
        "train_start": train_start,
        "train_stop": train_stop,
        "input_ids_sha256": _canonical_sha256(full_ids, dtype="<i8"),
        "response_mask_sha256": _canonical_sha256(response_mask, dtype="u1"),
        "old_log_probs_sha256": _canonical_sha256(behavior, dtype="<f4"),
        "rollout_log_probs_sha256": _canonical_sha256(behavior, dtype="<f4"),
    }


def _audit_path(root: Path, attempt_id: str = "fixture") -> Path:
    return root / "audit" / f"attempt_{attempt_id}" / "step_00000001.jsonl"


def _refresh_attempt_config_hash(root: Path, attempt_id: str = "fixture") -> None:
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    rendered = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    path = root / "audit" / f"attempt_{attempt_id}" / "attempt.json"
    attempt = json.loads(path.read_text(encoding="utf-8"))
    attempt["resolved_config"] = config
    attempt["resolved_config_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    _write_json(path, attempt)


def _fixture(
    root: Path,
    *,
    include_continuation: bool = True,
    statistic: str = "mean",
    threshold_mode: str = "stddev",
    max_seed_window_stddevs: float = 15.0,
    nodes: int = 1,
    schema_version: int = 5,
    critique_grpo_grouping: str = "per_original",
    enable_positive_compression: bool = True,
) -> None:
    attempt_id = "fixture"
    invocation_id = "invocation"
    completion = {
        "status": "completed",
        "invocation_id": invocation_id,
        "audit_attempt_id": attempt_id,
        "wall_seconds": 2.0,
    }
    _write_json(root / "status.json", completion)
    _write_json(root / "completed.json", completion)
    branch_config = {
        "num_critiques": 2,
        "critique_grpo_grouping": critique_grpo_grouping,
        "enable_positive_compression": enable_positive_compression,
        "num_positive_critiques": 2,
        "positive_compression_target": 0.25,
        "learnability_logprob_statistic": statistic,
        "min_seed_window_percentile": 0.2,
        "full_credit_seed_window_percentile": 0.5,
        "min_continuation_tokens": 128,
    }
    if schema_version >= 5:
        branch_config.update(
            learnability_threshold_mode=threshold_mode,
            max_seed_window_stddevs=max_seed_window_stddevs,
        )
    _write_json(
        root / "resolved_config.json",
        {
            "algorithm": {"branch_revision_grpo": branch_config},
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {
                "rollout": {"n": 2},
                "actor": {"policy_loss": {"loss_mode": "dppo_tv"}},
            },
            "trainer": {"nnodes": nodes, "n_gpus_per_node": 8},
        },
    )
    resolved_config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    resolved_config_json = json.dumps(resolved_config, sort_keys=True, default=str, ensure_ascii=False)
    attempt_dir = root / "audit" / f"attempt_{attempt_id}"
    _write_json(
        attempt_dir / "attempt.json",
        {
            "schema_version": schema_version,
            "attempt_id": attempt_id,
            "starting_global_step": 1,
            "resolved_config_sha256": hashlib.sha256(resolved_config_json.encode("utf-8")).hexdigest(),
            "resolved_config": resolved_config,
            "hostname": "fixture",
            "pid": 1,
        },
    )

    events: list[dict] = []
    actor_sources: list[dict] = []
    original_rewards = [0.0, 1.0, 0.0, 1.0] + [1.0] * 12
    prompt_pass_at_1 = {
        "prompt-0": 0.5,
        "prompt-1": 0.5,
        **{f"prompt-{index}": 1.0 for index in range(2, 8)},
    }
    originals: dict[str, dict] = {}
    for original_index, original_reward in enumerate(original_rewards):
        rollout_id = f"p:{original_index}"
        prompt_group_id = f"prompt-{original_index // 2}"
        prompt_ids = [10, original_index // 2]
        solution_ids = [100 + original_index, 200 + original_index]
        solution_log_probs = [-0.5, -0.5]
        original = {
            "event": "original",
            "rollout_id": rollout_id,
            "prompt_group_id": prompt_group_id,
            "source_row": original_index,
            "prompt_ids": prompt_ids,
            "solution_ids": solution_ids,
            "solution_log_probs": solution_log_probs,
            "editable_solution_length": 2,
            "reward": original_reward,
        }
        originals[rollout_id] = original
        events.append(original)
        actor_sources.append(
            {
                "actor_row_id": f"original:{rollout_id}",
                "kind": "original",
                "group_id": f"solution:{prompt_group_id}",
                "reward": original_reward,
                "full_ids": [*prompt_ids, *solution_ids],
                "train_start": len(prompt_ids),
                "behavior_log_probs": solution_log_probs,
            }
        )

    if include_continuation:
        reference_scores = np.asarray(
            [_aggregate(original["solution_log_probs"], statistic) for original in originals.values()],
            dtype=np.float64,
        )
        if schema_version == 2:
            reference_windows = [
                {
                    "rollout_id": rollout_id,
                    "start": 0,
                    "score": float(score),
                    "weight": 1.0 / len(originals),
                }
                for (rollout_id, _), score in zip(originals.items(), reference_scores, strict=True)
            ]
            reference_payload = {
                "sampled_windows": len(reference_windows),
                "windows": reference_windows,
            }
        else:
            reference_payload = {
                "window_weighting": "uniform_per_window",
                "total_windows": len(originals),
                "rollout_window_counts": [{"rollout_id": rollout_id, "windows": 1} for rollout_id in originals],
                "window_scores_sha256": _canonical_sha256(reference_scores, dtype="<f8"),
            }
            if schema_version >= 5:
                reference_payload.update(
                    population_mean=float(np.mean(reference_scores, dtype=np.float64)),
                    population_stddev=float(np.std(reference_scores, dtype=np.float64, ddof=0)),
                )
        events.append(
            {
                "event": "learnability_reference",
                "reference_key": f"{statistic}:2",
                "logprob_statistic": statistic,
                "seed_tokens": 2,
                "eligible_rollouts": len(originals),
                **reference_payload,
            }
        )

    continuation_count = 0
    structurally_valid_count = 0
    valid_recovery_count = 0
    accepted_recovery_count = 0
    successful_recoveries = 0.0
    self_critique_rewards: list[float] = []
    for original_index, original_reward in enumerate(original_rewards):
        if original_reward == 1.0 and not enable_positive_compression:
            continue
        rollout_id = f"p:{original_index}"
        original = originals[rollout_id]
        prompt_group_id = original["prompt_group_id"]
        objective = "recovery" if original_reward == 0.0 else "compression"
        baseline = prompt_pass_at_1[prompt_group_id]
        critique_prompt_ids = [*original["prompt_ids"], *original["solution_ids"], 999]
        for critique_index in range(2):
            valid = include_continuation and (
                (original_index == 0 and critique_index in {0, 1}) or (original_index == 1 and critique_index == 0)
            )
            accepted = valid and critique_index == 0
            outcome = 1.0 if accepted else 0.0
            self_critique_rewards.append(outcome - baseline)
            objective_credit = 0.4026955278742087 if accepted and objective == "compression" else outcome
            reward = outcome - baseline if objective == "recovery" else objective_credit
            critique_ids = [700 + original_index, 800 + critique_index]
            branch_prefix_ids = [original["solution_ids"][0]] if valid else []
            if valid and schema_version >= 4 and original_index == 0:
                branch_prefix_ids = []
            prefix_ids = [600 + original_index * 2 + critique_index] if valid and schema_version >= 4 else []
            continuation_prefix_ids = [*branch_prefix_ids, *prefix_ids] if valid and schema_version >= 4 else []
            replacement_ids = [
                500 + original_index * 4 + critique_index * 2,
                501 + original_index * 4 + critique_index * 2,
            ]
            replacement_ids = replacement_ids if valid else []
            replacement_log_probs = ([-0.1, -0.1] if accepted else [-1.0, -1.0]) if valid else []
            revision_context_ids = continuation_prefix_ids if schema_version >= 4 else branch_prefix_ids
            revised_prefix_ids = [*revision_context_ids, *replacement_ids] if valid else []
            generated_ids = [900 + original_index * 2 + critique_index] if accepted else []
            generated_log_probs = [-0.2] if accepted else []
            if valid:
                structurally_valid_count += 1
                if objective == "recovery":
                    valid_recovery_count += 1
                seed_score = _aggregate(replacement_log_probs, statistic)
                percentile = 1.0 if accepted else 0.0
                reward_weight = 1.0 if accepted else 0.0
                learnability_event = {
                    "event": "learnability",
                    "score_source": "vllm_prompt_logprobs",
                    "reference_key": f"{statistic}:2",
                    "rollout_id": rollout_id,
                    "objective": objective,
                    "critique_index": critique_index,
                    "seed_tokens": 2,
                    "logprob_statistic": statistic,
                    "seed_score": seed_score,
                    "scoring_prompt_ids": [*original["prompt_ids"], *revision_context_ids, *replacement_ids],
                    "prompt_logprob_start": len(original["prompt_ids"]) + len(revision_context_ids),
                    "scored_token_ids": replacement_ids,
                    "scored_token_log_probs": replacement_log_probs,
                    "percentile": percentile,
                    "reward_weight": reward_weight,
                    "accepted": accepted,
                    "eligible_rollouts": len(originals),
                    ("sampled_windows" if schema_version == 2 else "total_windows"): len(originals),
                }
                if schema_version >= 5:
                    learnability_event.update(
                        threshold_mode=threshold_mode,
                        reference_mean=-0.5,
                        reference_stddev=0.0,
                        stddevs_below_mean=None,
                        acceptance_floor=-0.5 if threshold_mode == "stddev" else None,
                        max_seed_window_stddevs=max_seed_window_stddevs,
                    )
                events.append(learnability_event)
            compression_fraction = objective_credit * 0.25 if accepted and objective == "compression" else None
            compression_credit = objective_credit if accepted and objective == "compression" else None
            edit_strings = (
                {
                    "prefix": "source prefix" if valid else "",
                    "prefix_plus_new_continuation": "source prefixgood" if valid else "",
                    "new_continuation": "good" if valid else "",
                    "prefix_ids": prefix_ids,
                    "continuation_prefix_ids": continuation_prefix_ids,
                }
                if schema_version >= 4
                else {
                    "branch": "bad" if valid else "",
                    "new_continuation": "good" if valid else "",
                }
            )
            critique = {
                "event": "critique",
                "actor_row_id": f"critique:{rollout_id}:{critique_index}",
                "continuation_actor_row_id": f"continuation:{rollout_id}:{critique_index}",
                "rollout_id": rollout_id,
                "prompt_group_id": prompt_group_id,
                "objective": objective,
                "critique_index": critique_index,
                "reward": reward,
                "objective_credit": objective_credit,
                "continuation_outcome": outcome,
                "prompt_pass_at_1": baseline,
                "learnability_accepted": accepted,
                "learnability_percentile": 1.0 if accepted else 0.0 if valid else None,
                "learnability_weight": 1.0 if accepted else 0.0,
                "compression_fraction": compression_fraction,
                "compression_credit": compression_credit,
                "generated_continuation_tokens": len(generated_ids),
                "continuation_reward_evaluated": accepted,
                "continuation_wasted_by_learnability": valid and not accepted,
                "parse_reason": "valid" if valid else "tag_count",
                **edit_strings,
                "branch_prefix_ids": branch_prefix_ids,
                "new_continuation_ids": replacement_ids,
                "new_continuation_log_probs": replacement_log_probs,
                "revised_prefix_ids": revised_prefix_ids,
                "generated_continuation_ids": generated_ids,
                "generated_continuation_log_probs": generated_log_probs,
                "critique_prompt_ids": critique_prompt_ids,
                "critique_ids": critique_ids,
                "critique_log_probs": [-0.2, -0.3],
            }
            events.append(critique)
            actor_sources.append(
                {
                    "actor_row_id": critique["actor_row_id"],
                    "kind": "critique",
                    "group_id": "critique:batch" if critique_grpo_grouping == "batch" else f"critique:{rollout_id}",
                    "reward": reward,
                    "full_ids": [*critique_prompt_ids, *critique_ids],
                    "train_start": len(critique_prompt_ids),
                    "behavior_log_probs": critique["critique_log_probs"],
                }
            )
            if accepted:
                continuation_count += 1
                if objective == "recovery":
                    accepted_recovery_count += 1
                    successful_recoveries += outcome
                continuation = {
                    "event": "continuation",
                    "actor_row_id": critique["continuation_actor_row_id"],
                    "rollout_id": rollout_id,
                    "objective": objective,
                    "critique_index": critique_index,
                    "reward": outcome,
                    "revised_prefix_ids": revised_prefix_ids,
                    "continuation_ids": generated_ids,
                    "continuation_log_probs": generated_log_probs,
                    "continuation_max_tokens": 128,
                    "compression_fraction": compression_fraction,
                    "compression_credit": compression_credit,
                }
                events.append(continuation)
                actor_sources.append(
                    {
                        "actor_row_id": continuation["actor_row_id"],
                        "kind": "continuation",
                        "group_id": f"solution:{prompt_group_id}",
                        "reward": outcome,
                        "full_ids": [*original["prompt_ids"], *revised_prefix_ids, *generated_ids],
                        "train_start": len(original["prompt_ids"]) + len(revised_prefix_ids),
                        "behavior_log_probs": generated_log_probs,
                    }
                )

    selected_original_count = 2 + (14 if enable_positive_compression else 0)
    critique_count = selected_original_count * 2
    rows = 16 + critique_count + continuation_count
    padding = (-rows) % (nodes * 8)
    actor_sources.extend(
        {
            "actor_row_id": f"padding:{index}",
            "kind": "padding",
            "group_id": f"padding:{index}",
            "reward": 0.0,
            "full_ids": [0],
            "train_start": None,
            "behavior_log_probs": [],
        }
        for index in range(padding)
    )
    response_width = max(len(source["full_ids"]) for source in actor_sources) - 1
    balanced_sources = list(reversed(actor_sources))
    events.extend(
        [
            {
                "event": "actor_batch",
                "rows": rows,
                "original": 16,
                "critiques": critique_count,
                "continuations": continuation_count,
                "padding": padding,
                "pad_token_id": 0,
                "policy_loss_mode": "dppo_tv",
                "actor_rows": [
                    _actor_audit_row(source, row_index=index, response_width=response_width)
                    for index, source in enumerate(balanced_sources)
                ],
            },
            {
                "event": "iteration",
                "originals": 16,
                "incorrect": 2,
                "correct": 14,
                "positive_compression_enabled": enable_positive_compression,
                "critique_grpo_grouping": critique_grpo_grouping,
                "learnability_logprob_statistic": statistic,
                **(
                    {
                        "learnability_threshold_mode": threshold_mode,
                        "max_seed_window_stddevs": max_seed_window_stddevs,
                    }
                    if schema_version >= 5
                    else {}
                ),
                "original_rewards": original_rewards,
                "prompt_pass_at_1": prompt_pass_at_1,
            },
            {"event": "step_complete"},
        ]
    )
    events = [
        {"schema_version": schema_version, "attempt_id": attempt_id, "global_step": 1, **event} for event in events
    ]
    _write_jsonl(_audit_path(root, attempt_id), events)
    _write_jsonl(
        root / "metrics.jsonl",
        [
            {
                "step": 1,
                "data": {
                    "branch_revision/originals": 16.0,
                    "branch_revision/incorrect_originals": 2.0,
                    "branch_revision/correct_originals": 14.0,
                    "branch_revision/critiques": float(critique_count),
                    "branch_revision/recovery_critiques": 4.0,
                    "branch_revision/compression_critiques": 28.0 if enable_positive_compression else 0.0,
                    "branch_revision/critique_grpo_grouping_is_batch": float(critique_grpo_grouping == "batch"),
                    "branch_revision/critique_grpo_group_count": float(
                        1 if critique_grpo_grouping == "batch" else selected_original_count
                    ),
                    "branch_revision/critique_grpo_group_size_mean": float(
                        critique_count if critique_grpo_grouping == "batch" else 2
                    ),
                    "branch_revision/critique_grpo_group_size_max": float(
                        critique_count if critique_grpo_grouping == "batch" else 2
                    ),
                    "branch_revision/valid_edits": float(structurally_valid_count),
                    "branch_revision/learnability_accepted_edits": float(continuation_count),
                    "branch_revision/continuations": float(continuation_count),
                    "branch_revision/self_critique_reward/mean": (
                        sum(self_critique_rewards) / len(self_critique_rewards) if self_critique_rewards else 0.0
                    ),
                    "branch_revision/flip/success_per_valid_continuation": (
                        successful_recoveries / accepted_recovery_count if accepted_recovery_count else 0.0
                    ),
                    "branch_revision/flip/success_per_continuation": (
                        successful_recoveries / valid_recovery_count if valid_recovery_count else 0.0
                    ),
                    "branch_revision/policy_loss_is_dppo_tv": 1.0,
                    "actor/grad_norm": 1.0,
                    "actor/pg_loss": 0.1,
                },
            }
        ],
    )


def _add_prompt_logprob_admission_evidence(root: Path, *, capacity: int = 8192) -> None:
    config_path = root / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["actor_rollout_ref"]["rollout"]["prompt_logprob_max_inflight_tokens"] = capacity
    _write_json(config_path, config)

    path = _audit_path(root)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    admissions = []
    request_sequence = 0
    for event in events:
        if event["event"] != "learnability":
            continue
        request_sequence += 1
        prompt_tokens = len(event["scoring_prompt_ids"])
        admission = {
            "server_id": "replica:0:node:0",
            "capacity": capacity,
            "request_sequence": request_sequence,
            "prompt_tokens": prompt_tokens,
            "charged_tokens": min(prompt_tokens, capacity),
            "wait_seconds": 0.0,
            "inflight_prompt_tokens_at_grant": prompt_tokens,
            "inflight_charged_tokens_at_grant": min(prompt_tokens, capacity),
            "high_water_prompt_tokens": prompt_tokens,
            "high_water_charged_tokens": min(prompt_tokens, capacity),
            "oversized": prompt_tokens > capacity,
        }
        event["prompt_logprob_admission"] = admission
        admissions.append(admission)
    assert admissions
    summary = {
        "schema_version": events[0]["schema_version"],
        "attempt_id": events[0]["attempt_id"],
        "global_step": events[0]["global_step"],
        "event": "prompt_logprob_admission_summary",
        "capacity": capacity,
        "requests": len(admissions),
        "prompt_tokens": sum(item["prompt_tokens"] for item in admissions),
        "per_server": {
            "replica:0:node:0": {
                "requests": len(admissions),
                "prompt_tokens": sum(item["prompt_tokens"] for item in admissions),
                "max_inflight_prompt_tokens": max(item["high_water_prompt_tokens"] for item in admissions),
                "max_inflight_charged_tokens": max(item["high_water_charged_tokens"] for item in admissions),
                "max_wait_seconds": 0.0,
            }
        },
    }
    events.insert(next(index for index, event in enumerate(events) if event["event"] == "actor_batch"), summary)
    _write_jsonl(path, events)
    _refresh_attempt_config_hash(root)


def _duplicate_fixture_step(root: Path) -> None:
    config_path = root / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_logprob_capacity = config["actor_rollout_ref"]["rollout"].get("prompt_logprob_max_inflight_tokens")
    config["trainer"]["total_training_steps"] = 2
    _write_json(config_path, config)
    first_events = [json.loads(line) for line in _audit_path(root).read_text(encoding="utf-8").splitlines()]
    second_events = [{**event, "global_step": 2} for event in first_events]
    if prompt_logprob_capacity is not None and not any(
        event["event"] == "prompt_logprob_admission_summary" for event in second_events
    ):
        second_events.insert(
            -2,
            {
                "schema_version": second_events[0]["schema_version"],
                "attempt_id": second_events[0]["attempt_id"],
                "global_step": 2,
                "event": "prompt_logprob_admission_summary",
                "prompt_tokens": 1,
            },
        )
    _write_jsonl(root / "audit" / "attempt_fixture" / "step_00000002.jsonl", second_events)
    metric_rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    _write_jsonl(root / "metrics.jsonl", [*metric_rows, *[{**row, "step": 2} for row in metric_rows]])
    _refresh_attempt_config_hash(root)


def _policy_actor_batch(source: dict, *, policy: str, kinds: set[str], dp_size: int = 8) -> dict:
    rows = [copy.deepcopy(row) for row in source["actor_rows"] if row["kind"] in kinds]
    response_width = int(rows[0]["response_width"])
    padding = (-len(rows)) % dp_size
    for index in range(padding):
        rows.append(
            _actor_audit_row(
                {
                    "actor_row_id": f"padding:{index}",
                    "kind": "padding",
                    "group_id": f"padding:{index}",
                    "reward": 0.0,
                    "full_ids": [0],
                    "train_start": None,
                    "behavior_log_probs": [],
                },
                row_index=len(rows),
                response_width=response_width,
            )
        )
    for index, row in enumerate(rows):
        row["balanced_row_index"] = index
    counts = Counter(row["kind"] for row in rows)
    return {
        **{key: value for key, value in source.items() if key not in {"actor_rows", "padding"}},
        "policy": policy,
        "rows": len(rows) - padding,
        "original": counts["original"],
        "critiques": counts["critique"],
        "continuations": counts["continuation"],
        "padding": padding,
        "actor_rows": rows,
    }


def _separate_critique_policy_fixture(root: Path) -> None:
    _fixture(root)
    config_path = root / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"]["total_training_steps"] = 2
    config["algorithm"]["branch_revision_grpo"].update(
        separate_critique_model=True,
        critique_warmup_steps=1,
        critique_model_nnodes=1,
        critique_model_n_gpus_per_node=8,
    )
    _write_json(config_path, config)
    _refresh_attempt_config_hash(root)

    base_events = [json.loads(line) for line in _audit_path(root).read_text(encoding="utf-8").splitlines()]
    base_batch = next(event for event in base_events if event["event"] == "actor_batch")
    base_iteration = next(event for event in base_events if event["event"] == "iteration")
    source_events = [
        event for event in base_events if event["event"] not in {"actor_batch", "iteration", "step_complete"}
    ]
    critique_batch = _policy_actor_batch(base_batch, policy="critique_actor", kinds={"critique"})
    main_batch = _policy_actor_batch(base_batch, policy="actor", kinds={"original", "continuation"})
    for step in (1, 2):
        iteration = {
            **base_iteration,
            "separate_critique_model": True,
            "critique_warmup_steps": 1,
            "critique_warmup_active": step == 1,
            "main_actor_updated": step == 2,
            "critique_actor_updated": True,
        }
        policy_batches = [critique_batch] if step == 1 else [main_batch, critique_batch]
        events = [*source_events, *policy_batches, iteration, {"event": "step_complete"}]
        events = [
            {
                "schema_version": base_events[0]["schema_version"],
                "attempt_id": base_events[0]["attempt_id"],
                **event,
                "global_step": step,
            }
            for event in events
        ]
        _write_jsonl(root / "audit" / "attempt_fixture" / f"step_{step:08d}.jsonl", events)

    metric = json.loads((root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metric["step"] = 2
    metric["data"].update(
        {
            "branch_revision/separate_critique_model": 1.0,
            "branch_revision/critique_warmup_active": 0.0,
            "branch_revision/main_actor_updated": 1.0,
            "branch_revision/critique_actor_updated": 1.0,
            "branch_revision/main_actor_rows": float(main_batch["rows"]),
            "branch_revision/critique_model_rows": float(critique_batch["rows"]),
            "critique_actor/grad_norm": 1.0,
            "critique_actor/pg_loss": 0.2,
        }
    )
    _write_jsonl(root / "metrics.jsonl", [metric])
    checkpoint_manifest = {
        "global_step": 2,
        "checkpoint_root": "/output/smoke_tests/branch_revision_grpo/fixture/checkpoints",
        "latest_step": 2,
        "actor_files": 25,
        "critique_actor_files": 25,
        "dataloader_bytes": 128,
    }
    completion = json.loads((root / "completed.json").read_text(encoding="utf-8"))
    completion["checkpoint_manifest"] = checkpoint_manifest
    _write_json(root / "completed.json", completion)
    _write_json(root / "status.json", completion)
    _write_json(root / "checkpoint_manifest.json", checkpoint_manifest)


def test_verifier_accepts_complete_live_contract(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = verify(tmp_path)
    assert result["status"] == "verified"


def test_verifier_accepts_one_iteration_level_critique_grpo_group(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        critique_grpo_grouping="batch",
        enable_positive_compression=False,
    )
    result = verify(tmp_path)
    assert result["status"] == "verified"
    assert result["critique_grpo_grouping"] == "batch"

    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    actor_batch = next(event for event in events if event["event"] == "actor_batch")
    critique_row = next(row for row in actor_batch["actor_rows"] if row["kind"] == "critique")
    critique_row["group_id"] = "critique:per-rank-0"
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="does not match its source tensors"):
        verify(tmp_path)


def test_verifier_proves_separate_critique_policy_warmup_and_post_warmup_batches(tmp_path: Path) -> None:
    _separate_critique_policy_fixture(tmp_path)
    result = verify(tmp_path)
    assert result["status"] == "verified"
    assert result["selected_global_step"] == 2


def test_verifier_validates_prompt_logprob_admission_evidence(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _add_prompt_logprob_admission_evidence(tmp_path)
    result = verify(tmp_path)
    assert result["status"] == "verified"

    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    learnability = next(event for event in events if event["event"] == "learnability")
    learnability["prompt_logprob_admission"]["high_water_charged_tokens"] = 8193
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="violates the prompt-logprob admission budget"):
        verify(tmp_path)


def test_verifier_rejects_admission_evidence_when_budget_is_disabled(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    learnability = next(event for event in events if event["event"] == "learnability")
    learnability["prompt_logprob_admission"] = {"capacity": 8192}
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="unbounded prompt-logprob scoring unexpectedly retained"):
        verify(tmp_path)


def test_verifier_requires_completed_step_range_and_tie_breaks_on_first_step(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _duplicate_fixture_step(tmp_path)
    result = verify(tmp_path)
    assert result["selected_global_step"] == 1
    assert len(result["audit_files"]) == 2

    (tmp_path / "audit" / "attempt_fixture" / "step_00000002.jsonl").unlink()
    with pytest.raises(ValueError, match="do not match the attempt range"):
        verify(tmp_path)
    assert result["valid_edits"] == 3
    assert result["successful_compression_credit"] == pytest.approx(0.4026955278742087)
    assert result["learnability_threshold_mode"] == "stddev"
    assert result["max_seed_window_stddevs"] == 15.0


def test_verifier_checks_admission_evidence_in_every_completed_step(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _add_prompt_logprob_admission_evidence(tmp_path)
    _duplicate_fixture_step(tmp_path)
    second_path = tmp_path / "audit" / "attempt_fixture" / "step_00000002.jsonl"
    events = [json.loads(line) for line in second_path.read_text(encoding="utf-8").splitlines()]
    learnability = next(event for event in events if event["event"] == "learnability")
    learnability["prompt_logprob_admission"]["high_water_charged_tokens"] = 8193
    _write_jsonl(second_path, events)
    with pytest.raises(ValueError, match="violates the prompt-logprob admission budget"):
        verify(tmp_path)


def test_verifier_uses_resumed_attempt_step_range(tmp_path: Path) -> None:
    _fixture(tmp_path)
    config_path = tmp_path / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"]["total_training_steps"] = 3
    _write_json(config_path, config)

    attempt_path = tmp_path / "audit" / "attempt_fixture" / "attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["starting_global_step"] = 3
    _write_json(attempt_path, attempt)
    _refresh_attempt_config_hash(tmp_path)

    first_path = _audit_path(tmp_path)
    events = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(
        first_path.with_name("step_00000003.jsonl"),
        [{**event, "global_step": 3} for event in events],
    )
    first_path.unlink()
    metric_rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    _write_jsonl(tmp_path / "metrics.jsonl", [{**row, "step": 3} for row in metric_rows])

    result = verify(tmp_path)
    assert result["selected_global_step"] == 3
    assert len(result["audit_files"]) == 1


def test_verifier_accepts_padding_for_the_full_two_node_data_parallel_world(tmp_path: Path) -> None:
    _fixture(tmp_path, nodes=2)
    result = verify(tmp_path)
    assert result["status"] == "verified"
    assert result["padding_rows"] == 14


def test_verifier_retains_legacy_schema_v2_support(tmp_path: Path) -> None:
    _fixture(tmp_path, schema_version=2)
    assert verify(tmp_path)["status"] == "verified"


def test_verifier_retains_legacy_schema_v3_support(tmp_path: Path) -> None:
    _fixture(tmp_path, schema_version=3)
    assert verify(tmp_path)["status"] == "verified"


def test_verifier_retains_legacy_schema_v4_support(tmp_path: Path) -> None:
    _fixture(tmp_path, schema_version=4)
    result = verify(tmp_path)
    assert result["status"] == "verified"
    assert result["learnability_threshold_mode"] == "percentile"


def test_verifier_accepts_explicit_schema_v5_percentile_mode(tmp_path: Path) -> None:
    _fixture(tmp_path, threshold_mode="percentile")
    result = verify(tmp_path)
    assert result["status"] == "verified"
    assert result["learnability_threshold_mode"] == "percentile"


def test_verifier_rejects_schema_v4_joint_text_that_does_not_extend_prefix(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    critique = next(event for event in events if event["event"] == "critique" and event["parse_reason"] == "valid")
    critique["prefix_plus_new_continuation"] = "different"
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="inconsistent prefix/joint boundaries"):
        verify(tmp_path)


def test_verifier_rejects_schema_v4_corrupted_continuation_prefix_boundary(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    critique = next(event for event in events if event["event"] == "critique" and event["parse_reason"] == "valid")
    critique["continuation_prefix_ids"] = [999]
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="inconsistent prefix/joint boundaries"):
        verify(tmp_path)


def test_verifier_accepts_minimum_logprob_evidence_and_post_balance_reordering(tmp_path: Path) -> None:
    _fixture(tmp_path, statistic="min")
    result = verify(tmp_path)
    assert result["learnability_logprob_statistic"] == "min"
    assert result["audit_attempt_id"] == "fixture"


def test_verifier_accepts_native_clipped_ppo_evidence(tmp_path: Path) -> None:
    _fixture(tmp_path)
    config_path = tmp_path / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] = "vanilla"
    _write_json(config_path, config)
    _refresh_attempt_config_hash(tmp_path)
    audit_path = _audit_path(tmp_path)
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    next(event for event in events if event["event"] == "actor_batch")["policy_loss_mode"] = "vanilla"
    _write_jsonl(audit_path, events)
    metrics_path = tmp_path / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    metrics[0]["data"]["branch_revision/policy_loss_is_dppo_tv"] = 0.0
    _write_jsonl(metrics_path, metrics)
    assert verify(tmp_path)["policy_loss_mode"] == "vanilla"


def test_verifier_accepts_only_expected_native_runtime_config_normalization(tmp_path: Path) -> None:
    _fixture(tmp_path)
    config_path = tmp_path / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"] = {"nnodes": 1, "n_gpus_per_node": 8, "total_training_steps": 1}
    config["reward"] = {"reward_model": {"enable_resource_pool": False, "nnodes": 0, "n_gpus_per_node": 1}}
    config["actor_rollout_ref"]["actor"]["optim"] = {"total_training_steps": -1}
    config["critic"] = {"optim": {"total_training_steps": -1}}
    _write_json(config_path, config)
    runtime_config = copy.deepcopy(config)
    runtime_config["reward"]["reward_model"].update(nnodes=1, n_gpus_per_node=8)
    runtime_config["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] = 1
    runtime_config["critic"]["optim"]["total_training_steps"] = 1
    attempt_path = tmp_path / "audit" / "attempt_fixture" / "attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["resolved_config"] = runtime_config
    rendered = json.dumps(runtime_config, sort_keys=True, default=str, ensure_ascii=False)
    attempt["resolved_config_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    _write_json(attempt_path, attempt)
    assert verify(tmp_path)["status"] == "verified"

    attempt["resolved_config"]["trainer"]["nnodes"] = 2
    rendered = json.dumps(attempt["resolved_config"], sort_keys=True, default=str, ensure_ascii=False)
    attempt["resolved_config_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    _write_json(attempt_path, attempt)
    with pytest.raises(ValueError, match="does not match the resolved configuration"):
        verify(tmp_path)


def test_verifier_rejects_smoke_without_a_valid_revision(tmp_path: Path) -> None:
    _fixture(tmp_path, include_continuation=False)
    with pytest.raises(ValueError, match="no learnability-accepted"):
        verify(tmp_path)


def test_integrity_verifier_accepts_complete_zero_signal_run(tmp_path: Path) -> None:
    _fixture(tmp_path, include_continuation=False)
    result = verify(tmp_path, require_algorithm_signal=False)
    assert result["status"] == "integrity-verified"
    assert result["algorithm_signal_required"] is False
    assert result["learnability_accepted_edits"] == 0
    assert result["successful_revisions"] == 0.0


def test_verifier_rejects_zero_signal_optimizer_step(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["data"]["actor/grad_norm"] = 0.0
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="positive learning signal"):
        verify(tmp_path)


def test_verifier_rejects_centered_critique_reward_on_revised_solution_row(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    actor_batch = next(event for event in events if event["event"] == "actor_batch")
    continuation = next(row for row in actor_batch["actor_rows"] if row["kind"] == "continuation")
    continuation["reward"] = 0.5
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="does not match its source tensors"):
        verify(tmp_path)


def test_verifier_rejects_critique_baseline_from_wrong_prompt_group(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    critique = next(event for event in events if event["event"] == "critique")
    critique["prompt_group_id"] = "prompt-7"
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="wrong original prompt group"):
        verify(tmp_path)


def test_verifier_requires_vllm_prompt_logprobs_for_learnability(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    learnability = next(event for event in events if event["event"] == "learnability")
    learnability["score_source"] = "actor_forward"
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="did not use vLLM prompt log probabilities"):
        verify(tmp_path)


def test_verifier_rejects_corrupted_exhaustive_reference_hash(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    reference = next(event for event in events if event["event"] == "learnability_reference")
    reference["window_scores_sha256"] = "0" * 64
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="corrupted exhaustive score hash"):
        verify(tmp_path)


@pytest.mark.parametrize("field", ["population_mean", "population_stddev"])
def test_verifier_rejects_corrupted_population_statistics(field: str, tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    reference = next(event for event in events if event["event"] == "learnability_reference")
    reference[field] = 1.0
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="corrupted population statistics"):
        verify(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold_mode", "percentile"),
        ("reference_mean", 1.0),
        ("reference_stddev", 1.0),
        ("stddevs_below_mean", 1.0),
        ("acceptance_floor", 1.0),
        ("max_seed_window_stddevs", 14.0),
    ],
)
def test_verifier_rejects_corrupted_stddev_learnability_evidence(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    learnability = next(event for event in events if event["event"] == "learnability")
    learnability[field] = value
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="corrupted standard-deviation evidence"):
        verify(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("window_weighting", "uniform_per_rollout", "uniform per-window mass"),
        ("total_windows", 15, "not exhaustive"),
        ("rollout_window_counts", [], "not exhaustive"),
    ],
)
def test_verifier_rejects_non_exhaustive_reference_contract(
    field: str,
    value: object,
    message: str,
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    reference = next(event for event in events if event["event"] == "learnability_reference")
    reference[field] = value
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match=message):
        verify(tmp_path)


def test_verifier_rejects_corrupted_prompt_scoring_slice(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    learnability = next(event for event in events if event["event"] == "learnability")
    learnability["prompt_logprob_start"] -= 1
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="corrupted prompt-scoring slice"):
        verify(tmp_path)


def test_verifier_rejects_generated_suffix_for_learnability_rejected_edit(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    critique = next(
        event
        for event in events
        if event["event"] == "critique" and event["parse_reason"] == "valid" and not event["learnability_accepted"]
    )
    critique["generated_continuation_ids"] = [42]
    critique["generated_continuation_log_probs"] = [-0.5]
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="unexpectedly generated a continuation"):
        verify(tmp_path)


@pytest.mark.parametrize("field", ["response_mask_sha256", "old_log_probs_sha256", "rollout_log_probs_sha256"])
def test_verifier_rejects_corrupted_actor_tensor_hash(field: str, tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    actor_batch = next(event for event in events if event["event"] == "actor_batch")
    next(row for row in actor_batch["actor_rows"] if row["kind"] == "continuation")[field] = "0" * 64
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="does not match its source tensors"):
        verify(tmp_path)


def test_verifier_uses_explicit_completed_attempt_and_preserves_incomplete_prior_attempt(tmp_path: Path) -> None:
    _fixture(tmp_path)
    prior = tmp_path / "audit" / "attempt_prior"
    _write_json(
        prior / "attempt.json",
        {"schema_version": 2, "attempt_id": "prior", "starting_global_step": 1},
    )
    _write_jsonl(
        prior / "step_00000001.jsonl",
        [{"schema_version": 2, "attempt_id": "prior", "global_step": 1, "event": "original"}],
    )
    _write_json(tmp_path / "failed.json", {"status": "failed", "invocation_id": "older"})
    assert verify(tmp_path)["audit_attempt_id"] == "fixture"


def test_verifier_rejects_missing_attempt_selection_or_step_completion(tmp_path: Path) -> None:
    _fixture(tmp_path)
    completed = json.loads((tmp_path / "completed.json").read_text(encoding="utf-8"))
    completed.pop("audit_attempt_id")
    _write_json(tmp_path / "completed.json", completed)
    with pytest.raises(ValueError, match="omitted its audit attempt ID"):
        verify(tmp_path)

    _fixture(tmp_path)
    path = _audit_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(path, [event for event in events if event["event"] != "step_complete"])
    with pytest.raises(ValueError, match="audit step is incomplete"):
        verify(tmp_path)


def test_extra_args_contains_no_async_or_critic_training() -> None:
    rendered = _extra_args("/output/evidence")
    assert "critic.enable=false" in rendered
    assert "launch_reward_fn_async=false" in rendered
    assert "actor_rollout_ref.rollout.temperature=1.0" in rendered
    assert "actor_rollout_ref.rollout.repetition_penalty=1.0" in rendered
    assert "enable_positive_compression=true" in rendered
    assert "learnability_logprob_statistic=mean" in rendered
    assert "learnability_threshold_mode=stddev" in rendered
    assert "max_seed_window_stddevs=15.0" in rendered
    assert "learnability_windows_per_rollout" not in rendered
    assert "critique_max_response_length=2560" in rendered
    assert "min_continuation_tokens=128" in rendered
    assert "prompt_logprob_max_inflight_tokens=null" in rendered
    assert "gpu_memory_utilization=0.6" in rendered
