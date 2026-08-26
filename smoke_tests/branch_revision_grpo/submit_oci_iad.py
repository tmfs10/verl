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
"""Dry-run-first cluster launcher for branch-revision GRPO smoke."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from smoke_tests.intermediate_mc_value.topology.submit_oci_iad import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_LAUNCHER,
    DEFAULT_PYTHON,
    TRAIN_DATA,
    VAL_DATA,
    _command_sha256,
    _git_provenance,
    _parse_job_id,
    _replace_ssh_tunnel_host,
    _replace_verl_container,
    _resolve_ssh_hostname,
    _run,
    _sha256,
)

DEFAULT_VERL = Path("/home/siddjain/workspace/verl/verl_branch_revision_grpo")
DEFAULT_REWARD = Path("/home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py")
MODEL_PATH = "/hf_models/Qwen3-1.7B"
SUPPORTED_MODEL_PATHS = {MODEL_PATH, "/hf_models/Qwen3-4B"}


@dataclass(frozen=True)
class SmokeClusterProfile:
    """Cluster-specific paths and scheduler policy for the shared smoke."""

    cluster_name: str
    config_filename: str
    ssh_alias: str
    remote_output_root: PurePosixPath
    verl_container: str
    replace_source_container: bool
    supported_model_paths: tuple[str, ...] = tuple(sorted(SUPPORTED_MODEL_PATHS))
    default_model_path: str = MODEL_PATH
    allowed_partitions: tuple[str, ...] = ("interactive",)
    max_interactive_nodes: int = 2


OCI_IAD_PROFILE = SmokeClusterProfile(
    cluster_name="oci-iad",
    config_filename="oci-iad.yaml",
    ssh_alias="iad-2",
    remote_output_root=PurePosixPath("/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output"),
    verl_container="/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-verl-0.7.0.sqsh",
    replace_source_container=True,
)


def _prepare_execution_config(profile: SmokeClusterProfile, source_dir: Path, local_run_dir: Path) -> Path:
    source = source_dir.expanduser().resolve() / profile.config_filename
    if not source.is_file():
        raise FileNotFoundError(f"missing authoritative {profile.cluster_name} config: {source}")
    target_host = _resolve_ssh_hostname(profile.ssh_alias)
    updated, original_host = _replace_ssh_tunnel_host(source.read_text(encoding="utf-8"), target_host)
    updated, original_container = _replace_verl_container(updated, profile.verl_container)
    if not profile.replace_source_container and original_container != profile.verl_container:
        raise ValueError(
            f"{profile.cluster_name} source VeRL container changed: "
            f"{original_container!r} != {profile.verl_container!r}"
        )
    destination_dir = local_run_dir / "cluster_config"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / profile.config_filename
    destination.write_text(updated, encoding="utf-8")
    (destination_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": _sha256(source),
                "source_host": original_host,
                "ssh_alias": profile.ssh_alias,
                "resolved_host": target_host,
                "source_verl_container": original_container,
                "execution_verl_container": profile.verl_container,
                "execution_config_sha256": _sha256(destination),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination_dir


def _remote_host_output_path(profile: SmokeClusterProfile, container_output: str) -> PurePosixPath:
    path = PurePosixPath(container_output)
    try:
        relative = path.relative_to("/output")
    except ValueError as exc:
        raise ValueError(f"remote output must be under /output: {container_output}") from exc
    return profile.remote_output_root / relative


def _ssh(profile: SmokeClusterProfile, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", profile.ssh_alias, command],
        capture_output=True,
        text=True,
        check=False,
    )


def _force_no_requeue(profile: SmokeClusterProfile, job_id: str) -> str:
    result = _ssh(
        profile,
        f"scontrol update JobId={shlex.quote(job_id)} Requeue=0 && scontrol show job -o {shlex.quote(job_id)}",
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"could not force Requeue=0 for job {job_id}")
    if "Requeue=0" not in result.stdout:
        raise RuntimeError(f"scheduler did not record Requeue=0 for job {job_id}: {result.stdout.strip()}")
    return result.stdout.strip()


def _extra_args(
    remote_evidence: str,
    *,
    n_prompts: int = 8,
    n_samples: int = 4,
    num_critiques: int = 4,
    ppo_mini_batch_size: int = 8,
    critique_grpo_grouping: str = "per_original",
    critique_advantage_mode: str = "grpo",
    critique_prompt_weighting: str = "equal_prompt",
    recovery_reference_mode: str = "successful_original",
    recovery_reference_selection_seed: int = 0,
    enable_recovery: bool = True,
    enable_positive_compression: bool = True,
    model_path: str = MODEL_PATH,
    loss_mode: str = "dppo_tv",
    learnability_logprob_statistic: str = "mean",
    learnability_threshold_mode: str = "stddev",
    max_seed_window_stddevs: float = 15.0,
    nodes: int = 1,
    max_prompt_length: int = 1024,
    max_response_length: int = 2048,
    max_model_len: int = 8192,
    critique_max_response_length: int = 2560,
    max_tokens_per_gpu: int = 8192,
    prompt_logprob_max_inflight_tokens: int | None = None,
    gpu_memory_utilization: float = 0.6,
    training_steps: int = 1,
    separate_critique_model: bool = False,
    critique_warmup_steps: int = 0,
    critique_model_nnodes: int = 1,
    critique_model_n_gpus_per_node: int = 8,
    resume_from_path: str | None = None,
    expected_resume_step: int = 0,
) -> str:
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError("loss_mode must be dppo_tv or vanilla")
    if critique_grpo_grouping not in {"per_original", "batch"}:
        raise ValueError("critique_grpo_grouping must be per_original or batch")
    if critique_advantage_mode not in {"grpo", "pass_at_1"}:
        raise ValueError("critique_advantage_mode must be grpo or pass_at_1")
    if critique_prompt_weighting not in {"equal_prompt", "headroom"}:
        raise ValueError("critique_prompt_weighting must be equal_prompt or headroom")
    if recovery_reference_mode not in {"none", "successful_original"}:
        raise ValueError("recovery_reference_mode must be none or successful_original")
    if (
        not isinstance(recovery_reference_selection_seed, int)
        or isinstance(recovery_reference_selection_seed, bool)
        or recovery_reference_selection_seed < 0
    ):
        raise ValueError("recovery_reference_selection_seed must be a nonnegative integer")
    if not isinstance(enable_recovery, bool):
        raise ValueError("enable_recovery must be boolean")
    if not enable_recovery and not enable_positive_compression:
        raise ValueError("recovery, positive compression, or both must be enabled")
    if not enable_recovery and recovery_reference_mode != "none":
        raise ValueError("recovery_reference_mode must be none when recovery is disabled")
    if critique_advantage_mode == "pass_at_1" and enable_positive_compression:
        raise ValueError("pass_at_1 critique advantages require recovery-only generation")
    if not isinstance(enable_positive_compression, bool):
        raise ValueError("enable_positive_compression must be boolean")
    if learnability_logprob_statistic not in {"mean", "min"}:
        raise ValueError("learnability_logprob_statistic must be mean or min")
    if learnability_threshold_mode not in {"stddev", "percentile"}:
        raise ValueError("learnability_threshold_mode must be stddev or percentile")
    if not math.isfinite(max_seed_window_stddevs) or max_seed_window_stddevs < 0.0:
        raise ValueError("max_seed_window_stddevs must be finite and nonnegative")
    if not math.isfinite(gpu_memory_utilization) or not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be finite and inside (0, 1)")
    resume_mode = "resume_path" if resume_from_path is not None else "disable"
    actor_nodes = nodes - critique_model_nnodes if separate_critique_model else nodes
    overrides = [
        "~critic.append_solution_to_prompt",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.intermediate_mc_value.enable=false",
        "algorithm.branch_revision_grpo.enable=true",
        f"algorithm.branch_revision_grpo.separate_critique_model={str(separate_critique_model).lower()}",
        f"algorithm.branch_revision_grpo.critique_warmup_steps={critique_warmup_steps}",
        f"algorithm.branch_revision_grpo.critique_model_nnodes={critique_model_nnodes}",
        f"algorithm.branch_revision_grpo.critique_model_n_gpus_per_node={critique_model_n_gpus_per_node}",
        f"algorithm.branch_revision_grpo.num_critiques={num_critiques}",
        f"algorithm.branch_revision_grpo.critique_grpo_grouping={critique_grpo_grouping}",
        f"algorithm.branch_revision_grpo.critique_advantage_mode={critique_advantage_mode}",
        f"algorithm.branch_revision_grpo.critique_prompt_weighting={critique_prompt_weighting}",
        f"algorithm.branch_revision_grpo.recovery_reference_mode={recovery_reference_mode}",
        f"algorithm.branch_revision_grpo.recovery_reference_selection_seed={recovery_reference_selection_seed}",
        f"algorithm.branch_revision_grpo.enable_recovery={str(enable_recovery).lower()}",
        "algorithm.branch_revision_grpo.critique_invalid_penalty=0.20",
        "algorithm.branch_revision_grpo.critique_learnability_rejection_penalty=0.05",
        "algorithm.branch_revision_grpo.critique_advantage_rms_floor=0.10",
        "algorithm.branch_revision_grpo.critique_advantage_clip=5.0",
        "algorithm.branch_revision_grpo.critique_prompt_headroom_exponent=1.0",
        f"algorithm.branch_revision_grpo.enable_positive_compression={str(enable_positive_compression).lower()}",
        f"algorithm.branch_revision_grpo.num_positive_critiques={num_critiques}",
        "algorithm.branch_revision_grpo.positive_compression_target=0.25",
        f"algorithm.branch_revision_grpo.learnability_logprob_statistic={learnability_logprob_statistic}",
        f"algorithm.branch_revision_grpo.learnability_threshold_mode={learnability_threshold_mode}",
        f"algorithm.branch_revision_grpo.max_seed_window_stddevs={max_seed_window_stddevs}",
        "algorithm.branch_revision_grpo.min_seed_window_percentile=0.20",
        "algorithm.branch_revision_grpo.full_credit_seed_window_percentile=0.50",
        f"algorithm.branch_revision_grpo.critique_max_response_length={critique_max_response_length}",
        "algorithm.branch_revision_grpo.branch_max_tokens=128",
        "algorithm.branch_revision_grpo.new_continuation_max_tokens=256",
        "algorithm.branch_revision_grpo.min_continuation_tokens=128",
        f"algorithm.branch_revision_grpo.audit_output_dir={remote_evidence}/audit",
        f"data.train_batch_size={n_prompts}",
        f"++data.gen_batch_size={n_prompts}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=true",
        "data.filter_overlong_prompts_workers=8",
        "data.dataloader_num_workers=0",
        "data.truncation=error",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_tokens_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={n_samples}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.repetition_penalty=1.0",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_tokens_per_gpu}",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.prompt_logprob_max_inflight_tokens="
        f"{'null' if prompt_logprob_max_inflight_tokens is None else prompt_logprob_max_inflight_tokens}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.enable_chunked_prefill=true",
        "actor_rollout_ref.rollout.enable_prefix_caching=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "critic.enable=false",
        "reward.reward_model.enable=false",
        "++reward.reward_model.launch_reward_fn_async=false",
        "trainer.use_legacy_worker_impl=enable",
        f"trainer.nnodes={actor_nodes}",
        "trainer.critic_warmup=0",
        "trainer.logger=[file]",
        "trainer.project_name=branch_revision_grpo_smoke",
        f"trainer.experiment_name=branch_revision_{loss_mode}_{learnability_logprob_statistic}_{learnability_threshold_mode}",
        f"trainer.default_local_dir={remote_evidence}/../checkpoints",
        f"trainer.total_training_steps={training_steps}",
        "trainer.total_epochs=1",
        "trainer.val_before_train=false",
        f"trainer.save_freq={training_steps if separate_critique_model else -1}",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        "trainer.rollout_data_dir=null",
        "trainer.validation_data_dir=null",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.resume_from_path={'null' if resume_from_path is None else resume_from_path}",
        f"trainer.expected_resume_step={expected_resume_step}",
        "trainer.load_dataloader_state_on_resume=true",
        "trainer.balance_batch=true",
        f"+branch_revision_smoke.output_dir={remote_evidence}",
        f"+branch_revision_smoke.n_prompts={n_prompts}",
        f"+branch_revision_smoke.n_samples={n_samples}",
        f"+branch_revision_smoke.num_critiques={num_critiques}",
        f"+branch_revision_smoke.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"+branch_revision_smoke.critique_grpo_grouping={critique_grpo_grouping}",
        f"+branch_revision_smoke.critique_advantage_mode={critique_advantage_mode}",
        f"+branch_revision_smoke.critique_prompt_weighting={critique_prompt_weighting}",
        f"+branch_revision_smoke.recovery_reference_mode={recovery_reference_mode}",
        f"+branch_revision_smoke.recovery_reference_selection_seed={recovery_reference_selection_seed}",
        f"+branch_revision_smoke.enable_recovery={str(enable_recovery).lower()}",
        f"+branch_revision_smoke.enable_positive_compression={str(enable_positive_compression).lower()}",
        f"+branch_revision_smoke.model_path={model_path}",
        f"+branch_revision_smoke.loss_mode={loss_mode}",
        f"+branch_revision_smoke.learnability_logprob_statistic={learnability_logprob_statistic}",
        f"+branch_revision_smoke.learnability_threshold_mode={learnability_threshold_mode}",
        f"+branch_revision_smoke.max_seed_window_stddevs={max_seed_window_stddevs}",
        f"+branch_revision_smoke.nodes={nodes}",
        f"+branch_revision_smoke.max_prompt_length={max_prompt_length}",
        f"+branch_revision_smoke.max_response_length={max_response_length}",
        f"+branch_revision_smoke.max_model_len={max_model_len}",
        f"+branch_revision_smoke.critique_max_response_length={critique_max_response_length}",
        f"+branch_revision_smoke.max_tokens_per_gpu={max_tokens_per_gpu}",
        "+branch_revision_smoke.prompt_logprob_max_inflight_tokens="
        f"{'null' if prompt_logprob_max_inflight_tokens is None else prompt_logprob_max_inflight_tokens}",
        f"+branch_revision_smoke.gpu_memory_utilization={gpu_memory_utilization}",
        f"+branch_revision_smoke.training_steps={training_steps}",
        f"+branch_revision_smoke.separate_critique_model={str(separate_critique_model).lower()}",
        f"+branch_revision_smoke.critique_warmup_steps={critique_warmup_steps}",
        f"+branch_revision_smoke.critique_model_nnodes={critique_model_nnodes}",
        f"+branch_revision_smoke.critique_model_n_gpus_per_node={critique_model_n_gpus_per_node}",
        f"+branch_revision_smoke.resume_mode={resume_mode}",
        f"+branch_revision_smoke.resume_from_path={'null' if resume_from_path is None else resume_from_path}",
        f"+branch_revision_smoke.expected_resume_step={expected_resume_step}",
    ]
    return " ".join(overrides)


def build_command(
    *,
    profile: SmokeClusterProfile = OCI_IAD_PROFILE,
    run_tag: str,
    dry_run: bool,
    python: Path,
    launcher: Path,
    verl_root: Path,
    reward_file: Path,
    config_dir: Path,
    n_prompts: int = 8,
    n_samples: int = 4,
    num_critiques: int = 4,
    ppo_mini_batch_size: int = 8,
    critique_grpo_grouping: str = "per_original",
    critique_advantage_mode: str = "grpo",
    critique_prompt_weighting: str = "equal_prompt",
    recovery_reference_mode: str = "successful_original",
    recovery_reference_selection_seed: int = 0,
    enable_recovery: bool = True,
    enable_positive_compression: bool = True,
    seed: int = 43,
    model_path: str = MODEL_PATH,
    loss_mode: str = "dppo_tv",
    learnability_logprob_statistic: str = "mean",
    learnability_threshold_mode: str = "stddev",
    max_seed_window_stddevs: float = 15.0,
    nodes: int = 1,
    max_prompt_length: int = 1024,
    max_response_length: int = 2048,
    max_model_len: int = 8192,
    critique_max_response_length: int = 2560,
    max_tokens_per_gpu: int = 8192,
    prompt_logprob_max_inflight_tokens: int | None = None,
    gpu_memory_utilization: float = 0.6,
    training_steps: int = 1,
    partition: str | None = None,
    separate_critique_model: bool = False,
    critique_warmup_steps: int = 0,
    critique_model_nnodes: int = 1,
    critique_model_n_gpus_per_node: int = 8,
    resume_from_path: str | None = None,
    expected_resume_step: int = 0,
) -> tuple[list[str], str]:
    positive = {
        "n_prompts": n_prompts,
        "n_samples": n_samples,
        "num_critiques": num_critiques,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "nodes": nodes,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "max_model_len": max_model_len,
        "critique_max_response_length": critique_max_response_length,
        "max_tokens_per_gpu": max_tokens_per_gpu,
        "training_steps": training_steps,
        "critique_model_nnodes": critique_model_nnodes,
        "critique_model_n_gpus_per_node": critique_model_n_gpus_per_node,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if prompt_logprob_max_inflight_tokens is not None and (
        isinstance(prompt_logprob_max_inflight_tokens, bool)
        or not isinstance(prompt_logprob_max_inflight_tokens, int)
        or prompt_logprob_max_inflight_tokens <= 0
    ):
        raise ValueError("prompt_logprob_max_inflight_tokens must be null or a positive integer")
    if not isinstance(separate_critique_model, bool):
        raise ValueError("separate_critique_model must be boolean")
    if critique_grpo_grouping not in {"per_original", "batch"}:
        raise ValueError("critique_grpo_grouping must be per_original or batch")
    if critique_advantage_mode not in {"grpo", "pass_at_1"}:
        raise ValueError("critique_advantage_mode must be grpo or pass_at_1")
    if critique_prompt_weighting not in {"equal_prompt", "headroom"}:
        raise ValueError("critique_prompt_weighting must be equal_prompt or headroom")
    if recovery_reference_mode not in {"none", "successful_original"}:
        raise ValueError("recovery_reference_mode must be none or successful_original")
    if (
        not isinstance(recovery_reference_selection_seed, int)
        or isinstance(recovery_reference_selection_seed, bool)
        or recovery_reference_selection_seed < 0
    ):
        raise ValueError("recovery_reference_selection_seed must be a nonnegative integer")
    if not isinstance(enable_recovery, bool):
        raise ValueError("enable_recovery must be boolean")
    if not enable_recovery and not enable_positive_compression:
        raise ValueError("recovery, positive compression, or both must be enabled")
    if not enable_recovery and recovery_reference_mode != "none":
        raise ValueError("recovery_reference_mode must be none when recovery is disabled")
    if critique_advantage_mode == "pass_at_1" and enable_positive_compression:
        raise ValueError("pass_at_1 critique advantages require recovery-only generation")
    if not isinstance(enable_positive_compression, bool):
        raise ValueError("enable_positive_compression must be boolean")
    if (
        not isinstance(critique_warmup_steps, int)
        or isinstance(critique_warmup_steps, bool)
        or critique_warmup_steps < 0
    ):
        raise ValueError("critique_warmup_steps must be a nonnegative integer")
    if separate_critique_model and critique_model_nnodes >= nodes:
        raise ValueError("separate critique policy requires at least one actor node and one critique node")
    actor_nodes = nodes - critique_model_nnodes if separate_critique_model else nodes
    if n_samples < 2 and not (
        separate_critique_model and training_steps > 0 and training_steps <= critique_warmup_steps
    ):
        raise ValueError("n_samples=1 requires a separate critique-policy warmup-only smoke")
    if num_critiques < 2:
        raise ValueError("num_critiques must be at least 2 for GRPO")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(expected_resume_step, int) or isinstance(expected_resume_step, bool) or expected_resume_step < 0:
        raise ValueError("expected_resume_step must be a nonnegative integer")
    if resume_from_path is None:
        if expected_resume_step != 0:
            raise ValueError("a fresh smoke must use expected_resume_step=0")
    else:
        resume_path = PurePosixPath(resume_from_path)
        if not resume_path.is_absolute() or not str(resume_path).startswith("/output/"):
            raise ValueError("resume_from_path must be an absolute mounted /output path")
        if resume_path.name != f"global_step_{expected_resume_step}" or expected_resume_step <= 0:
            raise ValueError("resume_from_path must end in global_step_<expected_resume_step>")
        if training_steps <= expected_resume_step:
            raise ValueError("resumed smoke total training steps must exceed expected_resume_step")
    if model_path not in profile.supported_model_paths:
        raise ValueError(f"model_path must be one of {sorted(profile.supported_model_paths)!r}")
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError("loss_mode must be dppo_tv or vanilla")
    if learnability_logprob_statistic not in {"mean", "min"}:
        raise ValueError("learnability_logprob_statistic must be mean or min")
    if learnability_threshold_mode not in {"stddev", "percentile"}:
        raise ValueError("learnability_threshold_mode must be stddev or percentile")
    if not math.isfinite(max_seed_window_stddevs) or max_seed_window_stddevs < 0.0:
        raise ValueError("max_seed_window_stddevs must be finite and nonnegative")
    if not math.isfinite(gpu_memory_utilization) or not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be finite and inside (0, 1)")
    if nodes > 4:
        raise ValueError("branch-revision smoke supports at most four nodes")
    if partition is not None and partition not in profile.allowed_partitions:
        if profile.allowed_partitions == ("interactive",):
            raise ValueError(
                f"{profile.cluster_name} branch-revision smoke partition must be interactive when explicitly selected"
            )
        allowed = ", ".join(profile.allowed_partitions)
        raise ValueError(f"{profile.cluster_name} branch-revision smoke partition must be one of {allowed}")
    if partition == "interactive" and nodes > profile.max_interactive_nodes:
        rendered_limit = "two" if profile.max_interactive_nodes == 2 else str(profile.max_interactive_nodes)
        raise ValueError(
            f"{profile.cluster_name} interactive branch-revision smoke supports at most {rendered_limit} nodes"
        )
    if max_prompt_length + max_response_length >= max_model_len:
        raise ValueError("max_prompt_length + max_response_length must be smaller than max_model_len")
    if critique_max_response_length >= max_model_len:
        raise ValueError("critique_max_response_length must be smaller than max_model_len")
    if max_tokens_per_gpu < max_prompt_length + max_response_length:
        raise ValueError("max_tokens_per_gpu must fit one maximum-length original sequence")
    remote_output = f"/output/smoke_tests/branch_revision_grpo/{run_tag}"
    remote_evidence = f"{remote_output}/evidence"
    command = [
        str(python),
        str(launcher),
        "--cluster",
        profile.cluster_name,
        "--config_dir",
        str(config_dir),
        "--explicit_output_dir",
        remote_output,
        "--output_base_dir",
        "/output/smoke_tests/branch_revision_grpo",
        "--local_verl_folder",
        str(verl_root),
        "--script_module",
        "smoke_tests.branch_revision_grpo.main_ppo_smoke",
        "--reward_file",
        str(reward_file),
        "--ground_truth_solution_key",
        "solution",
        "--expname",
        "branch-revision-grpo-smoke",
        "--time_limit",
        "02:00:00",
        "--nodes",
        str(nodes),
        "--trainer_nodes",
        str(actor_nodes),
        "--gpus",
        "8",
        "--actor_model",
        model_path,
        "--critic_model",
        model_path,
        "--prompt_data",
        TRAIN_DATA,
        "--eval_data",
        VAL_DATA,
        "--n_prompts",
        str(n_prompts),
        "--n_samples",
        str(n_samples),
        "--n_val_samples",
        "1",
        "--val_batch_size",
        "8",
        "--max_prompt_len",
        str(max_prompt_length),
        "--max_len",
        str(max_prompt_length + max_response_length),
        "--max_tokens_per_gpu",
        str(max_tokens_per_gpu),
        "--num_epochs",
        "1",
        "--num_training_jobs",
        "1",
        "--num_ppo_iter",
        "1",
        "--actor_lr",
        "2e-6",
        "--clip_ae",
        "0.2,0.2",
        "--infer_server",
        "vllm",
        "--sequence_parallel_size",
        "1",
        "--T",
        "1.0",
        "--val_T",
        "1.0",
        "--val_top_p",
        "1.0",
        "--save_freq",
        "-1",
        "--test_freq",
        "-1",
        "--ae",
        "grpo",
        "--seed",
        str(seed),
        "--no_sandbox",
        "--no_requeue",
        "--disable_val_before_train",
        "--omit_noncore_algorithm_overrides",
        "--skip_runtime_package_install",
        "--extra_args",
        _extra_args(
            remote_evidence,
            n_prompts=n_prompts,
            n_samples=n_samples,
            num_critiques=num_critiques,
            ppo_mini_batch_size=ppo_mini_batch_size,
            critique_grpo_grouping=critique_grpo_grouping,
            critique_advantage_mode=critique_advantage_mode,
            critique_prompt_weighting=critique_prompt_weighting,
            recovery_reference_mode=recovery_reference_mode,
            recovery_reference_selection_seed=recovery_reference_selection_seed,
            enable_recovery=enable_recovery,
            enable_positive_compression=enable_positive_compression,
            model_path=model_path,
            loss_mode=loss_mode,
            learnability_logprob_statistic=learnability_logprob_statistic,
            learnability_threshold_mode=learnability_threshold_mode,
            max_seed_window_stddevs=max_seed_window_stddevs,
            nodes=nodes,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_model_len=max_model_len,
            critique_max_response_length=critique_max_response_length,
            max_tokens_per_gpu=max_tokens_per_gpu,
            prompt_logprob_max_inflight_tokens=prompt_logprob_max_inflight_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            training_steps=training_steps,
            separate_critique_model=separate_critique_model,
            critique_warmup_steps=critique_warmup_steps,
            critique_model_nnodes=critique_model_nnodes,
            critique_model_n_gpus_per_node=critique_model_n_gpus_per_node,
            resume_from_path=resume_from_path,
            expected_resume_step=expected_resume_step,
        ),
    ]
    if nodes <= 2:
        command.append("--add_interactive")
    if partition is not None:
        command.extend(["--partition", partition])
    if dry_run:
        command.append("--dry_run")
    return command, remote_output


def _job_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"no submitted smoke job record: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not str(value.get("job_id", "")).isdigit():
        raise ValueError(f"invalid smoke job record: {value!r}")
    return value


def main(profile: SmokeClusterProfile = OCI_IAD_PROFILE) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "submit", "status", "collect", "verify", "verify-integrity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--verl-root", type=Path, default=DEFAULT_VERL)
    parser.add_argument("--reward-file", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--num-critiques", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=8)
    parser.add_argument("--critique-grpo-grouping", choices=("per_original", "batch"), default="per_original")
    parser.add_argument("--critique-advantage-mode", choices=("grpo", "pass_at_1"), default="grpo")
    parser.add_argument("--critique-prompt-weighting", choices=("equal_prompt", "headroom"), default="equal_prompt")
    parser.add_argument(
        "--recovery-reference-mode",
        choices=("none", "successful_original"),
        default="successful_original",
    )
    parser.add_argument("--recovery-reference-selection-seed", type=int, default=0)
    parser.add_argument("--disable-recovery", action="store_true")
    parser.add_argument("--disable-positive-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--model-path",
        choices=sorted(profile.supported_model_paths),
        default=profile.default_model_path,
    )
    parser.add_argument("--loss-mode", choices=("dppo_tv", "vanilla"), default="dppo_tv")
    parser.add_argument("--learnability-logprob-statistic", choices=("mean", "min"), default="mean")
    parser.add_argument("--learnability-threshold-mode", choices=("stddev", "percentile"), default="stddev")
    parser.add_argument("--max-seed-window-stddevs", type=float, default=15.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--critique-max-response-length", type=int, default=2560)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=8192)
    admission_group = parser.add_mutually_exclusive_group()
    admission_group.add_argument(
        "--prompt-logprob-max-inflight-tokens",
        type=int,
        default=None,
        help="Optionally enable a per-server weighted prompt-logprob token cap; uncapped scoring is the default.",
    )
    admission_group.add_argument(
        "--disable-prompt-logprob-admission",
        action="store_true",
        help="Retained for command compatibility; uncapped prompt-logprob scoring is already the default.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--separate-critique-model", action="store_true")
    parser.add_argument("--critique-warmup-steps", type=int, default=0)
    parser.add_argument("--critique-model-nnodes", type=int, default=1)
    parser.add_argument("--critique-model-n-gpus-per-node", type=int, default=8)
    parser.add_argument("--resume-from-path")
    parser.add_argument("--expected-resume-step", type=int, default=0)
    parser.add_argument("--partition", choices=("interactive",))
    args = parser.parse_args()

    local_run_dir = args.local_run_dir.expanduser().resolve()
    repo_root = args.verl_root.expanduser().resolve()
    if local_run_dir == repo_root or repo_root in local_run_dir.parents:
        raise ValueError("runtime smoke artifacts must be outside the VeRL workspace")
    local_run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = local_run_dir / "dry_run.ok.json"
    job_path = local_run_dir / "job.json"

    if args.action in {"status", "collect"}:
        record = _job_record(job_path)
        job_id = record["job_id"]
        if args.action == "status":
            result = _ssh(
                profile,
                f"sacct -X -j {shlex.quote(job_id)} -n -P "
                "-o JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList | head -1",
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or f"could not query Slurm job {job_id}")
            print(result.stdout.strip() or "not-visible")
            return
        destination = local_run_dir / "collected"
        destination.mkdir(parents=True, exist_ok=True)
        source = _remote_host_output_path(profile, record["remote_output"]) / "evidence"
        result = subprocess.run(["rsync", "-a", f"{profile.ssh_alias}:{source}/", f"{destination}/"], check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
        print(json.dumps({"status": "collected", "source": str(source), "destination": str(destination)}))
        return

    if args.action in {"verify", "verify-integrity"}:
        from smoke_tests.branch_revision_grpo.verify_smoke import verify

        integrity_only = args.action == "verify-integrity"
        result = verify(local_run_dir / "collected", require_algorithm_signal=not integrity_only)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        output_name = "integrity_verified.json" if integrity_only else "verified.json"
        (local_run_dir / output_name).write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return

    execution_config_dir = _prepare_execution_config(profile, args.config_dir, local_run_dir)
    prompt_logprob_max_inflight_tokens = (
        None if args.disable_prompt_logprob_admission else args.prompt_logprob_max_inflight_tokens
    )
    submit_command, remote_output = build_command(
        profile=profile,
        run_tag=args.run_tag,
        dry_run=False,
        python=args.python,
        launcher=args.launcher,
        verl_root=repo_root,
        reward_file=args.reward_file,
        config_dir=execution_config_dir,
        n_prompts=args.n_prompts,
        n_samples=args.n_samples,
        num_critiques=args.num_critiques,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        critique_grpo_grouping=args.critique_grpo_grouping,
        critique_advantage_mode=args.critique_advantage_mode,
        critique_prompt_weighting=args.critique_prompt_weighting,
        recovery_reference_mode=args.recovery_reference_mode,
        recovery_reference_selection_seed=args.recovery_reference_selection_seed,
        enable_recovery=not args.disable_recovery,
        enable_positive_compression=not args.disable_positive_compression,
        seed=args.seed,
        model_path=args.model_path,
        loss_mode=args.loss_mode,
        learnability_logprob_statistic=args.learnability_logprob_statistic,
        learnability_threshold_mode=args.learnability_threshold_mode,
        max_seed_window_stddevs=args.max_seed_window_stddevs,
        nodes=args.nodes,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_model_len,
        critique_max_response_length=args.critique_max_response_length,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        prompt_logprob_max_inflight_tokens=prompt_logprob_max_inflight_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        training_steps=args.training_steps,
        partition=args.partition,
        separate_critique_model=args.separate_critique_model,
        critique_warmup_steps=args.critique_warmup_steps,
        critique_model_nnodes=args.critique_model_nnodes,
        critique_model_n_gpus_per_node=args.critique_model_n_gpus_per_node,
        resume_from_path=args.resume_from_path,
        expected_resume_step=args.expected_resume_step,
    )
    git = _git_provenance(repo_root)
    provenance = {
        "run_tag": args.run_tag,
        "git": git,
        "launcher_sha256": _sha256(args.launcher),
        "execution_config_sha256": _sha256(execution_config_dir / profile.config_filename),
        "submit_command_sha256": _command_sha256(submit_command),
        "remote_output": remote_output,
        "cluster_name": profile.cluster_name,
        "ssh_alias": profile.ssh_alias,
        "remote_output_root": str(profile.remote_output_root),
    }

    if args.action == "dry-run":
        dry_command, _ = build_command(
            profile=profile,
            run_tag=args.run_tag,
            dry_run=True,
            python=args.python,
            launcher=args.launcher,
            verl_root=repo_root,
            reward_file=args.reward_file,
            config_dir=execution_config_dir,
            n_prompts=args.n_prompts,
            n_samples=args.n_samples,
            num_critiques=args.num_critiques,
            ppo_mini_batch_size=args.ppo_mini_batch_size,
            critique_grpo_grouping=args.critique_grpo_grouping,
            critique_advantage_mode=args.critique_advantage_mode,
            critique_prompt_weighting=args.critique_prompt_weighting,
            recovery_reference_mode=args.recovery_reference_mode,
            recovery_reference_selection_seed=args.recovery_reference_selection_seed,
            enable_recovery=not args.disable_recovery,
            enable_positive_compression=not args.disable_positive_compression,
            seed=args.seed,
            model_path=args.model_path,
            loss_mode=args.loss_mode,
            learnability_logprob_statistic=args.learnability_logprob_statistic,
            learnability_threshold_mode=args.learnability_threshold_mode,
            max_seed_window_stddevs=args.max_seed_window_stddevs,
            nodes=args.nodes,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            max_model_len=args.max_model_len,
            critique_max_response_length=args.critique_max_response_length,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            prompt_logprob_max_inflight_tokens=prompt_logprob_max_inflight_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            training_steps=args.training_steps,
            partition=args.partition,
            separate_critique_model=args.separate_critique_model,
            critique_warmup_steps=args.critique_warmup_steps,
            critique_model_nnodes=args.critique_model_nnodes,
            critique_model_n_gpus_per_node=args.critique_model_n_gpus_per_node,
            resume_from_path=args.resume_from_path,
            expected_resume_step=args.expected_resume_step,
        )
        result = _run(dry_command, local_run_dir / "dry_run.log")
        if result.returncode:
            raise SystemExit(result.returncode)
        marker_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "dry-run-complete", "command_sha256": provenance["submit_command_sha256"]}))
        return

    if not marker_path.is_file():
        raise ValueError(f"missing successful dry-run marker: {marker_path}")
    prior = json.loads(marker_path.read_text(encoding="utf-8"))
    if prior != provenance:
        raise ValueError("submission inputs changed after the successful dry-run")
    if job_path.exists():
        raise ValueError(f"smoke job is already recorded in {job_path}")
    result = _run(submit_command, local_run_dir / "submit.log")
    if result.returncode:
        raise SystemExit(result.returncode)
    job_id = _parse_job_id(local_run_dir / "submit.log")
    scheduler_contract = _force_no_requeue(profile, job_id)
    (local_run_dir / "scheduler_contract.txt").write_text(scheduler_contract + "\n", encoding="utf-8")
    job_path.write_text(
        json.dumps({"job_id": job_id, "remote_output": remote_output}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "submitted", "job_id": job_id, "remote_output": remote_output}))


if __name__ == "__main__":
    main()
