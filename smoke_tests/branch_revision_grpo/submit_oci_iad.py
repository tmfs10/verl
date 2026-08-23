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
"""Dry-run-first OCI-IAD launcher for branch-revision GRPO smoke."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path

from smoke_tests.intermediate_mc_value.topology.submit_oci_iad import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_LAUNCHER,
    DEFAULT_PYTHON,
    REMOTE_OUTPUT_ROOT,
    SSH_ALIAS,
    TRAIN_DATA,
    VAL_DATA,
    _command_sha256,
    _force_no_requeue,
    _git_provenance,
    _parse_job_id,
    _prepare_execution_config,
    _remote_host_output_path,
    _run,
    _sha256,
    _ssh,
)

DEFAULT_VERL = Path("/home/siddjain/workspace/verl/verl_branch_revision_grpo")
DEFAULT_REWARD = Path("/home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py")
MODEL_PATH = "/hf_models/Qwen3-1.7B"
SUPPORTED_MODEL_PATHS = {MODEL_PATH, "/hf_models/Qwen3-4B"}


def _extra_args(
    remote_evidence: str,
    *,
    n_prompts: int = 8,
    n_samples: int = 4,
    num_critiques: int = 4,
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
) -> str:
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError("loss_mode must be dppo_tv or vanilla")
    if learnability_logprob_statistic not in {"mean", "min"}:
        raise ValueError("learnability_logprob_statistic must be mean or min")
    if learnability_threshold_mode not in {"stddev", "percentile"}:
        raise ValueError("learnability_threshold_mode must be stddev or percentile")
    if not math.isfinite(max_seed_window_stddevs) or max_seed_window_stddevs < 0.0:
        raise ValueError("max_seed_window_stddevs must be finite and nonnegative")
    overrides = [
        "~critic.append_solution_to_prompt",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.intermediate_mc_value.enable=false",
        "algorithm.branch_revision_grpo.enable=true",
        f"algorithm.branch_revision_grpo.num_critiques={num_critiques}",
        "algorithm.branch_revision_grpo.enable_positive_compression=true",
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
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
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
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.enable_chunked_prefill=true",
        "actor_rollout_ref.rollout.enable_prefix_caching=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "critic.enable=false",
        "reward.reward_model.enable=false",
        "++reward.reward_model.launch_reward_fn_async=false",
        "trainer.use_legacy_worker_impl=enable",
        "trainer.critic_warmup=0",
        "trainer.logger=[file]",
        "trainer.project_name=branch_revision_grpo_smoke",
        f"trainer.experiment_name=branch_revision_{loss_mode}_{learnability_logprob_statistic}_{learnability_threshold_mode}",
        f"trainer.default_local_dir={remote_evidence}/checkpoints",
        "trainer.total_training_steps=1",
        "trainer.total_epochs=1",
        "trainer.val_before_train=false",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        "trainer.rollout_data_dir=null",
        "trainer.validation_data_dir=null",
        "trainer.resume_mode=disable",
        "trainer.balance_batch=true",
        f"+branch_revision_smoke.output_dir={remote_evidence}",
        f"+branch_revision_smoke.n_prompts={n_prompts}",
        f"+branch_revision_smoke.n_samples={n_samples}",
        f"+branch_revision_smoke.num_critiques={num_critiques}",
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
    ]
    return " ".join(overrides)


def build_command(
    *,
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
) -> tuple[list[str], str]:
    positive = {
        "n_prompts": n_prompts,
        "n_samples": n_samples,
        "num_critiques": num_critiques,
        "nodes": nodes,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "max_model_len": max_model_len,
        "critique_max_response_length": critique_max_response_length,
        "max_tokens_per_gpu": max_tokens_per_gpu,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 for a GRPO acceptance group")
    if num_critiques < 2:
        raise ValueError("num_critiques must be at least 2 for GRPO")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if model_path not in SUPPORTED_MODEL_PATHS:
        raise ValueError(f"model_path must be one of {sorted(SUPPORTED_MODEL_PATHS)!r}")
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError("loss_mode must be dppo_tv or vanilla")
    if learnability_logprob_statistic not in {"mean", "min"}:
        raise ValueError("learnability_logprob_statistic must be mean or min")
    if learnability_threshold_mode not in {"stddev", "percentile"}:
        raise ValueError("learnability_threshold_mode must be stddev or percentile")
    if not math.isfinite(max_seed_window_stddevs) or max_seed_window_stddevs < 0.0:
        raise ValueError("max_seed_window_stddevs must be finite and nonnegative")
    if nodes > 2:
        raise ValueError("this interactive-capable smoke launcher supports at most two nodes")
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
        "oci-iad",
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
        "--add_interactive",
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
        ),
    ]
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


def main() -> None:
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
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--model-path", choices=sorted(SUPPORTED_MODEL_PATHS), default=MODEL_PATH)
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
                f"sacct -X -j {shlex.quote(job_id)} -n -P "
                "-o JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList | head -1"
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or f"could not query Slurm job {job_id}")
            print(result.stdout.strip() or "not-visible")
            return
        destination = local_run_dir / "collected"
        destination.mkdir(parents=True, exist_ok=True)
        source = _remote_host_output_path(record["remote_output"]) / "evidence"
        result = subprocess.run(["rsync", "-a", f"{SSH_ALIAS}:{source}/", f"{destination}/"], check=False)
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

    execution_config_dir = _prepare_execution_config(args.config_dir, local_run_dir)
    submit_command, remote_output = build_command(
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
    )
    git = _git_provenance(repo_root)
    provenance = {
        "run_tag": args.run_tag,
        "git": git,
        "launcher_sha256": _sha256(args.launcher),
        "execution_config_sha256": _sha256(execution_config_dir / "oci-iad.yaml"),
        "submit_command_sha256": _command_sha256(submit_command),
        "remote_output": remote_output,
        "ssh_alias": SSH_ALIAS,
        "remote_output_root": str(REMOTE_OUTPUT_ROOT),
    }

    if args.action == "dry-run":
        dry_command, _ = build_command(
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
    scheduler_contract = _force_no_requeue(job_id)
    (local_run_dir / "scheduler_contract.txt").write_text(scheduler_contract + "\n", encoding="utf-8")
    job_path.write_text(
        json.dumps({"job_id": job_id, "remote_output": remote_output}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "submitted", "job_id": job_id, "remote_output": remote_output}))


if __name__ == "__main__":
    main()
