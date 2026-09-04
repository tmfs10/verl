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
"""Dry-run-first two-node CW-DFW random-continuation evaluator."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from smoke_tests.branch_revision_grpo.submit_cw_dfw import CW_DFW_PROFILE
from smoke_tests.branch_revision_grpo.submit_oci_iad import (
    _force_no_requeue,
    _prepare_execution_config,
    _remote_host_output_path,
    _ssh,
)
from smoke_tests.intermediate_mc_value.topology.submit_oci_iad import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_LAUNCHER,
    DEFAULT_PYTHON,
    TRAIN_DATA,
    VAL_DATA,
    _command_sha256,
    _git_provenance,
    _parse_job_id,
    _run,
    _sha256,
)

DEFAULT_VERL = Path("/home/siddjain/workspace/verl/verl_branch_revision_grpo")
DEFAULT_REWARD = Path("/home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py")
MODEL_PATH = "/hf_models/Qwen/Qwen3-4B"


def _extra_args(
    remote_output: str,
    *,
    prompts: int,
    rollouts: int,
    points: int,
    continuations: int,
    seed: int,
) -> str:
    world_size = 16
    ppo_mini_batch_size = max(prompts, (world_size + rollouts - 1) // rollouts)
    overrides = [
        "~critic.append_solution_to_prompt",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.intermediate_mc_value.enable=false",
        "algorithm.branch_revision_grpo.enable=false",
        "algorithm.random_continuation_baseline.enable=true",
        f"algorithm.random_continuation_baseline.points_per_rollout={points}",
        f"algorithm.random_continuation_baseline.continuations_per_mark={continuations}",
        "algorithm.random_continuation_baseline.min_prefix_fraction=0.10",
        "algorithm.random_continuation_baseline.min_continuation_tokens=128",
        "algorithm.random_continuation_baseline.structural_boundaries_only=true",
        f"algorithm.random_continuation_baseline.selection_seed={seed}",
        "algorithm.random_continuation_baseline.bootstrap_samples=10000",
        f"algorithm.random_continuation_baseline.audit_output_dir={remote_output}/audit",
        f"data.train_batch_size={prompts}",
        f"++data.gen_batch_size={prompts}",
        "data.max_prompt_length=2048",
        "data.max_response_length=8192",
        "data.filter_overlong_prompts=true",
        "data.filter_overlong_prompts_workers=16",
        "data.dataloader_num_workers=0",
        "data.truncation=error",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={rollouts}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "+actor_rollout_ref.rollout.repetition_penalty=1.0",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.max_model_len=32768",
        "actor_rollout_ref.rollout.max_num_batched_tokens=32768",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.enable_chunked_prefill=true",
        "actor_rollout_ref.rollout.enable_prefix_caching=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "critic.enable=false",
        "reward.reward_model.enable=false",
        "++reward.reward_model.launch_reward_fn_async=false",
        "trainer.use_legacy_worker_impl=enable",
        "trainer.nnodes=2",
        "trainer.n_gpus_per_node=8",
        "trainer.critic_warmup=0",
        "trainer.logger=[file]",
        "trainer.project_name=random_continuation_baseline",
        "trainer.experiment_name=random_continuation_baseline_cw_dfw",
        f"trainer.default_local_dir={remote_output}/checkpoints",
        "trainer.total_training_steps=1",
        "trainer.total_epochs=1",
        "trainer.val_before_train=false",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        "trainer.rollout_data_dir=null",
        "trainer.validation_data_dir=null",
        "trainer.resume_mode=disable",
        "trainer.resume_from_path=null",
        "trainer.balance_batch=false",
        f"+random_continuation_run.output_dir={remote_output}",
        f"+random_continuation_run.model_path={MODEL_PATH}",
        f"+random_continuation_run.n_prompts={prompts}",
        f"+random_continuation_run.rollouts_per_prompt={rollouts}",
        f"+random_continuation_run.points_per_rollout={points}",
        f"+random_continuation_run.continuations_per_mark={continuations}",
        f"+random_continuation_run.seed={seed}",
        "+random_continuation_run.nodes=2",
        "+random_continuation_run.max_prompt_length=2048",
        "+random_continuation_run.max_response_length=8192",
        "+random_continuation_run.max_model_len=32768",
    ]
    return " ".join(overrides)


def build_command(
    *,
    run_tag: str,
    local_config_dir: Path,
    verl_root: Path,
    reward_file: Path,
    prompts: int,
    rollouts: int,
    points: int,
    continuations: int,
    seed: int,
    dry_run: bool,
) -> tuple[list[str], str]:
    cardinalities = {
        "prompts": prompts,
        "rollouts": rollouts,
        "points": points,
        "continuations": continuations,
    }
    for name, value in cardinalities.items():
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # The shared submitter statically requires --n_prompts to be divisible by
    # the 16 trainer ranks. The authoritative batch size is the explicit Hydra
    # override and run contract emitted below; round up only this outer-launcher
    # placeholder so a four-prompt, evaluation-only smoke remains possible.
    trainer_world_size = 16
    launcher_prompts = ((prompts + trainer_world_size - 1) // trainer_world_size) * trainer_world_size
    if len(run_tag) > 80:
        raise ValueError("run tag is too long to keep generated scheduler identities below 128 characters")
    remote_output = f"/output/smoke_tests/random_continuation_baseline/{run_tag}"
    command = [
        str(DEFAULT_PYTHON),
        str(DEFAULT_LAUNCHER),
        "--cluster",
        "cw-dfw",
        "--config_dir",
        str(local_config_dir),
        "--explicit_output_dir",
        remote_output,
        "--output_base_dir",
        "/output/smoke_tests/random_continuation_baseline",
        "--local_verl_folder",
        str(verl_root),
        "--script_module",
        "smoke_tests.random_continuation_baseline.main_ppo",
        "--reward_file",
        str(reward_file),
        "--ground_truth_solution_key",
        "solution",
        "--expname",
        "random-continuation-baseline",
        "--time_limit",
        "04:00:00",
        "--nodes",
        "2",
        "--trainer_nodes",
        "2",
        "--gpus",
        "8",
        "--actor_model",
        MODEL_PATH,
        "--critic_model",
        MODEL_PATH,
        "--prompt_data",
        TRAIN_DATA,
        "--eval_data",
        VAL_DATA,
        "--n_prompts",
        str(launcher_prompts),
        "--n_samples",
        str(rollouts),
        "--n_val_samples",
        "1",
        "--val_batch_size",
        "8",
        "--max_prompt_len",
        "2048",
        "--max_len",
        "10240",
        "--max_tokens_per_gpu",
        "32768",
        "--num_epochs",
        "1",
        "--num_training_jobs",
        "1",
        "--num_ppo_iter",
        "1",
        "--actor_lr",
        "1e-6",
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
        "--add_interactive",
        "--partition",
        "interactive",
        "--extra_args",
        _extra_args(
            remote_output,
            prompts=prompts,
            rollouts=rollouts,
            points=points,
            continuations=continuations,
            seed=seed,
        ),
    ]
    if dry_run:
        command.append("--dry_run")
    return command, remote_output


def _job_record(path: Path) -> dict[str, str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not str(record.get("job_id", "")).isdigit():
        raise ValueError(f"invalid job record: {record!r}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "submit", "status", "collect", "verify"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, default=DEFAULT_VERL)
    parser.add_argument("--reward-file", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--prompts", type=int, default=256)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--continuations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    local_run_dir = args.local_run_dir.expanduser().resolve()
    verl_root = args.verl_root.expanduser().resolve()
    if local_run_dir == verl_root or verl_root in local_run_dir.parents:
        raise ValueError("runtime evidence must be outside the VeRL workspace")
    local_run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = local_run_dir / "dry_run.ok.json"
    job_path = local_run_dir / "job.json"

    if args.action in {"status", "collect", "verify"}:
        record = _job_record(job_path)
        if args.action == "status":
            result = _ssh(
                CW_DFW_PROFILE,
                f"sacct -X -j {shlex.quote(record['job_id'])} -n -P "
                "-o JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList | head -1",
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            print(result.stdout.strip() or "not-visible")
            return
        if args.action == "collect":
            destination = local_run_dir / "collected"
            destination.mkdir(parents=True, exist_ok=True)
            source = _remote_host_output_path(CW_DFW_PROFILE, record["remote_output"])
            result = subprocess.run(["rsync", "-a", f"dfw:{source}/", f"{destination}/"], check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
            print(json.dumps({"status": "collected", "source": str(source), "destination": str(destination)}))
            return
        from smoke_tests.random_continuation_baseline.verify import verify

        result = verify(local_run_dir / "collected")
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        (local_run_dir / "verified.json").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return

    execution_config_dir = _prepare_execution_config(CW_DFW_PROFILE, args.config_dir, local_run_dir)
    submit_command, remote_output = build_command(
        run_tag=args.run_tag,
        local_config_dir=execution_config_dir,
        verl_root=verl_root,
        reward_file=args.reward_file,
        prompts=args.prompts,
        rollouts=args.rollouts,
        points=args.points,
        continuations=args.continuations,
        seed=args.seed,
        dry_run=False,
    )
    provenance = {
        "run_tag": args.run_tag,
        "git": _git_provenance(verl_root),
        "launcher_sha256": _sha256(DEFAULT_LAUNCHER),
        "execution_config_sha256": _sha256(execution_config_dir / "cw-dfw.yaml"),
        "submit_command_sha256": _command_sha256(submit_command),
        "remote_output": remote_output,
    }
    if args.action == "dry-run":
        dry_command, _ = build_command(
            run_tag=args.run_tag,
            local_config_dir=execution_config_dir,
            verl_root=verl_root,
            reward_file=args.reward_file,
            prompts=args.prompts,
            rollouts=args.rollouts,
            points=args.points,
            continuations=args.continuations,
            seed=args.seed,
            dry_run=True,
        )
        result = _run(dry_command, local_run_dir / "dry_run.log")
        if result.returncode:
            raise SystemExit(result.returncode)
        marker_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "dry-run-complete", "remote_output": remote_output}))
        return
    if not marker_path.is_file() or json.loads(marker_path.read_text(encoding="utf-8")) != provenance:
        raise ValueError("submission inputs do not match a successful dry run")
    if job_path.exists():
        raise ValueError(f"job already recorded in {job_path}")
    result = _run(submit_command, local_run_dir / "submit.log")
    if result.returncode:
        raise SystemExit(result.returncode)
    job_id = _parse_job_id(local_run_dir / "submit.log")
    scheduler_contract = _force_no_requeue(CW_DFW_PROFILE, job_id)
    if "Partition=interactive" not in scheduler_contract or "NumNodes=2" not in scheduler_contract:
        raise RuntimeError(f"submitted scheduler contract drifted: {scheduler_contract}")
    (local_run_dir / "scheduler_contract.txt").write_text(scheduler_contract + "\n", encoding="utf-8")
    job_path.write_text(
        json.dumps({"job_id": job_id, "remote_output": remote_output}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "submitted", "job_id": job_id, "remote_output": remote_output}))


if __name__ == "__main__":
    main()
