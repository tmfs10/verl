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

"""Dry-run-first OCI-IAD launcher for intermediate-MC topology candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_PYTHON = Path("/home/siddjain/anaconda3/envs/skills_latest/bin/python")
DEFAULT_LAUNCHER = Path("/home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit.py")
DEFAULT_VERL = Path("/home/siddjain/workspace/verl/verl_intermediate_mc_value_model")
DEFAULT_REWARD = Path("/home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py")
DEFAULT_CONFIG_DIR = Path("/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen")
TRAIN_DATA = "/data/prime_rl/opsd_math_30k/openthoughts_math_30k_opsd_full.jsonl"
VAL_DATA = "/data/rl/mathgen/comp_math_verl.jsonl"
TRAIN_ROWS = 29427
TRAIN_SHA256 = "f79a42fe155218db2f1927ee903afd101929724f2d0516352bdbb91cdb139178"
SSH_ALIAS = "iad-2"
VERL_CONTAINER = "/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-verl-0.7.0.sqsh"
REMOTE_OUTPUT_ROOT = PurePosixPath("/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output")


def _resolve_ssh_hostname(alias: str) -> str:
    result = subprocess.run(["ssh", "-G", alias], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"could not resolve SSH alias {alias!r}: {result.stderr.strip()}")
    hostnames = [
        line.split(maxsplit=1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("hostname ") and len(line.split(maxsplit=1)) == 2
    ]
    if len(hostnames) != 1:
        raise ValueError(f"SSH alias {alias!r} resolved to {len(hostnames)} hostnames: {hostnames!r}")
    return hostnames[0]


def _replace_ssh_tunnel_host(config_text: str, target_host: str) -> tuple[str, str]:
    pattern = re.compile(r"(?m)^(  host:\s*)([^#\s]+)(\s*(?:#.*)?)$")
    matches = pattern.findall(config_text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one two-space-indented SSH host in oci-iad.yaml, found {len(matches)}")
    original_host = matches[0][1]
    updated = pattern.sub(lambda match: f"{match.group(1)}{target_host}{match.group(3)}", config_text, count=1)
    return updated, original_host


def _replace_verl_container(config_text: str, target_container: str) -> tuple[str, str]:
    pattern = re.compile(r"(?m)^(  verl:\s*)([^#\s]+)(\s*(?:#.*)?)$")
    matches = pattern.findall(config_text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one active VeRL container in oci-iad.yaml, found {len(matches)}")
    original_container = matches[0][1]
    updated = pattern.sub(lambda match: f"{match.group(1)}{target_container}{match.group(3)}", config_text, count=1)
    return updated, original_container


def _prepare_execution_config(source_dir: Path, local_run_dir: Path) -> Path:
    source = source_dir.expanduser().resolve() / "oci-iad.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"missing authoritative OCI-IAD config: {source}")
    target_host = _resolve_ssh_hostname(SSH_ALIAS)
    updated, original_host = _replace_ssh_tunnel_host(source.read_text(encoding="utf-8"), target_host)
    updated, original_container = _replace_verl_container(updated, VERL_CONTAINER)
    destination_dir = local_run_dir / "cluster_config"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    destination.write_text(updated, encoding="utf-8")
    (destination_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": _sha256(source),
                "source_host": original_host,
                "ssh_alias": SSH_ALIAS,
                "resolved_host": target_host,
                "source_verl_container": original_container,
                "execution_verl_container": VERL_CONTAINER,
                "execution_config_sha256": _sha256(destination),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_provenance(verl_root: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(verl_root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
        return result.stdout.strip()

    status = git("status", "--porcelain")
    if status:
        raise ValueError("remote dry-run/submission requires a clean VeRL feature worktree")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    if head != upstream:
        raise ValueError(f"VeRL feature HEAD {head} is not pushed to its configured upstream {upstream}")
    return {"head": head, "upstream": upstream}


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        candidate = json.loads(line)
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate candidate id {candidate_id}")
        seen.add(candidate_id)
        contract = candidate["contract"]
        expected = {
            "prompts_per_iteration": 64,
            "rollouts_per_prompt": 8,
            "solutions_per_iteration": 512,
            "prompt_tokens": 4096,
            "solution_tokens": 8192,
            "critique_tokens": 8192,
            "continuations_per_mark": 1,
            "max_marks": 1,
            "temperature": 1.0,
            "synchronous": True,
            "feature_enabled": True,
        }
        if contract != expected:
            raise ValueError(f"{candidate_id}: benchmark contract drifted: {contract!r}")
        if candidate["workload"]["num_critiques"] not in {0, 4}:
            raise ValueError(f"{candidate_id}: only M0 and M4 are allowed")
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"empty candidate manifest: {path}")
    return candidates


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _extra_args(candidate: dict[str, Any], remote_output: str, *, allow_memory_gated: bool = False) -> str:
    workload = candidate["workload"]
    topology = candidate["topology"]
    profile = candidate["batch_profile"]
    seed = int(candidate["seed"])
    candidate_id = candidate["candidate_id"]
    if profile["gated_by_memory_headroom"] and not allow_memory_gated:
        raise ValueError(
            f"{candidate_id}: {profile['profile_id']} is gated by measured memory headroom and cannot be bulk-launched"
        )

    actor_micro = "null" if profile["actor_dynamic"] else "1"
    critic_micro = "null" if profile["critic_dynamic"] else "1"
    overrides = [
        # The shared launcher still injects this deleted, non-VeRL critic key.
        # Remove it after the launcher's base overrides instead of weakening
        # FSDPCriticConfig to accept obsolete configuration.
        "~critic.append_solution_to_prompt",
        "algorithm.adv_estimator=gae",
        "algorithm.gamma=1.0",
        "algorithm.lam=1.0",
        "algorithm.use_kl_in_reward=false",
        "algorithm.intermediate_mc_value.enable=true",
        f"algorithm.intermediate_mc_value.critic_head={workload['critic_head']}",
        f"algorithm.intermediate_mc_value.mark_selector={workload['mark_selector']}",
        f"algorithm.intermediate_mc_value.num_critiques={workload['num_critiques']}",
        "algorithm.intermediate_mc_value.continuations_per_mark=1",
        "algorithm.intermediate_mc_value.max_marks=1",
        "algorithm.intermediate_mc_value.critique_max_response_length=8192",
        "algorithm.intermediate_mc_value.mark_start_fraction=0.05",
        "algorithm.intermediate_mc_value.mark_end_fraction=0.90",
        "algorithm.intermediate_mc_value.min_mark_gap=32",
        "algorithm.intermediate_mc_value.variance_scope=rollout",
        "algorithm.intermediate_mc_value.variance_random_probability=0.05",
        f"algorithm.intermediate_mc_value.selection_seed={seed}",
        "algorithm.intermediate_mc_value.audit_output_dir=null",
        "data.train_batch_size=64",
        "++data.gen_batch_size=64",
        "data.max_prompt_length=4096",
        "data.max_response_length=8192",
        "data.filter_overlong_prompts=true",
        "data.filter_overlong_prompts_workers=16",
        "data.dataloader_num_workers=0",
        "data.truncation=error",
        "actor_rollout_ref.model.use_remove_padding=true",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_bool(profile['gradient_checkpointing'])}",
        f"actor_rollout_ref.actor.strategy={topology['strategy']}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor_micro}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.use_dynamic_bsz={_bool(profile['actor_dynamic'])}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={profile['actor_token_cap']}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.policy_loss.loss_mode=dppo_tv",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={topology['sequence_parallel_size']}",
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={topology['actor_fsdp_size']}",
        f"actor_rollout_ref.actor.fsdp_config.reshard_after_forward={_bool(profile['reshard_after_forward'])}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={topology['rollout_tp']}",
        "actor_rollout_ref.rollout.max_model_len=24576",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={profile['rollout_batched_tokens']}",
        f"actor_rollout_ref.rollout.max_num_seqs={profile['rollout_max_num_seqs']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={profile['rollout_gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.enforce_eager={_bool(profile['rollout_enforce_eager'])}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=true",
        "actor_rollout_ref.rollout.enable_prefix_caching=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "critic.enable=true",
        f"critic.strategy={topology['strategy']}",
        "critic.ppo_mini_batch_size=64",
        f"critic.ppo_micro_batch_size_per_gpu={critic_micro}",
        f"critic.forward_micro_batch_size_per_gpu={critic_micro}",
        "critic.ppo_epochs=1",
        f"critic.use_dynamic_bsz={_bool(profile['critic_dynamic'])}",
        f"critic.ppo_max_token_len_per_gpu={profile['critic_token_cap']}",
        f"critic.forward_max_token_len_per_gpu={profile['critic_token_cap']}",
        f"critic.ulysses_sequence_parallel_size={topology['sequence_parallel_size']}",
        f"critic.model.fsdp_config.fsdp_size={topology['critic_fsdp_size']}",
        f"critic.model.fsdp_config.reshard_after_forward={_bool(profile['reshard_after_forward'])}",
        "critic.model.fsdp_config.param_offload=false",
        "critic.model.fsdp_config.optimizer_offload=false",
        f"critic.model.enable_gradient_checkpointing={_bool(profile['gradient_checkpointing'])}",
        "critic.model.use_remove_padding=true",
        "reward.reward_model.enable=false",
        "++reward.reward_model.launch_reward_fn_async=false",
        "trainer.use_legacy_worker_impl=enable",
        "trainer.critic_warmup=0",
        "trainer.logger=[file]",
        "trainer.project_name=intermediate_mc_topology",
        f"trainer.experiment_name={candidate_id}",
        f"trainer.default_local_dir={remote_output}/checkpoints",
        f"trainer.total_training_steps={candidate['total_steps']}",
        "trainer.total_epochs=1",
        "trainer.val_before_train=false",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        "trainer.rollout_data_dir=null",
        "trainer.validation_data_dir=null",
        "trainer.resume_mode=disable",
        "trainer.balance_batch=true",
        f"+topology_benchmark.output_dir={remote_output}/benchmark",
        f"++topology_benchmark.candidate_id={candidate_id}",
        f"++topology_benchmark.stabilization_steps={candidate['stabilization_steps']}",
        f"++topology_benchmark.measured_steps={candidate['measured_steps']}",
        f"++topology_benchmark.expected_train_rows={TRAIN_ROWS}",
        f"++topology_benchmark.expected_train_sha256={TRAIN_SHA256}",
        f"++topology_benchmark.expected_model_path={workload['model_path']}",
        f"++topology_benchmark.expected_critic_head={workload['critic_head']}",
        f"++topology_benchmark.expected_mark_selector={workload['mark_selector']}",
        f"++topology_benchmark.expected_num_critiques={workload['num_critiques']}",
        f"++topology_benchmark.expected_nodes={topology['nodes']}",
        f"++topology_benchmark.expected_strategy={topology['strategy']}",
        f"++topology_benchmark.expected_actor_fsdp_size={topology['actor_fsdp_size']}",
        f"++topology_benchmark.expected_critic_fsdp_size={topology['critic_fsdp_size']}",
        f"++topology_benchmark.expected_rollout_tp={topology['rollout_tp']}",
        f"++topology_benchmark.expected_sequence_parallel_size={topology['sequence_parallel_size']}",
        f"++topology_benchmark.expected_actor_dynamic={_bool(profile['actor_dynamic'])}",
        f"++topology_benchmark.expected_critic_dynamic={_bool(profile['critic_dynamic'])}",
        f"++topology_benchmark.expected_actor_token_cap={profile['actor_token_cap']}",
        f"++topology_benchmark.expected_critic_token_cap={profile['critic_token_cap']}",
        f"++topology_benchmark.expected_rollout_batched_tokens={profile['rollout_batched_tokens']}",
        f"++topology_benchmark.expected_rollout_max_num_seqs={profile['rollout_max_num_seqs']}",
        f"++topology_benchmark.expected_rollout_gpu_memory_utilization={profile['rollout_gpu_memory_utilization']}",
        f"++topology_benchmark.expected_rollout_enforce_eager={_bool(profile['rollout_enforce_eager'])}",
        f"++topology_benchmark.expected_gradient_checkpointing={_bool(profile['gradient_checkpointing'])}",
        f"++topology_benchmark.expected_reshard_after_forward={_bool(profile['reshard_after_forward'])}",
    ]
    return " ".join(overrides)


def build_command(
    candidate: dict[str, Any],
    *,
    run_tag: str,
    dry_run: bool,
    python: Path,
    launcher: Path,
    verl_root: Path,
    reward_file: Path,
    config_dir: Path,
    allow_memory_gated: bool = False,
) -> tuple[list[str], str]:
    candidate_id = candidate["candidate_id"]
    topology = candidate["topology"]
    model_path = candidate["workload"]["model_path"]
    remote_output = f"/output/smoke_tests/intermediate_mc_topology/{run_tag}/{candidate_id}"
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
        "/output/smoke_tests/intermediate_mc_topology",
        "--local_verl_folder",
        str(verl_root),
        "--script_module",
        "smoke_tests.intermediate_mc_value.topology.main_ppo_topology",
        "--reward_file",
        str(reward_file),
        "--ground_truth_solution_key",
        "solution",
        "--expname",
        candidate_id,
        "--time_limit",
        "04:00:00",
        "--nodes",
        str(topology["nodes"]),
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
        "64",
        "--n_samples",
        "8",
        "--n_val_samples",
        "1",
        "--val_batch_size",
        "64",
        "--max_prompt_len",
        "4096",
        "--max_len",
        "12288",
        "--max_tokens_per_gpu",
        "24576",
        "--num_epochs",
        "1",
        "--num_training_jobs",
        "1",
        "--num_ppo_iter",
        "1",
        "--critic_num_ppo_iter",
        "1",
        "--actor_lr",
        "2e-6",
        "--critic_lr",
        "1e-5",
        "--clip_ae",
        "0.2,0.2",
        "--infer_server",
        "vllm",
        "--sequence_parallel_size",
        str(topology["sequence_parallel_size"]),
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
        "gae",
        "--seed",
        str(candidate["seed"]),
        "--no_sandbox",
        "--no_requeue",
        "--disable_val_before_train",
        "--omit_noncore_algorithm_overrides",
        "--skip_runtime_package_install",
        "--extra_args",
        _extra_args(candidate, remote_output, allow_memory_gated=allow_memory_gated),
    ]
    if int(topology["nodes"]) == 2:
        command.append("--add_interactive")
    if dry_run:
        command.append("--dry_run")
    return command, remote_output


def _selected(candidates: list[dict[str, Any]], ids: set[str], max_candidates: int) -> list[dict[str, Any]]:
    selected = [candidate for candidate in candidates if not ids or candidate["candidate_id"] in ids]
    if ids - {candidate["candidate_id"] for candidate in selected}:
        raise ValueError(
            f"unknown candidate ids: {sorted(ids - {candidate['candidate_id'] for candidate in selected})}"
        )
    if max_candidates > 0:
        selected = selected[:max_candidates]
    if not selected:
        raise ValueError("candidate selection is empty")
    return selected


def _run(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    print("[command]", shlex.join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_env = os.environ.copy()
    # The local validation environment may prepend Python 3.12 packages while
    # the configured cluster launcher uses Python 3.10. The launcher receives
    # the VeRL checkout explicitly and must not inherit that incompatible path.
    launcher_env.pop("PYTHONPATH", None)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=launcher_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
            handle.flush()
        returncode = process.wait()
    return subprocess.CompletedProcess(command, returncode, "", "")


def _parse_job_id(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"slurm_tunnel://nemo_run/(\d+)|Submitted batch job (\d+)", text)
    job_ids = [first or second for first, second in matches]
    if not job_ids:
        raise RuntimeError(f"could not parse submitted Slurm job id from {log_path}")
    return job_ids[-1]


def _load_submitted(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        candidate_id, job_id, remote_output = line.split("\t")
        records[candidate_id] = {"job_id": job_id, "remote_output": remote_output}
    return records


def _remote_host_output_path(container_output: str) -> PurePosixPath:
    """Translate the container's /output mount into its iad-2 login-node path."""

    container_path = PurePosixPath(container_output)
    try:
        relative = container_path.relative_to("/output")
    except ValueError as error:
        raise ValueError(f"remote output is outside the pinned /output mount: {container_output!r}") from error
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError(f"invalid remote output below /output: {container_output!r}")
    return REMOTE_OUTPUT_ROOT.joinpath(relative)


def _ssh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", SSH_ALIAS, command], capture_output=True, text=True, check=False)


def _requeue_value(scontrol_output: str) -> int:
    match = re.search(r"(?:^|\s)Requeue=(\d+)(?:\s|$)", scontrol_output)
    if match is None:
        raise ValueError(f"scontrol output has no Requeue field: {scontrol_output!r}")
    return int(match.group(1))


def _force_no_requeue(job_id: str) -> str:
    if not job_id.isdigit():
        raise ValueError(f"invalid Slurm job id {job_id!r}")
    update = _ssh(f"scontrol update JobId={job_id} Requeue=0")
    if update.returncode:
        _ssh(f"scancel {job_id}")
        raise RuntimeError(
            f"failed to enforce Requeue=0 for job {job_id}; job was cancelled: "
            f"{update.stderr.strip() or update.stdout.strip()}"
        )
    query = _ssh(f"scontrol show job {job_id} -o")
    try:
        requeue = _requeue_value(query.stdout) if query.returncode == 0 else None
    except ValueError:
        requeue = None
    if requeue != 0:
        _ssh(f"scancel {job_id}")
        detail = query.stderr.strip() or query.stdout.strip()
        raise RuntimeError(f"job {job_id} did not retain Requeue=0 after enforcement; job was cancelled: {detail}")
    return query.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "dry-run", "submit", "status", "collect"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--verl-root", type=Path, default=DEFAULT_VERL)
    parser.add_argument("--reward-file", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--allow-memory-gated",
        action="store_true",
        help="Allow P05 only after prior measurements prove sufficient GPU-memory headroom.",
    )
    parser.add_argument(
        "--submit-all",
        action="store_true",
        help="Explicitly permit submission of the entire selected manifest instead of a bounded wave.",
    )
    args = parser.parse_args()

    repo_root = args.verl_root.expanduser().resolve()
    local_run_dir = args.local_run_dir.expanduser().resolve()
    if local_run_dir == repo_root or repo_root in local_run_dir.parents:
        raise ValueError("runtime benchmark artifacts must be outside the VeRL workspace")
    local_run_dir.mkdir(parents=True, exist_ok=True)
    execution_config_dir = args.config_dir
    execution_config_sha256 = None
    if args.action in {"render", "dry-run", "submit"}:
        execution_config_dir = _prepare_execution_config(args.config_dir, local_run_dir)
        execution_config_sha256 = _sha256(execution_config_dir / "oci-iad.yaml")
    candidates = _selected(_load_manifest(args.manifest), set(args.candidate_id), args.max_candidates)
    manifest_sha = _sha256(args.manifest)
    selection_ids = [candidate["candidate_id"] for candidate in candidates]
    marker_path = local_run_dir / "dry_run.ok.json"
    jobs_path = local_run_dir / "jobs.tsv"

    if args.action == "submit" and not args.submit_all and not args.candidate_id and args.max_candidates <= 0:
        raise ValueError("submit requires --candidate-id, --max-candidates, or the explicit --submit-all override")

    if args.action == "render":
        output = local_run_dir / "rendered_commands.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for index, candidate in enumerate(candidates, start=1):
                command, remote_output = build_command(
                    candidate,
                    run_tag=args.run_tag,
                    dry_run=False,
                    python=args.python,
                    launcher=args.launcher,
                    verl_root=args.verl_root,
                    reward_file=args.reward_file,
                    config_dir=execution_config_dir,
                    allow_memory_gated=args.allow_memory_gated,
                )
                handle.write(
                    json.dumps(
                        {"candidate_id": candidate["candidate_id"], "command": command, "remote_output": remote_output},
                        sort_keys=True,
                    )
                    + "\n"
                )
                print(f"[render {index}/{len(candidates)}] {candidate['candidate_id']}", flush=True)
        print(json.dumps({"rendered": len(candidates), "output": str(output)}, sort_keys=True))
        return

    if args.action == "dry-run":
        git_provenance = _git_provenance(args.verl_root)
        command_hashes: dict[str, str] = {}
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = candidate["candidate_id"]
            print(f"[dry-run {index}/{len(candidates)}] {candidate_id}", flush=True)
            command, _ = build_command(
                candidate,
                run_tag=args.run_tag,
                dry_run=True,
                python=args.python,
                launcher=args.launcher,
                verl_root=args.verl_root,
                reward_file=args.reward_file,
                config_dir=execution_config_dir,
                allow_memory_gated=args.allow_memory_gated,
            )
            result = _run(command, local_run_dir / "dry_run_logs" / f"{candidate_id}.log")
            if result.returncode:
                raise SystemExit(result.returncode)
            submit_command, _ = build_command(
                candidate,
                run_tag=args.run_tag,
                dry_run=False,
                python=args.python,
                launcher=args.launcher,
                verl_root=args.verl_root,
                reward_file=args.reward_file,
                config_dir=execution_config_dir,
                allow_memory_gated=args.allow_memory_gated,
            )
            command_hashes[candidate_id] = _command_sha256(submit_command)
        marker_path.write_text(
            json.dumps(
                {
                    "manifest_sha256": manifest_sha,
                    "ssh_alias": SSH_ALIAS,
                    "execution_config_sha256": execution_config_sha256,
                    "launcher_sha256": _sha256(args.launcher),
                    "allow_memory_gated": args.allow_memory_gated,
                    "git": git_provenance,
                    "command_sha256": command_hashes,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "dry-run-complete", "candidates": len(candidates)}, sort_keys=True))
        return

    if args.action == "submit":
        if not marker_path.is_file():
            raise ValueError(f"missing successful dry-run marker: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        git_provenance = _git_provenance(args.verl_root)
        if marker.get("manifest_sha256") != manifest_sha:
            raise ValueError("dry-run marker does not match the current manifest")
        if marker.get("ssh_alias") != SSH_ALIAS:
            raise ValueError("dry-run marker does not match the current SSH alias")
        if marker.get("execution_config_sha256") != execution_config_sha256:
            raise ValueError("dry-run marker does not match the current execution-only cluster config")
        if marker.get("launcher_sha256") != _sha256(args.launcher):
            raise ValueError("dry-run marker does not match the current cluster launcher")
        if marker.get("allow_memory_gated") != args.allow_memory_gated:
            raise ValueError("dry-run marker does not match the current memory-gate setting")
        if marker.get("git") != git_provenance:
            raise ValueError("dry-run marker does not match the clean, pushed VeRL revision")
        dry_run_hashes = marker.get("command_sha256", {})
        missing_dry_runs = set(selection_ids) - set(dry_run_hashes)
        if missing_dry_runs:
            raise ValueError(f"selected candidates were not dry-run: {sorted(missing_dry_runs)}")
        submitted = _load_submitted(jobs_path)
        if not jobs_path.exists():
            jobs_path.write_text("candidate_id\tjob_id\tremote_output\n", encoding="utf-8")
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = candidate["candidate_id"]
            if candidate_id in submitted:
                print(f"[submit {index}/{len(candidates)}] already submitted {candidate_id}", flush=True)
                continue
            print(f"[submit {index}/{len(candidates)}] {candidate_id}", flush=True)
            command, remote_output = build_command(
                candidate,
                run_tag=args.run_tag,
                dry_run=False,
                python=args.python,
                launcher=args.launcher,
                verl_root=args.verl_root,
                reward_file=args.reward_file,
                config_dir=execution_config_dir,
                allow_memory_gated=args.allow_memory_gated,
            )
            if _command_sha256(command) != dry_run_hashes[candidate_id]:
                raise ValueError(f"{candidate_id}: submission command changed after its successful dry-run")
            log_path = local_run_dir / "submit_logs" / f"{candidate_id}.log"
            result = _run(command, log_path)
            if result.returncode:
                raise SystemExit(result.returncode)
            job_id = _parse_job_id(log_path)
            scheduler_contract = _force_no_requeue(job_id)
            contract_dir = local_run_dir / "scheduler_contracts"
            contract_dir.mkdir(parents=True, exist_ok=True)
            (contract_dir / f"{candidate_id}.txt").write_text(scheduler_contract + "\n", encoding="utf-8")
            with jobs_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{candidate_id}\t{job_id}\t{remote_output}\n")
            print(f"[submitted] candidate={candidate_id} job={job_id}", flush=True)
        return

    submitted = _load_submitted(jobs_path)
    if not submitted:
        raise ValueError(f"no submitted jobs recorded in {jobs_path}")
    if args.action == "status":
        failures = 0
        for index, (candidate_id, record) in enumerate(submitted.items(), start=1):
            job_id = record["job_id"]
            result = _ssh(
                f"sacct -X -j {shlex.quote(job_id)} -n -P -o JobIDRaw,State,ElapsedRaw,AllocTRES,NodeList | head -1"
            )
            if result.returncode:
                failures += 1
                state = result.stderr.strip() or "ssh-failed"
            else:
                state = result.stdout.strip() or "not-visible"
            print(f"[status {index}/{len(submitted)}] {candidate_id}\t{job_id}\t{state}", flush=True)
        if failures:
            raise SystemExit(1)
        return

    collect_root = local_run_dir / "collected"
    collect_root.mkdir(parents=True, exist_ok=True)
    for index, (candidate_id, record) in enumerate(submitted.items(), start=1):
        destination = collect_root / candidate_id
        destination.mkdir(parents=True, exist_ok=True)
        host_output = _remote_host_output_path(record["remote_output"])
        benchmark_output = host_output / "benchmark"
        source = f"{SSH_ALIAS}:{benchmark_output}/"
        print(f"[collect {index}/{len(submitted)}] {candidate_id}", flush=True)
        result = subprocess.run(["rsync", "-a", source, str(destination) + "/"], check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
        metrics_source = benchmark_output / "metrics.jsonl"
        used_legacy_fallback = False
        if not (destination / "metrics.jsonl").is_file():
            # Runs created before the TaskRunner runtime-env fix wrote the file
            # logger under NeMo Run's bundled working directory. Recover exactly
            # one such artifact and reject ambiguity instead of guessing.
            legacy_root = host_output / "job"
            legacy_pattern = f"*/nemo-run/code/intermediate_mc_topology/{candidate_id}.jsonl"
            query = _ssh(f"find {shlex.quote(str(legacy_root))} -type f -path {shlex.quote(legacy_pattern)} -print")
            if query.returncode:
                raise RuntimeError(query.stderr.strip() or f"could not search legacy metrics below {legacy_root}")
            matches = [line for line in query.stdout.splitlines() if line.strip()]
            if len(matches) != 1:
                raise RuntimeError(
                    f"{candidate_id}: expected exactly one legacy metrics file below {legacy_root}, got {matches!r}"
                )
            metrics_source = PurePosixPath(matches[0])
            result = subprocess.run(
                ["rsync", "-a", f"{SSH_ALIAS}:{metrics_source}", str(destination / "metrics.jsonl")],
                check=False,
            )
            if result.returncode:
                raise SystemExit(result.returncode)
            used_legacy_fallback = True
        (destination / "collection_provenance.json").write_text(
            json.dumps(
                {
                    "benchmark_source": str(benchmark_output),
                    "metrics_sha256": _sha256(destination / "metrics.jsonl"),
                    "metrics_source": str(metrics_source),
                    "used_legacy_metrics_fallback": used_legacy_fallback,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
