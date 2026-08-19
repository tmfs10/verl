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

import json
from pathlib import Path

import pytest

from smoke_tests.intermediate_mc_value.topology import submit_oci_iad as submit_module
from smoke_tests.intermediate_mc_value.topology.analyze import (
    _apply_token_drift_rejection,
    _recommendations,
    aggregate_repeats,
    analyze_candidate,
)
from smoke_tests.intermediate_mc_value.topology.matrix import (
    BATCH_PROFILES,
    WORKLOADS,
    Candidate,
    finalist_repeats,
    four_node_expansion,
    four_node_finalist_repeats,
    memory_expansion,
    topology_by_id,
    two_node_core,
)
from smoke_tests.intermediate_mc_value.topology.submit_oci_iad import (
    _extra_args,
    _remote_host_output_path,
    _replace_ssh_tunnel_host,
    _replace_verl_container,
    _requeue_value,
    build_command,
)


def _candidate(num_critiques: int = 0) -> Candidate:
    workload = next(
        workload
        for workload in WORKLOADS
        if workload.model == "qwen3-1p7b"
        and workload.recipe == "scalar_random"
        and workload.num_critiques == num_critiques
    )
    return Candidate("test", workload, topology_by_id("T01"), BATCH_PROFILES["P01"], 1234)


def test_two_node_core_is_bounded_and_keeps_m0_and_m4_separate() -> None:
    candidates = two_node_core(1234)
    assert len(candidates) == 76
    assert len({candidate.candidate_id for candidate in candidates}) == 76
    assert {candidate.workload.num_critiques for candidate in candidates} == {0, 4}
    assert all(candidate.topology.nodes == 2 for candidate in candidates)
    assert all(candidate.batch_profile.profile_id == "P01" for candidate in candidates)
    assert sum(candidate.workload.recipe == "beta_variance" for candidate in candidates) == 28


def test_finalists_use_three_seeds_and_four_measured_steps() -> None:
    candidates = finalist_repeats(
        "qwen3-4b-beta_variance-m4",
        "T05",
        "P03",
        [1234, 2345, 3456],
    )
    assert [candidate.seed for candidate in candidates] == [1234, 2345, 3456]
    assert all(candidate.stabilization_steps == 1 for candidate in candidates)
    assert all(candidate.measured_steps == 4 for candidate in candidates)


def test_four_node_promotion_preserves_then_expands_fsdp_groups() -> None:
    candidates = four_node_expansion("qwen3-1p7b-scalar_random-m0", ["T07"], "P01", 1234)
    assert [(candidate.topology.actor_fsdp_size, candidate.topology.critic_fsdp_size) for candidate in candidates] == [
        (8, 16),
        (16, 32),
    ]
    assert all(candidate.topology.nodes == 4 for candidate in candidates)
    assert all(candidate.topology.source_topology == "T07" for candidate in candidates)


def test_memory_profile_is_explicit_and_four_node_finalists_repeat() -> None:
    memory = memory_expansion("qwen3-4b-beta_variance-m4", "T05", 1234)
    assert len(memory) == 1
    assert memory[0].batch_profile.gated_by_memory_headroom is True

    finalists = four_node_finalist_repeats(
        "qwen3-4b-beta_variance-m4",
        "T05",
        "expanded",
        "P03",
        [1234, 2345, 3456],
    )
    assert [candidate.seed for candidate in finalists] == [1234, 2345, 3456]
    assert all(candidate.topology.topology_id == "N4_T05_X" for candidate in finalists)
    assert all(candidate.measured_steps == 4 for candidate in finalists)


def test_launcher_contract_keeps_feature_enabled_for_m0_and_has_no_fully_async_trainer() -> None:
    candidate = _candidate(0).to_dict()
    command, remote_output = build_command(
        candidate,
        run_tag="test-tag",
        dry_run=True,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=Path("/configs"),
    )
    joined = " ".join(command)
    assert "algorithm.intermediate_mc_value.enable=true" in joined
    assert "algorithm.intermediate_mc_value.num_critiques=0" in joined
    assert "algorithm.intermediate_mc_value.continuations_per_mark=1" in joined
    assert "algorithm.intermediate_mc_value.max_marks=1" in joined
    assert "actor_rollout_ref.rollout.temperature=1.0" in joined
    assert "/data/prime_rl/opsd_math_30k/openthoughts_math_30k_opsd_full.jsonl" in joined
    assert "trainer.critic_warmup=0" in joined
    assert "trainer.logger=[file]" in joined
    assert "~critic.append_solution_to_prompt" in joined
    assert "--skip_runtime_package_install" in command
    assert "fully_async" not in joined
    assert command[-1] == "--dry_run"
    assert remote_output.endswith(candidate["candidate_id"])


def test_execution_config_changes_only_the_authorized_login_host() -> None:
    source = """executor: slurm

ssh_tunnel:
  host: draco-oci-login-01.draco-oci-iad.nvidia.com
  user: siddjain

account: nemotron_reason_code

containers:
  verl: /containers/missing-verl.sqsh
"""
    target = "draco-oci-login-02.draco-oci-iad.nvidia.com"
    updated, original = _replace_ssh_tunnel_host(source, target)
    assert original == "draco-oci-login-01.draco-oci-iad.nvidia.com"
    assert updated == source.replace(original, target)

    container = "/containers/verl-vllm-0.12.sqsh"
    updated, original_container = _replace_verl_container(updated, container)
    assert original_container == "/containers/missing-verl.sqsh"
    assert updated == source.replace(original, target).replace(original_container, container)


def test_oci_runtime_is_the_a100_preflighted_shared_verl_image() -> None:
    assert submit_module.VERL_CONTAINER == (
        "/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-verl-0.7.0.sqsh"
    )


def test_collector_translates_only_the_pinned_output_mount() -> None:
    assert str(_remote_host_output_path("/output/smoke_tests/run/candidate")) == (
        "/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/run/candidate"
    )
    with pytest.raises(ValueError, match="outside the pinned /output mount"):
        _remote_host_output_path("/tmp/run")
    with pytest.raises(ValueError, match="invalid remote output"):
        _remote_host_output_path("/output/../tmp/run")


def test_scheduler_requeue_contract_parser() -> None:
    assert _requeue_value("JobId=123 JobState=PENDING Requeue=0 Restarts=0") == 0
    assert _requeue_value("JobId=123 JobState=PENDING Requeue=1 Restarts=0") == 1
    with pytest.raises(ValueError, match="no Requeue field"):
        _requeue_value("JobId=123 JobState=PENDING")


def test_scheduler_requeue_enforcement_verifies_and_cancels_on_failure(monkeypatch) -> None:
    commands: list[str] = []

    def success(command: str):
        commands.append(command)
        stdout = "JobId=123 JobState=PENDING Requeue=0 Restarts=0" if command.startswith("scontrol show") else ""
        return submit_module.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(submit_module, "_ssh", success)
    assert "Requeue=0" in submit_module._force_no_requeue("123")
    assert commands == ["scontrol update JobId=123 Requeue=0", "scontrol show job 123 -o"]

    commands.clear()

    def rejected(command: str):
        commands.append(command)
        stdout = "JobId=123 JobState=PENDING Requeue=1 Restarts=0" if command.startswith("scontrol show") else ""
        return submit_module.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(submit_module, "_ssh", rejected)
    with pytest.raises(RuntimeError, match="job was cancelled"):
        submit_module._force_no_requeue("123")
    assert commands[-1] == "scancel 123"


def test_memory_aggressive_profile_cannot_be_bulk_launched() -> None:
    candidate = _candidate(4).to_dict()
    candidate["batch_profile"] = BATCH_PROFILES["P05"].__dict__
    with pytest.raises(ValueError, match="gated by measured memory headroom"):
        _extra_args(candidate, "/output/run")


def _hardware(host: str) -> dict[str, object]:
    lines = "\n".join(f"{index}, GPU-{host}-{index}, NVIDIA A100-SXM4-80GB, 81920, 550.00" for index in range(8))
    return {
        "hostname": host,
        "nvidia_smi_query": {"returncode": 0, "stdout": lines, "stderr": ""},
    }


def _write_successful_run(
    root: Path,
    candidate: dict[str, object],
    *,
    step_seconds: tuple[float, float, float] = (120.0, 100.0, 110.0),
    generation_tokens: float = 100000.0,
) -> None:
    root.mkdir(parents=True)
    (root / "completed.json").write_text("{}\n", encoding="utf-8")
    nodes = int(candidate["topology"]["nodes"])
    (root / "ray_hardware_after.json").write_text(
        json.dumps([_hardware(f"node-{index}") for index in range(nodes)]) + "\n",
        encoding="utf-8",
    )
    num_critiques = int(candidate["workload"]["num_critiques"])
    with (root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for step, seconds in enumerate(step_seconds, start=1):
            metrics = {
                "training/global_step": step,
                "intermediate_mc/warmup": 0.0,
                "intermediate_mc/bundles": 512,
                "intermediate_mc/critiques": 512 * num_critiques,
                "intermediate_mc/selected_marks": 512,
                "intermediate_mc/continuation_attempts": 512.0,
                "intermediate_mc/tokens/critique_input": 0.0 if num_critiques == 0 else 400000.0,
                "intermediate_mc/tokens/critique_output": 0.0 if num_critiques == 0 else 300000.0,
                "intermediate_mc/tokens/generation_output": generation_tokens,
                "intermediate_mc/tokens/critic_input": 200000.0,
                "intermediate_mc/tokens/actor_train": 100000.0,
                "actor/grad_norm": 1.0,
                "critic/grad_norm": 1.0,
                "timing_s/step": seconds,
                "timing_s/gen": seconds * 0.5,
                "timing_s/intermediate_mc_continuations": seconds * 0.1,
                "timing_s/values": seconds * 0.05,
                "timing_s/update_critic": seconds * 0.15,
                "timing_s/update_actor": seconds * 0.2,
                "perf/max_memory_reserved_gb": 62.0,
            }
            handle.write(json.dumps({"step": step, "data": metrics}) + "\n")


def test_analyzer_accepts_complete_m0_and_computes_measured_iteration_rate(tmp_path: Path) -> None:
    candidate = _candidate(0).to_dict()
    run_dir = tmp_path / candidate["candidate_id"]
    _write_successful_run(run_dir, candidate)
    result = analyze_candidate(candidate, run_dir)
    assert result["valid"] is True
    assert result["median_step_seconds"] == 105.0
    assert result["iterations_per_hour"] == pytest.approx(3600.0 / 105.0)
    assert result["gpu_hours_per_iteration"] == pytest.approx(16 * 105.0 / 3600.0)
    assert result["estimated_memory_headroom_gb"] == 18.0


def test_analyzer_rejects_m0_with_critique_tokens(tmp_path: Path) -> None:
    candidate = _candidate(0).to_dict()
    run_dir = tmp_path / candidate["candidate_id"]
    _write_successful_run(run_dir, candidate)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    records[1]["data"]["intermediate_mc/tokens/critique_output"] = 1.0
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    result = analyze_candidate(candidate, run_dir)
    assert result["valid"] is False
    assert any("M0 requires" in reason for reason in result["rejection_reasons"])


def test_matched_token_drift_over_ten_percent_is_rejected(tmp_path: Path) -> None:
    first = _candidate(0).to_dict()
    second_candidate = Candidate(
        "test",
        _candidate(0).workload,
        topology_by_id("T02"),
        BATCH_PROFILES["P01"],
        1234,
    )
    second = second_candidate.to_dict()
    first_dir = tmp_path / first["candidate_id"]
    second_dir = tmp_path / second["candidate_id"]
    _write_successful_run(first_dir, first, generation_tokens=100000.0)
    _write_successful_run(second_dir, second, generation_tokens=140000.0)
    results = [analyze_candidate(first, first_dir), analyze_candidate(second, second_dir)]
    _apply_token_drift_rejection(results)
    assert all(not result["valid"] for result in results)
    assert aggregate_repeats(results) == []


def test_recommendation_uses_p95_inside_three_percent_tie() -> None:
    topology = _candidate(0).to_dict()["topology"]
    profile = _candidate(0).to_dict()["batch_profile"]
    base = {
        "workload_id": "qwen3-1p7b-scalar_random-m0",
        "nodes": 2,
        "profile_id": "P01",
        "batch_profile": profile,
        "median_step_seconds_ci95": None,
    }
    raw_best = {
        **base,
        "topology_id": "T01",
        "topology": {**topology, "topology_id": "T01"},
        "iterations_per_hour": 36.0,
        "p95_step_seconds": 120.0,
    }
    stable_tie = {
        **base,
        "topology_id": "T02",
        "topology": {**topology, "topology_id": "T02"},
        "iterations_per_hour": 35.1,
        "p95_step_seconds": 105.0,
    }
    recommendation = _recommendations([raw_best, stable_tie])[0]
    assert recommendation["raw_best_topology_id"] == "T01"
    assert recommendation["selected_topology_id"] == "T02"
