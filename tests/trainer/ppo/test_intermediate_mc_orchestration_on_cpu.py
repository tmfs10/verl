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

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput
from verl.experimental.agent_loop.intermediate_mc_agent_loop import (
    INTERMEDIATE_MC_CHILD_FIELD,
    ContinuationGeneration,
    CritiqueGeneration,
    IntermediateMCAgentLoop,
    IntermediateMCGenerationRecord,
)
from verl.trainer.config import IntermediateMCValueConfig
from verl.trainer.ppo.intermediate_mc_value import build_critic_context, build_unconditioned_critic_context
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer_intermediate_mc import (
    IntermediateMCValueController,
    _Bundle,
    configured_optimizer_rows,
    validate_intermediate_mc_runtime_config,
)
from verl.workers.rollout.replica import TokenOutput


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9


def _controller(*, num_critiques=2, critic_head="scalar", mark_selector="random"):
    controller = IntermediateMCValueController.__new__(IntermediateMCValueController)
    controller.feature = IntermediateMCValueConfig(
        num_critiques=num_critiques,
        critic_head=critic_head,
        mark_selector=mark_selector,
        continuations_per_mark=2,
        max_marks=1,
        min_mark_gap=1,
    )
    controller.config = OmegaConf.create(
        {
            "algorithm": {"gamma": 1.0, "lam": 1.0},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 1},
                "rollout": {
                    "n": 1,
                    "prompt_length": 8,
                    "response_length": 8,
                    "max_model_len": 64,
                },
            },
            "critic": {
                "ppo_mini_batch_size": 1,
                "forward_micro_batch_size_per_gpu": 1,
                "forward_max_token_len_per_gpu": 1024,
                "use_dynamic_bsz": False,
            },
            "trainer": {"critic_warmup": 2, "balance_batch": False},
        }
    )
    controller.tokenizer = _Tokenizer()
    controller.critique_delimiter_ids = [6] if num_critiques > 0 else []
    controller.solution_delimiter_ids = [7]
    controller.critique_instruction_ids = [30, 31] if num_critiques > 0 else []
    controller.critic_context_limit = 64
    controller.audit_path = None
    controller.trainer = SimpleNamespace(global_steps=1)
    return controller


def _bundle(*, reward=1.0, num_critiques=2):
    if num_critiques > 0:
        contexts = [
            build_critic_context(
                [11, 12],
                [40 + index, 50 + index],
                [21, 22, 23],
                critique_delimiter_ids=[6],
                solution_delimiter_ids=[7],
            )
            for index in range(num_critiques)
        ]
    else:
        contexts = [
            build_unconditioned_critic_context(
                [11, 12],
                [21, 22, 23],
                solution_delimiter_ids=[7],
            )
        ]
    critic_streams = max(1, num_critiques)
    return _Bundle(
        order=0,
        dataset_index=10,
        rollout_id="rollout",
        prompt_group_id="prompt",
        source_row=0,
        prompt_ids=[11, 12],
        solution_ids=[21, 22, 23],
        solution_log_probs=[-0.1, -0.2, -0.3],
        terminal_reward=reward,
        critique_ids=[[40 + index, 50 + index] for index in range(num_critiques)],
        critique_log_probs=[[-0.4, -0.5] for _ in range(num_critiques)],
        contexts=contexts,
        critic_values=[
            [0.2 + 0.1 * index, 0.4 + 0.1 * index, 0.6 + 0.1 * index, 0.8 + 0.1 * index]
            for index in range(critic_streams)
        ],
        critic_variances=[None] * critic_streams,
    )


def _source() -> DataProto:
    prompts = torch.tensor([[0, 11, 12]])
    responses = torch.tensor([[21, 22, 23, 0, 0, 0, 0, 0]])
    response_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0]]),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "advantages": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        },
        non_tensors={
            "raw_prompt": np.array([[{"role": "user", "content": "q"}]], dtype=object),
            "intermediate_mc_rollout_id": np.array(["rollout"], dtype=object),
        },
    )


def test_critic_targets_train_s0_dense_states_and_eos_for_every_critique() -> None:
    controller = _controller()
    bundle = _bundle()
    bundle.per_mark_targets = {2: 0.5}
    bundle.dense_targets = {1: 0.5, 2: 0.5}
    batch = controller._make_critic_batch([bundle])
    controller._set_critic_targets(batch, [bundle])
    assert batch.batch["critic_target_mask"].tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    assert batch.batch["critic_targets"][0].tolist() == pytest.approx([0.75, 0.5, 0.5, 1.0])


def test_warmup_and_all_failed_marks_train_s0_and_eos_only() -> None:
    controller = _controller()
    bundle = _bundle(reward=0.25)
    batch = controller._make_critic_batch([bundle])
    controller._set_critic_targets(batch, [bundle])
    assert batch.batch["critic_target_mask"].tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
    assert batch.batch["critic_targets"][:, [0, 3]].tolist() == [[0.25, 0.25], [0.25, 0.25]]


def test_critic_batch_preserves_explicit_s0_plus_solution_positions() -> None:
    batch = _controller()._make_critic_batch([_bundle()])
    assert batch.batch["critic_positions"][0].tolist() == [5, 6, 7, 8]
    assert batch.batch["critic_position_mask"][0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert isinstance(batch, DataProto)


def test_actor_packing_has_one_real_prompt_token_and_only_output_tokens_train() -> None:
    controller = _controller()
    actor_batch = controller._make_actor_batch(_source(), [_bundle()])
    assert len(actor_batch) == 3
    assert actor_batch.batch["prompts"].shape[1] == 1
    assert actor_batch.non_tensor_batch["intermediate_mc_actor_kind"].tolist() == [
        "solution",
        "critique",
        "critique",
    ]
    assert actor_batch.batch["response_mask"][0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert actor_batch.batch["response_mask"][1].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    torch.testing.assert_close(actor_batch.batch["old_log_probs"], actor_batch.batch["rollout_log_probs"])
    assert actor_batch.batch["response_mask"].sum().item() == 3 + 2 + 2
    assert "continuation" not in actor_batch.non_tensor_batch["intermediate_mc_actor_kind"].tolist()


def test_no_self_critique_uses_one_unconditioned_critic_stream_and_solution_only_actor() -> None:
    controller = _controller(num_critiques=0)
    bundle = _bundle(num_critiques=0)
    assert len(bundle.contexts) == 1
    assert bundle.contexts[0].token_ids == [11, 12, 7, 21, 22, 23]
    critic_batch = controller._make_critic_batch([bundle])
    assert len(critic_batch) == 1
    assert critic_batch.batch["critic_positions"].tolist() == [[2, 3, 4, 5]]

    actor_batch = controller._make_actor_batch(_source(), [bundle])
    assert len(actor_batch) == 1
    assert actor_batch.non_tensor_batch["intermediate_mc_actor_kind"].tolist() == ["solution"]
    assert actor_batch.batch["response_mask"].sum().item() == len(bundle.solution_ids)


def test_throughput_token_metrics_cover_all_generation_and_training_lanes() -> None:
    controller = _controller()
    bundle = _bundle()
    bundle.continuations = [
        ContinuationGeneration(mark=1, sample_index=0, token_ids=(61, 62, 63)),
        ContinuationGeneration(mark=2, sample_index=0, token_ids=(71, 72)),
    ]
    bundle.failed_continuations = [(2, 1)]
    critic_batch = controller._make_critic_batch([bundle])
    actor_batch = controller._make_actor_batch(_source(), [bundle])

    metrics = controller._token_metrics(
        [bundle],
        critic_batch=critic_batch,
        actor_batch=actor_batch,
    )

    assert metrics == {
        "intermediate_mc/tokens/prompt": 2.0,
        "intermediate_mc/tokens/solution_output": 3.0,
        "intermediate_mc/tokens/critique_input": 14.0,
        "intermediate_mc/tokens/critique_output": 4.0,
        "intermediate_mc/tokens/continuation_input": 11.0,
        "intermediate_mc/tokens/continuation_output": 5.0,
        "intermediate_mc/tokens/generation_input": 27.0,
        "intermediate_mc/tokens/generation_output": 12.0,
        "intermediate_mc/tokens/critic_input": 18.0,
        "intermediate_mc/tokens/actor_input": 23.0,
        "intermediate_mc/tokens/actor_train": 7.0,
        "intermediate_mc/continuation_attempts": 3.0,
    }


def test_no_self_critique_token_metrics_have_no_critique_lane() -> None:
    controller = _controller(num_critiques=0)
    bundle = _bundle(num_critiques=0)
    bundle.continuations = [ContinuationGeneration(mark=2, sample_index=0, token_ids=(71,))]
    critic_batch = controller._make_critic_batch([bundle])
    actor_batch = controller._make_actor_batch(_source(), [bundle])

    metrics = controller._token_metrics(
        [bundle],
        critic_batch=critic_batch,
        actor_batch=actor_batch,
    )

    assert metrics["intermediate_mc/tokens/critique_input"] == 0.0
    assert metrics["intermediate_mc/tokens/critique_output"] == 0.0
    assert metrics["intermediate_mc/tokens/critic_input"] == 6.0
    assert metrics["intermediate_mc/tokens/actor_input"] == 5.0
    assert metrics["intermediate_mc/tokens/actor_train"] == 3.0


def test_no_self_critique_builds_one_unconditioned_context_from_empty_child_record() -> None:
    controller = _controller(num_critiques=0)
    record = IntermediateMCGenerationRecord(
        rollout_id="rollout",
        critiques=(),
        selected_marks=(),
        continuations=(),
        failed_continuations=(),
        selector_diagnostics=(),
    )
    bundle = controller._build_bundles(_source(), {"rollout": record}, [1.0])[0]
    assert bundle.critique_ids == []
    assert bundle.critique_log_probs == []
    assert len(bundle.contexts) == 1
    assert bundle.contexts[0].critique_range == (2, 2)
    assert bundle.contexts[0].token_ids == [11, 12, 7, 21, 22, 23]
    assert bundle.critic_values == [None]


def _causal_log_probs(input_ids: torch.Tensor, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.randn(64, 7, generator=generator)
    projection = torch.randn(7, 64, generator=generator)
    hidden = embedding[input_ids].cumsum(dim=0)
    logits = hidden[:-1] @ projection
    return torch.log_softmax(logits, dim=-1).gather(1, input_ids[1:].unsqueeze(1)).squeeze(1)


def test_packed_and_conventional_causal_log_probs_are_identical() -> None:
    actor_batch = _controller()._make_actor_batch(_source(), [_bundle()])
    full = torch.tensor([11, 12, 21, 22, 23])
    conventional = _causal_log_probs(full)[1:]
    packed_full = actor_batch.batch["input_ids"][0, : len(full)]
    packed_all = _causal_log_probs(packed_full)
    packed = packed_all[actor_batch.batch["response_mask"][0, : len(full) - 1].bool()]
    torch.testing.assert_close(packed, conventional)


def test_optimizer_batches_fail_instead_of_adding_dummy_rows() -> None:
    controller = _controller()
    controller.trainer = SimpleNamespace(_get_dp_size=lambda _worker, _role: 2)
    controller.config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    batch = controller._make_actor_batch(_source(), [_bundle()])
    with pytest.raises(ValueError, match="optimizer padding is forbidden"):
        controller._validate_optimizer_batch(batch, role="actor", worker_group=object())
    assert len(batch) == 3


def test_native_warmup_uses_global_step_without_feature_checkpoint_state() -> None:
    controller = _controller()
    batch = DataProto.from_dict(non_tensors={"prompt_group_id": np.array(["p", "p"], dtype=object)})
    controller.trainer.global_steps = 2
    controller.prepare_generation_batch(batch)
    assert batch.non_tensor_batch["intermediate_mc_warmup"].tolist() == [True, True]
    controller.trainer.global_steps = 3
    controller.prepare_generation_batch(batch)
    assert batch.non_tensor_batch["intermediate_mc_warmup"].tolist() == [False, False]
    assert not hasattr(RayPPOTrainer, "_save_additional_trainer_state")


def test_child_record_is_extracted_once_and_removed_before_reward_logging() -> None:
    controller = _controller()
    record = IntermediateMCGenerationRecord(
        rollout_id="r",
        critiques=(CritiqueGeneration((1,), (-0.1,)), CritiqueGeneration((2,), (-0.2,))),
        selected_marks=(1,),
        continuations=(ContinuationGeneration(1, 0, (3,)),),
        failed_continuations=(),
        selector_diagnostics=(),
    )
    values = np.empty(1, dtype=object)
    values[0] = record
    # Real generation outputs always have a tensor batch; retain that invariant so
    # DataProto.__len__ continues to identify the one generated solution after the
    # controller removes its private non-tensor child record.
    output = DataProto.from_dict(
        tensors={"responses": torch.ones((1, 1), dtype=torch.long)},
        non_tensors={INTERMEDIATE_MC_CHILD_FIELD: values},
    )
    extracted = controller.extract_generation_records(output)
    assert extracted == {"r": record}
    assert INTERMEDIATE_MC_CHILD_FIELD not in output.non_tensor_batch


def _runtime_config():
    return OmegaConf.create(
        {
            "algorithm": {
                "intermediate_mc_value": {
                    "_target_": "verl.trainer.config.IntermediateMCValueConfig",
                    "enable": True,
                    "critic_head": "scalar",
                    "mark_selector": "random",
                },
                "adv_estimator": "gae",
                "gamma": 1.0,
                "use_kl_in_reward": False,
                "rollout_correction": {"rollout_is": None, "rollout_rs": None, "bypass_mode": False},
                "opsd": {"enable": False},
            },
            "actor_rollout_ref": {
                "model": {"trust_remote_code": False, "override_config": {}},
                "actor": {
                    "strategy": "fsdp",
                    "ppo_mini_batch_size": 1,
                    "ulysses_sequence_parallel_size": 1,
                    "use_kl_loss": False,
                    "use_rollout_log_probs": False,
                    "use_prefix_grouper": False,
                    "router_replay": {"mode": "none"},
                    "policy_loss": {"loss_mode": "vanilla"},
                },
                "rollout": {
                    "name": "vllm",
                    "n": 1,
                    "temperature": 1.0,
                    "calculate_log_probs": False,
                    "max_model_len": 64,
                    "logprobs_mode": "processed_logprobs",
                    "skip_rollout": False,
                    "enable_rollout_routing_replay": False,
                    "multi_turn": {"enable": False},
                    "val_kwargs": {"temperature": 1.0},
                },
            },
            "critic": {
                "strategy": "fsdp",
                "enable": None,
                "cliprange_value": 0.2,
                "ppo_mini_batch_size": 1,
                "ulysses_sequence_parallel_size": 1,
                "model": {"path": "local-critic", "tokenizer_path": None},
            },
            "trainer": {
                "use_legacy_worker_impl": "auto",
                "critic_warmup": 30,
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
            "reward": {
                "reward_manager": {"source": "register", "name": "naive"},
                "reward_kwargs": {},
                "reward_model": {
                    "enable": False,
                    "launch_reward_fn_async": False,
                    "reward_loop_source": None,
                    "reward_loop_module_path": None,
                    "reward_loop_class_name": None,
                    "enable_resource_pool": False,
                },
            },
            "data": {
                "train_batch_size": 1,
                "gen_batch_size": 1,
                "use_dataset_responses": False,
                "max_prompt_length": 16,
                "max_response_length": 16,
            },
        }
    )


def test_runtime_config_keeps_native_loss_and_enables_behavior_log_probs() -> None:
    config = _runtime_config()
    validate_intermediate_mc_runtime_config(config)
    assert config.critic.enable is True
    assert config.actor_rollout_ref.rollout.calculate_log_probs is True
    assert config.actor_rollout_ref.actor.use_rollout_log_probs is True
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


@pytest.mark.parametrize("num_critiques", [0, 2])
def test_configured_optimizer_rows_cover_no_critique_and_self_critique_modes(num_critiques) -> None:
    config = _runtime_config()
    config.algorithm.intermediate_mc_value.num_critiques = num_critiques
    feature = IntermediateMCValueConfig(enable=True, num_critiques=num_critiques)
    rows = configured_optimizer_rows(config, feature)
    assert rows == {
        "solutions": 1,
        "critic": max(1, num_critiques),
        "actor": 1 + num_critiques,
        "actor_critiques": num_critiques,
    }


def test_disabled_runtime_config_is_a_strict_noop() -> None:
    config = _runtime_config()
    config.algorithm.intermediate_mc_value.enable = False
    config.reward.reward_manager.name = "conditional_logprob"
    config.critic.model.path = "hdfs://unsupported/critic"
    validate_intermediate_mc_runtime_config(config)
    assert config.critic.enable is None
    assert config.actor_rollout_ref.rollout.calculate_log_probs is False
    assert config.actor_rollout_ref.actor.use_rollout_log_probs is False


def test_intermediate_mc_disables_streaming_reward_loop_and_preserves_rollout_identity() -> None:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _runtime_config()
    trainer.reward_fn = SimpleNamespace()
    trainer.use_rm = False
    trainer.intermediate_mc_controller = object()
    assert trainer._should_enable_agent_reward_loop() is False

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = None
    output = _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2]]),
        response_ids=torch.tensor([[3, 0]]),
        response_mask=torch.tensor([[1, 0]]),
        response_logprobs=torch.tensor([[-0.1, 0.0]]),
        input_ids=torch.tensor([[1, 2, 3, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        position_ids=torch.tensor([[0, 1, 2, 0]]),
        num_turns=1,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )
    processed = worker._postprocess(
        [output],
        input_non_tensor_batch={"intermediate_mc_rollout_id": np.array(["rollout:0"], dtype=object)},
    )
    assert processed.non_tensor_batch["intermediate_mc_rollout_id"].tolist() == ["rollout:0"]

    trainer.intermediate_mc_controller = None
    assert trainer._should_enable_agent_reward_loop() is True


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("algorithm.gamma", 0.99, "gamma=1"),
        ("actor_rollout_ref.rollout.temperature", 0.9, "temperature=1.0"),
        ("actor_rollout_ref.rollout.max_model_len", None, "explicit positive"),
        ("reward.reward_model.launch_reward_fn_async", True, "blocking iteration barrier"),
        ("reward.reward_manager.name", "conditional_logprob", "conditional_logprob"),
        ("critic.model.path", "hdfs://models/critic", "HDFS"),
        ("critic.model.tokenizer_path", "hdfs://models/tokenizer", "HDFS"),
        (
            "reward.reward_kwargs.use_response_logprob_reward_for_uniform_outcome_groups",
            True,
            "grouped reward transformations",
        ),
        ("reward.reward_kwargs.use_shortest_success_reward", True, "grouped reward transformations"),
        ("reward.reward_kwargs.use_longest_success_penalty_reward", True, "grouped reward transformations"),
        ("critic.cliprange_value", 1.1, "normalized epsilon"),
    ],
)
def test_runtime_config_rejects_unsupported_modes(path, value, message) -> None:
    config = _runtime_config()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match=message):
        validate_intermediate_mc_runtime_config(config)


def test_runtime_config_rejects_optimizer_cardinality_before_worker_allocation() -> None:
    config = _runtime_config()
    config.algorithm.intermediate_mc_value.num_critiques = 2
    config.data.gen_batch_size = 2
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 4
    with pytest.raises(ValueError, match="actor rows .* global PPO minibatch"):
        validate_intermediate_mc_runtime_config(config)


def test_runtime_config_rejects_minibatch_that_cannot_be_sharded_across_dp() -> None:
    config = _runtime_config()
    config.algorithm.intermediate_mc_value.num_critiques = 0
    config.data.train_batch_size = 6
    config.data.gen_batch_size = 6
    config.trainer.n_gpus_per_node = 2
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 3
    config.critic.ppo_mini_batch_size = 2
    with pytest.raises(ValueError, match="actor global PPO minibatch .* actor DP size"):
        validate_intermediate_mc_runtime_config(config)


def _agent_loop(*, include_critiques=True):
    loop = IntermediateMCAgentLoop.__new__(IntermediateMCAgentLoop)
    loop.feature = IntermediateMCValueConfig(
        num_critiques=2,
        continuations_per_mark=2,
        max_marks=1,
        min_mark_gap=1,
    )
    loop.response_length = 8
    loop.max_model_len = 64
    loop.critique_instruction_ids = [30]
    loop.critique_delimiter_ids = [31]
    loop.solution_delimiter_ids = [32]
    return loop


def test_sampling_parameters_force_temperature_processed_logprobs_and_explicit_cap() -> None:
    params = IntermediateMCAgentLoop._sampling_params(
        {"temperature": 0.4, "logprobs": False, "max_new_tokens": 99},
        max_tokens=7,
    )
    assert params["temperature"] == 1.0
    assert params["logprobs"] is True
    assert params["max_tokens"] == 7
    assert "max_new_tokens" not in params


def test_no_self_critique_solution_stage_emits_no_critique_requests() -> None:
    loop = _agent_loop()
    loop.feature = IntermediateMCValueConfig(num_critiques=0, max_marks=0, min_mark_gap=1)
    generated_kinds: list[str] = []

    async def fake_generate(_route_key, _prompt, _params, *, max_tokens, kind):
        generated_kinds.append(kind)
        return TokenOutput(token_ids=[20, 21], log_probs=[-0.1, -0.2])

    loop._generate = fake_generate
    output = asyncio.run(
        loop.run(
            {"temperature": 1.0},
            prompt_ids_override=[10],
            intermediate_mc_stage="solution",
            intermediate_mc_rollout_id="rollout",
            intermediate_mc_critic_context_limit=64,
            intermediate_mc_warmup=False,
            intermediate_mc_global_step=1,
            index=0,
        )
    )
    record = output.extra_fields[INTERMEDIATE_MC_CHILD_FIELD]
    assert generated_kinds == ["solution"]
    assert record.critiques == ()
    assert record.continuations == ()


def test_critique_failure_drains_every_child_before_raising() -> None:
    loop = _agent_loop()
    completed: list[str] = []
    route_keys: list[str] = []

    async def fake_generate(route_key, _prompt, _params, *, max_tokens, kind):
        route_keys.append(route_key)
        await asyncio.sleep(0.002 if kind == "critique[0]" else 0.005)
        completed.append(kind)
        if kind == "critique[0]":
            raise RuntimeError("injected critique failure")
        return TokenOutput(token_ids=[1], log_probs=[-0.1])

    loop._generate = fake_generate
    with pytest.raises(RuntimeError, match="after draining"):
        asyncio.run(
            loop._generate_children(
                route_key="sticky",
                prompt_ids=[10],
                solution_ids=[20, 21, 22],
                solution_log_probs=[-0.1, -0.2, -0.3],
                selected_marks=[1],
                sampling_params={"temperature": 1.0},
                critic_context_limit=64,
                include_critiques=True,
            )
        )
    assert len(completed) == 4
    assert route_keys == ["sticky"] * 4


def test_individual_continuation_failure_is_omitted_after_drain() -> None:
    loop = _agent_loop()

    async def fake_generate(_route_key, _prompt, _params, *, max_tokens, kind):
        await asyncio.sleep(0)
        if kind.endswith(",0]"):
            raise RuntimeError("injected continuation failure")
        return TokenOutput(token_ids=[7], log_probs=[-0.1])

    loop._generate = fake_generate
    critiques, continuations, failures, _ = asyncio.run(
        loop._generate_children(
            route_key="sticky",
            prompt_ids=[10],
            solution_ids=[20, 21, 22],
            solution_log_probs=[-0.1, -0.2, -0.3],
            selected_marks=[1],
            sampling_params={"temperature": 1.0},
            critic_context_limit=64,
            include_critiques=False,
        )
    )
    assert critiques == ()
    assert [(item.mark, item.sample_index) for item in continuations] == [(1, 1)]
    assert failures == ((1, 0),)


def test_ema_marks_are_selected_from_average_critic_values_after_scoring() -> None:
    controller = _controller(mark_selector="ema")
    controller.feature = IntermediateMCValueConfig(
        num_critiques=2,
        mark_selector="ema",
        continuations_per_mark=2,
        max_marks=1,
        min_mark_gap=1,
        mark_start_fraction=0.0,
        mark_end_fraction=1.0,
        ema_alpha=1.0,
        ema_baseline_token=1,
        ema_ratio_up=2.0,
        ema_ratio_down=0.5,
    )
    bundle = _bundle()
    bundle.critic_values = [
        [0.4, 0.1, 0.5, 0.4],
        [0.6, 0.3, 0.7, 0.4],
    ]
    audit: list[tuple[str, dict]] = []
    controller._audit = lambda event, **payload: audit.append((event, payload))

    controller._select_ema_marks([bundle])
    assert bundle.marks == [2]
    selection = next(payload for event, payload in audit if event == "mark_selection" and "token" in payload)
    assert selection["value"] == pytest.approx(0.6)
    assert selection["reference"] == pytest.approx(0.2)
    assert selection["ratio"] == pytest.approx(3.0)

    request = controller._make_deferred_continuation_request(_source(), [bundle])
    assert request is not None
    assert request.non_tensor_batch["intermediate_mc_stage"].tolist() == ["continuations"]
    assert request.non_tensor_batch["intermediate_mc_selected_marks"][0] == [2]


@pytest.mark.parametrize("selector", ["ema", "variance"])
def test_critic_dependent_selectors_never_select_marks_in_initial_generation(selector) -> None:
    loop = _agent_loop()
    loop.feature = IntermediateMCValueConfig(
        critic_head="beta" if selector == "variance" else "scalar",
        mark_selector=selector,
        max_marks=1,
        min_mark_gap=1,
    )
    marks, diagnostics = loop._select_marks(
        [1, 2, 3],
        rollout_id="rollout",
        global_step=1,
        dataset_index=0,
    )
    assert marks == []
    assert diagnostics == []


def test_deferred_continuation_stage_profiles_and_cleans_up_on_generation_failure() -> None:
    controller = _controller(critic_head="beta", mark_selector="variance")
    bundle = _bundle()
    bundle.marks = [1]
    events: list[str] = []

    def fail_generation(_request):
        events.append("generate")
        raise RuntimeError("injected variance generation failure")

    controller.trainer = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(
            wake_up_replicas=lambda: events.append("wake"),
            sleep_replicas=lambda: events.append("sleep"),
        ),
        async_rollout_manager=SimpleNamespace(
            start_profile=lambda: events.append("start_profile"),
            stop_profile=lambda: events.append("stop_profile"),
            generate_sequences=fail_generation,
        ),
    )
    controller._make_deferred_continuation_request = lambda _source, _bundles: DataProto.from_dict(
        non_tensors={"request": np.array([1])}
    )
    with pytest.raises(RuntimeError, match="injected variance"):
        controller._generate_deferred_continuations(
            _source(),
            [bundle],
            {},
            profile_rollout=True,
        )
    assert events == ["wake", "start_profile", "generate", "sleep", "stop_profile"]


@pytest.mark.parametrize(
    ("wake_up", "failure", "expected_events"),
    [
        (True, "wake", ["wake", "sleep"]),
        (True, "start_profile", ["wake", "start_profile", "sleep", "stop_profile"]),
        (True, "sleep", ["wake", "start_profile", "generate", "sleep", "stop_profile"]),
        (True, "stop_profile", ["wake", "start_profile", "generate", "sleep", "stop_profile"]),
        (False, "start_profile", ["start_profile", "sleep", "stop_profile"]),
        (False, "generate", ["start_profile", "generate", "sleep", "stop_profile"]),
        (False, "sleep", ["start_profile", "generate", "sleep", "stop_profile"]),
        (False, "stop_profile", ["start_profile", "generate", "sleep", "stop_profile"]),
    ],
)
def test_blocking_rollout_lifecycle_attempts_independent_cleanup(
    wake_up: bool,
    failure: str,
    expected_events: list[str],
) -> None:
    controller = _controller()
    events: list[str] = []

    def operation(name: str, result=None):
        def run(*_args, **_kwargs):
            events.append(name)
            if failure == name:
                raise RuntimeError(f"injected {name} failure")
            return result

        return run

    controller.trainer = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(
            wake_up_replicas=operation("wake"),
            sleep_replicas=operation("sleep"),
        ),
        async_rollout_manager=SimpleNamespace(
            start_profile=operation("start_profile"),
            stop_profile=operation("stop_profile"),
            generate_sequences=operation("generate", DataProto()),
        ),
    )
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        controller._generate_sequences_with_lifecycle(
            DataProto(),
            profile_rollout=True,
            wake_up_replicas=wake_up,
        )
    assert events == expected_events


@pytest.mark.parametrize(("wake_up", "primary_failure"), [(True, "start_profile"), (False, "generate")])
def test_blocking_rollout_lifecycle_retains_every_simultaneous_cleanup_failure(
    wake_up: bool,
    primary_failure: str,
) -> None:
    controller = _controller()
    events: list[str] = []
    failures = {primary_failure, "sleep", "stop_profile"}

    def operation(name: str, result=None):
        def run(*_args, **_kwargs):
            events.append(name)
            if name in failures:
                raise RuntimeError(f"injected {name} failure")
            return result

        return run

    controller.trainer = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(
            wake_up_replicas=operation("wake"),
            sleep_replicas=operation("sleep"),
        ),
        async_rollout_manager=SimpleNamespace(
            start_profile=operation("start_profile"),
            stop_profile=operation("stop_profile"),
            generate_sequences=operation("generate", DataProto()),
        ),
    )
    with pytest.raises(RuntimeError, match=f"injected {primary_failure} failure") as error_info:
        controller._generate_sequences_with_lifecycle(
            DataProto(),
            profile_rollout=True,
            wake_up_replicas=wake_up,
        )

    notes = error_info.value.__notes__
    assert any("rollout sleep cleanup" in note and "injected sleep failure" in note for note in notes)
    assert any("rollout profile cleanup" in note and "injected stop_profile failure" in note for note in notes)
    expected_events = ["wake"] if wake_up else []
    expected_events.append("start_profile")
    if primary_failure == "generate":
        expected_events.append("generate")
    expected_events.extend(("sleep", "stop_profile"))
    assert events == expected_events


def test_blocking_rollout_lifecycle_retains_secondary_cleanup_failure_without_primary_error() -> None:
    controller = _controller()

    def fail_sleep():
        raise RuntimeError("injected sleep failure")

    def fail_stop_profile():
        raise RuntimeError("injected stop_profile failure")

    controller.trainer = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(sleep_replicas=fail_sleep),
        async_rollout_manager=SimpleNamespace(
            start_profile=lambda: None,
            stop_profile=fail_stop_profile,
            generate_sequences=lambda _request: DataProto(),
        ),
    )
    with pytest.raises(RuntimeError, match="injected sleep failure") as error_info:
        controller._generate_sequences_with_lifecycle(
            DataProto(),
            profile_rollout=True,
            wake_up_replicas=False,
        )
    assert any(
        "rollout profile cleanup" in note and "injected stop_profile failure" in note
        for note in error_info.value.__notes__
    )
