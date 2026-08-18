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

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.config import IntermediateMCValueConfig
from verl.trainer.ppo.intermediate_mc_value import build_critic_context
from verl.trainer.ppo.ray_trainer_intermediate_mc import (
    IntermediateMCRayPPOTrainer,
    _Bundle,
    validate_intermediate_mc_runtime_config,
)


def _trainer(recipe="scalar_random", warmup=30, continuations_per_mark=1):
    trainer = IntermediateMCRayPPOTrainer.__new__(IntermediateMCRayPPOTrainer)
    trainer.feature = IntermediateMCValueConfig(
        recipe=recipe,
        num_critiques=2,
        continuations_per_mark=continuations_per_mark,
        critic_warmup_updates=warmup,
    )
    trainer.config = SimpleNamespace(
        algorithm=SimpleNamespace(gamma=1.0, lam=1.0),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_mini_batch_size=1),
            rollout=SimpleNamespace(n=1, prompt_length=4, response_length=5),
        ),
        critic=SimpleNamespace(
            ppo_mini_batch_size=1,
            forward_micro_batch_size_per_gpu=1,
            forward_max_token_len_per_gpu=1024,
            use_dynamic_bsz=False,
        ),
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9)
    trainer.global_steps = 7
    trainer.critic_update_count = 0
    trainer._tokenizer_fingerprint = "tokenizer"
    trainer._audit_path = None
    return trainer


def _bundle(reward=1.0):
    contexts = [
        build_critic_context(
            [1],
            [2],
            [3, 4, 5],
            critique_delimiter_ids=[6],
            solution_delimiter_ids=[7],
        )
        for _ in range(2)
    ]
    return _Bundle(
        order=0,
        dataset_index=10,
        rollout_id="rollout",
        prompt_group_id="prompt",
        source_row=0,
        prompt_ids=[1],
        solution_ids=[3, 4, 5],
        terminal_reward=reward,
        contexts=contexts,
        critic_values=[[0.2, 0.4, 0.6, 0.8], [0.4, 0.6, 0.8, 1.0]],
    )


def test_warmup_supervises_only_terminal_solution_position() -> None:
    trainer = _trainer()
    bundle = _bundle()
    critic_batch, mapping = trainer._make_critic_batch([bundle])
    trainer._set_warmup_targets(critic_batch, mapping, [bundle])
    assert critic_batch.batch["critic_target_mask"].tolist() == [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert critic_batch.batch["critic_targets"][:, 3].tolist() == [1.0, 1.0]


def test_dense_marks_supervise_earlier_tokens_and_never_delimiter() -> None:
    trainer = _trainer()
    bundle = _bundle()
    bundle.dense_targets = {1: 0.25, 2: 0.75}
    critic_batch, mapping = trainer._make_critic_batch([bundle])
    trainer._set_training_targets(critic_batch, mapping, [bundle])
    assert critic_batch.batch["critic_target_mask"].tolist() == [
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]
    assert critic_batch.batch["critic_targets"][0].tolist() == [0.0, 0.25, 0.75, 1.0]


def test_solution_advantages_average_critiques_and_use_unsupervised_delimiter() -> None:
    trainer = _trainer()
    advantages = trainer._solution_advantages([_bundle()], response_width=5)
    assert advantages.shape == (1, 5)
    assert torch.count_nonzero(advantages[:, 3:]) == 0
    # Whitening is over exactly the three solution tokens.
    torch.testing.assert_close(advantages[0, :3].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[0, :3].var(unbiased=False), torch.tensor(1.0), atol=1e-6, rtol=0)


def test_checkpoint_contract_restores_exact_critic_update_count(tmp_path) -> None:
    trainer = _trainer(recipe="beta_variance")
    trainer.critic_update_count = 29
    trainer._save_additional_trainer_state(str(tmp_path))
    state = json.loads((tmp_path / trainer.STATE_FILENAME).read_text())
    assert state["critic_update_count"] == 29
    trainer._validate_additional_trainer_state(str(tmp_path))
    trainer.critic_update_count = 0
    trainer._load_additional_trainer_state(str(tmp_path))
    assert trainer.critic_update_count == 29


def test_checkpoint_contract_mismatch_fails_before_restore(tmp_path) -> None:
    trainer = _trainer(recipe="scalar_random")
    trainer._save_additional_trainer_state(str(tmp_path))
    incompatible = _trainer(recipe="beta_variance")
    with pytest.raises(RuntimeError, match="contract"):
        incompatible._validate_additional_trainer_state(str(tmp_path))


def test_dummy_critic_padding_has_zero_target_mass() -> None:
    trainer = _trainer()
    trainer.config.critic.ppo_mini_batch_size = 4
    bundle = _bundle()
    critic_batch, mapping = trainer._make_critic_batch([bundle])
    trainer._set_warmup_targets(critic_batch, mapping, [bundle])
    padded = trainer._pad_critic_batch(critic_batch)
    assert len(padded) == 4
    assert padded.batch["critic_target_mask"][-2:].sum().item() == 0.0


def test_continuation_rows_are_not_part_of_actor_key_contract() -> None:
    assert IntermediateMCRayPPOTrainer._actor_keys() == [
        "prompts",
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "rollout_log_probs",
    ]
    assert "continuation" not in " ".join(IntermediateMCRayPPOTrainer._actor_keys())


def test_context_batch_positions_include_delimiter_then_every_solution_token() -> None:
    trainer = _trainer()
    batch, _ = trainer._make_critic_batch([_bundle()])
    assert batch.batch["critic_positions"][0].tolist() == [3, 4, 5, 6]
    assert batch.batch["critic_position_mask"][0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert isinstance(batch, DataProto)


def _runtime_config():
    return OmegaConf.create(
        {
            "algorithm": {
                "intermediate_mc_value": {
                    "_target_": "verl.trainer.config.IntermediateMCValueConfig",
                    "enable": True,
                    "recipe": "scalar_random",
                    "actor_loss_mode": "dppo_tv",
                },
                "adv_estimator": "gae",
                "use_kl_in_reward": False,
                "rollout_correction": {"rollout_is": None, "rollout_rs": None, "bypass_mode": False},
                "opsd": {"enable": False},
            },
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "use_kl_loss": False,
                    "use_rollout_log_probs": False,
                    "policy_loss": {"loss_mode": "vanilla"},
                },
                "rollout": {
                    "n": 1,
                    "temperature": 1.0,
                    "calculate_log_probs": False,
                    "multi_turn": {"enable": False},
                    "val_kwargs": {"temperature": 1.0},
                },
            },
            "critic": {"strategy": "fsdp", "enable": None},
            "trainer": {"use_legacy_worker_impl": "auto", "critic_warmup": 0},
            "reward": {
                "reward_model": {
                    "enable": False,
                    "launch_reward_fn_async": False,
                    "reward_loop_source": None,
                    "reward_loop_module_path": None,
                    "reward_loop_class_name": None,
                }
            },
            "data": {"use_dataset_responses": False},
        }
    )


def test_runtime_config_forces_recorded_behavior_probs_and_dppo_default() -> None:
    config = _runtime_config()
    validate_intermediate_mc_runtime_config(config)
    assert config.critic.enable is True
    assert config.actor_rollout_ref.rollout.calculate_log_probs is True
    assert config.actor_rollout_ref.actor.use_rollout_log_probs is True
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "dppo_tv"


def test_disabled_runtime_config_is_a_strict_noop() -> None:
    config = _runtime_config()
    config.algorithm.intermediate_mc_value.enable = False
    validate_intermediate_mc_runtime_config(config)
    assert config.critic.enable is None
    assert config.actor_rollout_ref.rollout.calculate_log_probs is False
    assert config.actor_rollout_ref.actor.use_rollout_log_probs is False
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


def test_runtime_config_rejects_rollout_correction_and_non_unit_temperature() -> None:
    config = _runtime_config()
    config.algorithm.rollout_correction.rollout_is = "sequence"
    with pytest.raises(ValueError, match="rollout correction"):
        validate_intermediate_mc_runtime_config(config)
    config = _runtime_config()
    config.actor_rollout_ref.rollout.temperature = 0.9
    with pytest.raises(ValueError, match="temperature=1.0"):
        validate_intermediate_mc_runtime_config(config)


def test_actor_batch_contains_only_solutions_and_critiques_with_behavior_denominator() -> None:
    trainer = _trainer()
    width = 5
    tensor_row = {
        "prompts": torch.tensor([[0, 1]]),
        "responses": torch.tensor([[3, 4, 5, 0, 0]]),
        "response_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        "input_ids": torch.tensor([[0, 1, 3, 4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0, 0]]),
        "rollout_log_probs": torch.tensor([[-0.1, -0.2, -0.3, 0.0, 0.0]]),
    }
    source = DataProto.from_dict(
        tensors=tensor_row,
        non_tensors={"source_only": np.array(["solution"], dtype=object)},
    )
    bundle = _bundle()
    bundle.critique_rows = []
    for offset in (0.0, 1.0):
        critique_tensors = {key: value.clone() for key, value in tensor_row.items()}
        critique_tensors["rollout_log_probs"] += offset
        bundle.critique_rows.append(
            DataProto.from_dict(
                tensors=critique_tensors,
                non_tensors={"critique_only": np.array([offset], dtype=object)},
            )
        )
    actor_batch = trainer._make_actor_batch(source, [bundle])
    assert len(actor_batch) == 3
    assert actor_batch.non_tensor_batch == {}
    torch.testing.assert_close(actor_batch.batch["old_log_probs"], actor_batch.batch["rollout_log_probs"])
    assert actor_batch.batch["response_mask"].shape == (3, width)


class _CheckpointManager:
    def __init__(self):
        self.sleep_calls = 0

    def sleep_replicas(self) -> None:
        self.sleep_calls += 1


def _request_source() -> DataProto:
    raw_prompt = np.empty(1, dtype=object)
    raw_prompt[0] = [{"role": "user", "content": "q"}]
    return DataProto.from_dict(non_tensors={"raw_prompt": raw_prompt})


def _generated_continuation_row() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 0, 0, 1]]),
            "responses": torch.tensor([[8, 9, 0, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0, 0, 0]]),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2, 0.0, 0.0, 0.0]]),
            "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0]]),
            "input_ids": torch.tensor([[0, 0, 0, 1, 8, 9, 0, 0, 0]]),
            "position_ids": torch.tensor([[0, 0, 0, 0, 1, 2, 0, 0, 0]]),
        }
    )


def test_continuation_generation_failure_still_sleeps_inference_replicas() -> None:
    trainer = _trainer()
    trainer.checkpoint_manager = _CheckpointManager()
    bundle = _bundle()
    bundle.marks = [1]

    def fail_generation(_request, **_kwargs):
        raise RuntimeError("injected generation failure")

    trainer._generate_rows_with_isolation = fail_generation
    with pytest.raises(RuntimeError, match="injected"):
        trainer._run_continuations(_request_source(), [bundle])
    assert trainer.checkpoint_manager.sleep_calls == 1


def test_partial_continuation_failure_averages_only_successes() -> None:
    trainer = _trainer(continuations_per_mark=2)
    trainer.checkpoint_manager = _CheckpointManager()
    bundle = _bundle()
    bundle.marks = [1]
    captured_prompts = []

    def generate(request, **_kwargs):
        captured_prompts.extend(request.non_tensor_batch["prompt_ids_override"].tolist())
        return [_generated_continuation_row(), None]

    trainer._generate_rows_with_isolation = generate
    trainer._continuation_rewards_with_isolation = lambda _batch: [0.75]
    trainer._run_continuations(_request_source(), [bundle])
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert captured_prompts == [[1, 3], [1, 3]]
    assert bundle.per_mark_targets == {1: 0.75}
    assert bundle.dense_targets == {1: 0.75}


def test_all_failed_continuations_omit_the_mark() -> None:
    trainer = _trainer(continuations_per_mark=2)
    trainer.checkpoint_manager = _CheckpointManager()
    bundle = _bundle()
    bundle.marks = [1]
    trainer._generate_rows_with_isolation = lambda _request, **_kwargs: [None, None]
    trainer._run_continuations(_request_source(), [bundle])
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert bundle.per_mark_targets == {}
    assert bundle.dense_targets == {}
