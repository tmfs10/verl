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
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.branch_revision_agent_loop import (
    BRANCH_REVISION_CHILD_FIELD,
    BranchRevisionAgentLoop,
    BranchRevisionCritiqueGeneration,
    BranchRevisionGenerationRecord,
)
from verl.trainer.config import BRANCH_REVISION_CRITIQUE_PROMPT, BranchRevisionGRPOConfig
from verl.trainer.ppo import ray_trainer_branch_revision as branch_controller_module
from verl.trainer.ppo.branch_revision_grpo import (
    decode_exact,
    encode_followup_user_turn,
    parse_branch_revision,
    strip_terminal_eos,
    validate_binary_reward_row,
)
from verl.trainer.ppo.core_algos import compute_policy_loss_dppo_tv, compute_policy_loss_vanilla
from verl.trainer.ppo.ray_trainer_branch_revision import (
    BranchRevisionGRPOController,
    _Bundle,
    validate_branch_revision_runtime_config,
)
from verl.workers.actor import dp_actor
from verl.workers.rollout.replica import TokenOutput


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 100 for char in text]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join("<eos>" if int(token) == self.eos_token_id else chr(int(token) - 100) for token in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        rendered = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return self.encode(rendered) if tokenize else rendered


TOKENIZER = _CharTokenizer()


def _ids(text: str) -> list[int]:
    return TOKENIZER.encode(text, add_special_tokens=False)


def test_agent_loop_package_registers_branch_revision_in_a_fresh_worker_process() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import verl.experimental.agent_loop; "
            "from verl.experimental.agent_loop.agent_loop import _agent_loop_registry; "
            "assert 'branch_revision_agent' in _agent_loop_registry, _agent_loop_registry"
        ),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    assert result.returncode == 0, result.stderr


VALID_ANALYSIS = (
    "1. PRUNING: The setup pruned candidates; the algebra did not.\n"
    "2. SWITCH: Switch at the equality and use an invariant.\n"
    "3. EDIT: Replace the equality with the invariant.\n"
)


def _structured(branch: str, replacement: str, prefix: str = VALID_ANALYSIS) -> str:
    return f"{prefix}<branch>{branch}</branch>\n<new continuation>{replacement}</new continuation>"


def test_critique_prompt_is_bounded_and_has_no_copyable_placeholder_values() -> None:
    assert "do not continue it or solve the problem again" in BRANCH_REVISION_CRITIQUE_PROMPT
    assert "three short numbered paragraphs" in BRANCH_REVISION_CRITIQUE_PROMPT
    assert "open <branch> before the numbered critique" in BRANCH_REVISION_CRITIQUE_PROMPT
    assert "inside <branch> is not analysis" in BRANCH_REVISION_CRITIQUE_PROMPT
    assert "the exact quoted section to replace" not in BRANCH_REVISION_CRITIQUE_PROMPT
    assert "the replacement text" not in BRANCH_REVISION_CRITIQUE_PROMPT
    assert "not write anything after" in BRANCH_REVISION_CRITIQUE_PROMPT


def test_critique_instruction_is_a_new_user_turn_with_visible_assistant_output() -> None:
    ids = encode_followup_user_turn(BRANCH_REVISION_CRITIQUE_PROMPT, TOKENIZER)
    text = decode_exact(ids, TOKENIZER)
    assert text.startswith("</assistant><user>--- BEGIN CRITIQUE TASK ---")
    assert text.endswith("</user><assistant><think>\n\n</think>\n\n")


def test_followup_boundary_uses_the_actual_conversation_context() -> None:
    class ContextSensitiveTokenizer(_CharTokenizer):
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        ):
            rendered = "".join(
                f"<{index}:{message['role']}>{message['content']}</{message['role']}>"
                for index, message in enumerate(messages)
            )
            if add_generation_prompt:
                rendered += f"<{len(messages)}:assistant>"
                if not enable_thinking:
                    rendered += "<think>\n\n</think>\n\n"
            return self.encode(rendered) if tokenize else rendered

    tokenizer = ContextSensitiveTokenizer()
    default_suffix = decode_exact(encode_followup_user_turn("review", tokenizer), tokenizer)
    contextual_suffix = decode_exact(
        encode_followup_user_turn(
            "review",
            tokenizer,
            prior_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            assistant_content="attempt",
        ),
        tokenizer,
    )
    assert default_suffix.startswith("</assistant><1:user>")
    assert contextual_suffix.startswith("</assistant><3:user>")
    assert default_suffix != contextual_suffix


def test_exact_parser_preserves_whitespace_unicode_and_truncates_at_unique_branch() -> None:
    solution = "use α\n  dead end\nthen waste"
    critique = _structured("  dead end\n", "  try β instead\n")
    parsed = parse_branch_revision(
        _ids(solution) + [TOKENIZER.eos_token_id],
        _ids(critique) + [TOKENIZER.eos_token_id],
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.branch_text == "  dead end\n"
    assert parsed.new_continuation_text == "  try β instead\n"
    assert parsed.revised_text == "use α\n  try β instead\n"
    assert decode_exact(parsed.revised_prefix_ids, TOKENIZER) == parsed.revised_text


def test_terminal_eos_stripping_preserves_interior_special_token() -> None:
    token_ids = [*_ids("a"), TOKENIZER.eos_token_id, *_ids("b"), TOKENIZER.eos_token_id]
    stripped = strip_terminal_eos(token_ids, TOKENIZER)
    assert stripped == token_ids[:-1]
    assert decode_exact(stripped, TOKENIZER) == "a<eos>b"


@pytest.mark.parametrize(
    ("critique", "reason"),
    [
        ("<branch>x</branch>", "tag_count"),
        ("<branch>x</branch>\n<new continuation>new</new continuation>", "critique_section_count"),
        (
            "2. SWITCH: switch\n1. PRUNING: prune\n3. EDIT: edit\n"
            "<branch>x</branch><new continuation>new</new continuation>",
            "critique_section_order",
        ),
        (
            "preamble\n" + VALID_ANALYSIS + "<branch>x</branch><new continuation>new</new continuation>",
            "text_before_critique",
        ),
        (
            "1. PRUNING: \n2. SWITCH: switch\n3. EDIT: edit\n"
            "<branch>x</branch><new continuation>new</new continuation>",
            "empty_pruning",
        ),
        (_structured("", "new"), "empty_branch"),
        (_structured("x", "   "), "empty_new_continuation"),
        (_structured("same", "new"), "branch_not_unique"),
        (_structured("x", "new") + " trailing", "text_after_tags"),
        ("<branch>x</branch> not whitespace <new continuation>new</new continuation>", "text_between_tags"),
    ],
)
def test_parser_fails_closed_with_reason(critique: str, reason: str) -> None:
    solution = "same same" if "same" in critique else "x solution"
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(critique),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert not parsed.valid
    assert parsed.reason == reason


def test_parser_enforces_token_caps_without_trimming_contents() -> None:
    parsed = parse_branch_revision(
        _ids("abcd rest"),
        _ids(_structured("abcd", "new")),
        TOKENIZER,
        branch_max_tokens=3,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "branch_token_cap"


@pytest.mark.parametrize(("row", "expected"), [([0.0, 0.0], 0.0), ([0.0, 1.0, 0.0], 1.0)])
def test_binary_reward_validation_accepts_only_one_terminal_unit(row, expected) -> None:
    assert validate_binary_reward_row(row, tolerance=1e-6) == expected


@pytest.mark.parametrize("row", [[0.2], [1.0, 1.0], [-1.0, 1.0], [float("nan")]])
def test_binary_reward_validation_rejects_shaping_and_multiple_components(row) -> None:
    with pytest.raises(ValueError):
        validate_binary_reward_row(row, tolerance=1e-6)


def _runtime_config(loss_mode="dppo_tv"):
    return OmegaConf.create(
        {
            "algorithm": {
                "branch_revision_grpo": {
                    "_target_": "verl.trainer.config.BranchRevisionGRPOConfig",
                    "enable": True,
                    "num_critiques": 4,
                    "critique_max_response_length": 16,
                    "branch_max_tokens": 8,
                    "new_continuation_max_tokens": 8,
                    "min_continuation_tokens": 8,
                    "reward_tolerance": 1e-6,
                    "critique_prompt": BRANCH_REVISION_CRITIQUE_PROMPT,
                    "audit_output_dir": None,
                },
                "intermediate_mc_value": {"enable": False},
                "opsd": {"enable": False},
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "rollout_correction": {"rollout_is": None, "rollout_rs": None, "bypass_mode": False},
            },
            "actor_rollout_ref": {
                "model": {"trust_remote_code": False, "override_config": {}},
                "actor": {
                    "strategy": "fsdp",
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "use_kl_loss": False,
                    "use_rollout_log_probs": False,
                    "use_prefix_grouper": False,
                    "router_replay": {"mode": "none"},
                    "policy_loss": {"loss_mode": loss_mode},
                },
                "rollout": {
                    "name": "vllm",
                    "n": 2,
                    "temperature": 1.0,
                    "calculate_log_probs": False,
                    "max_model_len": 128,
                    "logprobs_mode": "processed_logprobs",
                    "skip_rollout": False,
                    "enable_rollout_routing_replay": False,
                    "multi_turn": {"enable": False},
                    "val_kwargs": {"temperature": 1.0},
                },
            },
            "critic": {"enable": False},
            "trainer": {"use_legacy_worker_impl": "auto"},
            "reward": {
                "reward_manager": {"source": "register", "name": "naive"},
                "reward_kwargs": {},
                "reward_model": {
                    "enable": False,
                    "launch_reward_fn_async": False,
                    "reward_loop_source": None,
                    "reward_loop_module_path": None,
                    "reward_loop_class_name": None,
                },
            },
            "data": {
                "use_dataset_responses": False,
                "max_prompt_length": 16,
                "max_response_length": 32,
            },
        }
    )


@pytest.mark.parametrize("loss_mode", ["dppo_tv", "vanilla"])
def test_runtime_config_accepts_both_native_policy_losses_and_forces_behavior_logprobs(loss_mode) -> None:
    config = _runtime_config(loss_mode)
    validate_branch_revision_runtime_config(config)
    assert config.critic.enable is False
    assert config.actor_rollout_ref.rollout.calculate_log_probs is True
    assert config.actor_rollout_ref.actor.use_rollout_log_probs is True
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == loss_mode


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("algorithm.intermediate_mc_value.enable", True, "mutually exclusive"),
        ("algorithm.adv_estimator", "gae", "adv_estimator=grpo"),
        ("actor_rollout_ref.rollout.n", 1, "rollout.n>=2"),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "gpg", "dppo_tv and vanilla"),
        ("actor_rollout_ref.rollout.temperature", 0.9, "temperature=1.0"),
        ("actor_rollout_ref.actor.use_kl_loss", True, "does not support"),
        ("reward.reward_model.launch_reward_fn_async", True, "blocking iteration barrier"),
        ("algorithm.branch_revision_grpo.min_continuation_tokens", 33, "smaller than"),
    ],
)
def test_runtime_config_rejects_unsafe_interactions(path, value, message) -> None:
    config = _runtime_config()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match=message):
        validate_branch_revision_runtime_config(config)


def _loop() -> BranchRevisionAgentLoop:
    loop = BranchRevisionAgentLoop.__new__(BranchRevisionAgentLoop)
    loop.feature = BranchRevisionGRPOConfig(
        enable=True,
        num_critiques=2,
        critique_max_response_length=256,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
        min_continuation_tokens=128,
    )
    loop.response_length = 256
    loop.max_model_len = 2048
    loop.tokenizer = TOKENIZER
    return loop


def test_child_sampling_always_uses_temperature_one_and_processed_logprobs() -> None:
    params = BranchRevisionAgentLoop._sampling_params(
        {"temperature": 0.2, "logprobs": False, "max_new_tokens": 999, "top_p": 0.9},
        max_tokens=17,
    )
    assert params == {"temperature": 1.0, "logprobs": True, "max_tokens": 17, "top_p": 0.9}


def test_agent_loop_generates_all_critiques_then_one_continuation_per_valid_edit() -> None:
    loop = _loop()
    calls: list[tuple[str, int, float, bool]] = []

    async def fake_generate(_route, prompt, params, *, max_tokens, kind):
        calls.append((kind, max_tokens, params.get("temperature", 1.0), params.get("logprobs", True)))
        if kind.startswith("critique"):
            decoded_prompt = decode_exact(prompt, TOKENIZER)
            assert "</assistant><user>--- BEGIN CRITIQUE TASK ---" in decoded_prompt
            assert decoded_prompt.endswith("</user><assistant><think>\n\n</think>\n\n")
        if kind == "critique[0]":
            text = _structured("dead", "better")
            return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))
        if kind == "critique[1]":
            text = "invalid structure"
            return TokenOutput(token_ids=_ids(text), log_probs=[-0.3] * len(_ids(text)))
        assert kind == "continuation[0]"
        assert decode_exact(prompt[-len(_ids("start better")) :], TOKENIZER) == "start better"
        return TokenOutput(token_ids=_ids(" solved"), log_probs=[-0.4] * len(_ids(" solved")))

    loop._generate = fake_generate
    solution_ids = _ids("start dead and waste")
    output = asyncio.run(
        loop.run(
            {"temperature": 0.2, "logprobs": False},
            branch_revision_rollout_id="p:0",
            branch_revision_parent_prompt_ids=_ids("q"),
            branch_revision_parent_solution_ids=solution_ids,
            branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
            raw_prompt=[{"role": "user", "content": "q"}],
        )
    )
    record = output.extra_fields[BRANCH_REVISION_CHILD_FIELD]
    assert list(record.critique_prompt_ids[: len(_ids("qstart dead and waste"))]) == _ids("qstart dead and waste")
    assert decode_exact(record.critique_prompt_ids, TOKENIZER).endswith(
        "</assistant><user>--- BEGIN CRITIQUE TASK ---"
        + BRANCH_REVISION_CRITIQUE_PROMPT.removeprefix("--- BEGIN CRITIQUE TASK ---")
        + "</user><assistant><think>\n\n</think>\n\n"
    )
    assert [critique.parse_reason for critique in record.critiques] == ["valid", "tag_count"]
    assert len(record.critiques[0].continuation_ids) > 0
    assert record.critiques[0].continuation_max_tokens >= loop.feature.min_continuation_tokens
    assert record.critiques[1].continuation_ids == ()
    assert [call[0] for call in calls] == ["critique[0]", "critique[1]", "continuation[0]"]


def test_agent_loop_rejects_edits_without_the_configured_continuation_budget() -> None:
    loop = _loop()
    loop.response_length = 100
    calls: list[str] = []

    async def fake_generate(_route, _prompt, _params, *, max_tokens, kind):
        del max_tokens
        calls.append(kind)
        text = _structured("dead", "better")
        return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))

    loop._generate = fake_generate
    solution_ids = _ids("start dead and waste")
    output = asyncio.run(
        loop.run(
            {},
            branch_revision_rollout_id="p:0",
            branch_revision_parent_prompt_ids=_ids("q"),
            branch_revision_parent_solution_ids=solution_ids,
            branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
            raw_prompt=[{"role": "user", "content": "q"}],
        )
    )
    record = output.extra_fields[BRANCH_REVISION_CHILD_FIELD]
    assert [critique.parse_reason for critique in record.critiques] == [
        "insufficient_continuation_budget",
        "insufficient_continuation_budget",
    ]
    assert calls == ["critique[0]", "critique[1]"]


def test_agent_loop_drains_all_sibling_failures_before_raising() -> None:
    loop = _loop()
    completed: list[str] = []

    async def fake_generate(_route, _prompt, _params, *, max_tokens, kind):
        del max_tokens
        await asyncio.sleep(0.001 if kind == "critique[0]" else 0.003)
        completed.append(kind)
        if kind == "critique[0]":
            raise RuntimeError("injected")
        return TokenOutput(token_ids=_ids("invalid"), log_probs=[-0.1] * len(_ids("invalid")))

    loop._generate = fake_generate
    solution_ids = _ids("solution")
    with pytest.raises(RuntimeError, match="after draining every request"):
        asyncio.run(
            loop.run(
                {},
                branch_revision_rollout_id="p:0",
                branch_revision_parent_prompt_ids=_ids("q"),
                branch_revision_parent_solution_ids=solution_ids,
                branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
                raw_prompt=[{"role": "user", "content": "q"}],
            )
        )
    assert sorted(completed) == ["critique[0]", "critique[1]"]


def _controller() -> BranchRevisionGRPOController:
    controller = BranchRevisionGRPOController.__new__(BranchRevisionGRPOController)
    controller.feature = BranchRevisionGRPOConfig(enable=True, num_critiques=2)
    controller.config = OmegaConf.create(
        {
            "algorithm": {"norm_adv_by_std_in_grpo": True},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "policy_loss": {"loss_mode": "dppo_tv"},
                    "clip_ratio": 0.2,
                    "clip_ratio_low": None,
                    "clip_ratio_high": None,
                    "clip_ratio_c": 3.0,
                },
                "rollout": {"n": 2, "max_model_len": 2048},
            },
            "trainer": {"balance_batch": False},
        }
    )
    controller.tokenizer = TOKENIZER
    controller.audit_dir = None
    controller._initialized_audit_steps = set()
    controller.trainer = SimpleNamespace(
        actor_rollout_wg=object(),
        _get_dp_size=lambda _worker, _role: 2,
        global_steps=1,
    )
    return controller


def _critique(text: str, *, valid: bool, continuation: str = "") -> BranchRevisionCritiqueGeneration:
    critique_ids = _ids(text)
    if valid:
        return BranchRevisionCritiqueGeneration(
            token_ids=tuple(critique_ids),
            log_probs=tuple([-0.2] * len(critique_ids)),
            finish_reason="stop",
            parse_reason="valid",
            branch_text="dead",
            new_continuation_text="better",
            revised_prefix_ids=tuple(_ids("start better")),
            continuation_ids=tuple(_ids(continuation)),
            continuation_log_probs=tuple([-0.3] * len(_ids(continuation))),
            continuation_finish_reason="stop",
            continuation_max_tokens=128,
        )
    return BranchRevisionCritiqueGeneration(
        token_ids=tuple(critique_ids),
        log_probs=tuple([-0.2] * len(critique_ids)),
        finish_reason="stop",
        parse_reason="tag_count",
        branch_text="",
        new_continuation_text="",
        revised_prefix_ids=(),
    )


def test_actor_batch_uses_prompt_and_original_grpo_groups_and_masks_reused_revision_seed() -> None:
    controller = _controller()
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    invalid = _critique("invalid", valid=False)
    wrong = _Bundle(
        source_row=0,
        rollout_id="p:0",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and waste")),
        original_reward=0.0,
        record=BranchRevisionGenerationRecord(
            "p:0",
            (valid, invalid),
            tuple([*_ids("q"), *_ids("start dead and waste"), *_ids("<followup>")]),
        ),
        continuation_rewards={0: 1.0},
    )
    correct = _Bundle(
        source_row=1,
        rollout_id="p:1",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("correct"),
        solution_log_probs=[-0.1] * len(_ids("correct")),
        original_reward=1.0,
    )
    actor_batch, padding = controller._make_actor_batch([wrong, correct])
    kinds = actor_batch.non_tensor_batch["branch_revision_actor_kind"].tolist()
    assert kinds == ["original", "critique", "continuation", "critique", "original", "padding"]
    assert padding == 1
    continuation_row = kinds.index("continuation")
    assert actor_batch.batch["response_mask"][continuation_row].sum().item() == len(_ids(" solved"))
    assert actor_batch.batch["old_log_probs"][continuation_row][
        actor_batch.batch["response_mask"][continuation_row].bool()
    ].tolist() == pytest.approx([-0.3] * len(_ids(" solved")))
    assert actor_batch.batch["response_mask"][-1].sum().item() == 0
    assert actor_batch.batch["advantages"][-1].abs().sum().item() == 0
    critique_rows = [index for index, kind in enumerate(kinds) if kind == "critique"]
    expected_critique_prompt = list(wrong.record.critique_prompt_ids)
    packed_critique = actor_batch.batch["input_ids"][critique_rows[0]][
        actor_batch.batch["attention_mask"][critique_rows[0]].bool()
    ].tolist()
    assert packed_critique == [*expected_critique_prompt, *valid.token_ids]
    critique_rewards = actor_batch.non_tensor_batch["branch_revision_reward"][critique_rows].tolist()
    assert critique_rewards == pytest.approx([0.5, -0.5])
    assert actor_batch.non_tensor_batch["branch_revision_reward"][continuation_row] == pytest.approx(1.0)
    assert actor_batch.batch["advantages"][critique_rows[0]].max().item() > 0
    assert actor_batch.batch["advantages"][critique_rows[1]].min().item() < 0
    original_rows = [index for index, kind in enumerate(kinds) if kind == "original"]
    assert actor_batch.batch["advantages"][original_rows[0]].min().item() < 0
    assert actor_batch.batch["advantages"][original_rows[1]].max().item() > 0
    assert actor_batch.meta_info["use_global_loss_normalization"] is True


@pytest.mark.parametrize("loss_fn", [compute_policy_loss_vanilla, compute_policy_loss_dppo_tv])
def test_global_denominators_make_partitioned_policy_loss_equal_full_batch(loss_fn) -> None:
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
            "global_batch_info": {},
        }
    )
    old = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, 0.0, 0.0], [-1.5, -1.0, -0.2]])
    current = old + torch.tensor([[0.1, -0.1, 0.0], [0.05, 0.0, 0.0], [-0.05, 0.2, 0.1]])
    advantages = torch.tensor([[1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
    full, _ = loss_fn(old, current, advantages, mask, config=config)
    config.global_batch_info = {"dp_size": 2, "batch_num_tokens": int(mask.sum()), "global_batch_size": 3}
    left, _ = loss_fn(old[:2], current[:2], advantages[:2], mask[:2], config=config)
    right, _ = loss_fn(old[2:], current[2:], advantages[2:], mask[2:], config=config)
    torch.testing.assert_close((left + right) / 2, full)


def test_global_loss_info_uses_valid_tokens_and_sequences_and_restores(monkeypatch) -> None:
    config = SimpleNamespace(global_batch_info={"existing": 7}, loss_scale_factor=None)
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op, group: tensor.mul_(2))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    previous, tokens, sequences = dp_actor._set_global_loss_normalization(
        config,
        torch.tensor([[1, 1, 0], [0, 0, 0]]),
        object(),
    )
    assert (tokens, sequences) == (4, 2)
    assert config.global_batch_info == {
        "dp_size": 2,
        "batch_num_tokens": 4,
        "global_batch_size": 2,
        "loss_scale_factor": None,
    }
    dp_actor._restore_global_loss_normalization(config, previous)
    assert config.global_batch_info == {"existing": 7}


def test_global_loss_info_is_restored_when_actor_update_raises() -> None:
    actor = SimpleNamespace(config=SimpleNamespace(global_batch_info={"existing": 7}))
    data = SimpleNamespace(meta_info={"use_global_loss_normalization": True})

    @dp_actor._restore_global_loss_normalization_on_exit
    def injected_failure(self, _data):
        self.config.global_batch_info.clear()
        self.config.global_batch_info["temporary"] = 1
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        injected_failure(actor, data)
    assert actor.config.global_batch_info == {"existing": 7}


def _source_batch() -> tuple[object, torch.Tensor]:
    solution_texts = ["start dead and waste", "correct"]
    prompt_ids = _ids("q")
    response_ids = [_ids(text) for text in solution_texts]
    response_width = max(map(len, response_ids))
    prompts = torch.tensor([prompt_ids, prompt_ids], dtype=torch.long)
    responses = torch.zeros((2, response_width), dtype=torch.long)
    response_mask = torch.zeros_like(responses)
    rollout_log_probs = torch.zeros((2, response_width), dtype=torch.float32)
    for row, tokens in enumerate(response_ids):
        responses[row, : len(tokens)] = torch.tensor(tokens)
        response_mask[row, : len(tokens)] = 1
        rollout_log_probs[row, : len(tokens)] = -0.1
    attention_mask = torch.cat([torch.ones_like(prompts), response_mask], dim=1)
    from verl import DataProto

    source = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "rollout_log_probs": rollout_log_probs,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "position_ids": torch.arange(attention_mask.shape[1]).repeat(2, 1),
        },
        non_tensors={
            "prompt_group_id": np.array(["p", "p"], dtype=object),
            "branch_revision_rollout_id": np.array(["p:0", "p:1"], dtype=object),
            "raw_prompt": np.array(
                [[{"role": "user", "content": "q"}], [{"role": "user", "content": "q"}]],
                dtype=object,
            ),
        },
    )
    rewards = torch.zeros_like(responses, dtype=torch.float32)
    rewards[1, len(response_ids[1]) - 1] = 1.0
    return source, rewards


def test_full_controller_update_critiques_only_incorrect_and_trains_one_combined_batch(monkeypatch) -> None:
    controller = _controller()
    controller.config.actor_rollout_ref.rollout.prompt_length = 8
    controller.config.actor_rollout_ref.rollout.response_length = 256
    source, original_rewards = _source_batch()
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    invalid = _critique("invalid", valid=False)
    record = BranchRevisionGenerationRecord(
        "p:0",
        (valid, invalid),
        tuple([*_ids("q"), *_ids("start dead and waste"), *_ids("<followup>")]),
    )
    child_values = np.empty(1, dtype=object)
    child_values[0] = record
    from verl import DataProto

    child_output = DataProto.from_dict(
        tensors={"responses": torch.ones((1, 1), dtype=torch.long)},
        non_tensors={BRANCH_REVISION_CHILD_FIELD: child_values},
        meta_info={"timing": {"child_internal": 0.25}},
    )
    events: list[str] = []
    captured: dict[str, object] = {}

    def update_actor(batch):
        events.append("update_actor")
        captured["batch"] = batch
        return DataProto(meta_info={"metrics": {"actor/pg_loss": [0.5]}})

    controller.trainer = SimpleNamespace(
        global_steps=1,
        actor_rollout_wg=object(),
        checkpoint_manager=SimpleNamespace(
            update_weights=lambda _step: events.append("restore"),
            sleep_replicas=lambda: events.append("sleep"),
        ),
        async_rollout_manager=SimpleNamespace(
            generate_sequences=lambda request: events.append("generate") or child_output,
            start_profile=lambda: events.append("start_profile"),
            stop_profile=lambda: events.append("stop_profile"),
        ),
        _get_dp_size=lambda _worker, _role: 2,
        _update_actor=update_actor,
        reward_fn=object(),
    )

    def fake_compute_reward(batch, reward_fn, actor_wg):
        del reward_fn, actor_wg
        assert len(batch) == 1
        tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        tensor[0, int(batch.batch["response_mask"][0].sum()) - 1] = 1.0
        return tensor, {}

    monkeypatch.setattr(branch_controller_module, "compute_reward", fake_compute_reward)
    metrics: dict[str, float] = {}
    timing: dict[str, float] = {}
    assert controller.run_update(source, original_rewards, metrics, timing)
    assert events == ["restore", "generate", "sleep", "update_actor"]
    assert metrics["branch_revision/incorrect_originals"] == 1.0
    assert metrics["branch_revision/critiques"] == 2.0
    assert metrics["branch_revision/continuations"] == 1.0
    assert metrics["branch_revision/flip/success_per_all_critiques"] == 0.5
    assert metrics["branch_revision/flip/success_per_valid_continuation"] == 1.0
    assert timing["child_internal"] == 0.25
    actor_batch = captured["batch"]
    assert actor_batch.non_tensor_batch["branch_revision_actor_kind"].tolist().count("critique") == 2
    assert actor_batch.non_tensor_batch["branch_revision_actor_kind"].tolist().count("continuation") == 1
    assert "advantages" in source.batch and "returns" in source.batch
    response_mask = source.batch["response_mask"].bool()
    assert source.batch["advantages"][0][response_mask[0]].max().item() < 0.0
    assert source.batch["advantages"][1][response_mask[1]].min().item() > 0.0
