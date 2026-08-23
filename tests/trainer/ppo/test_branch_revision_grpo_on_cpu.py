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
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
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
    _gather_and_drain,
)
from verl.trainer.config import (
    BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT,
    BRANCH_REVISION_CRITIQUE_PROMPT,
    BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT,
    BranchRevisionGRPOConfig,
)
from verl.trainer.ppo import ray_trainer_branch_revision as branch_controller_module
from verl.trainer.ppo.branch_revision_grpo import (
    LearnabilityScore,
    aggregate_log_probs,
    build_learnability_reference,
    build_rollout_logprob_prefixes,
    decode_exact,
    encode_followup_user_turn,
    parse_branch_revision,
    score_seed_learnability,
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
from verl.workers.rollout.replica import (
    PROMPT_LOGPROBS_SLICE_START,
    TokenOutput,
    extract_chosen_prompt_log_probs,
)


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


VALID_ANALYSIS = "The setup narrowed the choices. A local alternative is justified here.\n"


def _structured(locator_prefix: str, new_continuation: str, analysis: str = VALID_ANALYSIS) -> str:
    joint = f"{locator_prefix}{new_continuation}"
    return (
        f"{analysis}<prefix>{locator_prefix}</prefix>\n<prefix + new continuation>{joint}</prefix + new continuation>"
    )


def test_objective_prompts_share_the_causal_and_exact_edit_contract() -> None:
    assert BRANCH_REVISION_CRITIQUE_PROMPT == BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT
    assert "attempted solution above is incorrect" in BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT
    assert "solution above is correct" in BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT
    for prompt in (BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT, BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT):
        assert "point immediately after <prefix> as the information boundary" in prompt
        assert "opening $$ and its matching closing $$" in prompt
        assert "inside a fenced code block" in prompt
        assert "trajectory counterfactually with everything after the prefix" in prompt
        assert "hidden. If work performed after the prefix" in prompt
        assert "genuine character prefix" in prompt
        assert "<prefix + new continuation>" in prompt
        assert "Do not compress a sequence of later developments" in prompt
        assert "After the free-form analysis" in prompt
        assert "nothing after the closing" in prompt
        assert "1. PRUNING:" not in prompt


def test_critique_instruction_is_a_new_user_turn_with_visible_assistant_output() -> None:
    ids = encode_followup_user_turn(BRANCH_REVISION_CRITIQUE_PROMPT, TOKENIZER)
    text = decode_exact(ids, TOKENIZER)
    assert text.startswith("</assistant><user>The attempted solution above is incorrect.")
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


def test_exact_parser_preserves_whitespace_unicode_and_substitutes_joint_text() -> None:
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
    assert parsed.prefix_text == "  dead end\n"
    assert parsed.prefix_plus_new_continuation_text == "  dead end\n  try β instead\n"
    assert parsed.new_continuation_text == "  try β instead\n"
    assert parsed.revised_text == "use α\n  dead end\n  try β instead\n"
    assert decode_exact(parsed.branch_prefix_ids, TOKENIZER) == "use α\n"
    assert decode_exact(parsed.prefix_ids, TOKENIZER) == "  dead end\n"
    assert decode_exact(parsed.continuation_prefix_ids, TOKENIZER) == "use α\n  dead end\n"
    assert decode_exact(parsed.new_continuation_ids, TOKENIZER) == "  try β instead\n"
    assert decode_exact(parsed.revised_prefix_ids, TOKENIZER) == parsed.revised_text


def test_parser_allows_the_generated_prefix_to_start_at_the_solution_boundary() -> None:
    parsed = parse_branch_revision(
        _ids("start here then waste"),
        _ids(_structured("start here", " and take a better local step")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.branch_prefix_ids == ()
    assert decode_exact(parsed.continuation_prefix_ids, TOKENIZER) == "start here"
    assert parsed.revised_text == "start here and take a better local step"


def test_terminal_eos_stripping_preserves_interior_special_token() -> None:
    token_ids = [*_ids("a"), TOKENIZER.eos_token_id, *_ids("b"), TOKENIZER.eos_token_id]
    stripped = strip_terminal_eos(token_ids, TOKENIZER)
    assert stripped == token_ids[:-1]
    assert decode_exact(stripped, TOKENIZER) == "a<eos>b"


@pytest.mark.parametrize(
    ("critique", "reason"),
    [
        ("<prefix>x</prefix>", "tag_count"),
        (
            _structured("", "new"),
            "empty_prefix",
        ),
        (
            VALID_ANALYSIS
            + "<prefix>x</prefix>\n"
            + "<prefix + new continuation>different</prefix + new continuation>",
            "prefix_not_prefix_of_joint",
        ),
        (_structured("x", "   "), "empty_new_continuation"),
        (_structured("x", "The final answer is 7"), "new_continuation_final_answer"),
        (_structured("x", "\\boxed{7}"), "new_continuation_final_answer"),
        (_structured("x", "#### 7"), "new_continuation_final_answer"),
        (_structured("same", "new"), "prefix_not_unique"),
        (_structured("zzz", "new"), "prefix_not_found"),
        (_structured("x", "new") + " trailing", "text_after_tags"),
        (
            VALID_ANALYSIS
            + "<prefix>x</prefix> not whitespace "
            + "<prefix + new continuation>xnew</prefix + new continuation>",
            "text_between_tags",
        ),
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


def test_parser_allows_empty_free_form_analysis() -> None:
    parsed = parse_branch_revision(
        _ids("x solution"),
        _ids("<prefix>x</prefix>\n<prefix + new continuation>x new</prefix + new continuation>"),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.analysis_text == ""
    assert parsed.prefix_text == "x"
    assert parsed.new_continuation_text == " new"


def test_parser_enforces_token_caps_without_trimming_contents() -> None:
    parsed = parse_branch_revision(
        _ids("abcd rest"),
        _ids(_structured("abcd", "new")),
        TOKENIZER,
        branch_max_tokens=3,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "prefix_token_cap"

    parsed = parse_branch_revision(
        _ids("a rest"),
        _ids(_structured("a", "long")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=3,
    )
    assert parsed.reason == "new_continuation_token_cap"


def test_parser_checks_final_answer_markers_only_in_the_appended_continuation() -> None:
    parsed = parse_branch_revision(
        _ids("The phrase final answer appears in this source span. Then waste."),
        _ids(_structured("The phrase final answer appears in this source span.", " Try a local check.")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.new_continuation_text == " Try a local check."


def test_parser_treats_overlapping_branch_occurrences_as_nonunique() -> None:
    parsed = parse_branch_revision(
        _ids("aaa"),
        _ids(_structured("aa", "new")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "prefix_not_unique"


def test_parser_recovers_a_unique_case_and_formatting_insensitive_branch() -> None:
    solution = "prefix\nSo N*K = 9,143\nthen waste"
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured("So, n * k = 9 143", "Check the next locally justified factor.")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.branch_start == len("prefix\n")
    assert parsed.revised_text == "prefix\nSo, n * k = 9 143Check the next locally justified factor."


def test_parser_accepts_a_unique_partial_prefix_without_a_length_threshold() -> None:
    solution = "abcqdef"
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured("Q completely different hindsight", "x")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.branch_start == 3
    assert parsed.revised_text == "abcQ completely different hindsightx"


def test_parser_rejects_a_tied_normalized_maximum() -> None:
    parsed = parse_branch_revision(
        _ids("first target then target again"),
        _ids(_structured("TARGET but reformatted", "new")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "prefix_not_unique"


def test_parser_rejects_a_branch_with_no_normalized_overlap() -> None:
    parsed = parse_branch_revision(
        _ids("abc"),
        _ids(_structured("$$", "new")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "prefix_not_found"


def test_normalized_match_searches_only_the_adjacent_formatting_gap_for_a_stable_boundary() -> None:
    class WhitespaceMergeTokenizer(_CharTokenizer):
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            index = 0
            while index < len(text):
                if text.startswith(" b", index):
                    result.append(900)
                    index += 2
                elif text.startswith(" x", index):
                    result.append(901)
                    index += 2
                else:
                    result.append(ord(text[index]) + 100)
                    index += 1
            return result

        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            pieces = []
            for token in token_ids:
                if int(token) == 900:
                    pieces.append(" b")
                elif int(token) == 901:
                    pieces.append(" x")
                else:
                    pieces.append(chr(int(token) - 100))
            return "".join(pieces)

    tokenizer = WhitespaceMergeTokenizer()
    parsed = parse_branch_revision(
        tokenizer.encode("a, b"),
        tokenizer.encode(_structured("B changed", "x")),
        tokenizer,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.branch_start == 2
    assert parsed.revised_text == "a,B changedx"


def test_parser_rejects_a_branch_inside_an_existing_token_boundary() -> None:
    class MergeTokenizer(_CharTokenizer):
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            index = 0
            while index < len(text):
                if text.startswith("ab", index):
                    result.append(900)
                    index += 2
                else:
                    result.append(ord(text[index]) + 100)
                    index += 1
            return result

        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join("ab" if int(token) == 900 else chr(int(token) - 100) for token in token_ids)

    tokenizer = MergeTokenizer()
    parsed = parse_branch_revision(
        tokenizer.encode("abc"),
        tokenizer.encode(_structured("b", "x")),
        tokenizer,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "branch_not_at_token_boundary"


def test_parser_rejects_a_replacement_that_retokenizes_the_preserved_prefix() -> None:
    class MergeTokenizer(_CharTokenizer):
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            index = 0
            while index < len(text):
                if text.startswith("ax", index):
                    result.append(901)
                    index += 2
                else:
                    result.append(ord(text[index]) + 100)
                    index += 1
            return result

        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join("ax" if int(token) == 901 else chr(int(token) - 100) for token in token_ids)

    tokenizer = MergeTokenizer()
    parsed = parse_branch_revision(
        tokenizer.encode("ba"),
        tokenizer.encode(_structured("a", "x")),
        tokenizer,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == "new_continuation_retokenizes_prefix"


@pytest.mark.parametrize(
    ("solution", "branch", "reason"),
    [
        ("before\n$$\nx + 1\n$$\nafter", "x + 1", "branch_inside_display_math"),
        ("before\n\\[\nx + 1\n\\]\nafter", "x + 1", "branch_inside_display_math"),
        (
            "before\n\\begin{align}\nx + 1\n\\end{align}\nafter",
            "x + 1",
            "branch_inside_latex_environment",
        ),
        (
            "before\n\\begin{align}\n\\begin{split}\nx + 1\n\\end{split}\n\\end{align}\nafter",
            "x + 1",
            "branch_inside_latex_environment",
        ),
        ("before\n```python\nx + 1\n```\nafter", "x + 1", "branch_inside_code_fence"),
        ("before\n~~~~\n$$ x + 1 $$\n~~~~\nafter", "$$ x + 1 $$", "branch_inside_code_fence"),
    ],
)
def test_parser_rejects_branch_points_inside_open_blocks(solution: str, branch: str, reason: str) -> None:
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured(branch, "take another local step")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert not parsed.valid
    assert parsed.reason == reason


@pytest.mark.parametrize(
    ("solution", "prefix", "reason"),
    [
        ("before\n$$\nx + 1\n$$\nafter", "before\n$$\nx + 1", "branch_inside_display_math"),
        ("before\n```python\nx + 1\n```\nafter", "before\n```python\nx + 1", "branch_inside_code_fence"),
        (
            "before\n\\begin{align}\nx + 1\n\\end{align}\nafter",
            "before\n\\begin{align}\nx + 1",
            "branch_inside_latex_environment",
        ),
    ],
)
def test_parser_rejects_prefixes_that_end_inside_open_blocks(
    solution: str,
    prefix: str,
    reason: str,
) -> None:
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured(prefix, "\nTake the next local step.")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == reason


def test_parser_accepts_a_prefix_containing_a_complete_display_block() -> None:
    prefix = "before\n$$\nx + 1\n$$"
    parsed = parse_branch_revision(
        _ids(f"{prefix}\nafter"),
        _ids(_structured(prefix, "\nTake the next local step.")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert parsed.revised_text == f"{prefix}\nTake the next local step."


@pytest.mark.parametrize(
    ("solution", "branch", "reason"),
    [
        ("before\n$$\nx + 1\n$$\nafter", "X plus one", "branch_inside_display_math"),
        ("before\n\\[\nx + 1\n\\]\nafter", "X plus one", "branch_inside_display_math"),
        (
            "before\n\\begin{align}\nx + 1\n\\end{align}\nafter",
            "X plus one",
            "branch_inside_latex_environment",
        ),
        ("before\n```python\nx + 1\n```\nafter", "X plus one", "branch_inside_code_fence"),
    ],
)
def test_normalized_match_does_not_move_an_inside_block_branch_before_its_opener(
    solution: str,
    branch: str,
    reason: str,
) -> None:
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured(branch, "take another local step")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.reason == reason


@pytest.mark.parametrize(
    ("solution", "branch", "expected_prefix"),
    [
        ("before\n$$\nx + 1\n$$\nafter", "$$\nX + 1\n$$", "before\n"),
        ("before\n\\[\nx + 1\n\\]\nafter", "\\[\nX + 1\n\\]", "before\n"),
        ("before\n```python\nx + 1\n```\nafter", "```python\nX + 1\n```", "before\n"),
    ],
)
def test_normalized_match_honors_an_explicitly_quoted_block_opener(
    solution: str,
    branch: str,
    expected_prefix: str,
) -> None:
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured(branch, "take another local step")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid
    assert decode_exact(parsed.branch_prefix_ids, TOKENIZER) == expected_prefix


@pytest.mark.parametrize(
    ("solution", "branch"),
    [
        ("before\n$$\nx + 1\n$$\nafter", "$$\nx + 1\n$$"),
        ("before\n$$\nx + 1\n$$\nafter", "after"),
        ("before \\$$ is literal\nafter", "after"),
        ("before\n```\ncode\n```\nafter", "after"),
    ],
)
def test_parser_allows_branch_points_at_balanced_block_boundaries(solution: str, branch: str) -> None:
    parsed = parse_branch_revision(
        _ids(solution),
        _ids(_structured(branch, "take another local step")),
        TOKENIZER,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
    )
    assert parsed.valid


def test_length_matched_learnability_enumerates_every_window_with_uniform_mass() -> None:
    prefixes = build_rollout_logprob_prefixes(
        ["short", "long"],
        [[-1.0, -1.0, -1.0], [-3.0] * 20],
    )
    first = build_learnability_reference(prefixes, window_size=2)
    second = build_learnability_reference(prefixes, window_size=2)
    torch.testing.assert_close(first.window_scores, second.window_scores)
    assert first.eligible_rollouts == 2
    assert first.rollout_window_counts == (("short", 2), ("long", 19))
    assert first.total_windows == 21
    score = score_seed_learnability(-2.0, first, minimum_percentile=0.0, full_credit_percentile=0.5)
    assert score.percentile == pytest.approx(19 / 21)


def test_mean_and_min_learnability_use_the_same_length_matched_windows() -> None:
    prefixes = build_rollout_logprob_prefixes(
        ["r"],
        [[-1.0, -9.0, -4.0, -4.0]],
    )
    mean_reference = build_learnability_reference(
        prefixes,
        window_size=2,
        logprob_statistic="mean",
    )
    min_reference = build_learnability_reference(
        prefixes,
        window_size=2,
        logprob_statistic="min",
    )
    assert mean_reference.window_scores.tolist() == pytest.approx([-5.0, -6.5, -4.0])
    assert min_reference.window_scores.tolist() == pytest.approx([-9.0, -9.0, -4.0])
    assert aggregate_log_probs([-1.0, -9.0], statistic="mean") == pytest.approx(-5.0)
    assert aggregate_log_probs([-1.0, -9.0], statistic="min") == pytest.approx(-9.0)


@pytest.mark.parametrize("statistic", ["mean", "min"])
def test_exhaustive_reference_matches_brute_force_for_every_length(statistic: str) -> None:
    rows = [
        [-1.25, -7.0, -2.5, -9.0, -3.0],
        [-4.0, -0.5, -6.0],
    ]
    prefixes = build_rollout_logprob_prefixes(["a", "b"], rows)
    for window_size in range(1, 6):
        reference = build_learnability_reference(
            prefixes,
            window_size=window_size,
            logprob_statistic=statistic,
        )
        expected = []
        expected_counts = []
        for rollout_id, row in zip(("a", "b"), rows, strict=True):
            normalized = torch.tensor(row, dtype=torch.float32).tolist()
            windows = [normalized[start : start + window_size] for start in range(len(row) - window_size + 1)]
            if windows:
                expected_counts.append((rollout_id, len(windows)))
            expected.extend(
                math.fsum(window) / len(window) if statistic == "mean" else min(window) for window in windows
            )
        assert reference.rollout_window_counts == tuple(expected_counts)
        assert reference.total_windows == len(expected)
        assert reference.window_scores.tolist() == pytest.approx(expected)


def test_mean_and_min_learnability_are_identical_for_one_token() -> None:
    prefixes = build_rollout_logprob_prefixes(["r"], [[-4.0, -2.0]])
    mean_reference = build_learnability_reference(prefixes, window_size=1, logprob_statistic="mean")
    min_reference = build_learnability_reference(prefixes, window_size=1, logprob_statistic="min")
    torch.testing.assert_close(mean_reference.window_scores, min_reference.window_scores)


@pytest.mark.parametrize("statistic", ["mean", "min"])
def test_learnability_rejects_when_no_original_has_a_full_window(statistic: str) -> None:
    prefixes = build_rollout_logprob_prefixes(["a", "b"], [[-1.0], [-2.0, -3.0]])
    reference = build_learnability_reference(prefixes, window_size=3, logprob_statistic=statistic)
    score = score_seed_learnability(
        -1.0,
        reference,
        minimum_percentile=0.2,
        full_credit_percentile=0.5,
    )
    assert reference.rollout_window_counts == ()
    assert reference.total_windows == 0
    assert reference.window_scores_sha256 == hashlib.sha256(b"").hexdigest()
    assert score.percentile == 0.0
    assert score.reward_weight == 0.0
    assert not score.accepted


def test_learnability_percentile_gates_and_ramps_reward_credit() -> None:
    prefixes = build_rollout_logprob_prefixes(["r"], [[-4.0, -3.0, -2.0, -1.0]])
    reference = build_learnability_reference(prefixes, window_size=1)
    rejected = score_seed_learnability(
        -4.1,
        reference,
        minimum_percentile=0.20,
        full_credit_percentile=0.50,
    )
    partial = score_seed_learnability(
        -3.0,
        reference,
        minimum_percentile=0.20,
        full_credit_percentile=0.70,
    )
    full = score_seed_learnability(
        -1.0,
        reference,
        minimum_percentile=0.20,
        full_credit_percentile=0.50,
    )
    assert not rejected.accepted and rejected.reward_weight == 0.0
    assert partial.accepted and partial.percentile == pytest.approx(0.5)
    assert partial.reward_weight == pytest.approx(0.6)
    assert full.accepted and full.reward_weight == 1.0


@pytest.mark.parametrize("statistic", ["mean", "min"])
def test_learnability_quantizes_both_sides_to_float32_before_comparison(statistic) -> None:
    raw_reference = -1.00000004
    raw_seed = -1.00000005
    assert raw_seed < raw_reference
    prefixes = build_rollout_logprob_prefixes(["r"], [[raw_reference]])
    reference = build_learnability_reference(
        prefixes,
        window_size=1,
        logprob_statistic=statistic,
    )
    score = score_seed_learnability(
        aggregate_log_probs([raw_seed], statistic=statistic),
        reference,
        minimum_percentile=0.2,
        full_credit_percentile=0.5,
    )
    assert score.seed_score == reference.window_scores.item() == -1.0
    assert score.percentile == 1.0


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
                    "enable_positive_compression": True,
                    "num_positive_critiques": 4,
                    "positive_compression_target": 0.25,
                    "learnability_logprob_statistic": "mean",
                    "min_seed_window_percentile": 0.20,
                    "full_credit_seed_window_percentile": 0.50,
                    "critique_max_response_length": 16,
                    "branch_max_tokens": 8,
                    "new_continuation_max_tokens": 8,
                    "min_continuation_tokens": 8,
                    "reward_tolerance": 1e-6,
                    "critique_prompt": BRANCH_REVISION_CRITIQUE_PROMPT,
                    "positive_critique_prompt": BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT,
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
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                    "calculate_log_probs": False,
                    "max_model_len": 128,
                    "logprobs_mode": "processed_logprobs",
                    "skip_rollout": False,
                    "enable_rollout_routing_replay": False,
                    "multi_turn": {"enable": False},
                    "val_kwargs": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
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


@pytest.mark.parametrize("statistic", ["mean", "min"])
def test_branch_revision_config_accepts_both_learnability_statistics(statistic) -> None:
    feature = BranchRevisionGRPOConfig(learnability_logprob_statistic=statistic)
    assert feature.learnability_logprob_statistic == statistic


def test_branch_revision_config_rejects_unknown_learnability_statistic() -> None:
    with pytest.raises(ValueError, match="must be mean or min"):
        BranchRevisionGRPOConfig(learnability_logprob_statistic="median")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("algorithm.intermediate_mc_value.enable", True, "mutually exclusive"),
        ("algorithm.adv_estimator", "gae", "adv_estimator=grpo"),
        ("actor_rollout_ref.rollout.n", 1, "rollout.n>=2"),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "gpg", "dppo_tv and vanilla"),
        ("actor_rollout_ref.rollout.temperature", 0.9, "temperature=1.0"),
        ("actor_rollout_ref.rollout.top_p", 0.9, "top_p=1"),
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


class _HeadroomTokenizer(_CharTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        headroom_marker="",
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        rendered += headroom_marker
        return self.encode(rendered) if tokenize else rendered


@pytest.mark.parametrize("critique_cap", [16, None])
def test_tokenizer_aware_context_headroom_accepts_exact_fit_and_rejects_one_token_short(critique_cap) -> None:
    config = _runtime_config()
    config.algorithm.branch_revision_grpo.critique_max_response_length = critique_cap
    config.data.apply_chat_template_kwargs = {"headroom_marker": "base"}
    config.data.train_apply_chat_template_kwargs = {"headroom_marker": "train-override"}
    tokenizer = _HeadroomTokenizer()
    followup_lengths = [
        len(
            encode_followup_user_turn(
                instruction,
                tokenizer,
                chat_template_kwargs={"headroom_marker": "train-override"},
            )
        )
        for instruction in (BRANCH_REVISION_CRITIQUE_PROMPT, BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT)
    ]
    effective_cap = int(critique_cap or config.data.max_response_length)
    required = (
        int(config.data.max_prompt_length)
        + int(config.data.max_response_length)
        + max(followup_lengths)
        + effective_cap
    )
    config.actor_rollout_ref.rollout.max_model_len = required
    validate_branch_revision_runtime_config(config, actor_tokenizer=tokenizer)
    config.actor_rollout_ref.rollout.max_model_len = required - 1
    with pytest.raises(ValueError, match=r"max_prompt=.*followup=.*critique_cap=.*required=.*limit="):
        validate_branch_revision_runtime_config(config, actor_tokenizer=tokenizer)


def _loop() -> BranchRevisionAgentLoop:
    loop = BranchRevisionAgentLoop.__new__(BranchRevisionAgentLoop)
    loop.feature = BranchRevisionGRPOConfig(
        enable=True,
        num_critiques=2,
        enable_positive_compression=True,
        num_positive_critiques=2,
        critique_max_response_length=256,
        branch_max_tokens=128,
        new_continuation_max_tokens=256,
        min_continuation_tokens=128,
    )
    loop.response_length = 256
    loop.max_model_len = 10000
    loop.tokenizer = TOKENIZER
    return loop


def test_child_sampling_always_uses_temperature_one_and_processed_logprobs() -> None:
    params = BranchRevisionAgentLoop._sampling_params(
        {"temperature": 0.2, "logprobs": False, "max_new_tokens": 999, "top_p": 0.9},
        max_tokens=17,
    )
    assert params == {
        "temperature": 1.0,
        "logprobs": True,
        "max_tokens": 17,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }

    scored = BranchRevisionAgentLoop._sampling_params(
        {"prompt_logprobs": -1, PROMPT_LOGPROBS_SLICE_START: 999},
        max_tokens=17,
        prompt_logprob_start=23,
    )
    assert scored["prompt_logprobs"] == 1
    assert scored[PROMPT_LOGPROBS_SLICE_START] == 23


def test_extract_chosen_prompt_log_probs_slices_and_checks_observed_tokens() -> None:
    class Logprob:
        def __init__(self, value):
            self.logprob = value

    token_ids = [10, 11, 12, 13]
    prompt_logprobs = [
        None,
        {99: Logprob(-0.1), 11: Logprob(-1.1)},
        {12: Logprob(-1.2)},
        {13: Logprob(-1.3), 98: Logprob(-0.2)},
    ]
    selected_ids, selected_logprobs = extract_chosen_prompt_log_probs(token_ids, prompt_logprobs, start=2)
    assert selected_ids == [12, 13]
    assert selected_logprobs == pytest.approx([-1.2, -1.3])
    with pytest.raises(RuntimeError, match="omit observed token"):
        extract_chosen_prompt_log_probs(token_ids, [None, {11: -1.0}, {99: -1.0}, {13: -1.0}], start=2)


def test_agent_loop_generates_all_critiques_then_one_continuation_per_valid_edit() -> None:
    loop = _loop()
    calls: list[tuple[str, int, float, bool]] = []

    async def fake_generate(_route, prompt, params, *, max_tokens, kind, prompt_logprob_start=None):
        calls.append((kind, max_tokens, params.get("temperature", 1.0), params.get("logprobs", True)))
        if kind.startswith("critique"):
            decoded_prompt = decode_exact(prompt, TOKENIZER)
            assert "</assistant><user>The attempted solution above is incorrect." in decoded_prompt
            assert decoded_prompt.endswith("</user><assistant><think>\n\n</think>\n\n")
        if kind == "critique[0]":
            text = _structured("dead", "better")
            return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))
        if kind == "critique[1]":
            text = "invalid structure"
            return TokenOutput(token_ids=_ids(text), log_probs=[-0.3] * len(_ids(text)))
        assert kind == "continuation[0]"
        assert decode_exact(prompt[-len(_ids("start deadbetter")) :], TOKENIZER) == "start deadbetter"
        seed_ids = _ids("better")
        assert prompt_logprob_start == len(prompt) - len(seed_ids)
        return TokenOutput(
            token_ids=_ids(" solved"),
            log_probs=[-0.4] * len(_ids(" solved")),
            prompt_log_prob_token_ids=seed_ids,
            prompt_log_probs=[-0.25] * len(seed_ids),
            prompt_log_prob_start=prompt_logprob_start,
        )

    loop._generate = fake_generate
    solution_ids = _ids("start dead and waste")
    output = asyncio.run(
        loop.run(
            {"temperature": 0.2, "logprobs": False},
            branch_revision_rollout_id="p:0",
            branch_revision_parent_objective="recovery",
            branch_revision_num_critiques=2,
            branch_revision_parent_prompt_ids=_ids("q"),
            branch_revision_parent_solution_ids=solution_ids,
            branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
            raw_prompt=[{"role": "user", "content": "q"}],
        )
    )
    record = output.extra_fields[BRANCH_REVISION_CHILD_FIELD]
    assert list(record.critique_prompt_ids[: len(_ids("qstart dead and waste"))]) == _ids("qstart dead and waste")
    assert decode_exact(record.critique_prompt_ids, TOKENIZER).endswith(
        "</assistant><user>" + BRANCH_REVISION_CRITIQUE_PROMPT + "</user><assistant><think>\n\n</think>\n\n"
    )
    assert [critique.parse_reason for critique in record.critiques] == ["valid", "tag_count"]
    assert record.objective == "recovery"
    assert len(record.critiques[0].continuation_ids) > 0
    assert record.critiques[0].continuation_max_tokens >= loop.feature.min_continuation_tokens
    assert record.critiques[0].new_continuation_log_probs == pytest.approx([-0.25] * len(_ids("better")))
    assert record.critiques[1].continuation_ids == ()
    assert [call[0] for call in calls] == ["critique[0]", "critique[1]", "continuation[0]"]


def test_agent_loop_gives_block_breaking_edits_no_continuation_path() -> None:
    loop = _loop()
    calls: list[str] = []

    async def fake_generate(_route, _prompt, _params, *, max_tokens, kind, prompt_logprob_start=None):
        del max_tokens
        assert prompt_logprob_start is None
        calls.append(kind)
        assert kind.startswith("critique")
        text = _structured("x + 1", "take another local step")
        return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))

    loop._generate = fake_generate
    solution_ids = _ids("before\n$$\nx + 1\n$$\nafter")
    output = asyncio.run(
        loop.run(
            {},
            branch_revision_rollout_id="p:0",
            branch_revision_parent_objective="recovery",
            branch_revision_num_critiques=2,
            branch_revision_parent_prompt_ids=_ids("q"),
            branch_revision_parent_solution_ids=solution_ids,
            branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
            raw_prompt=[{"role": "user", "content": "q"}],
        )
    )
    record = output.extra_fields[BRANCH_REVISION_CHILD_FIELD]
    assert [critique.parse_reason for critique in record.critiques] == [
        "branch_inside_display_math",
        "branch_inside_display_math",
    ]
    assert all(not critique.continuation_ids for critique in record.critiques)
    assert calls == ["critique[0]", "critique[1]"]


def test_agent_loop_rejects_edits_without_the_configured_continuation_budget() -> None:
    loop = _loop()
    loop.response_length = 100
    calls: list[str] = []

    async def fake_generate(_route, _prompt, _params, *, max_tokens, kind, prompt_logprob_start=None):
        del max_tokens
        assert prompt_logprob_start is None
        calls.append(kind)
        text = _structured("dead", "better")
        return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))

    loop._generate = fake_generate
    solution_ids = _ids("start dead and waste")
    output = asyncio.run(
        loop.run(
            {},
            branch_revision_rollout_id="p:0",
            branch_revision_parent_objective="recovery",
            branch_revision_num_critiques=2,
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


def test_agent_loop_uses_the_positive_compression_prompt_for_correct_parents() -> None:
    loop = _loop()

    async def fake_generate(_route, prompt, _params, *, max_tokens, kind, prompt_logprob_start=None):
        del max_tokens
        assert prompt_logprob_start is None
        assert kind.startswith("critique")
        decoded_prompt = decode_exact(prompt, TOKENIZER)
        assert "</assistant><user>The solution above is correct." in decoded_prompt
        text = "free analysis without tags"
        return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))

    loop._generate = fake_generate
    solution_ids = _ids("correct but verbose")
    output = asyncio.run(
        loop.run(
            {},
            branch_revision_rollout_id="p:1",
            branch_revision_parent_objective="compression",
            branch_revision_num_critiques=2,
            branch_revision_parent_prompt_ids=_ids("q"),
            branch_revision_parent_solution_ids=solution_ids,
            branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
            raw_prompt=[{"role": "user", "content": "q"}],
        )
    )
    record = output.extra_fields[BRANCH_REVISION_CHILD_FIELD]
    assert record.objective == "compression"
    assert len(record.critiques) == 2


def test_agent_loop_drains_all_sibling_failures_before_raising() -> None:
    loop = _loop()
    completed: list[str] = []

    async def fake_generate(_route, _prompt, _params, *, max_tokens, kind, prompt_logprob_start=None):
        del max_tokens
        assert prompt_logprob_start is None
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
                branch_revision_parent_objective="recovery",
                branch_revision_num_critiques=2,
                branch_revision_parent_prompt_ids=_ids("q"),
                branch_revision_parent_solution_ids=solution_ids,
                branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
                raw_prompt=[{"role": "user", "content": "q"}],
            )
        )
    assert sorted(completed) == ["critique[0]", "critique[1]"]


def test_agent_loop_validates_every_critique_before_launching_continuations() -> None:
    loop = _loop()
    calls: list[str] = []

    async def fake_generate(_route, _prompt, _params, *, max_tokens, kind, prompt_logprob_start=None):
        del max_tokens, prompt_logprob_start
        calls.append(kind)
        if kind == "critique[0]":
            text = _structured("dead", "better")
            return TokenOutput(token_ids=_ids(text), log_probs=[-0.2] * len(_ids(text)))
        if kind == "critique[1]":
            return TokenOutput(token_ids=_ids("broken"), log_probs=None)
        raise AssertionError("a continuation launched before every critique passed validation")

    loop._generate = fake_generate
    solution_ids = _ids("start dead and waste")
    with pytest.raises(RuntimeError, match="before continuation launch"):
        asyncio.run(
            loop.run(
                {},
                branch_revision_rollout_id="p:0",
                branch_revision_parent_objective="recovery",
                branch_revision_num_critiques=2,
                branch_revision_parent_prompt_ids=_ids("q"),
                branch_revision_parent_solution_ids=solution_ids,
                branch_revision_parent_solution_log_probs=[-0.1] * len(solution_ids),
                raw_prompt=[{"role": "user", "content": "q"}],
            )
        )
    assert calls == ["critique[0]", "critique[1]"]


def test_gather_and_drain_cancels_every_task_when_parent_is_cancelled() -> None:
    finalized: list[int] = []

    async def scenario() -> None:
        async def child(index: int) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                finalized.append(index)

        children = [asyncio.create_task(child(index)) for index in range(2)]
        parent = asyncio.create_task(_gather_and_drain(children, phase="critique", indices=[0, 1]))
        await asyncio.sleep(0)
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert all(task.done() for task in children)

    asyncio.run(scenario())
    assert sorted(finalized) == [0, 1]


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
    controller.audit_root = None
    controller.audit_dir = None
    controller.audit_attempt_id = None
    controller._initialized_audit_steps = set()
    controller.trainer = SimpleNamespace(
        actor_rollout_wg=object(),
        _get_dp_size=lambda _worker, _role: 2,
        global_steps=1,
    )
    return controller


def _enable_audit(controller: BranchRevisionGRPOController, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    controller.audit_root = str(root)
    controller.audit_dir = None
    controller.audit_attempt_id = None
    controller._initialized_audit_steps = set()


def test_audit_resume_creates_a_new_attempt_without_overwriting_incomplete_evidence(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    first = _controller()
    _enable_audit(first, audit_root)
    first._audit("original", rollout_id="p:0")
    first_attempt = first.audit_attempt_id

    second = _controller()
    _enable_audit(second, audit_root)
    second._audit("original", rollout_id="p:0")
    second._audit("step_complete")
    second_attempt = second.audit_attempt_id

    assert first_attempt and second_attempt and first_attempt != second_attempt
    first_metadata = json.loads((audit_root / f"attempt_{first_attempt}" / "attempt.json").read_text())
    rendered_config = json.dumps(first_metadata["resolved_config"], sort_keys=True, default=str, ensure_ascii=False)
    assert first_metadata["resolved_config_sha256"] == hashlib.sha256(rendered_config.encode()).hexdigest()
    first_events = (audit_root / f"attempt_{first_attempt}" / "step_00000001.jsonl").read_text(encoding="utf-8")
    second_events = (audit_root / f"attempt_{second_attempt}" / "step_00000001.jsonl").read_text(encoding="utf-8")
    assert '"event": "step_complete"' not in first_events
    assert '"event": "step_complete"' in second_events


def test_actor_audit_uses_the_actual_post_reorder_batch(tmp_path: Path) -> None:
    controller = _controller()
    _enable_audit(controller, tmp_path / "audit")
    bundles = [
        _Bundle(
            source_row=index,
            rollout_id=f"p:{index}",
            prompt_group_id="p",
            prompt_ids=_ids("q"),
            solution_ids=_ids("wrong" if index == 0 else "right"),
            solution_log_probs=[-0.1] * len(_ids("wrong" if index == 0 else "right")),
            original_reward=float(index),
        )
        for index in range(2)
    ]
    actor_batch, padding_rows = controller._make_actor_batch(bundles)
    actor_batch.reorder(torch.tensor([1, 0]))
    controller._audit_actor_batch(actor_batch, padding_rows=padding_rows)
    attempt = controller.audit_attempt_id
    path = tmp_path / "audit" / f"attempt_{attempt}" / "step_00000001.jsonl"
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert [row["actor_row_id"] for row in event["actor_rows"]] == ["original:p:1", "original:p:0"]
    assert [row["balanced_row_index"] for row in event["actor_rows"]] == [0, 1]


def _critique(text: str, *, valid: bool, continuation: str = "") -> BranchRevisionCritiqueGeneration:
    critique_ids = _ids(text)
    if valid:
        return BranchRevisionCritiqueGeneration(
            token_ids=tuple(critique_ids),
            log_probs=tuple([-0.2] * len(critique_ids)),
            finish_reason="stop",
            parse_reason="valid",
            prefix_text="dead",
            prefix_plus_new_continuation_text="deadbetter",
            new_continuation_text="better",
            branch_prefix_ids=tuple(_ids("start ")),
            prefix_ids=tuple(_ids("dead")),
            continuation_prefix_ids=tuple(_ids("start dead")),
            new_continuation_ids=tuple(_ids("better")),
            new_continuation_log_probs=tuple([-0.05] * len(_ids("better"))),
            revised_prefix_ids=tuple(_ids("start deadbetter")),
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
        prefix_text="",
        prefix_plus_new_continuation_text="",
        new_continuation_text="",
        branch_prefix_ids=(),
        prefix_ids=(),
        continuation_prefix_ids=(),
        new_continuation_ids=(),
        new_continuation_log_probs=(),
        revised_prefix_ids=(),
    )


def _learnability(*, percentile: float = 1.0, weight: float = 1.0, accepted: bool = True) -> LearnabilityScore:
    return LearnabilityScore(
        logprob_statistic="mean",
        seed_score=-0.1,
        percentile=percentile,
        reward_weight=weight,
        accepted=accepted,
        eligible_rollouts=2,
        total_windows=4,
    )


def test_learnability_fails_closed_without_aligned_vllm_prompt_scores() -> None:
    controller = _controller()
    valid = replace(
        _critique(_structured("dead", "better"), valid=True, continuation=" solved"),
        new_continuation_log_probs=(),
    )
    bundle = _Bundle(
        source_row=0,
        rollout_id="p:0",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and waste")),
        original_reward=0.0,
        record=BranchRevisionGenerationRecord(
            "p:0",
            "recovery",
            (valid, _critique("invalid", valid=False)),
            tuple(_ids("prompt")),
        ),
    )
    with pytest.raises(RuntimeError, match="one vLLM prompt log probability"):
        controller._score_seed_learnability([bundle])


def test_controller_uses_configured_minimum_logprob_statistic() -> None:
    controller = _controller()
    controller.feature = replace(controller.feature, learnability_logprob_statistic="min")
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    seed_values = [-0.01] * len(valid.new_continuation_ids)
    seed_values[2] = -0.75
    valid = replace(valid, new_continuation_log_probs=tuple(seed_values))
    bundle = _Bundle(
        source_row=0,
        rollout_id="p:0",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and waste")),
        original_reward=0.0,
        record=BranchRevisionGenerationRecord(
            "p:0",
            "recovery",
            (valid, _critique("invalid", valid=False)),
            tuple(_ids("prompt")),
        ),
    )
    controller._score_seed_learnability([bundle])
    assert bundle.learnability[0].logprob_statistic == "min"
    assert bundle.learnability[0].seed_score == pytest.approx(-0.75)


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
            "recovery",
            (valid, invalid),
            tuple([*_ids("q"), *_ids("start dead and waste"), *_ids("<followup>")]),
        ),
        learnability={0: _learnability()},
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


def test_positive_critique_reward_combines_compression_and_learnability_credit() -> None:
    controller = _controller()
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" done")
    invalid = _critique("invalid", valid=False)
    bundle = _Bundle(
        source_row=0,
        rollout_id="p:0",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and a very long waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and a very long waste")),
        original_reward=1.0,
        record=BranchRevisionGenerationRecord(
            "p:0",
            "compression",
            (valid, invalid),
            tuple([*_ids("q"), *_ids("start dead and a very long waste"), *_ids("<followup>")]),
        ),
        learnability={0: _learnability(percentile=0.35, weight=0.5)},
        continuation_rewards={0: 1.0},
        compression_fractions={0: 0.25},
        compression_credits={0: 1.0},
    )
    rows = controller._actor_rows([bundle])
    assert [row.kind for row in rows] == ["original", "critique", "continuation", "critique"]
    assert [row.reward for row in rows if row.kind == "critique"] == pytest.approx([0.5, 0.0])
    assert next(row for row in rows if row.kind == "continuation").reward == 1.0


def test_recovery_critique_reward_applies_learnability_before_prompt_baseline() -> None:
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
        record=BranchRevisionGenerationRecord("p:0", "recovery", (valid, invalid), tuple(_ids("prompt"))),
        learnability={0: _learnability(weight=0.5)},
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
    rows = controller._actor_rows([wrong, correct])
    assert [row.reward for row in rows if row.kind == "critique"] == pytest.approx([0.0, -0.5])


def test_positive_continuation_credit_uses_completed_editable_length(monkeypatch) -> None:
    controller = _controller()
    controller.feature = BranchRevisionGRPOConfig(
        enable=True,
        num_critiques=2,
        enable_positive_compression=True,
        num_positive_critiques=2,
        positive_compression_target=0.5,
    )
    controller.config.actor_rollout_ref.rollout.prompt_length = 8
    controller.config.actor_rollout_ref.rollout.response_length = 256
    controller.trainer.reward_fn = object()
    source, _ = _source_batch()
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    bundle = _Bundle(
        source_row=1,
        rollout_id="p:1",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and a very long waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and a very long waste")),
        original_reward=1.0,
        record=BranchRevisionGenerationRecord(
            "p:1",
            "compression",
            (valid, _critique("invalid", valid=False)),
            tuple(_ids("prompt")),
        ),
        learnability={0: _learnability()},
    )

    def fake_compute_reward(batch, reward_fn, actor_wg):
        del reward_fn, actor_wg
        tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        tensor[0, int(batch.batch["response_mask"][0].sum()) - 1] = 1.0
        return tensor, {}

    monkeypatch.setattr(branch_controller_module, "compute_reward", fake_compute_reward)
    controller._evaluate_continuations(source, [bundle])
    original_length = len(_ids("start dead and a very long waste"))
    revised_length = len(_ids("start deadbetter solved"))
    expected_fraction = (original_length - revised_length) / original_length
    assert bundle.compression_fractions[0] == pytest.approx(expected_fraction)
    assert bundle.compression_credits[0] == pytest.approx(min(expected_fraction / 0.5, 1.0))


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


def test_child_request_selects_correct_rollouts_only_when_positive_compression_is_enabled() -> None:
    controller = _controller()
    source, reward_tensor = _source_batch()
    rewards = controller._original_rewards(reward_tensor)
    bundles = controller._build_bundles(source, rewards)
    negative_only = controller._make_child_request(source, bundles)
    assert len(negative_only) == 1
    assert negative_only.non_tensor_batch["branch_revision_parent_objective"].tolist() == ["recovery"]

    controller.feature = BranchRevisionGRPOConfig(
        enable=True,
        num_critiques=2,
        enable_positive_compression=True,
        num_positive_critiques=3,
    )
    both = controller._make_child_request(source, bundles)
    assert len(both) == 2
    assert both.non_tensor_batch["branch_revision_parent_objective"].tolist() == ["recovery", "compression"]
    assert both.non_tensor_batch["branch_revision_num_critiques"].tolist() == [2, 3]


def test_low_learnability_edit_is_not_rewarded_or_solution_trained(monkeypatch) -> None:
    controller = _controller()
    source, _ = _source_batch()
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    invalid = _critique("invalid", valid=False)
    rejected = _Bundle(
        source_row=0,
        rollout_id="p:0",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and waste")),
        original_reward=0.0,
        record=BranchRevisionGenerationRecord(
            "p:0",
            "recovery",
            (valid, invalid),
            tuple([*_ids("q"), *_ids("start dead and waste"), *_ids("<followup>")]),
        ),
        learnability={0: _learnability(percentile=0.1, weight=0.0, accepted=False)},
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
    monkeypatch.setattr(
        branch_controller_module,
        "compute_reward",
        lambda *_args, **_kwargs: pytest.fail("rejected continuation reached the reward function"),
    )
    controller._evaluate_continuations(source, [rejected, correct])
    assert rejected.continuation_rewards == {}
    rows = controller._actor_rows([rejected, correct])
    assert all(row.kind != "continuation" for row in rows)
    assert [row.reward for row in rows if row.kind == "critique"] == [-0.5, -0.5]


def test_success_per_continuation_counts_learnability_rejections_in_its_denominator() -> None:
    controller = _controller()
    first = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    second = _critique(_structured("dead", "alternate"), valid=True, continuation=" failed")
    bundle = _Bundle(
        source_row=0,
        rollout_id="p:0",
        prompt_group_id="p",
        prompt_ids=_ids("q"),
        solution_ids=_ids("start dead and waste"),
        solution_log_probs=[-0.1] * len(_ids("start dead and waste")),
        original_reward=0.0,
        record=BranchRevisionGenerationRecord(
            "p:0",
            "recovery",
            (first, second),
            tuple([*_ids("q"), *_ids("start dead and waste"), *_ids("<followup>")]),
        ),
        learnability={
            0: _learnability(),
            1: _learnability(percentile=0.1, weight=0.0, accepted=False),
        },
        continuation_rewards={0: 1.0},
    )
    actor_batch, padding_rows = controller._make_actor_batch([bundle])
    metrics = controller._metrics([bundle], actor_batch, padding_rows)
    assert metrics["branch_revision/flip/success_per_valid_continuation"] == 1.0
    assert metrics["branch_revision/flip/success_per_continuation"] == 0.5


def test_full_controller_update_critiques_only_incorrect_and_trains_one_combined_batch(monkeypatch) -> None:
    controller = _controller()
    controller.config.actor_rollout_ref.rollout.prompt_length = 8
    controller.config.actor_rollout_ref.rollout.response_length = 256
    source, original_rewards = _source_batch()
    valid = _critique(_structured("dead", "better"), valid=True, continuation=" solved")
    invalid = _critique("invalid", valid=False)
    record = BranchRevisionGenerationRecord(
        "p:0",
        "recovery",
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
        actor_rollout_wg=SimpleNamespace(world_size=2),
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
    assert metrics["branch_revision/flip/success_per_continuation"] == 1.0
    assert timing["child_internal"] == 0.25
    actor_batch = captured["batch"]
    assert actor_batch.non_tensor_batch["branch_revision_actor_kind"].tolist().count("critique") == 2
    assert actor_batch.non_tensor_batch["branch_revision_actor_kind"].tolist().count("continuation") == 1
    assert "advantages" in source.batch and "returns" in source.batch
    response_mask = source.batch["response_mask"].bool()
    assert source.batch["advantages"][0][response_mask[0]].max().item() < 0.0
    assert source.batch["advantages"][1][response_mask[1]].min().item() > 0.0
