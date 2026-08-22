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
"""Blocking composite child rollout for branch-revision GRPO."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.trainer.config import BranchRevisionGRPOConfig
from verl.trainer.ppo.branch_revision_grpo import (
    decode_exact,
    encode_followup_user_turn,
    normalize_log_probs_float32,
    parse_branch_revision,
    strip_terminal_eos,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import PROMPT_LOGPROBS_SLICE_START, TokenOutput

BRANCH_REVISION_AGENT_NAME = "branch_revision_agent"
BRANCH_REVISION_CHILD_FIELD = "__branch_revision_children__"


@dataclass(frozen=True)
class BranchRevisionCritiqueGeneration:
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    finish_reason: str | None
    parse_reason: str
    prefix_text: str
    prefix_plus_new_continuation_text: str
    new_continuation_text: str
    branch_prefix_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    continuation_prefix_ids: tuple[int, ...]
    new_continuation_ids: tuple[int, ...]
    new_continuation_log_probs: tuple[float, ...]
    revised_prefix_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...] = ()
    continuation_log_probs: tuple[float, ...] = ()
    continuation_finish_reason: str | None = None
    continuation_max_tokens: int = 0

    @property
    def valid(self) -> bool:
        return self.parse_reason == "valid"


@dataclass(frozen=True)
class BranchRevisionGenerationRecord:
    rollout_id: str
    objective: str
    critiques: tuple[BranchRevisionCritiqueGeneration, ...]
    critique_prompt_ids: tuple[int, ...]


def _as_int_list(value: Any, name: str) -> list[int]:
    result = [int(token) for token in normalize_token_ids(value)]
    if not result:
        raise ValueError(f"{name} must contain at least one token")
    return result


def _as_float_list(value: Any, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"{name} is required")
    result = [float(item) for item in value]
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


async def _gather_and_drain(
    tasks: list[asyncio.Task],
    *,
    phase: str,
    indices: list[int],
) -> list[Any]:
    """Await every request and leave no child task live when the parent exits."""

    if len(tasks) != len(indices):
        raise ValueError("branch-revision task labels do not match the task count")
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException as primary_error:
        for task in tasks:
            if not task.done():
                task.cancel()
        cleanup_results = await asyncio.gather(*tasks, return_exceptions=True)
        cleanup_errors = [
            result
            for result in cleanup_results
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
        ]
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note) and cleanup_errors:
            add_note(f"{phase} task cleanup also returned: {cleanup_errors!r}")
        raise
    errors = [
        f"{phase}[{index}]: {result!r}"
        for index, result in zip(indices, results, strict=True)
        if isinstance(result, BaseException)
    ]
    if errors:
        raise RuntimeError(
            f"branch-revision {phase} generation failed after draining every request: " + "; ".join(errors)
        )
    return results


@register(BRANCH_REVISION_AGENT_NAME)
class BranchRevisionAgentLoop(AgentLoopBase):
    """Generate IID critiques, parse exact edits, then generate revised suffixes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.branch_revision_grpo,
            dataclass_type=BranchRevisionGRPOConfig,
        )
        self.response_length = int(self.rollout_config.response_length)
        self.max_model_len = int(self.rollout_config.max_model_len)

    @staticmethod
    def _sampling_params(
        base: dict[str, Any],
        *,
        max_tokens: int,
        prompt_logprob_start: int | None = None,
    ) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("branch-revision generation max_tokens must be positive")
        result = dict(base)
        result.pop("max_new_tokens", None)
        result["max_tokens"] = int(max_tokens)
        result["temperature"] = 1.0
        result["top_p"] = 1.0
        result["top_k"] = -1
        result["repetition_penalty"] = 1.0
        result["logprobs"] = True
        result.pop("prompt_logprobs", None)
        result.pop(PROMPT_LOGPROBS_SLICE_START, None)
        if prompt_logprob_start is not None:
            if isinstance(prompt_logprob_start, bool) or not isinstance(prompt_logprob_start, int):
                raise ValueError("branch-revision prompt_logprob_start must be an integer")
            if prompt_logprob_start <= 0:
                raise ValueError("branch-revision prompt_logprob_start must be positive")
            result["prompt_logprobs"] = 1
            result[PROMPT_LOGPROBS_SLICE_START] = prompt_logprob_start
        return result

    async def _generate(
        self,
        route_key: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        kind: str,
        prompt_logprob_start: int | None = None,
    ) -> TokenOutput:
        if len(prompt_ids) + max_tokens > self.max_model_len:
            raise ValueError(
                f"{kind} exceeds rollout.max_model_len: prompt={len(prompt_ids)} "
                f"max_tokens={max_tokens} limit={self.max_model_len}"
            )
        return await self.server_manager.generate(
            request_id=route_key,
            prompt_ids=prompt_ids,
            sampling_params=self._sampling_params(
                sampling_params,
                max_tokens=max_tokens,
                prompt_logprob_start=prompt_logprob_start,
            ),
        )

    @staticmethod
    def _validated_output(output: TokenOutput, *, cap: int, kind: str) -> tuple[list[int], list[float]]:
        token_ids = [int(token) for token in output.token_ids[:cap]]
        if not token_ids:
            raise RuntimeError(f"{kind} generation returned no tokens")
        if output.log_probs is None or len(output.log_probs) < len(token_ids):
            raise RuntimeError(f"{kind} generation did not return one processed log probability per token")
        log_probs = [float(value) for value in output.log_probs[: len(token_ids)]]
        return token_ids, log_probs

    @staticmethod
    def _finish_reason(output: TokenOutput) -> str | None:
        value = output.extra_fields.get("finish_reason", output.stop_reason)
        return None if value is None else str(value)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.monotonic()
        rollout_id = str(kwargs["branch_revision_rollout_id"])
        objective = str(kwargs["branch_revision_parent_objective"])
        if objective not in {"recovery", "compression"}:
            raise ValueError(f"unknown branch-revision objective {objective!r}")
        num_critiques = int(kwargs["branch_revision_num_critiques"])
        expected_critiques = (
            self.feature.num_critiques if objective == "recovery" else self.feature.num_positive_critiques
        )
        if num_critiques != expected_critiques:
            raise ValueError(
                f"branch-revision {objective} request asked for {num_critiques} critiques; "
                f"expected {expected_critiques}"
            )
        prompt_ids = _as_int_list(kwargs["branch_revision_parent_prompt_ids"], "parent prompt")
        solution_ids = _as_int_list(kwargs["branch_revision_parent_solution_ids"], "parent solution")
        solution_log_probs = _as_float_list(
            kwargs["branch_revision_parent_solution_log_probs"],
            "parent solution log probabilities",
        )
        if len(solution_ids) != len(solution_log_probs):
            raise ValueError("parent solution tokens and behavior log probabilities must have equal lengths")

        editable_solution_ids = strip_terminal_eos(solution_ids, self.tokenizer)
        if not editable_solution_ids:
            raise ValueError("branch-revision solution is empty after removing its terminal EOS boundary")
        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("branch-revision child generation requires the original raw_prompt messages")
        critique_instruction = (
            self.feature.critique_prompt if objective == "recovery" else self.feature.positive_critique_prompt
        )
        critique_instruction_ids = encode_followup_user_turn(
            critique_instruction,
            self.tokenizer,
            prior_messages=list(raw_prompt),
            assistant_content=decode_exact(editable_solution_ids, self.tokenizer),
            chat_template_kwargs=kwargs.get("chat_template_kwargs"),
        )
        critique_prompt = [*prompt_ids, *editable_solution_ids, *critique_instruction_ids]
        requested_cap = int(self.feature.critique_max_response_length or self.response_length)
        critique_cap = min(requested_cap, self.max_model_len - len(critique_prompt))
        if critique_cap <= 0:
            raise ValueError(
                "branch-revision critique has no context capacity: "
                f"prompt={len(critique_prompt)} limit={self.max_model_len}"
            )

        critique_tasks = [
            asyncio.create_task(
                self._generate(
                    f"{rollout_id}:critique:{index}",
                    critique_prompt,
                    sampling_params,
                    max_tokens=critique_cap,
                    kind=f"critique[{index}]",
                )
            )
            for index in range(num_critiques)
        ]
        critique_results = await _gather_and_drain(
            critique_tasks,
            phase="critique",
            indices=list(range(num_critiques)),
        )

        parsed_records: list[dict[str, Any] | None] = [None] * num_critiques
        continuation_specs: list[tuple[int, int]] = []
        processing_errors: list[str] = []
        num_preempted = 0
        for index, raw_output in enumerate(critique_results):
            try:
                if not isinstance(raw_output, TokenOutput):
                    raise TypeError(f"critique[{index}] returned unexpected type {type(raw_output)!r}")
                num_preempted += int(raw_output.num_preempted or 0)
                critique_ids, critique_log_probs = self._validated_output(
                    raw_output,
                    cap=critique_cap,
                    kind=f"critique[{index}]",
                )
                parsed = parse_branch_revision(
                    solution_ids,
                    critique_ids,
                    self.tokenizer,
                    branch_max_tokens=self.feature.branch_max_tokens,
                    new_continuation_max_tokens=self.feature.new_continuation_max_tokens,
                )
                parse_reason = parsed.reason
                branch_prefix_ids = list(parsed.branch_prefix_ids)
                prefix_ids = list(parsed.prefix_ids)
                continuation_prefix_ids = list(parsed.continuation_prefix_ids)
                new_continuation_ids = list(parsed.new_continuation_ids)
                revised_prefix_ids = list(parsed.revised_prefix_ids)
                continuation_max_tokens = 0
                if parsed.valid:
                    continuation_max_tokens = min(
                        self.response_length - len(revised_prefix_ids),
                        self.max_model_len - len(prompt_ids) - len(revised_prefix_ids),
                    )
                    if continuation_max_tokens < self.feature.min_continuation_tokens:
                        parse_reason = "insufficient_continuation_budget"
                        branch_prefix_ids = []
                        prefix_ids = []
                        continuation_prefix_ids = []
                        new_continuation_ids = []
                        revised_prefix_ids = []
                        continuation_max_tokens = 0
                parsed_records[index] = {
                    "token_ids": critique_ids,
                    "log_probs": critique_log_probs,
                    "finish_reason": self._finish_reason(raw_output),
                    "parse_reason": parse_reason,
                    "prefix_text": parsed.prefix_text,
                    "prefix_plus_new_continuation_text": parsed.prefix_plus_new_continuation_text,
                    "new_continuation_text": parsed.new_continuation_text,
                    "branch_prefix_ids": branch_prefix_ids,
                    "prefix_ids": prefix_ids,
                    "continuation_prefix_ids": continuation_prefix_ids,
                    "new_continuation_ids": new_continuation_ids,
                    "revised_prefix_ids": revised_prefix_ids,
                    "continuation_max_tokens": continuation_max_tokens,
                }
                if parse_reason == "valid":
                    if continuation_max_tokens < self.feature.min_continuation_tokens:
                        raise RuntimeError("valid branch revision unexpectedly lacks its minimum continuation capacity")
                    continuation_specs.append((index, continuation_max_tokens))
            except Exception as error:
                processing_errors.append(f"critique[{index}]: {error!r}")
        if processing_errors:
            raise RuntimeError(
                "branch-revision critique validation failed before continuation launch: " + "; ".join(processing_errors)
            )
        if any(record is None for record in parsed_records):
            raise RuntimeError("branch-revision critique validation did not produce every parsed record")
        complete_records = [record for record in parsed_records if record is not None]

        continuation_indices = [index for index, _ in continuation_specs]
        continuation_tasks = [
            asyncio.create_task(
                self._generate(
                    f"{rollout_id}:continuation:{index}",
                    [*prompt_ids, *complete_records[index]["revised_prefix_ids"]],
                    sampling_params,
                    max_tokens=max_tokens,
                    kind=f"continuation[{index}]",
                    prompt_logprob_start=len(prompt_ids)
                    + len(complete_records[index]["continuation_prefix_ids"]),
                )
            )
            for index, max_tokens in continuation_specs
        ]

        continuation_results = await _gather_and_drain(
            continuation_tasks,
            phase="continuation",
            indices=continuation_indices,
        )
        for index, raw_output in zip(continuation_indices, continuation_results, strict=True):
            if not isinstance(raw_output, TokenOutput):
                raise TypeError(f"continuation[{index}] returned unexpected type {type(raw_output)!r}")
            num_preempted += int(raw_output.num_preempted or 0)
            max_tokens = self.response_length - len(complete_records[index]["revised_prefix_ids"])
            continuation_ids, continuation_log_probs = self._validated_output(
                raw_output,
                cap=min(max_tokens, int(complete_records[index]["continuation_max_tokens"])),
                kind=f"continuation[{index}]",
            )
            expected_seed_ids = [int(token) for token in complete_records[index]["new_continuation_ids"]]
            prompt_seed_ids = raw_output.prompt_log_prob_token_ids
            prompt_seed_log_probs = raw_output.prompt_log_probs
            expected_start = len(prompt_ids) + len(complete_records[index]["continuation_prefix_ids"])
            if raw_output.prompt_log_prob_start != expected_start:
                raise RuntimeError(
                    f"continuation[{index}] prompt log-probability slice starts at "
                    f"{raw_output.prompt_log_prob_start!r}; expected {expected_start}"
                )
            if prompt_seed_ids is None or [int(token) for token in prompt_seed_ids] != expected_seed_ids:
                raise RuntimeError(
                    f"continuation[{index}] prompt log probabilities are not aligned to replacement tokens"
                )
            if prompt_seed_log_probs is None or len(prompt_seed_log_probs) != len(expected_seed_ids):
                raise RuntimeError(
                    f"continuation[{index}] did not return one prompt log probability per replacement token"
                )
            if any(value is None for value in prompt_seed_log_probs):
                raise RuntimeError(
                    f"continuation[{index}] replacement token unexpectedly lacks a prompt log probability"
                )
            normalized_seed_log_probs = normalize_log_probs_float32(
                value for value in prompt_seed_log_probs if value is not None
            )
            complete_records[index]["continuation_ids"] = continuation_ids
            complete_records[index]["continuation_log_probs"] = continuation_log_probs
            complete_records[index]["continuation_finish_reason"] = self._finish_reason(raw_output)
            complete_records[index]["new_continuation_log_probs"] = [
                float(value) for value in normalized_seed_log_probs.tolist()
            ]

        critiques = tuple(
            BranchRevisionCritiqueGeneration(
                token_ids=tuple(record["token_ids"]),
                log_probs=tuple(record["log_probs"]),
                finish_reason=record["finish_reason"],
                parse_reason=record["parse_reason"],
                prefix_text=record["prefix_text"],
                prefix_plus_new_continuation_text=record["prefix_plus_new_continuation_text"],
                new_continuation_text=record["new_continuation_text"],
                branch_prefix_ids=tuple(record["branch_prefix_ids"]),
                prefix_ids=tuple(record["prefix_ids"]),
                continuation_prefix_ids=tuple(record["continuation_prefix_ids"]),
                new_continuation_ids=tuple(record["new_continuation_ids"]),
                new_continuation_log_probs=tuple(record.get("new_continuation_log_probs", ())),
                revised_prefix_ids=tuple(record["revised_prefix_ids"]),
                continuation_ids=tuple(record.get("continuation_ids", ())),
                continuation_log_probs=tuple(record.get("continuation_log_probs", ())),
                continuation_finish_reason=record.get("continuation_finish_reason"),
                continuation_max_tokens=int(record["continuation_max_tokens"]),
            )
            for record in complete_records
        )
        record = BranchRevisionGenerationRecord(
            rollout_id=rollout_id,
            objective=objective,
            critiques=critiques,
            critique_prompt_ids=tuple(critique_prompt),
        )
        extra_fields = {
            BRANCH_REVISION_CHILD_FIELD: record,
            "turn_scores": [],
            "tool_rewards": [],
            "stop_reason": "completed",
            "finish_reason": "branch_revision_children_complete",
        }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=solution_ids,
            response_mask=[1] * len(solution_ids),
            response_logprobs=solution_log_probs,
            routed_experts=None,
            multi_modal_data={},
            num_turns=3,
            metrics={
                "generate_sequences": time.monotonic() - started,
                "tool_calls": 0.0,
                "num_preempted": num_preempted,
            },
            extra_fields=extra_fields,
        )
