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
BRANCH_REVISION_SCORE_FIELD = "__branch_revision_score__"
BRANCH_REVISION_CONTINUATION_FIELD = "__branch_revision_continuation__"


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
    critique_prompt_ids: tuple[int, ...] = ()
    reference_rollout_id: str | None = None
    reference_solution_ids: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        return self.parse_reason == "valid"


@dataclass(frozen=True)
class BranchRevisionGenerationRecord:
    rollout_id: str
    objective: str
    critiques: tuple[BranchRevisionCritiqueGeneration, ...]
    critique_prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class BranchRevisionScoreGeneration:
    rollout_id: str
    critique_index: int
    prompt_logprob_start: int
    scored_token_ids: tuple[int, ...]
    scored_token_log_probs: tuple[float, ...]
    admission: dict[str, Any] | None


@dataclass(frozen=True)
class BranchRevisionContinuationGeneration:
    rollout_id: str
    critique_index: int
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    finish_reason: str | None
    max_tokens: int


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
    """Run one blocking branch-revision critique, score, or continuation phase."""

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
        response_logprobs: bool = True,
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
        result["logprobs"] = response_logprobs
        result.pop("prompt_logprobs", None)
        result.pop(PROMPT_LOGPROBS_SLICE_START, None)
        if prompt_logprob_start is not None:
            if isinstance(prompt_logprob_start, bool) or not isinstance(prompt_logprob_start, int):
                raise ValueError("branch-revision prompt_logprob_start must be an integer")
            if prompt_logprob_start <= 0:
                raise ValueError("branch-revision prompt_logprob_start must be positive")
            # Zero requests no alternatives. vLLM still includes the observed
            # prompt token, which is the only entry learnability scoring needs.
            result["prompt_logprobs"] = 0
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
        response_logprobs: bool = True,
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
                response_logprobs=response_logprobs,
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

    def _score_only_response(self, output: TokenOutput) -> list[int]:
        if output.token_ids:
            return [int(output.token_ids[0])]
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.tokenizer.pad_token_id
        if token_id is None:
            raise RuntimeError("branch-revision score-only output requires a tokenizer EOS or pad token")
        return [int(token_id)]

    async def _run_score_phase(
        self,
        sampling_params: dict[str, Any],
        *,
        started: float,
        **kwargs,
    ) -> AgentLoopOutput:
        rollout_id = str(kwargs["branch_revision_rollout_id"])
        critique_index = int(kwargs["branch_revision_critique_index"])
        route_key = str(kwargs["branch_revision_route_key"])
        prompt_ids = _as_int_list(kwargs["branch_revision_parent_prompt_ids"], "parent prompt")
        continuation_prefix_ids = _as_int_list(
            kwargs["branch_revision_continuation_prefix_ids"],
            "continuation prefix",
        )
        new_continuation_ids = _as_int_list(
            kwargs["branch_revision_new_continuation_ids"],
            "new continuation",
        )
        scoring_prompt = [*prompt_ids, *continuation_prefix_ids, *new_continuation_ids]
        prompt_logprob_start = len(prompt_ids) + len(continuation_prefix_ids)
        output = await self._generate(
            route_key,
            scoring_prompt,
            sampling_params,
            max_tokens=1,
            kind=f"score[{critique_index}]",
            prompt_logprob_start=prompt_logprob_start,
            response_logprobs=False,
        )
        if output.prompt_log_prob_start != prompt_logprob_start:
            raise RuntimeError("branch-revision score prompt-logprob boundary changed in transit")
        if (
            output.prompt_log_prob_token_ids is None
            or [int(token) for token in output.prompt_log_prob_token_ids] != new_continuation_ids
        ):
            raise RuntimeError("branch-revision score prompt log probabilities misalign with replacement tokens")
        if output.prompt_log_probs is None or len(output.prompt_log_probs) != len(new_continuation_ids):
            raise RuntimeError("branch-revision score did not return one prompt log probability per replacement token")
        if any(value is None for value in output.prompt_log_probs):
            raise RuntimeError("branch-revision replacement token unexpectedly lacks a prompt log probability")
        normalized = normalize_log_probs_float32(value for value in output.prompt_log_probs if value is not None)
        admission = output.extra_fields.get("prompt_logprob_admission")
        admission_capacity = self.rollout_config.prompt_logprob_max_inflight_tokens
        if admission_capacity is None:
            if admission is not None:
                raise RuntimeError("unbounded branch-revision score unexpectedly returned admission evidence")
        elif not isinstance(admission, dict):
            raise RuntimeError("branch-revision score is missing prompt-logprob admission evidence")
        record = BranchRevisionScoreGeneration(
            rollout_id=rollout_id,
            critique_index=critique_index,
            prompt_logprob_start=prompt_logprob_start,
            scored_token_ids=tuple(new_continuation_ids),
            scored_token_log_probs=tuple(float(value) for value in normalized.tolist()),
            admission=None if admission is None else dict(admission),
        )
        response_ids = self._score_only_response(output)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[0] * len(response_ids),
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data={},
            num_turns=1,
            metrics={
                "generate_sequences": time.monotonic() - started,
                "tool_calls": 0.0,
                "num_preempted": int(output.num_preempted or 0),
            },
            extra_fields={
                BRANCH_REVISION_SCORE_FIELD: record,
                "turn_scores": [],
                "tool_rewards": [],
                "stop_reason": "completed",
                "finish_reason": "branch_revision_score_complete",
            },
        )

    async def _run_continuation_phase(
        self,
        sampling_params: dict[str, Any],
        *,
        started: float,
        **kwargs,
    ) -> AgentLoopOutput:
        rollout_id = str(kwargs["branch_revision_rollout_id"])
        critique_index = int(kwargs["branch_revision_critique_index"])
        route_key = str(kwargs["branch_revision_route_key"])
        prompt_ids = _as_int_list(kwargs["branch_revision_parent_prompt_ids"], "parent prompt")
        revised_prefix_ids = _as_int_list(
            kwargs["branch_revision_revised_prefix_ids"],
            "revised prefix",
        )
        max_tokens = int(kwargs["branch_revision_continuation_max_tokens"])
        if max_tokens < self.feature.min_continuation_tokens:
            raise ValueError("accepted branch revision lacks its minimum continuation budget")
        output = await self._generate(
            route_key,
            [*prompt_ids, *revised_prefix_ids],
            sampling_params,
            max_tokens=max_tokens,
            kind=f"continuation[{critique_index}]",
        )
        token_ids, log_probs = self._validated_output(
            output,
            cap=max_tokens,
            kind=f"continuation[{critique_index}]",
        )
        record = BranchRevisionContinuationGeneration(
            rollout_id=rollout_id,
            critique_index=critique_index,
            token_ids=tuple(token_ids),
            log_probs=tuple(log_probs),
            finish_reason=self._finish_reason(output),
            max_tokens=max_tokens,
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_ids,
            response_mask=[1] * len(token_ids),
            response_logprobs=log_probs,
            routed_experts=None,
            multi_modal_data={},
            num_turns=1,
            metrics={
                "generate_sequences": time.monotonic() - started,
                "tool_calls": 0.0,
                "num_preempted": int(output.num_preempted or 0),
            },
            extra_fields={
                BRANCH_REVISION_CONTINUATION_FIELD: record,
                "turn_scores": [],
                "tool_rewards": [],
                "stop_reason": "completed",
                "finish_reason": "branch_revision_continuation_complete",
            },
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.monotonic()
        phase = str(kwargs.get("branch_revision_phase", "critique"))
        if phase == "score":
            return await self._run_score_phase(sampling_params, started=started, **kwargs)
        if phase == "continuation":
            return await self._run_continuation_phase(sampling_params, started=started, **kwargs)
        if phase != "critique":
            raise ValueError(f"unknown branch-revision child phase {phase!r}")
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

        reference_rollout_ids = [str(value) for value in kwargs.get("branch_revision_reference_rollout_ids", ())]
        raw_reference_solution_ids = list(kwargs.get("branch_revision_reference_solution_ids", ()))
        if self.feature.recovery_reference_mode == "successful_original" and objective == "recovery":
            if len(reference_rollout_ids) != num_critiques or len(raw_reference_solution_ids) != num_critiques:
                raise ValueError("successful-original recovery requires one reference per critique")
            reference_solution_ids = [
                _as_int_list(value, f"successful reference[{index}]")
                for index, value in enumerate(raw_reference_solution_ids)
            ]
            if any(reference_id == rollout_id for reference_id in reference_rollout_ids):
                raise ValueError("an incorrect rollout cannot be its own successful reference")
        else:
            if reference_rollout_ids or raw_reference_solution_ids:
                raise ValueError("reference-free branch revision unexpectedly received successful references")
            reference_solution_ids = []

        editable_solution_ids = strip_terminal_eos(solution_ids, self.tokenizer)
        if not editable_solution_ids:
            raise ValueError("branch-revision solution is empty after removing its terminal EOS boundary")
        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("branch-revision child generation requires the original raw_prompt messages")
        requested_cap = int(self.feature.critique_max_response_length or self.response_length)
        critique_prompts: list[list[int]] = []
        critique_caps: list[int] = []
        for index in range(num_critiques):
            if self.feature.recovery_reference_mode == "successful_original" and objective == "recovery":
                editable_reference = strip_terminal_eos(reference_solution_ids[index], self.tokenizer)
                if not editable_reference:
                    raise ValueError(f"successful reference[{index}] is empty after removing terminal EOS")
                reference_text = decode_exact(editable_reference, self.tokenizer)
                critique_instruction = self.feature.successful_reference_critique_prompt.replace(
                    "{successful_rollout}",
                    reference_text,
                )
            else:
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
            critique_cap = min(requested_cap, self.max_model_len - len(critique_prompt))
            if critique_cap <= 0:
                raise ValueError(
                    f"branch-revision critique[{index}] has no context capacity: "
                    f"prompt={len(critique_prompt)} limit={self.max_model_len}"
                )
            critique_prompts.append(critique_prompt)
            critique_caps.append(critique_cap)

        critique_tasks = [
            asyncio.create_task(
                self._generate(
                    f"{rollout_id}:critique:{index}",
                    critique_prompts[index],
                    sampling_params,
                    max_tokens=critique_caps[index],
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
        processing_errors: list[str] = []
        num_preempted = 0
        for index, raw_output in enumerate(critique_results):
            try:
                if not isinstance(raw_output, TokenOutput):
                    raise TypeError(f"critique[{index}] returned unexpected type {type(raw_output)!r}")
                num_preempted += int(raw_output.num_preempted or 0)
                critique_ids, critique_log_probs = self._validated_output(
                    raw_output,
                    cap=critique_caps[index],
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
                    "critique_prompt_ids": critique_prompts[index],
                    "reference_rollout_id": reference_rollout_ids[index] if reference_rollout_ids else None,
                    "reference_solution_ids": reference_solution_ids[index] if reference_solution_ids else [],
                }
                if parse_reason == "valid" and continuation_max_tokens < self.feature.min_continuation_tokens:
                    raise RuntimeError("valid branch revision unexpectedly lacks its minimum continuation capacity")
            except Exception as error:
                processing_errors.append(f"critique[{index}]: {error!r}")
        if processing_errors:
            raise RuntimeError(
                "branch-revision critique validation failed before score launch: " + "; ".join(processing_errors)
            )
        if any(record is None for record in parsed_records):
            raise RuntimeError("branch-revision critique validation did not produce every parsed record")
        complete_records = [record for record in parsed_records if record is not None]

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
                critique_prompt_ids=tuple(record["critique_prompt_ids"]),
                reference_rollout_id=record["reference_rollout_id"],
                reference_solution_ids=tuple(record["reference_solution_ids"]),
            )
            for record in complete_records
        )
        record = BranchRevisionGenerationRecord(
            rollout_id=rollout_id,
            objective=objective,
            critiques=critiques,
            critique_prompt_ids=(
                tuple(critique_prompts[0]) if all(prompt == critique_prompts[0] for prompt in critique_prompts) else ()
            ),
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
            num_turns=2,
            metrics={
                "generate_sequences": time.monotonic() - started,
                "tool_calls": 0.0,
                "num_preempted": num_preempted,
            },
            extra_fields=extra_fields,
        )
