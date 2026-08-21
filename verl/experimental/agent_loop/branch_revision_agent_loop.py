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
from verl.trainer.config import BRANCH_REVISION_CRITIQUE_PROMPT, BranchRevisionGRPOConfig
from verl.trainer.ppo.branch_revision_grpo import (
    decode_exact,
    encode_followup_user_turn,
    parse_branch_revision,
    strip_terminal_eos,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

BRANCH_REVISION_AGENT_NAME = "branch_revision_agent"
BRANCH_REVISION_CHILD_FIELD = "__branch_revision_children__"


@dataclass(frozen=True)
class BranchRevisionCritiqueGeneration:
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    finish_reason: str | None
    parse_reason: str
    branch_text: str
    new_continuation_text: str
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
    def _sampling_params(base: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("branch-revision generation max_tokens must be positive")
        result = dict(base)
        result.pop("max_new_tokens", None)
        result["max_tokens"] = int(max_tokens)
        result["temperature"] = 1.0
        result["logprobs"] = True
        return result

    async def _generate(
        self,
        route_key: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        kind: str,
    ) -> TokenOutput:
        if len(prompt_ids) + max_tokens > self.max_model_len:
            raise ValueError(
                f"{kind} exceeds rollout.max_model_len: prompt={len(prompt_ids)} "
                f"max_tokens={max_tokens} limit={self.max_model_len}"
            )
        return await self.server_manager.generate(
            request_id=route_key,
            prompt_ids=prompt_ids,
            sampling_params=self._sampling_params(sampling_params, max_tokens=max_tokens),
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
        critique_instruction_ids = encode_followup_user_turn(
            BRANCH_REVISION_CRITIQUE_PROMPT,
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
                    rollout_id,
                    critique_prompt,
                    sampling_params,
                    max_tokens=critique_cap,
                    kind=f"critique[{index}]",
                )
            )
            for index in range(self.feature.num_critiques)
        ]
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)
        critique_errors = [
            f"critique[{index}]: {result!r}"
            for index, result in enumerate(critique_results)
            if isinstance(result, BaseException)
        ]
        if critique_errors:
            raise RuntimeError(
                "branch-revision critique generation failed after draining every request: " + "; ".join(critique_errors)
            )

        parsed_records: list[dict[str, Any]] = []
        continuation_tasks: list[asyncio.Task] = []
        continuation_indices: list[int] = []
        num_preempted = 0
        for index, raw_output in enumerate(critique_results):
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
            revised_prefix_ids = list(parsed.revised_prefix_ids)
            continuation_max_tokens = 0
            if parsed.valid:
                continuation_max_tokens = min(
                    self.response_length - len(revised_prefix_ids),
                    self.max_model_len - len(prompt_ids) - len(revised_prefix_ids),
                )
                if continuation_max_tokens < self.feature.min_continuation_tokens:
                    parse_reason = "insufficient_continuation_budget"
                    revised_prefix_ids = []
                    continuation_max_tokens = 0
            parsed_records.append(
                {
                    "token_ids": critique_ids,
                    "log_probs": critique_log_probs,
                    "finish_reason": self._finish_reason(raw_output),
                    "parse_reason": parse_reason,
                    "branch_text": parsed.branch_text,
                    "new_continuation_text": parsed.new_continuation_text,
                    "revised_prefix_ids": revised_prefix_ids,
                    "continuation_max_tokens": continuation_max_tokens,
                }
            )
            if parse_reason != "valid":
                continue
            max_tokens = continuation_max_tokens
            if max_tokens < self.feature.min_continuation_tokens:
                raise RuntimeError("valid branch revision unexpectedly lacks its minimum continuation capacity")
            continuation_tasks.append(
                asyncio.create_task(
                    self._generate(
                        rollout_id,
                        [*prompt_ids, *revised_prefix_ids],
                        sampling_params,
                        max_tokens=max_tokens,
                        kind=f"continuation[{index}]",
                    )
                )
            )
            continuation_indices.append(index)

        continuation_results = (
            await asyncio.gather(*continuation_tasks, return_exceptions=True) if continuation_tasks else []
        )
        continuation_errors = [
            f"continuation[{index}]: {result!r}"
            for index, result in zip(continuation_indices, continuation_results, strict=True)
            if isinstance(result, BaseException)
        ]
        if continuation_errors:
            raise RuntimeError(
                "branch-revision continuation generation failed after draining every request: "
                + "; ".join(continuation_errors)
            )
        for index, raw_output in zip(continuation_indices, continuation_results, strict=True):
            if not isinstance(raw_output, TokenOutput):
                raise TypeError(f"continuation[{index}] returned unexpected type {type(raw_output)!r}")
            num_preempted += int(raw_output.num_preempted or 0)
            max_tokens = self.response_length - len(parsed_records[index]["revised_prefix_ids"])
            continuation_ids, continuation_log_probs = self._validated_output(
                raw_output,
                cap=min(max_tokens, int(parsed_records[index]["continuation_max_tokens"])),
                kind=f"continuation[{index}]",
            )
            parsed_records[index]["continuation_ids"] = continuation_ids
            parsed_records[index]["continuation_log_probs"] = continuation_log_probs
            parsed_records[index]["continuation_finish_reason"] = self._finish_reason(raw_output)

        critiques = tuple(
            BranchRevisionCritiqueGeneration(
                token_ids=tuple(record["token_ids"]),
                log_probs=tuple(record["log_probs"]),
                finish_reason=record["finish_reason"],
                parse_reason=record["parse_reason"],
                branch_text=record["branch_text"],
                new_continuation_text=record["new_continuation_text"],
                revised_prefix_ids=tuple(record["revised_prefix_ids"]),
                continuation_ids=tuple(record.get("continuation_ids", ())),
                continuation_log_probs=tuple(record.get("continuation_log_probs", ())),
                continuation_finish_reason=record.get("continuation_finish_reason"),
                continuation_max_tokens=int(record["continuation_max_tokens"]),
            )
            for record in parsed_records
        )
        record = BranchRevisionGenerationRecord(
            rollout_id=rollout_id,
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
