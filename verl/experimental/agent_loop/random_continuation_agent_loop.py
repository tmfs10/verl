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
"""One original rollout plus natural continuations from random valid prefixes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.trainer.config import RandomContinuationBaselineConfig
from verl.trainer.ppo.branch_revision_grpo import strip_terminal_eos
from verl.trainer.ppo.random_continuation_baseline import (
    RandomMarkSelection,
    select_structurally_valid_random_marks,
    stable_random,
    stable_seed,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

RANDOM_CONTINUATION_AGENT_NAME = "random_continuation_baseline_agent"
RANDOM_CONTINUATION_RECORD_FIELD = "__random_continuation_baseline_record__"


@dataclass(frozen=True)
class RandomContinuationGeneration:
    mark: int
    sample_index: int
    sampling_seed: int
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    finish_reason: str | None


@dataclass(frozen=True)
class RandomContinuationRecord:
    rollout_id: str
    original_sample_index: int
    original_sampling_seed: int
    editable_solution_length: int
    selection: RandomMarkSelection
    continuations: tuple[RandomContinuationGeneration, ...]
    failures: tuple[tuple[int, int, str], ...]


def _as_prompt_ids(value: Any) -> list[int]:
    result = [int(token) for token in normalize_token_ids(value)]
    if not result:
        raise ValueError("random-continuation prompt must contain at least one token")
    return result


@register(RANDOM_CONTINUATION_AGENT_NAME)
class RandomContinuationAgentLoop(AgentLoopBase):
    """Generate the complete evaluation payload inside one blocking agent turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.random_continuation_baseline,
            dataclass_type=RandomContinuationBaselineConfig,
        )
        self.response_length = int(self.rollout_config.response_length)
        self.max_model_len = int(self.rollout_config.max_model_len)

    @staticmethod
    def _sampling_params(base: dict[str, Any], *, max_tokens: int, sampling_seed: int) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("random-continuation max_tokens must be positive")
        result = dict(base)
        result.pop("max_new_tokens", None)
        result.update(
            {
                "max_tokens": int(max_tokens),
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "logprobs": True,
                "seed": int(sampling_seed),
            }
        )
        return result

    async def _generate(
        self,
        route_key: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        sampling_seed: int,
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
            sampling_params=self._sampling_params(
                sampling_params,
                max_tokens=max_tokens,
                sampling_seed=sampling_seed,
            ),
        )

    @staticmethod
    def _validated_output(output: TokenOutput, *, cap: int, kind: str) -> tuple[list[int], list[float]]:
        token_ids = [int(token) for token in output.token_ids[:cap]]
        if not token_ids:
            raise RuntimeError(f"{kind} generation returned no tokens")
        if output.log_probs is None or len(output.log_probs) < len(token_ids):
            raise RuntimeError(f"{kind} generation did not return one processed log probability per token")
        return token_ids, [float(value) for value in output.log_probs[: len(token_ids)]]

    async def _prompt_ids(self, kwargs: dict[str, Any]) -> list[int]:
        override = kwargs.get("prompt_ids_override")
        if override is not None:
            return _as_prompt_ids(override)
        return await self.apply_chat_template(list(kwargs["raw_prompt"]))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.monotonic()
        rollout_id = str(kwargs["random_continuation_rollout_id"])
        original_sample_index = int(kwargs["random_continuation_original_sample_index"])
        dataset_index = kwargs.get("index", rollout_id)
        prompt_ids = await self._prompt_ids(kwargs)
        original_sampling_seed = stable_seed(
            self.feature.selection_seed,
            dataset_index,
            original_sample_index,
            "original",
        )
        parent = await self._generate(
            f"{rollout_id}:original",
            prompt_ids,
            sampling_params,
            max_tokens=self.response_length,
            sampling_seed=original_sampling_seed,
            kind="original solution",
        )
        solution_ids, solution_log_probs = self._validated_output(
            parent,
            cap=self.response_length,
            kind="original solution",
        )
        editable_solution_ids = strip_terminal_eos(solution_ids, self.tokenizer)
        selection = select_structurally_valid_random_marks(
            editable_solution_ids,
            tokenizer=self.tokenizer,
            points_per_rollout=self.feature.points_per_rollout,
            min_prefix_fraction=self.feature.min_prefix_fraction,
            response_budget=self.response_length,
            min_continuation_tokens=self.feature.min_continuation_tokens,
            rng=stable_random(
                self.feature.selection_seed,
                dataset_index,
                original_sample_index,
                "marks",
            ),
            structural_boundaries_only=self.feature.structural_boundaries_only,
        )

        tasks: list[asyncio.Task] = []
        metadata: list[tuple[int, int]] = []
        for mark in selection.marks:
            max_tokens = self.response_length - mark
            for sample_index in range(self.feature.continuations_per_mark):
                continuation_seed = stable_seed(
                    self.feature.selection_seed,
                    dataset_index,
                    original_sample_index,
                    "continuation",
                    mark,
                    sample_index,
                )
                tasks.append(
                    asyncio.create_task(
                        self._generate(
                            f"{rollout_id}:continuation:{mark}:{sample_index}",
                            [*prompt_ids, *editable_solution_ids[:mark]],
                            sampling_params,
                            max_tokens=max_tokens,
                            sampling_seed=continuation_seed,
                            kind=f"random continuation[{mark},{sample_index}]",
                        )
                    )
                )
                metadata.append((mark, sample_index))

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        continuations: list[RandomContinuationGeneration] = []
        failures: list[tuple[int, int, str]] = []
        child_preempted = 0
        for (mark, sample_index), result in zip(metadata, results, strict=True):
            if isinstance(result, BaseException):
                failures.append((mark, sample_index, repr(result)))
                continue
            child_preempted += int(result.num_preempted or 0)
            try:
                token_ids, log_probs = self._validated_output(
                    result,
                    cap=self.response_length - mark,
                    kind=f"random continuation[{mark},{sample_index}]",
                )
            except Exception as error:
                failures.append((mark, sample_index, repr(error)))
                continue
            extra = dict(result.extra_fields or {})
            finish_reason = extra.get("finish_reason", result.stop_reason)
            continuations.append(
                RandomContinuationGeneration(
                    mark=mark,
                    sample_index=sample_index,
                    sampling_seed=stable_seed(
                        self.feature.selection_seed,
                        dataset_index,
                        original_sample_index,
                        "continuation",
                        mark,
                        sample_index,
                    ),
                    token_ids=tuple(token_ids),
                    log_probs=tuple(log_probs),
                    finish_reason=None if finish_reason is None else str(finish_reason),
                )
            )

        record = RandomContinuationRecord(
            rollout_id=rollout_id,
            original_sample_index=original_sample_index,
            original_sampling_seed=original_sampling_seed,
            editable_solution_length=len(editable_solution_ids),
            selection=selection,
            continuations=tuple(continuations),
            failures=tuple(failures),
        )
        extra_fields = dict(parent.extra_fields or {})
        extra_fields.update(
            {
                RANDOM_CONTINUATION_RECORD_FIELD: record,
                "turn_scores": [],
                "tool_rewards": [],
                "stop_reason": parent.stop_reason,
                "finish_reason": extra_fields.get("finish_reason", parent.stop_reason),
            }
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=solution_ids,
            response_mask=[1] * len(solution_ids),
            response_logprobs=solution_log_probs,
            routed_experts=parent.routed_experts,
            multi_modal_data={},
            num_turns=2,
            metrics={
                "generate_sequences": time.monotonic() - started,
                "tool_calls": 0.0,
                "num_preempted": int(parent.num_preempted or 0) + child_preempted,
            },
            extra_fields=extra_fields,
        )
