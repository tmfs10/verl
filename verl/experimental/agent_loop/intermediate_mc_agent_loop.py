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
"""Composite rollout used by synchronous intermediate-MC PPO training."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.trainer.config import INTERMEDIATE_MC_CRITIQUE_PROMPT
from verl.trainer.ppo.intermediate_mc_value import (
    CRITIQUE_DELIMITER,
    SOLUTION_DELIMITER,
    select_random_marks,
    stable_rng,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

INTERMEDIATE_MC_AGENT_NAME = "intermediate_mc_agent"
INTERMEDIATE_MC_CHILD_FIELD = "__intermediate_mc_children__"


@dataclass(frozen=True)
class CritiqueGeneration:
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]


@dataclass(frozen=True)
class ContinuationGeneration:
    mark: int
    sample_index: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class IntermediateMCGenerationRecord:
    rollout_id: str
    critiques: tuple[CritiqueGeneration, ...]
    selected_marks: tuple[int, ...]
    continuations: tuple[ContinuationGeneration, ...]
    failed_continuations: tuple[tuple[int, int], ...]
    selector_diagnostics: tuple[dict[str, Any], ...]


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


@register(INTERMEDIATE_MC_AGENT_NAME)
class IntermediateMCAgentLoop(AgentLoopBase):
    """Generate one solution and immediately fan out its synchronous child work.

    The surrounding trainer still observes a blocking generation call. Concurrency
    here is confined to independent requests belonging to the same iteration and
    every task is drained before success or failure is returned.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from verl.trainer.config import IntermediateMCValueConfig

        self.feature = omega_conf_to_dataclass(
            self.config.algorithm.intermediate_mc_value,
            dataclass_type=IntermediateMCValueConfig,
        )
        self.response_length = int(self.rollout_config.response_length)
        self.max_model_len = int(self.rollout_config.max_model_len)
        self.solution_delimiter_ids = self.tokenizer.encode(SOLUTION_DELIMITER, add_special_tokens=False)
        if not self.solution_delimiter_ids:
            raise ValueError("intermediate MC solution delimiter must tokenize non-empty")
        if self.feature.num_critiques > 0:
            self.critique_instruction_ids = self.tokenizer.encode(
                "\n\n" + INTERMEDIATE_MC_CRITIQUE_PROMPT,
                add_special_tokens=False,
            )
            self.critique_delimiter_ids = self.tokenizer.encode(CRITIQUE_DELIMITER, add_special_tokens=False)
            if not self.critique_instruction_ids or not self.critique_delimiter_ids:
                raise ValueError("intermediate MC critique delimiter and instruction must tokenize non-empty")
        else:
            self.critique_instruction_ids = []
            self.critique_delimiter_ids = []

    @staticmethod
    def _sampling_params(base: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("intermediate MC generation max_tokens must be positive")
        result = dict(base)
        result.pop("max_new_tokens", None)
        result["max_tokens"] = int(max_tokens)
        result["temperature"] = 1.0
        result["logprobs"] = True
        return result

    def _assert_capacity(self, prompt_ids: list[int], max_tokens: int, kind: str) -> None:
        requested = len(prompt_ids) + max_tokens
        if requested > self.max_model_len:
            raise ValueError(
                f"{kind} request exceeds rollout.max_model_len: "
                f"prompt={len(prompt_ids)} max_tokens={max_tokens} limit={self.max_model_len}"
            )

    async def _generate(
        self,
        route_key: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        kind: str,
    ) -> TokenOutput:
        self._assert_capacity(prompt_ids, max_tokens, kind)
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
        return token_ids, [float(value) for value in output.log_probs[: len(token_ids)]]

    async def _prompt_ids(self, kwargs: dict[str, Any]) -> list[int]:
        override = kwargs.get("prompt_ids_override")
        if override is not None:
            return _as_int_list(override, "prompt_ids_override")
        return await self.apply_chat_template(list(kwargs["raw_prompt"]))

    def _select_marks(
        self,
        solution_ids: list[int],
        *,
        rollout_id: str,
        global_step: int,
        dataset_index: object,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        k = self.feature.resolved_max_marks
        # EMA and variance depend on critic outputs and are selected by the
        # synchronous controller after critic inference.
        if self.feature.mark_selector != "random" or k == 0:
            return [], []
        rng = stable_rng(self.feature.selection_seed, global_step, dataset_index, rollout_id)
        marks = select_random_marks(
            len(solution_ids),
            k=k,
            min_gap=self.feature.min_mark_gap,
            start_fraction=self.feature.mark_start_fraction,
            end_fraction=self.feature.mark_end_fraction,
            rng=rng,
        )
        return marks, [{"token": mark, "reason": "random"} for mark in marks]

    async def _generate_children(
        self,
        *,
        route_key: str,
        prompt_ids: list[int],
        solution_ids: list[int],
        solution_log_probs: list[float],
        selected_marks: list[int],
        sampling_params: dict[str, Any],
        critic_context_limit: int,
        include_critiques: bool,
    ) -> tuple[
        tuple[CritiqueGeneration, ...],
        tuple[ContinuationGeneration, ...],
        tuple[tuple[int, int], ...],
        int,
    ]:
        tasks: list[asyncio.Task] = []
        task_metadata: list[tuple[str, int, int]] = []
        critique_cap = int(self.feature.critique_max_response_length or self.response_length)
        if include_critiques:
            critique_prompt = [*prompt_ids, *solution_ids, *self.critique_instruction_ids]
            actor_available = self.max_model_len - len(critique_prompt)
            critic_fixed = (
                len(prompt_ids)
                + len(self.critique_delimiter_ids)
                + len(self.solution_delimiter_ids)
                + len(solution_ids)
            )
            critic_available = critic_context_limit - critic_fixed
            critique_cap = min(critique_cap, actor_available, critic_available)
            if critique_cap <= 0:
                raise ValueError(
                    "self-critique has no capacity in the actor and critic context windows: "
                    f"actor_available={actor_available} critic_available={critic_available}"
                )
            for critique_index in range(self.feature.num_critiques):
                tasks.append(
                    asyncio.create_task(
                        self._generate(
                            route_key,
                            critique_prompt,
                            sampling_params,
                            max_tokens=critique_cap,
                            kind=f"critique[{critique_index}]",
                        )
                    )
                )
                task_metadata.append(("critique", critique_index, -1))
        for mark in selected_marks:
            if not 1 <= mark < len(solution_ids):
                raise ValueError(f"continuation mark {mark} must satisfy 1 <= mark < {len(solution_ids)}")
            max_tokens = self.response_length - mark
            continuation_prompt = [*prompt_ids, *solution_ids[:mark]]
            for sample_index in range(self.feature.continuations_per_mark):
                tasks.append(
                    asyncio.create_task(
                        self._generate(
                            route_key,
                            continuation_prompt,
                            sampling_params,
                            max_tokens=max_tokens,
                            kind=f"continuation[{mark},{sample_index}]",
                        )
                    )
                )
                task_metadata.append(("continuation", mark, sample_index))

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        critiques: list[CritiqueGeneration] = []
        continuations: list[ContinuationGeneration] = []
        failed_continuations: list[tuple[int, int]] = []
        critique_errors: list[str] = []
        num_preempted = 0
        for metadata, result in zip(task_metadata, results, strict=True):
            kind, first, second = metadata
            if isinstance(result, BaseException):
                if kind == "critique":
                    critique_errors.append(f"critique[{first}]: {result!r}")
                else:
                    failed_continuations.append((first, second))
                continue
            num_preempted += int(result.num_preempted or 0)
            try:
                cap = critique_cap if kind == "critique" else self.response_length - first
                token_ids, log_probs = self._validated_output(result, cap=cap, kind=kind)
            except Exception as error:
                if kind == "critique":
                    critique_errors.append(f"critique[{first}]: {error!r}")
                else:
                    failed_continuations.append((first, second))
                continue
            if kind == "critique":
                critiques.append(CritiqueGeneration(tuple(token_ids), tuple(log_probs)))
            else:
                continuations.append(ContinuationGeneration(first, second, tuple(token_ids)))
        if critique_errors:
            raise RuntimeError(
                "critique generation failed after draining all child requests: " + "; ".join(critique_errors)
            )
        if include_critiques and len(critiques) != self.feature.num_critiques:
            raise RuntimeError(
                f"expected {self.feature.num_critiques} critiques after draining children, got {len(critiques)}"
            )
        return tuple(critiques), tuple(continuations), tuple(failed_continuations), num_preempted

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.monotonic()
        stage = str(kwargs.get("intermediate_mc_stage", "solution"))
        rollout_id = str(kwargs["intermediate_mc_rollout_id"])
        critic_context_limit = int(kwargs["intermediate_mc_critic_context_limit"])
        if stage == "solution":
            prompt_ids = await self._prompt_ids(kwargs)
            parent = await self._generate(
                rollout_id,
                prompt_ids,
                sampling_params,
                max_tokens=self.response_length,
                kind="solution",
            )
            solution_ids, solution_log_probs = self._validated_output(
                parent,
                cap=self.response_length,
                kind="solution",
            )
            if bool(kwargs.get("intermediate_mc_warmup", False)):
                selected_marks, diagnostics = [], []
            else:
                selected_marks, diagnostics = self._select_marks(
                    solution_ids,
                    rollout_id=rollout_id,
                    global_step=int(kwargs.get("intermediate_mc_global_step", -1)),
                    dataset_index=kwargs.get("index", rollout_id),
                )
            critiques, continuations, failures, child_preempted = await self._generate_children(
                route_key=rollout_id,
                prompt_ids=prompt_ids,
                solution_ids=solution_ids,
                solution_log_probs=solution_log_probs,
                selected_marks=selected_marks,
                sampling_params=sampling_params,
                critic_context_limit=critic_context_limit,
                include_critiques=self.feature.num_critiques > 0,
            )
        elif stage == "continuations":
            prompt_ids = _as_int_list(kwargs["intermediate_mc_parent_prompt_ids"], "parent prompt")
            solution_ids = _as_int_list(kwargs["intermediate_mc_parent_solution_ids"], "parent solution")
            solution_log_probs = _as_float_list(
                kwargs["intermediate_mc_parent_solution_log_probs"],
                "parent solution log probabilities",
            )
            if len(solution_ids) != len(solution_log_probs):
                raise ValueError("parent solution tokens and log probabilities must have equal lengths")
            selected_marks = [int(mark) for mark in kwargs["intermediate_mc_selected_marks"]]
            diagnostics = []
            critiques, continuations, failures, child_preempted = await self._generate_children(
                route_key=rollout_id,
                prompt_ids=prompt_ids,
                solution_ids=solution_ids,
                solution_log_probs=solution_log_probs,
                selected_marks=selected_marks,
                sampling_params=sampling_params,
                critic_context_limit=critic_context_limit,
                include_critiques=False,
            )
            parent = TokenOutput(token_ids=solution_ids, log_probs=solution_log_probs)
        else:
            raise ValueError(f"unknown intermediate MC agent stage {stage!r}")

        record = IntermediateMCGenerationRecord(
            rollout_id=rollout_id,
            critiques=critiques,
            selected_marks=tuple(selected_marks),
            continuations=continuations,
            failed_continuations=failures,
            selector_diagnostics=tuple(diagnostics),
        )
        metrics = {
            "generate_sequences": time.monotonic() - started,
            "tool_calls": 0.0,
            "num_preempted": int(parent.num_preempted or 0) + child_preempted,
        }
        extra_fields = dict(parent.extra_fields)
        extra_fields.update(
            {
                INTERMEDIATE_MC_CHILD_FIELD: record,
                "turn_scores": [],
                "tool_rewards": [],
                "stop_reason": parent.stop_reason,
                "finish_reason": parent.extra_fields.get("finish_reason", parent.stop_reason),
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
            metrics=metrics,
            extra_fields=extra_fields,
        )
