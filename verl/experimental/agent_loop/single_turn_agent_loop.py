# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        response_length = int(kwargs.get("response_length_override", self.response_length))

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        if kwargs.get("prompt_ids_override", None) is not None:
            prompt_ids = normalize_token_ids(kwargs["prompt_ids_override"])
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                chat_template_kwargs=kwargs.get("chat_template_kwargs"),
            )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        response_mask = [1] * len(token_output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_output.token_ids[:response_length],
            response_mask=response_mask[:response_length],
            response_logprobs=token_output.log_probs[:response_length] if token_output.log_probs else None,
            routed_experts=(
                token_output.routed_experts[: len(prompt_ids) + response_length]
                if token_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=token_output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        finish_reason = output.extra_fields.get("finish_reason", token_output.stop_reason)
        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "stop_reason": token_output.stop_reason,
                "finish_reason": finish_reason,
            }
        )

        return output
