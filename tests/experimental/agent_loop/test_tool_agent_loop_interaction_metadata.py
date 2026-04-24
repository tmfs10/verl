# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from unittest.mock import AsyncMock

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.workers.rollout.replica import TokenOutput


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_interacting_state_merges_interaction_metadata():
    loop = object.__new__(ToolAgentLoop)
    loop.apply_chat_template = AsyncMock(return_value=[99])

    interaction = AsyncMock()
    interaction.generate_response = AsyncMock(
        return_value=(
            False,
            "Reflect and try again.",
            0.0,
            {"answer_history": ["41"], "entropy_bonus_coef": 0.5},
        )
    )

    agent_data = AgentData(
        messages=[{"role": "assistant", "content": "41"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-1",
        tools_kwargs={},
        interaction=interaction,
        interaction_kwargs={"name": "math_verify"},
    )
    agent_data.last_assistant_stop_reason = "completed"

    state = await ToolAgentLoop._handle_interacting_state(loop, agent_data)

    assert state == AgentState.GENERATING
    assert interaction.generate_response.await_count == 1
    _, call_kwargs = interaction.generate_response.await_args
    assert call_kwargs["stop_reason"] == "completed"
    assert call_kwargs["name"] == "math_verify"
    assert agent_data.extra_fields["answer_history"] == ["41"]
    assert agent_data.extra_fields["entropy_bonus_coef"] == 0.5
    assert agent_data.turn_scores == [0.0]
    assert agent_data.user_turns == 1
    assert agent_data.messages[-1] == {"role": "user", "content": "Reflect and try again."}
    assert agent_data.prompt_ids == [99]
    assert agent_data.response_mask == [0]


@pytest.mark.asyncio
async def test_interacting_state_rebuilds_prompt_when_interaction_requests_prompt_reset():
    loop = object.__new__(ToolAgentLoop)
    loop.apply_chat_template = AsyncMock(return_value=[7, 8, 9])
    loop.tool_schemas = []

    interaction = AsyncMock()
    interaction.generate_response = AsyncMock(
        return_value=(
            False,
            "Ignored in reset mode.",
            0.0,
            {
                "answer_history": ["41"],
                "reward_mode": "last_completed_turn",
                "reset_generation_prompt": True,
                "next_generation_messages": [{"role": "user", "content": "Question summary with past answers"}],
            },
        )
    )

    agent_data = AgentData(
        messages=[{"role": "assistant", "content": "41"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-reset",
        tools_kwargs={},
        interaction=interaction,
        interaction_kwargs={"name": "math_verify"},
    )
    agent_data.last_assistant_stop_reason = "completed"
    agent_data.turn_prompt_reset = True

    state = await ToolAgentLoop._handle_interacting_state(loop, agent_data)

    assert state == AgentState.GENERATING
    assert agent_data.messages == [{"role": "user", "content": "Question summary with past answers"}]
    assert agent_data.prompt_ids == [7, 8, 9]
    assert agent_data.response_mask == []
    assert agent_data.turn_scores == [0.0]
    assert agent_data.extra_fields["reward_mode"] == "last_completed_turn"
    assert agent_data.user_turns == 1


@pytest.mark.asyncio
async def test_generating_state_collects_terminal_interaction_metadata_on_turn_cap():
    loop = object.__new__(ToolAgentLoop)
    loop.response_length = 32
    loop.max_assistant_turns = 1
    loop.max_user_turns = 0
    loop.interaction_config_file = "interaction.json"
    loop.tools = {}
    loop.tool_parser = SimpleNamespace(extract_tool_calls=AsyncMock(return_value=(None, [])))
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(return_value=TokenOutput(token_ids=[1, 2], log_probs=[-0.1, -0.2], extra_fields={}))
    )
    loop.loop = asyncio.get_running_loop()
    loop.tokenizer = SimpleNamespace(decode=lambda token_ids, skip_special_tokens=True: "Final wrong answer")

    interaction = AsyncMock()
    interaction.generate_response = AsyncMock(
        return_value=(
            False,
            "Reflect and try again.",
            0.0,
            {"answer_history": ["41"], "entropy_bonus_coef": 0.5},
        )
    )

    agent_data = AgentData(
        messages=[{"role": "user", "content": "Solve the problem"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-2",
        tools_kwargs={},
        interaction=interaction,
        interaction_kwargs={"name": "math_verify"},
    )

    state = await ToolAgentLoop._handle_generating_state(loop, agent_data, sampling_params={})

    assert state == AgentState.TERMINATED
    interaction.generate_response.assert_awaited_once_with(
        "req-2",
        [
            {"role": "user", "content": "Solve the problem"},
            {"role": "assistant", "content": "Final wrong answer"},
        ],
        stop_reason="completed",
        name="math_verify",
    )
    assert agent_data.messages[-1] == {"role": "assistant", "content": "Final wrong answer"}
    assert agent_data.turn_scores == [0.0]
    assert agent_data.extra_fields["answer_history"] == ["41"]
    assert agent_data.extra_fields["entropy_bonus_coef"] == 0.5
    assert agent_data.user_turns == 0


def test_finalize_output_sequences_uses_selected_turn_rollout_for_prompt_reset_mode():
    loop = object.__new__(ToolAgentLoop)
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-finalize",
        tools_kwargs={},
    )
    agent_data.turn_prompt_reset = True
    agent_data.selected_turn_rollout = {
        "prompt_ids": [1, 2, 3],
        "response_ids": [4, 5],
        "response_logprobs": [-0.1, -0.2],
    }

    prompt_ids, response_ids, response_mask, response_logprobs = ToolAgentLoop._finalize_output_sequences(
        loop, agent_data
    )

    assert prompt_ids == [1, 2, 3]
    assert response_ids == [4, 5]
    assert response_mask == [1, 1]
    assert response_logprobs == [-0.1, -0.2]


@pytest.mark.anyio
async def test_run_injects_ground_truth_and_extra_info_into_interaction_kwargs():
    loop = object.__new__(ToolAgentLoop)
    loop.response_length = 32
    loop.per_turn_response_length = None
    loop.interaction_config_file = "interaction.json"
    loop.interaction_map = {"lean_goedel": AsyncMock()}
    loop.process_vision_info = AsyncMock(return_value={})
    loop._handle_pending_state = AsyncMock(return_value=AgentState.TERMINATED)
    interaction = loop.interaction_map["lean_goedel"]
    interaction.start_interaction = AsyncMock(return_value="req-goedel")
    interaction.finalize_interaction = AsyncMock()

    await ToolAgentLoop.run(
        loop,
        sampling_params={},
        raw_prompt=[{"role": "user", "content": "Initial Goedel prompt"}],
        extra_info={"problem_id": "Goedel-Pset-7", "interaction_kwargs": {"name": "lean_goedel"}},
        reward_model={"ground_truth": '{"formal_statement": "theorem foo : True := by sorry"}'},
        data_source="goedel_lean",
    )

    _, call_kwargs = interaction.start_interaction.await_args
    assert call_kwargs["raw_prompt"] == [{"role": "user", "content": "Initial Goedel prompt"}]
    assert call_kwargs["ground_truth"] == '{"formal_statement": "theorem foo : True := by sorry"}'
    assert call_kwargs["extra_info"]["problem_id"] == "Goedel-Pset-7"
    assert call_kwargs["data_source"] == "goedel_lean"
    interaction.finalize_interaction.assert_awaited_once()
