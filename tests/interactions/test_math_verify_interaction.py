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

from unittest.mock import patch

import pytest

from verl.interactions.math_verify_interaction import INVALID_ANSWER, MathVerifyInteraction


class TestMathVerifyInteraction:
    def setup_method(self):
        self.config = {"name": "math_verify"}
        self.interaction = MathVerifyInteraction(self.config)

    @pytest.mark.asyncio
    async def test_generate_response_tracks_answer_history_and_default_entropy_coef(self):
        instance_id = await self.interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "First attempt"}],
            )

        assert should_terminate is False
        assert response == "Your response is incorrect! You need to reflect on your answer and try again."
        assert reward == 0.0
        assert metadata["answer_history"] == ["41"]
        assert metadata["entropy_bonus_coef"] == 0.0
        assert metadata["last_extracted_answer"] == "41"
        assert metadata["last_answer_correct"] is False

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "43"},
        ):
            _, _, _, metadata = await self.interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Second attempt"}],
            )

        assert metadata["answer_history"] == ["41", "43"]

    @pytest.mark.asyncio
    async def test_generate_response_normalizes_missing_prediction(self):
        instance_id = await self.interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": None},
        ):
            _, _, _, metadata = await self.interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Unparseable attempt"}],
            )

        assert metadata["answer_history"] == [INVALID_ANSWER]
        assert metadata["last_extracted_answer"] == INVALID_ANSWER

    @pytest.mark.asyncio
    async def test_generate_response_terminates_when_answer_is_correct(self):
        instance_id = await self.interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": 1.0, "acc": True, "pred": "42"},
        ):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Correct answer"}],
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert reward == 1.0
        assert metadata["answer_history"] == ["42"]
        assert metadata["last_answer_correct"] is True

    @pytest.mark.asyncio
    async def test_repeat_until_stable_mode_stops_on_repeated_completed_answer(self):
        interaction = MathVerifyInteraction({"name": "math_verify", "interaction_mode": "repeat_until_stable"})
        instance_id = await interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            should_terminate, response, reward, metadata = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "First attempt"}],
                stop_reason="completed",
            )

        assert should_terminate is False
        assert response == interaction.repeat_until_stable_prompt
        assert reward == 0.0
        assert metadata["reward_mode"] == "last_completed_turn"
        assert metadata["has_last_completed_answer"] is True
        assert metadata["last_completed_answer"] == "41"
        assert metadata["last_completed_answer_correct"] is False

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            should_terminate, response, reward, metadata = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Repeated attempt"}],
                stop_reason="completed",
            )

        assert should_terminate is True
        assert response == "Your latest completed final answer repeats a previous completed answer. Stopping."
        assert reward == 0.0
        assert metadata["last_completed_answer"] == "41"

    @pytest.mark.asyncio
    async def test_repeat_until_stable_mode_ignores_interrupted_turn_for_final_reward_state(self):
        interaction = MathVerifyInteraction({"name": "math_verify", "interaction_mode": "repeat_until_stable"})
        instance_id = await interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": 1.0, "acc": True, "pred": "42"},
        ):
            await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Completed correct answer"}],
                stop_reason="completed",
            )

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            _, _, _, metadata = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Interrupted wrong answer"}],
                stop_reason="aborted",
            )

        assert metadata["last_turn_completed"] is False
        assert metadata["last_completed_answer"] == "42"
        assert metadata["last_completed_answer_correct"] is True

    @pytest.mark.asyncio
    async def test_question_with_past_answers_mode_builds_next_generation_prompt(self):
        interaction = MathVerifyInteraction({"name": "math_verify", "turn_context_mode": "question_with_past_answers"})
        raw_prompt = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 6 * 7?"},
        ]
        instance_id = await interaction.start_interaction(
            instance_id="test_instance",
            ground_truth="42",
            raw_prompt=raw_prompt,
            per_turn_response_length=6000,
        )

        assert "You only have 6000 tokens to produce an answer for this turn." in raw_prompt[-1]["content"]

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            should_terminate, _, reward, metadata = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "First attempt"}],
                stop_reason="completed",
            )

        assert should_terminate is False
        assert reward == 0.0
        assert metadata["reset_generation_prompt"] is True
        assert metadata["reward_mode"] == "last_completed_turn"
        assert metadata["next_generation_messages"][0] == {"role": "system", "content": "You are a math tutor."}
        assert "Question:\nWhat is 6 * 7?" in metadata["next_generation_messages"][-1]["content"]
        assert "1. 41" in metadata["next_generation_messages"][-1]["content"]
        assert "You only have 6000 tokens to produce an answer for this turn." in metadata["next_generation_messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_generate_response_appends_turn_budget_to_retry_prompts(self):
        interaction = MathVerifyInteraction({"name": "math_verify"})
        instance_id = await interaction.start_interaction(
            instance_id="test_instance",
            ground_truth="42",
            raw_prompt=[{"role": "user", "content": "What is 6 * 7?"}],
            per_turn_response_length=6000,
        )

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            should_terminate, response, reward, _ = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Wrong answer"}],
            )

        assert should_terminate is False
        assert reward == 0.0
        assert "You only have 6000 tokens to produce an answer for this turn." in response

    @pytest.mark.asyncio
    async def test_s2r_mode_requests_verification_after_answer_turn(self):
        interaction = MathVerifyInteraction({"name": "math_verify", "interaction_mode": "s2r"})
        instance_id = await interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            should_terminate, response, reward, metadata = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Initial attempt"}],
                stop_reason="completed",
            )

        assert should_terminate is False
        assert response == interaction.s2r_verify_prompt
        assert reward == -1.0
        assert metadata["reward_mode"] == "last_completed_turn"
        assert metadata["last_completed_answer"] == "41"
        assert metadata["s2r_processed_phase"] == "answer"
        assert metadata["s2r_next_phase"] == "verify"
        assert metadata["s2r_answer_correct_history"] == [False]
        assert metadata["s2r_verification_verdict_history"] == []

    @pytest.mark.asyncio
    async def test_s2r_mode_retries_after_matching_incorrect_verification(self):
        interaction = MathVerifyInteraction({"name": "math_verify", "interaction_mode": "s2r"})
        instance_id = await interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": -1.0, "acc": False, "pred": "41"},
        ):
            await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Wrong attempt"}],
                stop_reason="completed",
            )

        should_terminate, response, reward, metadata = await interaction.generate_response(
            instance_id,
            [{"role": "assistant", "content": "Therefore, the answer is incorrect."}],
            stop_reason="completed",
        )

        assert should_terminate is False
        assert response == interaction.s2r_retry_prompt
        assert reward == 1.0
        assert metadata["s2r_processed_phase"] == "verify"
        assert metadata["s2r_next_phase"] == "answer"
        assert metadata["s2r_last_verification_verdict"] == "incorrect"
        assert metadata["s2r_last_verification_matches_answer"] is True
        assert metadata["s2r_verification_verdict_history"] == ["incorrect"]
        assert metadata["s2r_verification_match_history"] == [True]

    @pytest.mark.asyncio
    async def test_s2r_mode_terminates_after_matching_correct_verification(self):
        interaction = MathVerifyInteraction({"name": "math_verify", "interaction_mode": "s2r"})
        instance_id = await interaction.start_interaction(instance_id="test_instance", ground_truth="42")

        with patch(
            "verl.utils.reward_score.math_dapo.compute_score",
            return_value={"score": 1.0, "acc": True, "pred": "42"},
        ):
            await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "Correct attempt"}],
                stop_reason="completed",
            )

        should_terminate, response, reward, metadata = await interaction.generate_response(
            instance_id,
            [{"role": "assistant", "content": "Therefore, the answer is correct."}],
            stop_reason="completed",
        )

        assert should_terminate is True
        assert response == "Stopping after self-verification."
        assert reward == 1.0
        assert metadata["reward_mode"] == "last_completed_turn"
        assert metadata["last_completed_answer"] == "42"
        assert metadata["last_completed_answer_correct"] is True
        assert metadata["s2r_last_verification_verdict"] == "correct"
        assert metadata["s2r_last_verification_matches_answer"] is True

    def test_s2r_mode_rejects_prompt_reset_turn_context(self):
        with pytest.raises(ValueError, match="interaction_mode='s2r'"):
            MathVerifyInteraction(
                {
                    "name": "math_verify",
                    "interaction_mode": "s2r",
                    "turn_context_mode": "question_with_past_answers",
                }
            )
