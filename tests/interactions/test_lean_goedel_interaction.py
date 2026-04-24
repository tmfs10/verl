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

import json
from unittest.mock import patch

import pytest

from verl.interactions.lean_goedel_interaction import LeanGoedelInteraction, build_submission_code


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_ground_truth() -> str:
    return json.dumps(
        {
            "header": "import Mathlib",
            "formal_statement": "theorem foo (h : True) : True := by sorry",
            "project_path": "/lean4/my_project_latest",
            "safe_verify": True,
        }
    )


class TestLeanGoedelInteraction:
    @pytest.mark.anyio
    async def test_build_submission_code_replaces_theorem_body_with_model_proof(self):
        ground_truth = json.loads(_make_ground_truth())
        solution = (
            "Proof plan.\n\n"
            "```lean4\n"
            "theorem foo (h : True) : True := by\n"
            "  exact h\n"
            "```"
        )

        submission = build_submission_code(solution, ground_truth, {})

        assert submission == "import Mathlib\n\ntheorem foo (h : True) : True := by\nexact h\n"

    @pytest.mark.anyio
    async def test_failure_generates_retry_prompt_with_full_history_reset(self):
        interaction = LeanGoedelInteraction({"name": "lean_goedel"})
        ground_truth = _make_ground_truth()
        raw_prompt = [{"role": "user", "content": "Initial Goedel prompt"}]
        instance_id = await interaction.start_interaction(
            instance_id="test_instance",
            ground_truth=ground_truth,
            raw_prompt=raw_prompt,
            extra_info={"problem_id": "Goedel-Pset-1"},
        )
        messages = raw_prompt + [{"role": "assistant", "content": "```lean4\ntheorem foo (h : True) : True := by\n  trivial\n```"}]

        with patch(
            "verl.interactions.lean_goedel_interaction.evaluate_lean_attempt",
            return_value={
                "final_ok": False,
                "compile_ok": False,
                "safe_verify_ok": False,
                "safe_verify_requested": True,
                "process_status": "failed",
                "request_error": "",
                "stdout": "",
                "stderr": "type mismatch",
                "safe_verify_stdout": "",
                "safe_verify_stderr": "",
                "host": "http://127.0.0.1:6000",
                "submission_code": "theorem foo (h : True) : True := by\ntrivial\n",
                "error_message": "type mismatch",
            },
        ):
            should_terminate, response, reward, metadata = await interaction.generate_response(
                instance_id,
                messages,
                stop_reason="completed",
            )

        assert should_terminate is False
        assert reward == 0.0
        assert "Round 1" in response
        assert "type mismatch" in response
        assert metadata["reset_generation_prompt"] is True
        assert metadata["last_turn_completed"] is True
        assert metadata["has_last_completed_proof"] is True
        assert metadata["last_completed_proof_correct"] is False
        assert metadata["next_generation_messages"] == messages + [{"role": "user", "content": response}]

    @pytest.mark.anyio
    async def test_success_terminates_without_retry_prompt(self):
        interaction = LeanGoedelInteraction({"name": "lean_goedel"})
        instance_id = await interaction.start_interaction(
            instance_id="test_instance",
            ground_truth=_make_ground_truth(),
            raw_prompt=[{"role": "user", "content": "Initial Goedel prompt"}],
            extra_info={},
        )

        with patch(
            "verl.interactions.lean_goedel_interaction.evaluate_lean_attempt",
            return_value={
                "final_ok": True,
                "compile_ok": True,
                "safe_verify_ok": True,
                "safe_verify_requested": True,
                "process_status": "completed",
                "request_error": "",
                "stdout": "",
                "stderr": "",
                "safe_verify_stdout": "",
                "safe_verify_stderr": "",
                "host": "http://127.0.0.1:6000",
                "submission_code": "theorem foo (h : True) : True := by\nexact h\n",
                "error_message": "",
            },
        ):
            should_terminate, response, reward, metadata = await interaction.generate_response(
                instance_id,
                [
                    {"role": "user", "content": "Initial Goedel prompt"},
                    {"role": "assistant", "content": "```lean4\ntheorem foo (h : True) : True := by\n  exact h\n```"},
                ],
                stop_reason="completed",
            )

        assert should_terminate is True
        assert response == "Your proof is correct!"
        assert reward == 1.0
        assert metadata["lean_turn_compile_acc"] == 1.0
        assert metadata["lean_turn_safe_verify_acc"] == 1.0
        assert "next_generation_messages" not in metadata

    @pytest.mark.anyio
    async def test_interrupted_turn_does_not_replace_last_completed_attempt(self):
        interaction = LeanGoedelInteraction({"name": "lean_goedel"})
        instance_id = await interaction.start_interaction(
            instance_id="test_instance",
            ground_truth=_make_ground_truth(),
            raw_prompt=[{"role": "user", "content": "Initial Goedel prompt"}],
            extra_info={},
        )

        with patch(
            "verl.interactions.lean_goedel_interaction.evaluate_lean_attempt",
            return_value={
                "final_ok": False,
                "compile_ok": True,
                "safe_verify_ok": False,
                "safe_verify_requested": True,
                "process_status": "completed",
                "request_error": "",
                "stdout": "",
                "stderr": "",
                "safe_verify_stdout": "SafeVerify failed",
                "safe_verify_stderr": "",
                "host": "http://127.0.0.1:6000",
                "submission_code": "submission-1",
                "error_message": "SafeVerify failed",
            },
        ):
            await interaction.generate_response(
                instance_id,
                [
                    {"role": "user", "content": "Initial Goedel prompt"},
                    {"role": "assistant", "content": "Attempt 1"},
                ],
                stop_reason="completed",
            )

        with patch(
            "verl.interactions.lean_goedel_interaction.evaluate_lean_attempt",
            return_value={
                "final_ok": False,
                "compile_ok": False,
                "safe_verify_ok": False,
                "safe_verify_requested": False,
                "process_status": "error",
                "request_error": "truncated output",
                "stdout": "",
                "stderr": "",
                "safe_verify_stdout": "",
                "safe_verify_stderr": "",
                "host": "http://127.0.0.1:6000",
                "submission_code": "submission-2",
                "error_message": "truncated output",
            },
        ):
            _, _, _, metadata = await interaction.generate_response(
                instance_id,
                [
                    {"role": "user", "content": "Initial Goedel prompt"},
                    {"role": "assistant", "content": "Attempt 2"},
                ],
                stop_reason="aborted",
            )

        assert metadata["last_turn_completed"] is False
        assert metadata["has_last_completed_proof"] is True
        assert metadata["last_completed_proof_correct"] is False
        assert metadata["lean_retry_round"] == 2
