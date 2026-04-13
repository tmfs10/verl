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

import logging
import os
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import math_dapo

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

INVALID_ANSWER = "[INVALID]"
TURN_TOKEN_BUDGET_PROMPT = "You only have {token_budget} tokens to produce an answer for this turn."
REPEAT_UNTIL_STABLE_PROMPT = (
    "Analyze your previous answers, which may or may not be correct, and produce a new refined answer. "
    "Conclude with your final answer in \\boxed{...}."
)
S2R_VERIFY_PROMPT = (
    "Wait, let me recheck my solution.\n\n"
    "Review the previous final answer to the original math problem. Determine whether that answer is correct. "
    "Use a genuine check rather than simply restating the same reasoning. End with exactly one verdict sentence:\n"
    "Therefore, the answer is correct.\n"
    "Therefore, the answer is incorrect."
)
S2R_RETRY_PROMPT = (
    "Let me try again.\n\n"
    "The previous answer may be wrong. Solve the original math problem again and provide a refined answer. "
    "Conclude with your final answer in \\boxed{...}."
)
S2R_CORRECT_PATTERNS = (
    "therefore, the answer is correct.",
    "is correct",
    "is indeed",
    "verified to be correct",
    "verified as correct",
    "appears to be correct",
    "answer checks out",
)
S2R_INCORRECT_PATTERNS = (
    "therefore, the answer is incorrect.",
    "is incorrect",
    "cannot verify",
    "is likely incorrect",
    "is unlikely correct",
    "answer is wrong",
    "incorrect answer",
    "not the correct answer",
    "answer is not correct",
    "solution is incorrect",
    "calculation is wrong",
    "result is incorrect",
)


class MathVerifyInteraction(BaseInteraction):
    """A multi-turn math interaction that verifies each assistant turn.

    This interaction tracks the normalized answer extracted from each assistant turn
    so the final reward function can optionally add an entropy bonus over the answer
    history when the trajectory ends incorrectly.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.strict_box_verify = bool(config.get("strict_box_verify", False))
        self.entropy_bonus_coef = float(config.get("entropy_bonus_coef", 0.0))
        self.interaction_mode = config.get("interaction_mode", "verifier")
        self.turn_context_mode = config.get("turn_context_mode", "full_history")
        self.turn_token_budget_prompt = config.get("turn_token_budget_prompt", TURN_TOKEN_BUDGET_PROMPT)
        self.repeat_until_stable_prompt = config.get("repeat_until_stable_prompt", REPEAT_UNTIL_STABLE_PROMPT)
        self.s2r_verify_prompt = config.get("s2r_verify_prompt", S2R_VERIFY_PROMPT)
        self.s2r_retry_prompt = config.get("s2r_retry_prompt", S2R_RETRY_PROMPT)

        if self.interaction_mode == "s2r" and self.turn_context_mode != "full_history":
            raise ValueError("interaction_mode='s2r' only supports turn_context_mode='full_history'.")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        raw_prompt: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        system_messages, question_text = self._extract_prompt_context(raw_prompt)
        turn_token_budget = self._normalize_turn_token_budget(kwargs.get("per_turn_response_length"))
        self._inject_turn_budget_into_initial_prompt(raw_prompt, turn_token_budget)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "answer_history": [],
            "completed_answer_history": [],
            "answer_correct_history": [],
            "verification_verdict_history": [],
            "verification_match_history": [],
            "last_pred": INVALID_ANSWER,
            "last_acc": False,
            "has_last_completed_answer": False,
            "last_completed_pred": INVALID_ANSWER,
            "last_completed_acc": False,
            "system_messages": system_messages,
            "question_text": question_text,
            "phase": "answer",
            "turn_token_budget": turn_token_budget,
        }
        return instance_id

    @staticmethod
    def _turn_completed(stop_reason: Optional[str]) -> bool:
        # Some backends do not populate stop_reason, so only explicit aborts count as interrupted.
        return stop_reason not in ("aborted", "abort")

    @staticmethod
    def _extract_prompt_context(raw_prompt: Optional[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], str]:
        if not raw_prompt:
            return [], ""

        system_messages = []
        question_parts = []
        non_system_messages = [message for message in raw_prompt if message.get("role") != "system"]
        single_user_message = len(non_system_messages) == 1 and non_system_messages[0].get("role") == "user"

        for message in raw_prompt:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                system_messages.append({"role": "system", "content": deepcopy(content)})
                continue
            if single_user_message and role == "user":
                question_parts.append(str(content))
            else:
                role_name = role.capitalize() if role else "Message"
                question_parts.append(f"{role_name}: {content}")

        return system_messages, "\n\n".join(question_parts)

    @staticmethod
    def _normalize_turn_token_budget(value: Optional[Any]) -> Optional[int]:
        if value is None:
            return None
        try:
            token_budget = int(value)
        except (TypeError, ValueError):
            return None
        return token_budget if token_budget > 0 else None

    def _format_turn_budget_reminder(self, token_budget: Optional[int]) -> Optional[str]:
        if token_budget is None:
            return None
        return self.turn_token_budget_prompt.format(token_budget=token_budget)

    def _append_turn_budget_reminder(self, prompt: str, token_budget: Optional[int]) -> str:
        reminder = self._format_turn_budget_reminder(token_budget)
        if not reminder:
            return prompt
        return f"{prompt}\n\n{reminder}"

    def _inject_turn_budget_into_initial_prompt(
        self, raw_prompt: Optional[list[dict[str, Any]]], token_budget: Optional[int]
    ) -> None:
        reminder = self._format_turn_budget_reminder(token_budget)
        if not raw_prompt or not reminder:
            return

        for message in reversed(raw_prompt):
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            if reminder in content:
                return
            message["content"] = f"{content}\n\n{reminder}" if content else reminder
            return

        raw_prompt.append({"role": "user", "content": reminder})

    def _build_summary_messages(self, instance_id: str) -> list[dict[str, Any]]:
        instance = self._instance_dict[instance_id]
        answer_lines = [f"{idx}. {answer}" for idx, answer in enumerate(instance["answer_history"], start=1)]
        if not answer_lines:
            answer_lines = ["None"]

        prompt = (
            f"Question:\n{instance['question_text']}\n\n"
            f"Past final answers, which may or may not be correct:\n" + "\n".join(answer_lines) + "\n\n"
            "Figure out the right answer. Analyze the past final answers if useful, then end with your final answer "
            "in \\boxed{...}."
        )
        prompt = self._append_turn_budget_reminder(prompt, instance.get("turn_token_budget"))
        return [*deepcopy(instance["system_messages"]), {"role": "user", "content": prompt}]

    @staticmethod
    def _extract_last_assistant_content(messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if item.get("role") == "assistant":
                return item.get("content") or ""
        return ""

    @staticmethod
    def _parse_s2r_verdict(content: str) -> Optional[str]:
        content_lower = content.lower()

        explicit_matches = []
        for verdict in ("correct", "incorrect"):
            needle = f"therefore, the answer is {verdict}."
            position = content_lower.rfind(needle)
            if position >= 0:
                explicit_matches.append((position, verdict))
        if explicit_matches:
            return max(explicit_matches)[1]

        correct_position = max((content_lower.rfind(pattern) for pattern in S2R_CORRECT_PATTERNS), default=-1)
        incorrect_position = max((content_lower.rfind(pattern) for pattern in S2R_INCORRECT_PATTERNS), default=-1)

        if correct_position < 0 and incorrect_position < 0:
            return None
        if incorrect_position > correct_position:
            return "incorrect"
        return "correct"

    def _build_last_completed_reward_data(self, instance_id: str) -> dict[str, Any]:
        instance = self._instance_dict[instance_id]
        return {
            "reward_mode": "last_completed_turn",
            "has_last_completed_answer": instance["has_last_completed_answer"],
            "last_completed_answer": instance["last_completed_pred"] if instance["has_last_completed_answer"] else None,
            "last_completed_answer_correct": instance["last_completed_acc"] if instance["has_last_completed_answer"] else False,
        }

    def _build_s2r_metadata(
        self,
        instance_id: str,
        *,
        processed_phase: str,
        next_phase: Optional[str],
        last_verification_verdict: Optional[str] = None,
        last_verification_matches_answer: Optional[bool] = None,
    ) -> dict[str, Any]:
        instance = self._instance_dict[instance_id]
        return {
            "s2r_processed_phase": processed_phase,
            "s2r_next_phase": next_phase,
            "s2r_answer_correct_history": deepcopy(instance["answer_correct_history"]),
            "s2r_verification_verdict_history": deepcopy(instance["verification_verdict_history"]),
            "s2r_verification_match_history": deepcopy(instance["verification_match_history"]),
            "s2r_last_verification_verdict": last_verification_verdict,
            "s2r_last_verification_matches_answer": last_verification_matches_answer,
        }

    async def _generate_s2r_verification_response(
        self, instance_id: str, stop_reason: Optional[str]
    ) -> tuple[bool, str, float, dict]:
        instance = self._instance_dict[instance_id]
        content = instance["response"]
        verdict = self._parse_s2r_verdict(content)
        verification_completed = self._turn_completed(stop_reason)
        answer_is_correct = bool(instance["has_last_completed_answer"] and instance["last_completed_acc"])
        verification_matches_answer = (
            verification_completed and verdict is not None and ((verdict == "correct") == answer_is_correct)
        )

        instance["verification_verdict_history"].append(verdict or INVALID_ANSWER)
        instance["verification_match_history"].append(bool(verification_matches_answer))

        additional_data = {
            "answer_history": deepcopy(instance["answer_history"]),
            "entropy_bonus_coef": self.entropy_bonus_coef,
            "last_extracted_answer": instance["last_pred"],
            "last_answer_correct": instance["last_acc"],
            "last_assistant_stop_reason": stop_reason,
            "last_turn_completed": verification_completed,
            **self._build_last_completed_reward_data(instance_id),
            **self._build_s2r_metadata(
                instance_id,
                processed_phase="verify",
                next_phase=None if verification_completed and verdict == "correct" else "answer",
                last_verification_verdict=verdict,
                last_verification_matches_answer=verification_matches_answer,
            ),
        }

        if not verification_completed or verdict != "correct":
            instance["phase"] = "answer"
            reward = 0.0 if verdict is None or not verification_completed else (1.0 if verification_matches_answer else -1.0)
            return (
                False,
                self._append_turn_budget_reminder(self.s2r_retry_prompt, instance.get("turn_token_budget")),
                reward,
                additional_data,
            )

        reward = 1.0 if verification_matches_answer else -1.0
        return True, "Stopping after self-verification.", reward, additional_data

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        stop_reason = kwargs.get("stop_reason")
        content = self._extract_last_assistant_content(messages)
        self._instance_dict[instance_id]["response"] = content
        turn_token_budget = self._instance_dict[instance_id].get("turn_token_budget")

        if self.interaction_mode == "s2r" and self._instance_dict[instance_id]["phase"] == "verify":
            return await self._generate_s2r_verification_response(instance_id, stop_reason)

        result = math_dapo.compute_score(
            content,
            self._instance_dict[instance_id]["ground_truth"],
            strict_box_verify=self.strict_box_verify,
        )
        pred = result.get("pred") or INVALID_ANSWER
        acc = bool(result.get("acc", False))
        turn_completed = self._turn_completed(stop_reason)

        self._instance_dict[instance_id]["answer_history"].append(pred)
        self._instance_dict[instance_id]["answer_correct_history"].append(acc)
        self._instance_dict[instance_id]["last_pred"] = pred
        self._instance_dict[instance_id]["last_acc"] = acc
        if turn_completed:
            self._instance_dict[instance_id]["completed_answer_history"].append(pred)
            self._instance_dict[instance_id]["has_last_completed_answer"] = True
            self._instance_dict[instance_id]["last_completed_pred"] = pred
            self._instance_dict[instance_id]["last_completed_acc"] = acc

        additional_data = {
            "answer_history": deepcopy(self._instance_dict[instance_id]["answer_history"]),
            "entropy_bonus_coef": self.entropy_bonus_coef,
            "last_extracted_answer": pred,
            "last_answer_correct": acc,
            "last_assistant_stop_reason": stop_reason,
            "last_turn_completed": turn_completed,
        }

        if (
            self.interaction_mode in {"repeat_until_stable", "s2r"}
            or self.turn_context_mode == "question_with_past_answers"
        ):
            additional_data.update(
                self._build_last_completed_reward_data(instance_id)
            )

        if self.interaction_mode == "s2r":
            if not turn_completed:
                self._instance_dict[instance_id]["phase"] = "answer"
                additional_data.update(
                    self._build_s2r_metadata(
                        instance_id,
                        processed_phase="answer",
                        next_phase="answer",
                    )
                )
                return False, self._append_turn_budget_reminder(self.s2r_retry_prompt, turn_token_budget), 0.0, additional_data

            self._instance_dict[instance_id]["phase"] = "verify"
            additional_data.update(
                self._build_s2r_metadata(
                    instance_id,
                    processed_phase="answer",
                    next_phase="verify",
                )
            )
            reward = 1.0 if acc else -1.0
            return False, self._append_turn_budget_reminder(self.s2r_verify_prompt, turn_token_budget), reward, additional_data

        if self.interaction_mode == "repeat_until_stable":
            completed_answer_history = self._instance_dict[instance_id]["completed_answer_history"]
            repeated_answer = turn_completed and pred in completed_answer_history[:-1]

            reward = 1.0 if turn_completed and acc else 0.0
            if repeated_answer:
                response = "Your latest completed final answer repeats a previous completed answer. Stopping."
                should_terminate_sequence = True
            else:
                response = self._append_turn_budget_reminder(self.repeat_until_stable_prompt, turn_token_budget)
                should_terminate_sequence = False
        else:
            if acc:
                response = "Your response is correct!"
                should_terminate_sequence = True
                reward = 1.0
            else:
                response = self._append_turn_budget_reminder(
                    "Your response is incorrect! You need to reflect on your answer and try again.",
                    turn_token_budget,
                )
                should_terminate_sequence = False
                reward = 0.0

        if self.turn_context_mode == "question_with_past_answers" and not should_terminate_sequence:
            additional_data["reset_generation_prompt"] = True
            additional_data["next_generation_messages"] = self._build_summary_messages(instance_id)

        return should_terminate_sequence, response, reward, additional_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        if self.interaction_mode in {"repeat_until_stable", "s2r"}:
            if not self._instance_dict[instance_id]["has_last_completed_answer"]:
                return 0.0
            return 1.0 if self._instance_dict[instance_id]["last_completed_acc"] else 0.0
        return 1.0 if self._instance_dict[instance_id]["last_acc"] else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
