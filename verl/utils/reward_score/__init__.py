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
import math
from collections import Counter

# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated

DEFAULT_ENTROPY_BONUS_COEF = 0.0
INVALID_ANSWER = "[INVALID]"


def _normalize_answer_history(answer_history):
    if answer_history is None:
        return []

    if isinstance(answer_history, str):
        answer_history = [answer_history]

    normalized_history = []
    for answer in answer_history:
        normalized_history.append(answer if answer not in (None, "") else INVALID_ANSWER)
    return normalized_history


def _compute_normalized_answer_entropy(answer_history) -> float:
    answer_history = _normalize_answer_history(answer_history)
    num_turns = len(answer_history)
    if num_turns <= 1:
        return 0.0

    entropy = 0.0
    for count in Counter(answer_history).values():
        probability = count / num_turns
        entropy -= probability * math.log(probability)

    return entropy / math.log(num_turns)


def _apply_math_answer_entropy_bonus(result, extra_info):
    if not extra_info or "answer_history" not in extra_info:
        return result

    if isinstance(result, dict):
        updated_result = dict(result)
        is_correct = bool(updated_result.get("acc", updated_result.get("score", 0.0) > 0))
    else:
        updated_result = {"score": float(result), "acc": float(result) > 0}
        is_correct = updated_result["acc"]

    if is_correct:
        return result

    entropy_bonus_coef = extra_info.get("entropy_bonus_coef", DEFAULT_ENTROPY_BONUS_COEF)
    entropy_bonus_coef = float(entropy_bonus_coef) if entropy_bonus_coef is not None else DEFAULT_ENTROPY_BONUS_COEF
    normalized_entropy = _compute_normalized_answer_entropy(extra_info.get("answer_history"))
    entropy_bonus = entropy_bonus_coef * normalized_entropy if entropy_bonus_coef > 0 else 0.0

    updated_result["score"] = entropy_bonus
    updated_result["acc"] = False
    updated_result["normalized_answer_entropy"] = normalized_entropy
    updated_result["answer_entropy_bonus"] = entropy_bonus
    updated_result["entropy_bonus_coef"] = entropy_bonus_coef
    return updated_result


def _get_last_completed_turn_reward(extra_info):
    if not extra_info or extra_info.get("reward_mode") != "last_completed_turn":
        return None

    has_last_completed_answer = bool(extra_info.get("has_last_completed_answer", False))
    pred = extra_info.get("last_completed_answer")
    acc = bool(extra_info.get("last_completed_answer_correct", False)) if has_last_completed_answer else False
    return {
        "score": 1.0 if acc else 0.0,
        "acc": acc,
        "pred": pred,
        "reward_mode": "last_completed_turn",
    }


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = _get_last_completed_turn_reward(extra_info)
        if res is None:
            res = math_reward.compute_score(solution_str, ground_truth)
            res = _apply_math_answer_entropy_bonus(res, extra_info)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = _get_last_completed_turn_reward(extra_info)
        if res is None:
            res = math_dapo.compute_score(solution_str, ground_truth)
            res = _apply_math_answer_entropy_bonus(res, extra_info)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
