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
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = [
    "AlgoConfig",
    "BRANCH_REVISION_CRITIQUE_PROMPT",
    "BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT",
    "BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT",
    "BranchRevisionGRPOConfig",
    "FilterGroupsConfig",
    "INTERMEDIATE_MC_CRITIQUE_PROMPT",
    "IntermediateMCValueConfig",
    "KLControlConfig",
    "OPSDAdvantageShapingConfig",
    "OPSDAuditConfig",
    "OPSDConfig",
    "OPSDSteeringConfig",
    "OPSDTokenKLLoggingConfig",
    "RolloutCorrectionConfig",
]


BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT = """\
The attempted solution above is incorrect. Analyze it with the goal of improving
its chance of solving the task within the remaining token budget.

Reason about which parts/steps meaningfully narrowed the available
possibilities or pruned the search space and which parts/steps did not. Explain when the trajectory should have
changed direction and what direction would have been more productive. Then
select an appropriate point at which to revise the trajectory.

Select a short coherent span of the attempted solution that immediately
precedes the new continuation you want to propose. Put this span inside
<prefix>. The text inside <prefix> will be used to locate the revision in the
attempted solution. At the matching location, the original trajectory from the
start of that prefix onward will be replaced by the text inside
<prefix + new continuation>.

Choose <prefix> so that both the point immediately before it and the point
immediately after it fall between coherent steps, not in the middle of a
thought or action. Neither boundary may be inside an open mathematical or
delimited block. In particular, do not place either boundary between an
opening $$ and its matching closing $$, between \\[ and \\], inside a
\\begin{...}...\\end{...} environment, or inside a fenced code block. If the
prefix includes such a block, include the complete block with both its opening
and closing delimiters.

Treat the point immediately after <prefix> as the information boundary for the
new continuation.

The new continuation may use an idea that also appears later in the trajectory
only when the task and the trajectory through the end of the prefix already
provide a reasonable basis for proposing that idea as the next direction.

Consider the trajectory counterfactually with everything after the prefix
hidden. If work performed after the prefix materially changed what was known,
narrowed the available possibilities, or made the idea substantially more
plausible, do not place that idea directly in the new continuation. Instead,
propose the next local step needed to make that progress. We do not want to
short-circuit useful exploration with hindsight.

Do not compress a sequence of later developments into one hindsight statement.
The new continuation should express one locally justified next move or
direction and leave subsequent work for the continuation rollout to perform.
In your analysis, explain why the information available through the end of the
prefix supports the proposed continuation.

Copy the text inside <prefix> character-for-character from one complete line or
short coherent span of the attempted solution. Include enough distinctive
adjacent text that the intended location is unique. Do not intentionally
correct, normalize, paraphrase, or reformat anything inside <prefix>.

Inside <prefix + new continuation>, first copy the entire text from <prefix>
character-for-character. It must be a genuine character prefix of
<prefix + new continuation>. Immediately after that copied prefix, append the
proposed new continuation, including whatever spacing or transition is needed
for it to flow naturally from the prefix. The appended text must be nonempty
and concise.

The appended new continuation must not state the final result in any form,
declare that the task has been completed, or contain a boxed answer, answer
delimiter, or final-answer phrase.

After the free-form analysis, end with exactly one <prefix> tag pair followed
by exactly one <prefix + new continuation> tag pair. Do not use either tag
anywhere else. Permit only whitespace between the two tag pairs, and write
nothing after the closing </prefix + new continuation> tag."""


BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT = """\
The solution above is correct. Analyze it with the goal of improving reasoning
efficiency: reducing the amount of reasoning needed to reach the correct result
without sacrificing correctness.

Reason about which parts/steps meaningfully narrowed the available
possibilities or pruned the search space and which parts/steps did not. Explain when the trajectory should have
changed direction and what direction would have been more productive. Then
select an appropriate point at which to revise the trajectory.

Select a short coherent span of the solution that immediately precedes the new
continuation you want to propose. Put this span inside <prefix>. The text inside
<prefix> will be used to locate the revision in the solution. At the matching
location, the original trajectory from the start of that prefix onward will be
replaced by the text inside <prefix + new continuation>.

Choose <prefix> so that both the point immediately before it and the point
immediately after it fall between coherent steps, not in the middle of a
thought or action. Neither boundary may be inside an open mathematical or
delimited block. In particular, do not place either boundary between an
opening $$ and its matching closing $$, between \\[ and \\], inside a
\\begin{...}...\\end{...} environment, or inside a fenced code block. If the
prefix includes such a block, include the complete block with both its opening
and closing delimiters.

Treat the point immediately after <prefix> as the information boundary for the
new continuation.

The new continuation may use an idea that also appears later in the trajectory
only when the task and the trajectory through the end of the prefix already
provide a reasonable basis for proposing that idea as the next direction.

Consider the trajectory counterfactually with everything after the prefix
hidden. If work performed after the prefix materially changed what was known,
narrowed the available possibilities, or made the idea substantially more
plausible, do not place that idea directly in the new continuation. Instead,
propose the next local step needed to make that progress. We do not want to
short-circuit useful exploration with hindsight.

Do not compress a sequence of later developments into one hindsight statement.
The new continuation should express one locally justified next move or
direction and leave subsequent work for the continuation rollout to perform.
In your analysis, explain why the information available through the end of the
prefix supports the proposed continuation.

Copy the text inside <prefix> character-for-character from one complete line or
short coherent span of the solution. Include enough distinctive adjacent text
that the intended location is unique. Do not intentionally correct, normalize,
paraphrase, or reformat anything inside <prefix>.

Inside <prefix + new continuation>, first copy the entire text from <prefix>
character-for-character. It must be a genuine character prefix of
<prefix + new continuation>. Immediately after that copied prefix, append the
proposed new continuation, including whatever spacing or transition is needed
for it to flow naturally from the prefix. The appended text must be nonempty
and concise.

The appended new continuation must not state the final result in any form,
declare that the task has been completed, or contain a boxed answer, answer
delimiter, or final-answer phrase.

After the free-form analysis, end with exactly one <prefix> tag pair followed
by exactly one <prefix + new continuation> tag pair. Do not use either tag
anywhere else. Permit only whitespace between the two tag pairs, and write
nothing after the closing </prefix + new continuation> tag."""


# Backwards-compatible name for the incorrect-rollout instruction.
BRANCH_REVISION_CRITIQUE_PROMPT = BRANCH_REVISION_INCORRECT_CRITIQUE_PROMPT


@dataclass
class BranchRevisionGRPOConfig(BaseConfig):
    """Synchronous policy-only GRPO over branch critiques and revised rollouts."""

    enable: bool = False
    separate_critique_model: bool = False
    critique_warmup_steps: int = 0
    critique_model_nnodes: int = 1
    critique_model_n_gpus_per_node: int = 8
    critique_grpo_grouping: str = "per_original"
    critique_advantage_mode: str = "grpo"
    critique_invalid_penalty: float = 0.20
    critique_learnability_rejection_penalty: float = 0.05
    critique_advantage_rms_floor: float = 0.10
    critique_advantage_clip: float = 5.0
    critique_prompt_headroom_exponent: float = 1.0
    num_critiques: int = 4
    enable_positive_compression: bool = False
    num_positive_critiques: int = 4
    positive_compression_target: float = 0.25
    learnability_logprob_statistic: str = "mean"
    learnability_threshold_mode: str = "stddev"
    max_seed_window_stddevs: float = 15.0
    min_seed_window_percentile: float = 0.20
    full_credit_seed_window_percentile: float = 0.50
    critique_max_response_length: Optional[int] = 8192
    branch_max_tokens: int = 128
    new_continuation_max_tokens: int = 256
    min_continuation_tokens: int = 128
    reward_tolerance: float = 1e-6
    critique_prompt: str = BRANCH_REVISION_CRITIQUE_PROMPT
    positive_critique_prompt: str = BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT
    audit_output_dir: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.separate_critique_model, bool):
            raise ValueError("algorithm.branch_revision_grpo.separate_critique_model must be boolean")
        if (
            not isinstance(self.critique_warmup_steps, int)
            or isinstance(self.critique_warmup_steps, bool)
            or self.critique_warmup_steps < 0
        ):
            raise ValueError("algorithm.branch_revision_grpo.critique_warmup_steps must be a non-negative integer")
        critique_resources = {
            "critique_model_nnodes": self.critique_model_nnodes,
            "critique_model_n_gpus_per_node": self.critique_model_n_gpus_per_node,
        }
        for name, value in critique_resources.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"algorithm.branch_revision_grpo.{name} must be a positive integer")
        if self.critique_grpo_grouping not in {"per_original", "batch"}:
            raise ValueError("algorithm.branch_revision_grpo.critique_grpo_grouping must be per_original or batch")
        if self.critique_advantage_mode not in {"grpo", "pass_at_1"}:
            raise ValueError("algorithm.branch_revision_grpo.critique_advantage_mode must be grpo or pass_at_1")
        nonnegative_floats = {
            "critique_invalid_penalty": self.critique_invalid_penalty,
            "critique_learnability_rejection_penalty": self.critique_learnability_rejection_penalty,
            "critique_prompt_headroom_exponent": self.critique_prompt_headroom_exponent,
        }
        for name, value in nonnegative_floats.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"algorithm.branch_revision_grpo.{name} must be finite and nonnegative")
        positive_floats = {
            "critique_advantage_rms_floor": self.critique_advantage_rms_floor,
            "critique_advantage_clip": self.critique_advantage_clip,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"algorithm.branch_revision_grpo.{name} must be finite and positive")
        positive_ints = {
            "num_critiques": self.num_critiques,
            "num_positive_critiques": self.num_positive_critiques,
            "branch_max_tokens": self.branch_max_tokens,
            "new_continuation_max_tokens": self.new_continuation_max_tokens,
            "min_continuation_tokens": self.min_continuation_tokens,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"algorithm.branch_revision_grpo.{name} must be a positive integer")
        if self.num_critiques < 2:
            raise ValueError("algorithm.branch_revision_grpo.num_critiques must be at least 2 for GRPO")
        if self.num_positive_critiques < 2:
            raise ValueError("algorithm.branch_revision_grpo.num_positive_critiques must be at least 2 for GRPO")
        if not isinstance(self.enable_positive_compression, bool):
            raise ValueError("algorithm.branch_revision_grpo.enable_positive_compression must be boolean")
        if self.critique_advantage_mode == "pass_at_1" and self.enable_positive_compression:
            raise ValueError(
                "algorithm.branch_revision_grpo.critique_advantage_mode=pass_at_1 currently supports recovery only"
            )
        if self.critique_max_response_length is not None and (
            not isinstance(self.critique_max_response_length, int)
            or isinstance(self.critique_max_response_length, bool)
            or self.critique_max_response_length <= 0
        ):
            raise ValueError("algorithm.branch_revision_grpo.critique_max_response_length must be positive or null")
        if not math.isfinite(self.reward_tolerance) or self.reward_tolerance <= 0.0:
            raise ValueError("algorithm.branch_revision_grpo.reward_tolerance must be finite and positive")
        if not math.isfinite(self.positive_compression_target) or not 0.0 < self.positive_compression_target <= 1.0:
            raise ValueError("algorithm.branch_revision_grpo.positive_compression_target must be in (0, 1]")
        if self.learnability_logprob_statistic not in {"mean", "min"}:
            raise ValueError("algorithm.branch_revision_grpo.learnability_logprob_statistic must be mean or min")
        if self.learnability_threshold_mode not in {"stddev", "percentile"}:
            raise ValueError("algorithm.branch_revision_grpo.learnability_threshold_mode must be stddev or percentile")
        if not math.isfinite(self.max_seed_window_stddevs) or self.max_seed_window_stddevs < 0.0:
            raise ValueError("algorithm.branch_revision_grpo.max_seed_window_stddevs must be finite and nonnegative")
        if not (
            math.isfinite(self.min_seed_window_percentile)
            and math.isfinite(self.full_credit_seed_window_percentile)
            and 0.0 <= self.min_seed_window_percentile < self.full_credit_seed_window_percentile <= 1.0
        ):
            raise ValueError("branch-revision seed percentiles must satisfy 0 <= min < full_credit <= 1")
        if self.critique_prompt != BRANCH_REVISION_CRITIQUE_PROMPT:
            raise ValueError("branch_revision_grpo.critique_prompt must exactly match the branch-revision instruction")
        if self.positive_critique_prompt != BRANCH_REVISION_CORRECT_CRITIQUE_PROMPT:
            raise ValueError(
                "branch_revision_grpo.positive_critique_prompt must exactly match the positive-compression instruction"
            )


INTERMEDIATE_MC_CRITIQUE_PROMPT = """You have been given above a question, the thought process, and the resulting
solution. The solution may or may not be correct. Analyze the thought process
and solution and at the end output your judgement on (a) whether the solution
is correct or not (b) what parts of the thought process were correct or led to
correct directions (c) what part of the thought process were dead-ends or
incorrect or didn't later enable moving in the correct direction."""


@dataclass
class IntermediateMCValueConfig(BaseConfig):
    """Synchronous self-critique and continuation supervision for PPO."""

    enable: bool = False
    critic_head: str = "scalar"
    mark_selector: str = "random"
    num_critiques: int = 4
    continuations_per_mark: int = 1
    max_marks: Optional[int] = None
    critique_max_response_length: Optional[int] = None
    mark_start_fraction: float = 0.05
    mark_end_fraction: float = 0.90
    min_mark_gap: int = 32
    ema_alpha: float = 0.1
    ema_baseline_token: int = 32
    ema_floor: float = 1e-4
    ema_ratio_up: float = 2.0
    ema_ratio_down: float = 0.5
    variance_scope: str = "rollout"
    variance_random_probability: float = 0.05
    selection_seed: int = 0
    max_reward: float = 1.0
    scalar_loss: str = "mse"
    beta_target_epsilon: float = 1e-4
    critique_normalization_epsilon: float = 1e-8
    critique_prompt: str = INTERMEDIATE_MC_CRITIQUE_PROMPT
    audit_output_dir: Optional[str] = None

    @property
    def num_critic_labels(self) -> int:
        return 1 if self.critic_head == "scalar" else 2

    @property
    def num_critic_streams(self) -> int:
        return max(1, self.num_critiques)

    @property
    def resolved_max_marks(self) -> int:
        if self.max_marks is not None:
            return self.max_marks
        return 4 if self.mark_selector == "ema" else 1

    def __post_init__(self):
        if self.critic_head not in {"scalar", "beta"}:
            raise ValueError("algorithm.intermediate_mc_value.critic_head must be scalar or beta")
        if self.mark_selector not in {"random", "ema", "variance"}:
            raise ValueError("algorithm.intermediate_mc_value.mark_selector must be random, ema, or variance")
        if self.mark_selector == "variance" and self.critic_head != "beta":
            raise ValueError("algorithm.intermediate_mc_value.mark_selector=variance requires critic_head=beta")
        if not isinstance(self.num_critiques, int) or isinstance(self.num_critiques, bool) or self.num_critiques < 0:
            raise ValueError("algorithm.intermediate_mc_value.num_critiques must be a non-negative integer")
        positive_ints = {
            "continuations_per_mark": self.continuations_per_mark,
            "min_mark_gap": self.min_mark_gap,
            "ema_baseline_token": self.ema_baseline_token,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"algorithm.intermediate_mc_value.{name} must be a positive integer")
        if self.max_marks is not None and (
            not isinstance(self.max_marks, int) or isinstance(self.max_marks, bool) or self.max_marks < 0
        ):
            raise ValueError("algorithm.intermediate_mc_value.max_marks must be null or a non-negative integer")
        if self.num_critiques == 1:
            warnings.warn(
                "intermediate_mc_value.num_critiques=1 makes the normalized critique advantage exactly zero",
                stacklevel=2,
            )
        if self.critique_max_response_length is not None:
            if (
                not isinstance(self.critique_max_response_length, int)
                or isinstance(self.critique_max_response_length, bool)
                or self.critique_max_response_length <= 0
            ):
                raise ValueError(
                    "algorithm.intermediate_mc_value.critique_max_response_length must be positive or null"
                )
        if not isinstance(self.selection_seed, int) or isinstance(self.selection_seed, bool):
            raise ValueError("algorithm.intermediate_mc_value.selection_seed must be an integer")
        if not 0.0 <= self.mark_start_fraction <= self.mark_end_fraction <= 1.0:
            raise ValueError("mark fractions must satisfy 0 <= start <= end <= 1")
        if not math.isfinite(self.ema_alpha) or not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be finite and in (0, 1]")
        if not math.isfinite(self.ema_floor) or self.ema_floor <= 0.0:
            raise ValueError("ema_floor must be finite and positive")
        if not math.isfinite(self.ema_ratio_up) or self.ema_ratio_up <= 1.0:
            raise ValueError("ema_ratio_up must be finite and greater than 1")
        if not math.isfinite(self.ema_ratio_down) or not 0.0 < self.ema_ratio_down < 1.0:
            raise ValueError("ema_ratio_down must be finite and in (0, 1)")
        if self.variance_scope not in {"rollout", "prompt", "batch"}:
            raise ValueError("variance_scope must be rollout, prompt, or batch")
        if not 0.0 <= self.variance_random_probability <= 1.0:
            raise ValueError("variance_random_probability must be in [0, 1]")
        if isinstance(self.max_reward, bool) or not math.isfinite(self.max_reward) or self.max_reward <= 0.0:
            raise ValueError("max_reward must be finite and positive")
        if self.scalar_loss not in {"mse", "bce"}:
            raise ValueError("scalar_loss must be mse or bce")
        if not 0.0 < self.beta_target_epsilon < 0.5:
            raise ValueError("beta_target_epsilon must be in (0, 0.5)")
        if not math.isfinite(self.critique_normalization_epsilon) or self.critique_normalization_epsilon <= 0:
            raise ValueError("critique_normalization_epsilon must be finite and positive")
        if self.critique_prompt != INTERMEDIATE_MC_CRITIQUE_PROMPT:
            raise ValueError("critique_prompt must exactly match the intermediate MC critique instruction")


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Configuration for Rollout Correction (addresses off-policy issues in RL training).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Rollout Correction handles off-policiness from multiple sources:
    1. Policy mismatch: Rollout policy (e.g., vLLM BF16) vs Training policy (e.g., FSDP FP32)
    2. Model update staleness: Rollout data collected from older policy checkpoints
    3. General off-policy scenarios: Any distribution shift between data collection and training

    For more details, see:
    "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
    https://richardli.xyz/rl-collapse

    This typed config replaces the old dict-based approach and provides:
    - Type safety and validation
    - Clear documentation of all parameters
    - Named factory methods for common presets (TIS, MIS, etc.)
    - Sensible defaults

    Args:
        rollout_is (Optional[str]): IS weight aggregation level.
            - None: No IS weights (metrics only)
            - "token": Per-token IS weights (low variance, biased)
            - "sequence": Per-sequence IS weights (unbiased, high variance)
            Default: "sequence"

        rollout_is_threshold (float): Upper threshold for IS weight truncation/rejection.
            Typical range: 1.5-5.0 for token level, 2.0-10.0 for sequence level.
            Default: 2.0

        rollout_is_batch_normalize (bool): Apply batch normalization to IS weights.
            - True: Normalize IS weights to have mean=1.0 within each batch
            - False: Use raw (truncated) IS weights (standard)
            - Reduces variance by ensuring average weight is 1.0 per batch
            - Only affects IS weight values, not rejection sampling
            Default: False (no batch normalization)

        rollout_rs (Optional[str]): Rejection sampling aggregation modes.
            Accepts a comma-delimited list (duplicates removed) of canonical options implemented in
            ``rollout_corr_helper``:
            - "token_k1": Token-level rejection with ``-log r`` (ratio thresholds supplied via
              ``rollout_rs_threshold`` as ``lower_upper``)
            - "token_k2": Token-level rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "token_k3": Token-level rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            - "seq_sum_k1": Sequence sum of ``-log r`` (ratio bounds)
            - "seq_sum_k2": Sequence sum of rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "seq_sum_k3": Sequence sum of rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            - "seq_mean_k1": Sequence mean of ``-log r`` (ratio bounds)
            - "seq_mean_k2": Sequence mean of rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "seq_mean_k3": Sequence mean of rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            - "seq_max_k2": Sequence max of rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "seq_max_k3": Sequence max of rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            names automatically. Default: None

        rollout_rs_threshold (Optional[Union[str, float]]): Threshold specification for rejection sampling.
            Provide one value per option (single entry is broadcast when multiple options are supplied).
            Ratio-based modes (``*k1``) expect ``lower_upper`` strings; supplying a single float implies
            only the upper ratio bound, with the lower bound inferred as its reciprocal. Divergence modes
            (k2/k3) expect positive upper bounds (float or string). Default: None

        bypass_mode (bool): Operating mode - bypass or decoupled.
            - True: Bypass mode - reuse rollout_log_prob as old_log_prob (2 policies)
              Uses compute_policy_loss_bypass_mode() with loss_type selection
            - False: Decoupled mode - compute old_log_prob separately (3 policies)
              Uses standard PPO loss with IS weight correction
            Default: False (decoupled mode)

        loss_type (str): Loss function type in bypass mode (bypass_mode=True).
            - "reinforce": REINFORCE-style policy gradient with explicit IS weights
              L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
            - "ppo_clip": PPO clipped objective (IS handled by ratio, no explicit weights)
              L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Default: "ppo_clip"

    Example:
        # Create with defaults
        config = RolloutCorrectionConfig()

        # Decoupled PPO mode presets (3 policies: π_rollout, π_old, π_θ)
        # IS weights correct for gap between π_old and π_rollout
        config = RolloutCorrectionConfig.decoupled_token_is()  # Token-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is()    # Seq-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is_rs() # Seq-MIS
        config = RolloutCorrectionConfig.decoupled_geo_rs()    # Geo-RS (ratio mode)

        # Bypass mode presets (2 policies: π_rollout = π_old, π_θ)
        # loss_type controls the loss function
        # PPO-clip presets (ratio handles IS, so no separate IS weights needed):
        config = RolloutCorrectionConfig.bypass_ppo_clip()              # PPO-clip only
        config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()       # PPO-clip + Geo-RS
        config = RolloutCorrectionConfig.bypass_ppo_clip_k3_rs()        # PPO-clip + K3-RS
        # REINFORCE presets (explicit IS weights):
        config = RolloutCorrectionConfig.bypass_pg_is()                 # REINFORCE + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs()             # REINFORCE + Geo-RS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_seq_tis()     # REINFORCE + Geo-RS + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_token_tis()   # REINFORCE + Geo-RS + Token-TIS

        # Decoupled Geometric ratio presets (length-normalized IS ratio)
        config = RolloutCorrectionConfig.decoupled_geo_rs_seq_tis()           # Decoupled Geo-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_geo_rs_token_tis()         # Decoupled Geo-RS + Token-TIS

        # Decoupled K3 KL Estimator presets (more stable for small KL values)
        config = RolloutCorrectionConfig.decoupled_k3_rs()                    # Decoupled K3-RS
        config = RolloutCorrectionConfig.decoupled_k3_rs_seq_tis()            # Decoupled K3-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_k3_rs_token_tis()          # Decoupled K3-RS + Token-TIS

    Reference:
        Liu, Li, Fu, Wang, Liu, Shen (2025)
        "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
        https://richardli.xyz/rl-collapse
    """

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_is_batch_normalize: bool = False
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[str | float] = None
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Token-level Importance Sampling.

        IS weight correction at token level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with token-level IS
        """
        return cls(rollout_is="token", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level Importance Sampling.

        IS weight correction at sequence level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence-level IS
        """
        return cls(rollout_is="sequence", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is_rs(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.5_2.0",
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level IS + Rejection Sampling.

        Sequence-level IS with sequence-level rejection sampling in decoupled mode.
        Rejects entire sequences based on sequence-level IS weight.

        Args:
            is_threshold (float): Upper threshold for IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Upper threshold for rejection sampling. Default: 0.5_2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence IS + RS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_sum_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Geometric Mean Rejection Sampling (ratio-based).

        Uses geometric mean IS ratio E[log(r)] for rejection sampling at sequence level.
        This is a ratio-based mode (ideal = 0.0) with [lower, upper] threshold bounds.
        Length-normalized but still uses IS ratio semantics.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_ppo_clip(cls) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss.

        PPO clipped objective in bypass mode. The PPO ratio = π_θ/π_rollout
        already handles IS correction, so no explicit IS weights are applied.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip
        """
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and Geometric Mean RS (ratio-based).

        PPO clipped objective in bypass mode with geometric mean IS ratio RS.
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and K3 Rejection Sampling.

        PPO clipped objective in bypass mode with K3 KL estimator RS to mask outliers.
        K3 is more stable than K1 for small KL values.
        The PPO ratio = π_θ/π_rollout already handles IS correction.

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + K3-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and IS Correction.

        Uses REINFORCE loss with explicit IS correction in bypass mode.
        No PPO clipping.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + IS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and Geometric Mean RS (ratio-based).

        REINFORCE with geometric mean IS ratio rejection sampling in bypass mode.
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Sequence-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Clipped Sequence Weight (debiasing). Uses E[log(r)] (ideal = 0.0).

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Token-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Token-level IS weights. Uses E[log(r)] (ideal = 0.0).

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_pg_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Sequence-level IS.

        Combines geometric mean IS ratio rejection with sequence-level IS
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Token-level IS.

        Combines geometric mean IS ratio rejection with token-level IS weights
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 KL Estimator Rejection Sampling.

        Uses K3 KL estimator at sequence level for rejection sampling.
        K3 = E[r - log(r) - 1] where r = π_train/π_rollout.
        More stable than geometric mean for small KL values.

        K3 >= 0 always (equals 0 when policies match exactly).

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
                Typical range: 0.001-0.1

        Returns:
            RolloutCorrectionConfig configured for K3 RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Sequence-level Truncated IS.

        Combines K3 KL estimator rejection with sequence-level IS weights.
        K3 provides more stable outlier detection than geometric mean.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Token-level Truncated IS.

        Combines K3 KL estimator rejection with token-level IS weights.
        K3 provides more stable outlier detection than geometric mean.
        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        """Disabled - Metrics Only Mode.

        Computes and logs off-policy metrics without applying correction.

        Returns:
            RolloutCorrectionConfig with all correction disabled
        """
        return cls(rollout_is=None, rollout_rs=None)


@dataclass
class OPSDAuditConfig(BaseConfig):
    """Opt-in, fail-fast numerical audit for OPSD updates."""

    enabled: bool = False
    output_dir: Optional[str] = None
    global_steps: list[int] = field(default_factory=lambda: [1, 2, 3])
    fail_fast: bool = True
    full_batch_ledger: bool = True
    reference_forward: bool = True
    reference_samples_per_rank: int = 1
    # Strict parity for the production remove-padding path against a compact
    # no-PAD oracle and synthetic extra-PAD variants.
    forward_max_abs_error: float = 5e-2
    forward_mean_abs_error: float = 5e-3
    # The Transformers dense padded FlashAttention path uses a different BF16
    # kernel shape. Keep it as an independently logged cross-kernel reference
    # with explicit, wider bounds; it is not the production PAD invariant.
    dense_forward_max_abs_error: float = 7.5e-1
    dense_forward_mean_abs_error: float = 3e-2
    dense_forward_fail_fast: bool = False
    # vLLM rollout and FSDP learner use different BF16 kernels. Bound and log
    # their sampled-token parity separately from same-path PAD invariance.
    behavior_forward_max_abs_error: float = 7.5e-1
    behavior_forward_mean_abs_error: float = 3e-2
    scalar_atol: float = 1e-5
    scalar_rtol: float = 1e-4

    def __post_init__(self):
        if any(step <= 0 for step in self.global_steps):
            raise ValueError(f"opsd.audit.global_steps must be positive, got {self.global_steps}")
        if self.reference_samples_per_rank <= 0:
            raise ValueError("opsd.audit.reference_samples_per_rank must be positive")
        for name in (
            "forward_max_abs_error",
            "forward_mean_abs_error",
            "dense_forward_max_abs_error",
            "dense_forward_mean_abs_error",
            "behavior_forward_max_abs_error",
            "behavior_forward_mean_abs_error",
            "scalar_atol",
            "scalar_rtol",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"opsd.audit.{name} must be non-negative")


@dataclass
class OPSDTokenKLLoggingConfig(BaseConfig):
    """Bounded production ledger for sampled token-level reverse KL.

    The worker logs only positions selected by the explicit response mask. It
    retains a small number of samples per rank and the largest-absolute sampled
    reverse-KL positions per retained response so an 8K production run does not
    emit an unbounded token dump.
    """

    enabled: bool = False
    output_dir: Optional[str] = None
    start_step: int = 1
    end_step: Optional[int] = None
    interval_steps: int = 1
    max_samples_per_rank: int = 1
    max_tokens_per_sample: int = 128

    def __post_init__(self):
        if self.start_step <= 0:
            raise ValueError(f"opsd.token_kl_logging.start_step must be positive, got {self.start_step}")
        if self.end_step is not None and self.end_step < self.start_step:
            raise ValueError(
                f"opsd.token_kl_logging.end_step must be at least start_step, got {self.end_step} < {self.start_step}"
            )
        if self.interval_steps <= 0:
            raise ValueError(f"opsd.token_kl_logging.interval_steps must be positive, got {self.interval_steps}")
        if self.max_samples_per_rank <= 0:
            raise ValueError(
                f"opsd.token_kl_logging.max_samples_per_rank must be positive, got {self.max_samples_per_rank}"
            )
        if self.max_tokens_per_sample <= 0:
            raise ValueError(
                f"opsd.token_kl_logging.max_tokens_per_sample must be positive, got {self.max_tokens_per_sample}"
            )


@dataclass
class OPSDGapDiagnosticsConfig(BaseConfig):
    """Sequence-level correctness separation for a steered OPSD teacher."""

    enabled: bool = False
    output_dir: Optional[str] = None
    interval_steps: int = 1
    crossfit_enabled: bool = True
    fold_seed: int = 1234
    full_sequence_ledger: bool = True

    def __post_init__(self):
        if self.interval_steps <= 0:
            raise ValueError(
                f"opsd.steering.gap_diagnostics.interval_steps must be positive, got {self.interval_steps}"
            )
        if self.fold_seed < 0:
            raise ValueError(f"opsd.steering.gap_diagnostics.fold_seed must be non-negative, got {self.fold_seed}")


@dataclass
class OPSDSteeringConfig(BaseConfig):
    """Configuration for same-prompt steering-vector teacher conditioning."""

    strict_contract: bool = False
    layer_fractions: str = ""
    expected_model_path: Optional[str] = None
    actor_model_path: Optional[str] = None
    expected_total_layers: Optional[int] = None
    expected_layer_indices: list[int] = field(default_factory=list)
    source_mode: str = "caa"
    # ``same_prompt`` constructs one CAA direction per prompt UID.  The
    # ``global_batch`` option constructs one direction from every correct and
    # incorrect rollout in the complete optimizer batch across all DP ranks.
    caa_scope: str = "same_prompt"
    correct_rollout_aggregation: str = "all"
    activation_aggregation: str = "per_rollout"
    # Policy-gradient steering differentiates a verifier-weighted,
    # sequence-balanced current-actor objective with respect to a shared
    # additive activation probe.  Only this explicit formulation is supported
    # until alternative objectives receive their own audits.
    gradient_objective: str = "grpo_advantage"
    gradient_aggregation: str = "per_rollout"
    scale: float = 1.0
    normalize: Optional[str] = None
    apply_positions: str = "all_nonpad"
    detach_vectors: bool = True
    gap_diagnostics: OPSDGapDiagnosticsConfig = field(default_factory=OPSDGapDiagnosticsConfig)

    def __post_init__(self):
        if isinstance(self.gap_diagnostics, dict):
            object.__setattr__(
                self,
                "gap_diagnostics",
                OPSDGapDiagnosticsConfig(**self.gap_diagnostics),
            )
        if self.expected_total_layers is not None and self.expected_total_layers <= 0:
            raise ValueError(
                f"opsd.steering.expected_total_layers must be positive when set, got {self.expected_total_layers}"
            )
        if any(index < 0 for index in self.expected_layer_indices):
            raise ValueError("opsd.steering.expected_layer_indices must contain only non-negative indexes")
        if len(set(self.expected_layer_indices)) != len(self.expected_layer_indices):
            raise ValueError("opsd.steering.expected_layer_indices must not contain duplicates")
        if self.source_mode not in {"caa", "positive", "policy_gradient"}:
            raise ValueError(
                f"Invalid opsd.steering.source_mode: {self.source_mode}. Must be caa, positive, or policy_gradient"
            )
        if self.caa_scope not in {"same_prompt", "global_batch"}:
            raise ValueError(f"Invalid opsd.steering.caa_scope: {self.caa_scope}. Must be same_prompt or global_batch")
        if self.source_mode not in {"caa", "policy_gradient"} and self.caa_scope != "same_prompt":
            raise ValueError(
                "opsd.steering.caa_scope=global_batch is defined only for source_mode=caa or policy_gradient"
            )
        if self.correct_rollout_aggregation not in {"first", "all"}:
            raise ValueError(
                "Invalid opsd.steering.correct_rollout_aggregation: "
                f"{self.correct_rollout_aggregation}. Must be first or all"
            )
        if self.activation_aggregation not in {"per_rollout", "pooled_tokens"}:
            raise ValueError(
                "Invalid opsd.steering.activation_aggregation: "
                f"{self.activation_aggregation}. Must be per_rollout or pooled_tokens"
            )
        if self.gradient_objective != "grpo_advantage":
            raise ValueError(
                f"Invalid opsd.steering.gradient_objective: {self.gradient_objective}. Must be grpo_advantage"
            )
        if self.gradient_aggregation != "per_rollout":
            raise ValueError(
                f"Invalid opsd.steering.gradient_aggregation: {self.gradient_aggregation}. Must be per_rollout"
            )
        if not math.isfinite(self.scale):
            raise ValueError(f"opsd.steering.scale must be finite, got {self.scale}")
        if self.normalize not in {None, "unit_norm", "rms"}:
            raise ValueError(f"Invalid opsd.steering.normalize: {self.normalize}. Must be unit_norm, rms, or null")
        if self.apply_positions not in {"all_nonpad", "response_only"}:
            raise ValueError(
                f"Invalid opsd.steering.apply_positions: {self.apply_positions}. Must be all_nonpad or response_only"
            )
        if not self.detach_vectors:
            raise ValueError(
                "opsd.steering.detach_vectors must be true; steering sources are a stop-gradient teacher signal"
            )


@dataclass
class OPSDAdvantageShapingConfig(BaseConfig):
    """Redistribute verifier-derived response advantage using detached teacher evidence.

    This is deliberately an RLVR objective, not a distillation objective.  The
    teacher-conditioned branch supplies only a stopped per-token score.  The
    ordinary student/original-prompt PPO loss remains the only actor objective.
    """

    enable: bool = False
    score_source: str = "teacher_minus_student_logprob"
    scale: float = 1.0
    normalize: Optional[str] = None
    clip_z: Optional[float] = None
    use_distill_mask: bool = True
    # Reweighting should preserve the verifier-determined update direction by
    # default.  Experiments may opt into sign flips explicitly.
    allow_token_sign_flip: bool = False
    # Optional symmetric cap on the token redistribution, expressed as a
    # fraction of the rollout's absolute sequence-level GRPO advantage. A
    # value of 1.0 enforces |delta A_t| <= |A|. ``None`` leaves only the
    # no-sign-flip bound active.
    max_delta_fraction: Optional[float] = None
    # ``None`` means every actual response token selected by the shaping mask.
    # This is a response-axis cap and can never select prompt tokens.
    max_response_tokens: Optional[int] = None
    student_rlvr_backward_scale: float = 1.0

    def __post_init__(self):
        if self.score_source != "teacher_minus_student_logprob":
            raise ValueError(
                "Invalid opsd.advantage_shaping.score_source: "
                f"{self.score_source}. Only teacher_minus_student_logprob is supported."
            )
        valid_normalize = {None, "none", "std", "mean_abs", "range"}
        if self.normalize not in valid_normalize:
            raise ValueError(
                "Invalid opsd.advantage_shaping.normalize: "
                f"{self.normalize}. Must be one of ['mean_abs', 'none', 'range', 'std'] or None"
            )
        if not math.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError(f"opsd.advantage_shaping.scale must be finite and non-negative, got {self.scale}")
        if self.clip_z is not None and (not math.isfinite(self.clip_z) or self.clip_z <= 0.0):
            raise ValueError(f"opsd.advantage_shaping.clip_z must be finite and positive when set, got {self.clip_z}")
        if self.max_delta_fraction is not None and (
            not math.isfinite(self.max_delta_fraction) or self.max_delta_fraction <= 0.0
        ):
            raise ValueError(
                "opsd.advantage_shaping.max_delta_fraction must be finite and positive when set, "
                f"got {self.max_delta_fraction}"
            )
        if self.max_response_tokens is not None and self.max_response_tokens <= 0:
            raise ValueError(
                f"opsd.advantage_shaping.max_response_tokens must be positive when set, got {self.max_response_tokens}"
            )
        if not math.isfinite(self.student_rlvr_backward_scale) or self.student_rlvr_backward_scale < 0.0:
            raise ValueError(
                "opsd.advantage_shaping.student_rlvr_backward_scale must be finite and non-negative, "
                f"got {self.student_rlvr_backward_scale}"
            )


@dataclass
class OPSDConfig(BaseConfig):
    """Configuration for On-Policy Self Distillation (OPSD).

    OPSD trains on model-generated responses, but compares the student policy
    (conditioned on the original prompt) against a teacher branch that is
    conditioned either on the original prompt plus the ground-truth answer, or
    on the original prompt plus a successful same-batch rollout in SDPO style.

    In ``opsd_rlvr`` mode, the teacher branch additionally receives the usual
    RLVR objective using verifier rewards, with optional off-policy correction
    against the student/original-prompt behavior policy.
    """

    enable: bool = False
    mode: str = "opsd"
    # ``direct_reverse_kl`` uses the baseline-reduced sampled score-function
    # surrogate. ``negative_kl_advantage`` freezes teacher-minus-current-actor
    # evidence and sends it through the actor's clipped PPO loss.
    # ``grpo_advantage_reweighting`` retains the verifier-derived GRPO scalar
    # and redistributes its token mass with centered teacher evidence.
    # ``None`` preserves compatibility with configurations that predate the
    # explicit selector: legacy advantage_shaping.enable=True resolves to GRPO
    # reweighting, otherwise direct reverse KL is the default.
    actor_objective: Optional[str] = None
    teacher_source: str = "ground_truth"
    teacher_model: str = "actor"
    teacher_ema_rate: float = 0.05
    teacher_ema_cpu_offload: bool = True
    ground_truth_field: str = "ground_truth_answer"
    teacher_prompt_style: str = "append_instruction"
    teacher_apply_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    sdpo_conditioning_mode: str = "prompt_append"
    steering: OPSDSteeringConfig = field(default_factory=OPSDSteeringConfig)
    teacher_prefix: str = "\n\nBelow is the ground truth answer:\n"
    teacher_suffix: str = "\n\nNow solve the problem"
    sdpo_success_prefix: str = "\n\nBelow is a successful previous attempt for this question:\n"
    sdpo_success_suffix: str = (
        "\n\nUse the successful previous attempt as implicit feedback and solve the problem again."
    )
    sdpo_distill_only_failed: bool = True
    sdpo_exclude_self_success: bool = True
    distill_loss: str = "sampled_reverse_kl"
    # Compatibility sentinels for pre-stabilization configs. These controls are
    # deliberately rejected when set: the former sparse union-top-k objective
    # represented opposing-only token mass both explicitly and in the tail
    # bucket, so it was not a valid shared probability partition. Full JSD is
    # disabled by the current reverse-KL-only policy, not by a known math bug.
    topk: Optional[int] = None
    distill_beta: Optional[float] = None
    distill_token_clip: Optional[float] = None
    distill_token_clip_tail: Optional[bool] = None
    distill_max_response_tokens: Optional[int] = None
    mix_weight: float = 0.5
    balance_mode: str = "none"
    balance_param_subset: str = "lm_head"
    # Only the explicit two-backward implementation is currently supported.
    # Keep this compatibility field so stale configs fail with a targeted
    # message instead of silently changing autograd semantics.
    separate_backward: bool = True
    distill_backward_scale: float = 1.0
    rlvr_backward_scale: float = 1.0
    rlvr_warmup_steps: int = 0
    # Optional supervised update for a separately optimized teacher. The
    # teacher sees the ground-truth-conditioned prompt and is teacher-forced on
    # verifier-successful student rollouts. This is an independent coefficient
    # in addition to the distillation/RLVR mixture.
    teacher_sft_weight: float = 0.0
    teacher_sft_target_scope: str = "thinking_and_answer"
    teacher_sft_success_field: str = "acc"
    teacher_sft_success_threshold: float = 0.5
    teacher_sft_think_end_tag: str = "</think>"
    offpolicy_is_mode: str = "sequence"
    offpolicy_is_clip: float = 2.0
    behavior_logprob_source: str = "rollout"
    max_prompt_length: Optional[int] = None
    truncation: Optional[str] = None
    text_only: bool = True
    log_diagnostics: bool = False
    debug_print_interval: int = 0
    debug_num_tokens: int = 8
    advantage_shaping: OPSDAdvantageShapingConfig = field(default_factory=OPSDAdvantageShapingConfig)
    audit: OPSDAuditConfig = field(default_factory=OPSDAuditConfig)
    token_kl_logging: OPSDTokenKLLoggingConfig = field(default_factory=OPSDTokenKLLoggingConfig)

    def __post_init__(self):
        if isinstance(self.steering, dict):
            object.__setattr__(self, "steering", OPSDSteeringConfig(**self.steering))
        if isinstance(self.advantage_shaping, dict):
            object.__setattr__(
                self,
                "advantage_shaping",
                OPSDAdvantageShapingConfig(**self.advantage_shaping),
            )
        if isinstance(self.audit, dict):
            object.__setattr__(self, "audit", OPSDAuditConfig(**self.audit))
        if isinstance(self.token_kl_logging, dict):
            object.__setattr__(
                self,
                "token_kl_logging",
                OPSDTokenKLLoggingConfig(**self.token_kl_logging),
            )

        valid_actor_objectives = {
            "direct_reverse_kl",
            "negative_kl_advantage",
            "grpo_advantage_reweighting",
        }
        if self.actor_objective is None:
            resolved_actor_objective = (
                "grpo_advantage_reweighting" if self.advantage_shaping.enable else "direct_reverse_kl"
            )
            object.__setattr__(self, "actor_objective", resolved_actor_objective)
        elif self.actor_objective not in valid_actor_objectives:
            raise ValueError(
                f"Invalid opsd.actor_objective: {self.actor_objective}. Must be one of {sorted(valid_actor_objectives)}"
            )
        if self.advantage_shaping.enable != (self.actor_objective == "grpo_advantage_reweighting"):
            raise ValueError(
                "opsd.advantage_shaping.enable is a compatibility alias and must equal "
                "(opsd.actor_objective == 'grpo_advantage_reweighting')"
            )

        valid_modes = {"opsd", "opsd_rlvr"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid opsd.mode: {self.mode}. Must be one of {sorted(valid_modes)}")

        valid_teacher_sources = {"ground_truth", "sdpo_success_rollout"}
        if self.teacher_source not in valid_teacher_sources:
            raise ValueError(
                f"Invalid opsd.teacher_source: {self.teacher_source}. Must be one of {sorted(valid_teacher_sources)}"
            )

        valid_sdpo_conditioning_modes = {"prompt_append", "steering"}
        if self.sdpo_conditioning_mode not in valid_sdpo_conditioning_modes:
            raise ValueError(
                "Invalid opsd.sdpo_conditioning_mode: "
                f"{self.sdpo_conditioning_mode}. Must be one of {sorted(valid_sdpo_conditioning_modes)}"
            )
        if self.sdpo_conditioning_mode == "steering":
            if self.teacher_source != "sdpo_success_rollout":
                raise ValueError(
                    "opsd.sdpo_conditioning_mode=steering requires opsd.teacher_source=sdpo_success_rollout"
                )
            if self.teacher_model != "actor":
                raise ValueError(
                    "The production steering-vector path requires teacher_model=actor so the only "
                    "privileged signal is the detached steering intervention"
                )
            if not self.steering.layer_fractions.strip():
                raise ValueError("opsd.steering.layer_fractions must be set for steering conditioning")
            if self.teacher_sft_weight != 0.0:
                raise ValueError("Steering-vector OPSD does not support teacher SFT")
            if self.sdpo_distill_only_failed:
                raise ValueError(
                    "Steering-vector OPSD is outcome-symmetric and requires opsd.sdpo_distill_only_failed=False"
                )
            if self.actor_objective in {"direct_reverse_kl", "negative_kl_advantage"}:
                if self.mode != "opsd":
                    raise ValueError(f"Steering actor_objective={self.actor_objective} requires opsd.mode=opsd")
            elif self.actor_objective == "grpo_advantage_reweighting":
                if self.mode != "opsd_rlvr":
                    raise ValueError("Steering actor_objective=grpo_advantage_reweighting requires opsd.mode=opsd_rlvr")
            if self.steering.strict_contract:
                strict_mismatches = {}
                expected_values = {
                    "correct_rollout_aggregation": "all",
                    "activation_aggregation": "per_rollout",
                    "normalize": "unit_norm",
                    "apply_positions": "response_only",
                    "detach_vectors": True,
                    "expected_model_path": "/hf_models/Qwen3-1.7B",
                    "actor_model_path": "/hf_models/Qwen3-1.7B",
                    "expected_total_layers": 28,
                    "expected_layer_indices": [9, 10],
                }
                for field_name, expected in expected_values.items():
                    actual = getattr(self.steering, field_name)
                    if actual != expected:
                        strict_mismatches[f"steering.{field_name}"] = (actual, expected)
                if self.steering.source_mode not in {"caa", "policy_gradient"}:
                    strict_mismatches["steering.source_mode"] = (
                        self.steering.source_mode,
                        "caa or policy_gradient",
                    )
                if self.steering.source_mode == "policy_gradient":
                    if self.steering.caa_scope != "global_batch":
                        strict_mismatches["steering.caa_scope"] = (
                            self.steering.caa_scope,
                            "global_batch",
                        )
                    if self.steering.gradient_objective != "grpo_advantage":
                        strict_mismatches["steering.gradient_objective"] = (
                            self.steering.gradient_objective,
                            "grpo_advantage",
                        )
                    if self.steering.gradient_aggregation != "per_rollout":
                        strict_mismatches["steering.gradient_aggregation"] = (
                            self.steering.gradient_aggregation,
                            "per_rollout",
                        )
                if self.steering.scale <= 0.0:
                    strict_mismatches["steering.scale"] = (
                        self.steering.scale,
                        "finite positive calibration coefficient",
                    )
                parent_expected = {
                    "sdpo_distill_only_failed": (self.sdpo_distill_only_failed, False),
                    "distill_max_response_tokens": (self.distill_max_response_tokens, None),
                    "balance_mode": (self.balance_mode, "none"),
                    "rlvr_backward_scale": (self.rlvr_backward_scale, 0.0),
                    "rlvr_warmup_steps": (self.rlvr_warmup_steps, 0),
                }
                if self.actor_objective == "grpo_advantage_reweighting":
                    parent_expected.update(
                        {
                            "mode": (self.mode, "opsd_rlvr"),
                            "distill_backward_scale": (self.distill_backward_scale, 0.0),
                            "mix_weight": (self.mix_weight, 1.0),
                            "advantage_shaping.enable": (self.advantage_shaping.enable, True),
                            "advantage_shaping.student_rlvr_backward_scale": (
                                self.advantage_shaping.student_rlvr_backward_scale,
                                1.0,
                            ),
                            "advantage_shaping.normalize": (
                                self.advantage_shaping.normalize,
                                None,
                            ),
                            "advantage_shaping.clip_z": (
                                self.advantage_shaping.clip_z,
                                None,
                            ),
                        }
                    )
                else:
                    parent_expected.update(
                        {
                            "mode": (self.mode, "opsd"),
                            "distill_backward_scale": (self.distill_backward_scale, 1.0),
                            "advantage_shaping.enable": (self.advantage_shaping.enable, False),
                        }
                    )
                strict_mismatches.update(
                    {field_name: values for field_name, values in parent_expected.items() if values[0] != values[1]}
                )
                if strict_mismatches:
                    raise ValueError(
                        "opsd.steering.strict_contract=True requires the audited outcome-symmetric CAA contract; "
                        f"mismatches={strict_mismatches}"
                    )

        valid_teacher_models = {"actor", "ema", "fixed", "separate"}
        if self.teacher_model not in valid_teacher_models:
            raise ValueError(
                f"Invalid opsd.teacher_model: {self.teacher_model}. Must be one of {sorted(valid_teacher_models)}"
            )
        if self.teacher_model == "ema" and self.mode == "opsd_rlvr":
            raise ValueError("opsd.teacher_model=ema is only supported with opsd.mode=opsd.")
        if self.teacher_model == "fixed" and self.mode == "opsd_rlvr":
            raise ValueError("opsd.teacher_model=fixed is only supported with opsd.mode=opsd.")
        if self.teacher_model == "separate" and self.mode != "opsd_rlvr":
            raise ValueError("opsd.teacher_model=separate requires opsd.mode=opsd_rlvr.")
        if not 0.0 <= self.teacher_ema_rate <= 1.0:
            raise ValueError(f"opsd.teacher_ema_rate must be in [0, 1], got {self.teacher_ema_rate}")

        valid_teacher_prompt_styles = {"append_instruction", "reference_solution_single_user"}
        if self.teacher_prompt_style not in valid_teacher_prompt_styles:
            raise ValueError(
                "Invalid opsd.teacher_prompt_style: "
                f"{self.teacher_prompt_style}. Must be one of {sorted(valid_teacher_prompt_styles)}"
            )

        if self.distill_loss == "topk_jsd":
            raise ValueError(
                "opsd.distill_loss=topk_jsd is disabled: the legacy union-top-k support "
                "double-counted opposing-only probability mass in its collapsed tail bucket. "
                "Use sampled_reverse_kl."
            )
        if self.distill_loss == "full_jsd":
            raise ValueError(
                "opsd.distill_loss=full_jsd is disabled by the current reverse-KL-only "
                "stabilization policy. Use sampled_reverse_kl."
            )
        if self.distill_loss != "sampled_reverse_kl":
            raise ValueError(f"Invalid opsd.distill_loss: {self.distill_loss}. Only sampled_reverse_kl is supported.")

        legacy_distill_controls = {
            "topk": self.topk,
            "distill_beta": self.distill_beta,
            "distill_token_clip": self.distill_token_clip,
            "distill_token_clip_tail": self.distill_token_clip_tail,
        }
        configured_legacy_controls = {
            name: value for name, value in legacy_distill_controls.items() if value is not None
        }
        if configured_legacy_controls:
            raise ValueError(
                "Top-k/JSD-only OPSD controls are disabled with sampled_reverse_kl: "
                f"{configured_legacy_controls}. Remove these overrides."
            )

        if self.distill_max_response_tokens is not None and self.distill_max_response_tokens <= 0:
            raise ValueError(
                f"opsd.distill_max_response_tokens must be positive when set, got {self.distill_max_response_tokens}"
            )

        if self.debug_print_interval < 0:
            raise ValueError(f"opsd.debug_print_interval must be non-negative, got {self.debug_print_interval}")

        if self.debug_num_tokens <= 0:
            raise ValueError(f"opsd.debug_num_tokens must be positive, got {self.debug_num_tokens}")

        if not 0.0 <= self.mix_weight <= 1.0:
            raise ValueError(f"opsd.mix_weight must be in [0, 1], got {self.mix_weight}")

        if self.advantage_shaping.enable:
            if self.mode != "opsd_rlvr":
                raise ValueError("opsd.advantage_shaping.enable=True requires opsd.mode=opsd_rlvr")
            if self.mix_weight != 1.0:
                raise ValueError("opsd.advantage_shaping.enable=True requires opsd.mix_weight=1.0")
            if self.balance_mode != "none":
                raise ValueError("opsd.advantage_shaping.enable=True requires opsd.balance_mode=none")
            if self.distill_backward_scale != 0.0:
                raise ValueError(
                    "opsd.advantage_shaping.enable=True requires opsd.distill_backward_scale=0.0; "
                    "teacher evidence is score-only and must not create a reverse-KL backward pass"
                )
            if self.teacher_model == "actor" and self.rlvr_backward_scale != 0.0:
                raise ValueError(
                    "Shared-teacher advantage shaping requires opsd.rlvr_backward_scale=0.0; "
                    "otherwise the ground-conditioned branch would also train the shared actor"
                )

        if self.actor_objective == "negative_kl_advantage" and self.mode != "opsd":
            raise ValueError("opsd.actor_objective=negative_kl_advantage requires opsd.mode=opsd")

        if self.distill_backward_scale < 0.0:
            raise ValueError(f"opsd.distill_backward_scale must be non-negative, got {self.distill_backward_scale}")

        if self.rlvr_backward_scale < 0.0:
            raise ValueError(f"opsd.rlvr_backward_scale must be non-negative, got {self.rlvr_backward_scale}")

        if self.rlvr_warmup_steps < 0:
            raise ValueError(f"opsd.rlvr_warmup_steps must be non-negative, got {self.rlvr_warmup_steps}")

        if not math.isfinite(self.teacher_sft_weight) or self.teacher_sft_weight < 0.0:
            raise ValueError(f"opsd.teacher_sft_weight must be finite and non-negative, got {self.teacher_sft_weight}")
        valid_teacher_sft_scopes = {"thinking_only", "thinking_and_answer"}
        if self.teacher_sft_target_scope not in valid_teacher_sft_scopes:
            raise ValueError(
                "Invalid opsd.teacher_sft_target_scope: "
                f"{self.teacher_sft_target_scope}. Must be one of {sorted(valid_teacher_sft_scopes)}"
            )
        if not self.teacher_sft_success_field.strip():
            raise ValueError("opsd.teacher_sft_success_field must be non-empty")
        if not math.isfinite(self.teacher_sft_success_threshold):
            raise ValueError(
                f"opsd.teacher_sft_success_threshold must be finite, got {self.teacher_sft_success_threshold}"
            )
        if not self.teacher_sft_think_end_tag:
            raise ValueError("opsd.teacher_sft_think_end_tag must be non-empty")
        if self.teacher_sft_weight > 0.0:
            if self.mode != "opsd_rlvr":
                raise ValueError("Teacher SFT requires opsd.mode=opsd_rlvr")
            if self.teacher_model != "separate":
                raise ValueError(
                    "Teacher SFT requires opsd.teacher_model=separate so its gradients cannot silently update the actor"
                )
            if self.teacher_source != "ground_truth":
                raise ValueError(
                    "Teacher SFT requires opsd.teacher_source=ground_truth so every supervised rollout uses the "
                    "ground-truth-conditioned prompt"
                )

        valid_balance_modes = {"none", "grad_norm"}
        if self.balance_mode not in valid_balance_modes:
            raise ValueError(
                f"Invalid opsd.balance_mode: {self.balance_mode}. Must be one of {sorted(valid_balance_modes)}"
            )
        if self.teacher_model == "separate" and self.balance_mode == "grad_norm":
            raise ValueError(
                "opsd.balance_mode=grad_norm is not supported with a separately optimized teacher; "
                "use balance_mode=none and control the actor/teacher optimizers independently."
            )
        if not self.separate_backward:
            raise ValueError(
                "opsd.separate_backward=False is disabled: the stabilized reverse-KL path uses "
                "explicit branch backward passes. Set separate_backward=True."
            )

        valid_balance_subsets = {"lm_head", "last_layer", "all"}
        if self.balance_param_subset not in valid_balance_subsets:
            raise ValueError(
                "Invalid opsd.balance_param_subset: "
                f"{self.balance_param_subset}. Must be one of {sorted(valid_balance_subsets)}"
            )

        valid_is_modes = {"none", "token", "sequence"}
        if self.offpolicy_is_mode not in valid_is_modes:
            raise ValueError(
                f"Invalid opsd.offpolicy_is_mode: {self.offpolicy_is_mode}. Must be one of {sorted(valid_is_modes)}"
            )

        if self.offpolicy_is_clip <= 0:
            raise ValueError(f"opsd.offpolicy_is_clip must be positive, got {self.offpolicy_is_clip}")

        valid_behavior_sources = {"rollout", "recompute"}
        if self.behavior_logprob_source not in valid_behavior_sources:
            raise ValueError(
                "Invalid opsd.behavior_logprob_source: "
                f"{self.behavior_logprob_source}. Must be one of {sorted(valid_behavior_sources)}"
            )

        valid_truncation = {None, "left", "right", "middle", "error"}
        if self.truncation not in valid_truncation:
            raise ValueError(
                f"Invalid opsd.truncation: {self.truncation}. Must be one of "
                f"{sorted(x for x in valid_truncation if x is not None)} or None"
            )


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_correction (Optional[RolloutCorrectionConfig]): Rollout Correction configuration.
            Addresses off-policy issues from policy mismatch, model staleness, and general distribution shifts.

            Set to None to disable entirely. Use factory methods for common presets:
            - RolloutCorrectionConfig.decoupled_token_is() - Decoupled mode with token-level IS
            - RolloutCorrectionConfig.decoupled_seq_is() - Decoupled mode with sequence-level IS
            - RolloutCorrectionConfig.decoupled_seq_is_rs() - Decoupled mode with sequence IS + RS
            - RolloutCorrectionConfig.decoupled_k1_rs() - Decoupled mode with K1-RS (divergence)
            - RolloutCorrectionConfig.decoupled_geo_rs() - Decoupled mode with Geo-RS (ratio)
            - RolloutCorrectionConfig.bypass_ppo_clip() - Bypass mode with PPO-clip
            - RolloutCorrectionConfig.bypass_ppo_clip_k1_rs() - Bypass mode with PPO-clip + K1-RS
            - RolloutCorrectionConfig.bypass_pg_is() - Bypass mode with REINFORCE + IS
            - RolloutCorrectionConfig.bypass_pg_k1_rs() - Bypass mode with REINFORCE + K1-RS

            For backward compatibility, you can still pass a dict, which will be converted to
            RolloutCorrectionConfig automatically.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Correction: corrects off-policy issues (policy mismatch, model staleness, distribution shifts)
    # Set to None to disable, use RolloutCorrectionConfig presets (e.g., .tis(), .mis()), or pass dict
    rollout_correction: Optional[RolloutCorrectionConfig] = None
    branch_revision_grpo: BranchRevisionGRPOConfig = field(default_factory=BranchRevisionGRPOConfig)
    intermediate_mc_value: IntermediateMCValueConfig = field(default_factory=IntermediateMCValueConfig)
    opsd: OPSDConfig = field(default_factory=OPSDConfig)
    # GDPO (Group reward-Decoupled Normalization Policy Optimization) settings.
    # gdpo_reward_keys: keys in non_tensor_batch (from compute_score's return dict) that
    #   correspond to individual reward dimensions, e.g. ["format_reward", "accuracy_reward"].
    # gdpo_reward_weights: per-dimension weights for aggregation (default: equal weights).
    gdpo_reward_keys: Optional[list[str]] = None
    gdpo_reward_weights: Optional[list[float]] = None
