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
"""Pure parsing and reward helpers for synchronous branch-revision GRPO."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PREFIX_OPEN = "<prefix>"
PREFIX_CLOSE = "</prefix>"
PREFIX_PLUS_CONTINUATION_OPEN = "<prefix + new continuation>"
PREFIX_PLUS_CONTINUATION_CLOSE = "</prefix + new continuation>"
_FOLLOWUP_SENTINEL = "__VERL_BRANCH_REVISION_PREVIOUS_ASSISTANT_7F4E65C1__"
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:"
    r"\\(?:boxed|fbox)\s*\{|"
    r"####|"
    r"</?answer(?:\s[^>]*)?>|"
    r"\bfinal\s+(?:answer|result)\b|"
    r"\banswer\s*(?:is\b|equals\b|=|:)|"
    r"\b(?:task|problem)\s+(?:is\s+)?(?:complete|completed|solved)\b"
    r")",
    flags=re.IGNORECASE,
)
_LATEX_ENVIRONMENT_PATTERN = re.compile(r"\\(begin|end)\{([^{}\r\n]+)\}")
_FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class ParsedBranchRevision:
    valid: bool
    reason: str
    solution_text: str
    critique_text: str
    analysis_text: str = ""
    prefix_text: str = ""
    prefix_plus_new_continuation_text: str = ""
    new_continuation_text: str = ""
    branch_start: int = -1
    revised_text: str = ""
    branch_prefix_ids: tuple[int, ...] = ()
    prefix_ids: tuple[int, ...] = ()
    continuation_prefix_ids: tuple[int, ...] = ()
    new_continuation_ids: tuple[int, ...] = ()
    revised_prefix_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RolloutLogProbPrefix:
    rollout_id: str
    log_probs: torch.Tensor
    prefix_sums: torch.Tensor
    min_sparse_table: tuple[torch.Tensor, ...]

    @property
    def token_count(self) -> int:
        return int(self.prefix_sums.numel() - 1)


@dataclass(frozen=True)
class LearnabilityReference:
    window_size: int
    logprob_statistic: str
    window_scores: torch.Tensor
    sorted_window_scores: torch.Tensor
    rollout_window_counts: tuple[tuple[str, int], ...]
    eligible_rollouts: int
    population_mean: float | None
    population_stddev: float | None

    @property
    def total_windows(self) -> int:
        return int(self.window_scores.numel())

    @property
    def window_scores_sha256(self) -> str:
        values = self.window_scores.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        canonical = np.asarray(values, dtype=np.dtype("<f8"))
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class LearnabilityScore:
    logprob_statistic: str
    threshold_mode: str
    seed_score: float
    percentile: float
    reference_mean: float | None
    reference_stddev: float | None
    stddevs_below_mean: float | None
    acceptance_floor: float | None
    max_seed_window_stddevs: float
    reward_weight: float
    accepted: bool
    eligible_rollouts: int
    total_windows: int


def terminal_eos_ids(tokenizer: Any) -> set[int]:
    raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, Iterable) and not isinstance(raw, int | str | bytes):
        return {int(token) for token in raw}
    return {int(raw)}


def strip_terminal_eos(token_ids: Iterable[int], tokenizer: Any) -> list[int]:
    """Remove only a terminal run of configured EOS ids, preserving interior specials."""

    result = [int(token) for token in token_ids]
    eos_ids = terminal_eos_ids(tokenizer)
    while result and result[-1] in eos_ids:
        result.pop()
    return result


def decode_exact(token_ids: Iterable[int], tokenizer: Any) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def encode_followup_user_turn(
    instruction: str,
    tokenizer: Any,
    *,
    prior_messages: Sequence[Mapping[str, Any]] | None = None,
    assistant_content: str | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Encode a genuine user follow-up plus assistant generation boundary.

    Deriving the suffix from the tokenizer's own chat template avoids hard-coding
    model-specific role tokens. Disabling hidden thinking for this turn keeps the
    generated critique itself visible and trainable; it does not change sampling
    temperature or the original solution rollout.
    """

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("branch-revision GRPO requires a tokenizer chat template")
    if (prior_messages is None) != (assistant_content is None):
        raise ValueError("prior_messages and assistant_content must be provided together")
    if _FOLLOWUP_SENTINEL in instruction or (assistant_content is not None and _FOLLOWUP_SENTINEL in assistant_content):
        raise ValueError("branch-revision text contains the reserved boundary sentinel")
    if prior_messages is None:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": _FOLLOWUP_SENTINEL},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [dict(message) for message in prior_messages]
        messages.extend(
            [
                {"role": "assistant", "content": f"{assistant_content}{_FOLLOWUP_SENTINEL}"},
                {"role": "user", "content": instruction},
            ]
        )
    template_kwargs = dict(chat_template_kwargs or {})
    template_kwargs["enable_thinking"] = False
    rendered = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not isinstance(rendered, str) or rendered.count(_FOLLOWUP_SENTINEL) != 1:
        raise ValueError("tokenizer chat template did not preserve the branch-revision boundary sentinel exactly")
    suffix = rendered.split(_FOLLOWUP_SENTINEL, 1)[1]
    if not suffix:
        raise ValueError("tokenizer chat template produced an empty branch-revision follow-up boundary")
    suffix_ids = [int(token) for token in tokenizer.encode(suffix, add_special_tokens=False)]
    if not suffix_ids or decode_exact(suffix_ids, tokenizer) != suffix:
        raise ValueError("branch-revision follow-up boundary does not round-trip through the tokenizer exactly")
    return suffix_ids


def build_rollout_logprob_prefixes(
    rollout_ids: Sequence[str],
    rollout_log_probs: Sequence[Sequence[float]],
    *,
    device: torch.device | str | None = None,
) -> tuple[RolloutLogProbPrefix, ...]:
    """Build exact mean and range-minimum indexes once per rollout and iteration."""

    if len(rollout_ids) != len(rollout_log_probs):
        raise ValueError("rollout ids and log-probability rows must have equal lengths")
    result: list[RolloutLogProbPrefix] = []
    for rollout_id, values in zip(rollout_ids, rollout_log_probs, strict=True):
        tensor = normalize_log_probs_float32(values, device=device)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"rollout {rollout_id!r} contains non-finite log probabilities")
        prefix_values = tensor.to(dtype=torch.float64)
        prefix = torch.cat([torch.zeros(1, dtype=torch.float64, device=tensor.device), prefix_values.cumsum(dim=0)])
        sparse_levels = [tensor]
        span = 1
        while span * 2 <= tensor.numel():
            previous = sparse_levels[-1]
            output_count = tensor.numel() - span * 2 + 1
            sparse_levels.append(torch.minimum(previous[:output_count], previous[span : span + output_count]))
            span *= 2
        result.append(RolloutLogProbPrefix(str(rollout_id), tensor, prefix, tuple(sparse_levels)))
    return tuple(result)


def normalize_log_probs_float32(
    values: Iterable[float],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Use one explicit precision boundary for rollout and prompt log-probabilities."""

    tensor = torch.as_tensor(list(values), dtype=torch.float32, device=device)
    if tensor.ndim != 1:
        raise ValueError("log probabilities must form a one-dimensional sequence")
    if not torch.isfinite(tensor).all():
        raise ValueError("log probabilities must all be finite")
    return tensor


def aggregate_log_probs(values: Iterable[float], *, statistic: str) -> float:
    """Aggregate float32-normalized token log-probabilities for learnability."""

    tensor = normalize_log_probs_float32(values)
    if tensor.numel() == 0:
        raise ValueError("cannot aggregate an empty log-probability sequence")
    if statistic == "mean":
        return float(tensor.to(dtype=torch.float64).mean().item())
    if statistic == "min":
        return float(tensor.amin().item())
    raise ValueError("learnability log-probability statistic must be mean or min")


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def branch_prefix_open_block_reason(prefix_text: str) -> str | None:
    """Return the structural block left open at a proposed branch boundary."""

    math_mode: str | None = None
    environment_stack: list[str] = []
    fence: tuple[str, int] | None = None
    for raw_line in prefix_text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        fence_match = _FENCE_PATTERN.match(line)
        if fence is not None:
            if fence_match is not None:
                run = fence_match.group(1)
                suffix = fence_match.group(2)
                if run[0] == fence[0] and len(run) >= fence[1] and not suffix.strip():
                    fence = None
            continue
        if math_mode is None and not environment_stack and fence_match is not None:
            run = fence_match.group(1)
            fence = (run[0], len(run))
            continue

        cursor = 0
        while cursor < len(line):
            if math_mode == "dollar":
                if line.startswith("$$", cursor) and not _is_escaped(line, cursor):
                    math_mode = None
                    cursor += 2
                else:
                    cursor += 1
                continue
            if math_mode == "bracket":
                if line.startswith(r"\]", cursor) and not _is_escaped(line, cursor):
                    math_mode = None
                    cursor += 2
                else:
                    cursor += 1
                continue
            if environment_stack:
                match = _LATEX_ENVIRONMENT_PATTERN.match(line, cursor)
                if match is None or _is_escaped(line, cursor):
                    cursor += 1
                    continue
                action, name = match.groups()
                if action == "begin":
                    environment_stack.append(name)
                elif environment_stack[-1] == name:
                    environment_stack.pop()
                cursor = match.end()
                continue

            if line.startswith("$$", cursor) and not _is_escaped(line, cursor):
                math_mode = "dollar"
                cursor += 2
                continue
            if line.startswith(r"\[", cursor) and not _is_escaped(line, cursor):
                math_mode = "bracket"
                cursor += 2
                continue
            match = _LATEX_ENVIRONMENT_PATTERN.match(line, cursor)
            if match is not None and not _is_escaped(line, cursor):
                action, name = match.groups()
                if action == "begin":
                    environment_stack.append(name)
                cursor = match.end()
                continue
            cursor += 1

    if fence is not None:
        return "branch_inside_code_fence"
    if math_mode is not None:
        return "branch_inside_display_math"
    if environment_stack:
        return "branch_inside_latex_environment"
    return None


def _normalize_branch_match_text(text: str) -> tuple[str, tuple[int, ...]]:
    """Lowercase alphanumerics and retain their source character offsets."""

    normalized: list[str] = []
    source_offsets: list[int] = []
    for source_offset, character in enumerate(text):
        if not character.isalnum():
            continue
        lowered = character.lower()
        normalized.extend(lowered)
        source_offsets.extend([source_offset] * len(lowered))
    return "".join(normalized), tuple(source_offsets)


def _z_prefix_lengths(pattern: str, text: str) -> list[int]:
    """Return the pattern-prefix match length at every text position in linear time."""

    if not pattern or not text:
        return [0] * len(text)
    # Normalized inputs contain only alphanumerics, so NUL is an unambiguous separator.
    combined = f"{pattern}\0{text}"
    z = [0] * len(combined)
    left = 0
    right = 0
    for index in range(1, len(combined)):
        if index <= right:
            z[index] = min(right - index + 1, z[index - left])
        while index + z[index] < len(combined) and combined[z[index]] == combined[index + z[index]]:
            z[index] += 1
        if index + z[index] - 1 > right:
            left = index
            right = index + z[index] - 1
    start = len(pattern) + 1
    return [min(value, len(pattern)) for value in z[start:]]


def _normalized_unique_prefix_location(branch_text: str, solution_text: str) -> tuple[int, int] | None:
    """Locate a uniquely best normalized branch prefix in the raw solution.

    The returned pair contains the raw source offset of the first matched
    alphanumeric and the raw offset immediately after the preceding source
    alphanumeric. The latter bounds boundary recovery to deleted formatting.
    """

    normalized_branch, _ = _normalize_branch_match_text(branch_text)
    normalized_solution, solution_offsets = _normalize_branch_match_text(solution_text)
    if not normalized_branch or not normalized_solution:
        return None
    match_lengths = _z_prefix_lengths(normalized_branch, normalized_solution)
    best_length = max(match_lengths, default=0)
    if best_length <= 0:
        return None
    best_positions = [index for index, value in enumerate(match_lengths) if value == best_length]
    if len(best_positions) != 1:
        return (-1, -1)
    normalized_start = best_positions[0]
    semantic_start = solution_offsets[normalized_start]
    previous_offset = solution_offsets[normalized_start - 1] if normalized_start else -1
    formatting_gap_start = previous_offset + 1
    return semantic_start, formatting_gap_start


def _matching_explicit_block_opener(
    branch_text: str,
    solution_text: str,
    *,
    formatting_gap_start: int,
    semantic_start: int,
    open_block_reason: str,
) -> int | None:
    """Find an explicitly quoted opener that makes the intended boundary external."""

    stripped = branch_text.lstrip()
    delimiter: str | None = None
    if open_block_reason == "branch_inside_display_math":
        if stripped.startswith("$$"):
            delimiter = "$$"
        elif stripped.startswith(r"\["):
            delimiter = r"\["
    elif open_block_reason == "branch_inside_code_fence":
        fence_match = _FENCE_PATTERN.match(stripped.splitlines()[0] if stripped else "")
        if fence_match is not None:
            delimiter = fence_match.group(1)
    elif open_block_reason == "branch_inside_latex_environment":
        environment_match = _LATEX_ENVIRONMENT_PATTERN.match(stripped)
        if environment_match is not None and environment_match.group(1) == "begin":
            delimiter = environment_match.group(0)
    if delimiter is None:
        return None
    opener = solution_text.rfind(delimiter, formatting_gap_start, semantic_start + 1)
    if opener < formatting_gap_start or _is_escaped(solution_text, opener):
        return None
    return opener


def _revision_at_branch_start(
    *,
    editable_solution_ids: list[int],
    solution_text: str,
    critique_text: str,
    analysis_text: str,
    prefix_text: str,
    prefix_plus_new_continuation_text: str,
    new_continuation_text: str,
    branch_start: int,
    tokenizer: Any,
    new_continuation_max_tokens: int,
) -> ParsedBranchRevision:
    branch_prefix_text = solution_text[:branch_start]
    open_block_reason = branch_prefix_open_block_reason(branch_prefix_text)
    if open_block_reason is not None:
        return _invalid(open_block_reason, solution_text, critique_text)
    branch_prefix_ids = [int(token) for token in tokenizer.encode(branch_prefix_text, add_special_tokens=False)]
    if decode_exact(branch_prefix_ids, tokenizer) != branch_prefix_text:
        return _invalid("branch_prefix_not_roundtrip_exact", solution_text, critique_text)
    if editable_solution_ids[: len(branch_prefix_ids)] != branch_prefix_ids:
        return _invalid("branch_not_at_token_boundary", solution_text, critique_text)
    continuation_prefix_text = branch_prefix_text + prefix_text
    continuation_prefix_ids = [
        int(token) for token in tokenizer.encode(continuation_prefix_text, add_special_tokens=False)
    ]
    if decode_exact(continuation_prefix_ids, tokenizer) != continuation_prefix_text:
        return _invalid("continuation_prefix_not_roundtrip_exact", solution_text, critique_text)
    if continuation_prefix_ids[: len(branch_prefix_ids)] != branch_prefix_ids:
        return _invalid("prefix_retokenizes_branch_prefix", solution_text, critique_text)
    prefix_ids = continuation_prefix_ids[len(branch_prefix_ids) :]
    if not prefix_ids:
        return _invalid("empty_prefix_tokens", solution_text, critique_text)
    open_continuation_reason = branch_prefix_open_block_reason(continuation_prefix_text)
    if open_continuation_reason is not None:
        return _invalid(open_continuation_reason, solution_text, critique_text)

    revised_text = branch_prefix_text + prefix_plus_new_continuation_text
    revised_prefix_ids = [int(token) for token in tokenizer.encode(revised_text, add_special_tokens=False)]
    if not revised_prefix_ids:
        return _invalid("empty_revised_prefix", solution_text, critique_text)
    if decode_exact(revised_prefix_ids, tokenizer) != revised_text:
        return _invalid("revision_not_roundtrip_exact", solution_text, critique_text)
    if revised_prefix_ids[: len(continuation_prefix_ids)] != continuation_prefix_ids:
        return _invalid("new_continuation_retokenizes_prefix", solution_text, critique_text)
    new_continuation_ids = revised_prefix_ids[len(continuation_prefix_ids) :]
    if not new_continuation_ids or len(new_continuation_ids) > new_continuation_max_tokens:
        return _invalid("new_continuation_token_cap", solution_text, critique_text)
    return ParsedBranchRevision(
        valid=True,
        reason="valid",
        solution_text=solution_text,
        critique_text=critique_text,
        analysis_text=analysis_text,
        prefix_text=prefix_text,
        prefix_plus_new_continuation_text=prefix_plus_new_continuation_text,
        new_continuation_text=new_continuation_text,
        branch_start=branch_start,
        revised_text=revised_text,
        branch_prefix_ids=tuple(branch_prefix_ids),
        prefix_ids=tuple(prefix_ids),
        continuation_prefix_ids=tuple(continuation_prefix_ids),
        new_continuation_ids=tuple(new_continuation_ids),
        revised_prefix_ids=tuple(revised_prefix_ids),
    )


def build_learnability_reference(
    prefixes: Sequence[RolloutLogProbPrefix],
    *,
    window_size: int,
    logprob_statistic: str = "mean",
) -> LearnabilityReference:
    """Enumerate every length-matched window with uniform per-window mass."""

    if window_size <= 0:
        raise ValueError("learnability window size must be positive")
    if logprob_statistic not in {"mean", "min"}:
        raise ValueError("learnability log-probability statistic must be mean or min")
    exhaustive_scores: list[torch.Tensor] = []
    rollout_window_counts: list[tuple[str, int]] = []
    for rollout in prefixes:
        candidate_count = rollout.token_count - window_size + 1
        if candidate_count <= 0:
            continue
        if logprob_statistic == "mean":
            scores = (rollout.prefix_sums[window_size:] - rollout.prefix_sums[:-window_size]) / float(window_size)
        else:
            level_index = window_size.bit_length() - 1
            span = 1 << level_index
            level = rollout.min_sparse_table[level_index]
            right_offset = window_size - span
            scores = torch.minimum(
                level[:candidate_count],
                level[right_offset : right_offset + candidate_count],
            ).to(dtype=torch.float64)
        if scores.numel() != candidate_count:
            raise RuntimeError("exhaustive learnability reference produced an incorrect window count")
        exhaustive_scores.append(scores)
        rollout_window_counts.append((rollout.rollout_id, candidate_count))
    if not exhaustive_scores:
        device = prefixes[0].prefix_sums.device if prefixes else None
        empty = torch.empty(0, dtype=torch.float64, device=device)
        return LearnabilityReference(
            window_size=window_size,
            logprob_statistic=logprob_statistic,
            window_scores=empty,
            sorted_window_scores=empty.clone(),
            rollout_window_counts=(),
            eligible_rollouts=0,
            population_mean=None,
            population_stddev=None,
        )
    window_scores = torch.cat(exhaustive_scores)
    population = window_scores.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    population_mean = float(np.mean(population, dtype=np.float64))
    population_stddev = float(np.std(population, dtype=np.float64, ddof=0))
    if not math.isfinite(population_mean) or not math.isfinite(population_stddev):
        raise RuntimeError("exhaustive learnability reference produced non-finite population statistics")
    return LearnabilityReference(
        window_size=window_size,
        logprob_statistic=logprob_statistic,
        window_scores=window_scores,
        sorted_window_scores=torch.sort(window_scores).values,
        rollout_window_counts=tuple(rollout_window_counts),
        eligible_rollouts=len(rollout_window_counts),
        population_mean=population_mean,
        population_stddev=population_stddev,
    )


def score_seed_learnability(
    seed_score: float,
    reference: LearnabilityReference,
    *,
    threshold_mode: str,
    max_seed_window_stddevs: float,
    minimum_percentile: float,
    full_credit_percentile: float,
) -> LearnabilityScore:
    """Gate a replacement seed by stddev or percentile against exhaustive windows."""

    if not math.isfinite(seed_score):
        raise ValueError("replacement seed log-probability score must be finite")
    if threshold_mode not in {"stddev", "percentile"}:
        raise ValueError("learnability threshold mode must be stddev or percentile")
    if not math.isfinite(max_seed_window_stddevs) or max_seed_window_stddevs < 0.0:
        raise ValueError("maximum learnability standard deviations must be finite and nonnegative")
    if not 0.0 <= minimum_percentile < full_credit_percentile <= 1.0:
        raise ValueError("learnability percentiles must satisfy 0 <= minimum < full credit <= 1")
    if reference.total_windows == 0:
        percentile = 0.0
    else:
        boundary = torch.searchsorted(
            reference.sorted_window_scores,
            torch.tensor(seed_score, dtype=torch.float64, device=reference.sorted_window_scores.device),
            right=True,
        )
        percentile = float(boundary.detach().cpu().item()) / reference.total_windows
        percentile = min(max(percentile, 0.0), 1.0)
    reference_mean = reference.population_mean
    reference_stddev = reference.population_stddev
    if reference_stddev is not None and reference_stddev > 0.0 and reference_mean is not None:
        stddevs_below_mean = (reference_mean - seed_score) / reference_stddev
    else:
        stddevs_below_mean = None
    if threshold_mode == "stddev":
        acceptance_floor = (
            reference_mean - max_seed_window_stddevs * reference_stddev
            if reference_mean is not None and reference_stddev is not None
            else None
        )
        accepted = acceptance_floor is not None and seed_score >= acceptance_floor
        reward_weight = float(accepted)
    else:
        acceptance_floor = None
        accepted = reference.total_windows > 0 and percentile >= minimum_percentile
        reward_weight = min(
            max((percentile - minimum_percentile) / (full_credit_percentile - minimum_percentile), 0.0),
            1.0,
        )
    return LearnabilityScore(
        logprob_statistic=reference.logprob_statistic,
        threshold_mode=threshold_mode,
        seed_score=float(seed_score),
        percentile=percentile,
        reference_mean=reference_mean,
        reference_stddev=reference_stddev,
        stddevs_below_mean=stddevs_below_mean,
        acceptance_floor=acceptance_floor,
        max_seed_window_stddevs=float(max_seed_window_stddevs),
        reward_weight=reward_weight,
        accepted=accepted,
        eligible_rollouts=reference.eligible_rollouts,
        total_windows=reference.total_windows,
    )


def _invalid(reason: str, solution_text: str, critique_text: str) -> ParsedBranchRevision:
    return ParsedBranchRevision(False, reason, solution_text, critique_text)


def parse_branch_revision(
    solution_ids: Iterable[int],
    critique_ids: Iterable[int],
    tokenizer: Any,
    *,
    branch_max_tokens: int,
    new_continuation_max_tokens: int,
) -> ParsedBranchRevision:
    """Parse a prefix-anchored joint edit with the existing tolerant locator."""

    editable_solution_ids = strip_terminal_eos(solution_ids, tokenizer)
    canonical_critique_ids = strip_terminal_eos(critique_ids, tokenizer)
    solution_text = decode_exact(editable_solution_ids, tokenizer)
    critique_text = decode_exact(canonical_critique_ids, tokenizer)

    tags = (PREFIX_OPEN, PREFIX_CLOSE, PREFIX_PLUS_CONTINUATION_OPEN, PREFIX_PLUS_CONTINUATION_CLOSE)
    if any(critique_text.count(tag) != 1 for tag in tags):
        return _invalid("tag_count", solution_text, critique_text)

    prefix_open = critique_text.index(PREFIX_OPEN)
    prefix_content_start = prefix_open + len(PREFIX_OPEN)
    prefix_close = critique_text.index(PREFIX_CLOSE)
    joint_open = critique_text.index(PREFIX_PLUS_CONTINUATION_OPEN)
    joint_content_start = joint_open + len(PREFIX_PLUS_CONTINUATION_OPEN)
    joint_close = critique_text.index(PREFIX_PLUS_CONTINUATION_CLOSE)
    if not (prefix_content_start <= prefix_close < joint_open and joint_content_start <= joint_close):
        return _invalid("tag_order", solution_text, critique_text)
    if critique_text[prefix_close + len(PREFIX_CLOSE) : joint_open].strip():
        return _invalid("text_between_tags", solution_text, critique_text)
    if critique_text[joint_close + len(PREFIX_PLUS_CONTINUATION_CLOSE) :].strip():
        return _invalid("text_after_tags", solution_text, critique_text)

    analysis_text = critique_text[:prefix_open]

    prefix_text = critique_text[prefix_content_start:prefix_close]
    joint_text = critique_text[joint_content_start:joint_close]
    if not prefix_text or not prefix_text.strip():
        return _invalid("empty_prefix", solution_text, critique_text)
    if not joint_text or not joint_text.strip():
        return _invalid("empty_prefix_plus_new_continuation", solution_text, critique_text)
    if not joint_text.startswith(prefix_text):
        return _invalid("prefix_not_prefix_of_joint", solution_text, critique_text)
    new_continuation_text = joint_text[len(prefix_text) :]
    if not new_continuation_text or not new_continuation_text.strip():
        return _invalid("empty_new_continuation", solution_text, critique_text)
    if _FINAL_ANSWER_PATTERN.search(new_continuation_text):
        return _invalid("new_continuation_final_answer", solution_text, critique_text)

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if not prefix_ids or len(prefix_ids) > branch_max_tokens:
        return _invalid("prefix_token_cap", solution_text, critique_text)
    branch_start = solution_text.find(prefix_text)
    if branch_start >= 0:
        if solution_text.find(prefix_text, branch_start + 1) >= 0:
            return _invalid("prefix_not_unique", solution_text, critique_text)
        return _revision_at_branch_start(
            editable_solution_ids=editable_solution_ids,
            solution_text=solution_text,
            critique_text=critique_text,
            analysis_text=analysis_text,
            prefix_text=prefix_text,
            prefix_plus_new_continuation_text=joint_text,
            new_continuation_text=new_continuation_text,
            branch_start=branch_start,
            tokenizer=tokenizer,
            new_continuation_max_tokens=new_continuation_max_tokens,
        )

    normalized_location = _normalized_unique_prefix_location(prefix_text, solution_text)
    if normalized_location is None:
        return _invalid("prefix_not_found", solution_text, critique_text)
    semantic_start, formatting_gap_start = normalized_location
    if semantic_start < 0:
        return _invalid("prefix_not_unique", solution_text, critique_text)

    semantic_open_reason = branch_prefix_open_block_reason(solution_text[:semantic_start])
    latest_boundary = semantic_start
    if semantic_open_reason is not None:
        explicit_opener = _matching_explicit_block_opener(
            prefix_text,
            solution_text,
            formatting_gap_start=formatting_gap_start,
            semantic_start=semantic_start,
            open_block_reason=semantic_open_reason,
        )
        if explicit_opener is None:
            return _invalid(semantic_open_reason, solution_text, critique_text)
        latest_boundary = explicit_opener

    candidate_failures: list[str] = []
    for candidate_start in range(latest_boundary, formatting_gap_start - 1, -1):
        if branch_prefix_open_block_reason(solution_text[:candidate_start]) is not None:
            continue
        candidate = _revision_at_branch_start(
            editable_solution_ids=editable_solution_ids,
            solution_text=solution_text,
            critique_text=critique_text,
            analysis_text=analysis_text,
            prefix_text=prefix_text,
            prefix_plus_new_continuation_text=joint_text,
            new_continuation_text=new_continuation_text,
            branch_start=candidate_start,
            tokenizer=tokenizer,
            new_continuation_max_tokens=new_continuation_max_tokens,
        )
        if candidate.valid:
            return candidate
        candidate_failures.append(candidate.reason)
        if candidate.reason == "new_continuation_token_cap":
            return candidate

    failure_priority = (
        "new_continuation_retokenizes_prefix",
        "prefix_retokenizes_branch_prefix",
        "continuation_prefix_not_roundtrip_exact",
        "revision_not_roundtrip_exact",
        "branch_not_at_token_boundary",
        "branch_prefix_not_roundtrip_exact",
    )
    for reason in failure_priority:
        if reason in candidate_failures:
            return _invalid(reason, solution_text, critique_text)
    return _invalid("branch_not_at_token_boundary", solution_text, critique_text)


def validate_binary_reward_row(values: Iterable[float], *, tolerance: float) -> float:
    """Require a single strict binary outcome rather than thresholding shaped rewards."""

    row = [float(value) for value in values]
    if not row or not all(math.isfinite(value) for value in row):
        raise ValueError("branch-revision reward row must contain only finite values")
    normalized: list[float] = []
    for value in row:
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=tolerance):
            normalized.append(0.0)
        elif math.isclose(value, 1.0, rel_tol=0.0, abs_tol=tolerance):
            normalized.append(1.0)
        else:
            raise ValueError(f"branch-revision rewards must be binary 0/1, got {value!r}")
    if sum(normalized) not in {0.0, 1.0}:
        raise ValueError("branch-revision reward rows may contain at most one unit outcome")
    return float(sum(normalized))
