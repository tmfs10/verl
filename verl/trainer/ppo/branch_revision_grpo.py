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

import torch

BRANCH_OPEN = "<branch>"
BRANCH_CLOSE = "</branch>"
NEW_CONTINUATION_OPEN = "<new continuation>"
NEW_CONTINUATION_CLOSE = "</new continuation>"
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


@dataclass(frozen=True)
class ParsedBranchRevision:
    valid: bool
    reason: str
    solution_text: str
    critique_text: str
    analysis_text: str = ""
    branch_text: str = ""
    new_continuation_text: str = ""
    branch_start: int = -1
    revised_text: str = ""
    branch_prefix_ids: tuple[int, ...] = ()
    new_continuation_ids: tuple[int, ...] = ()
    revised_prefix_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RolloutLogProbPrefix:
    rollout_id: str
    log_probs: torch.Tensor
    prefix_sums: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.prefix_sums.numel() - 1)


@dataclass(frozen=True)
class LearnabilityReference:
    window_size: int
    logprob_statistic: str
    window_scores: torch.Tensor
    window_weights: torch.Tensor
    window_rollout_ids: tuple[str, ...]
    window_starts: torch.Tensor
    eligible_rollouts: int

    @property
    def sampled_windows(self) -> int:
        return int(self.window_scores.numel())


@dataclass(frozen=True)
class LearnabilityScore:
    logprob_statistic: str
    seed_score: float
    percentile: float
    reward_weight: float
    accepted: bool
    eligible_rollouts: int
    sampled_windows: int


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
    """Normalize rollout log-probabilities and build cumulative sums once per iteration."""

    if len(rollout_ids) != len(rollout_log_probs):
        raise ValueError("rollout ids and log-probability rows must have equal lengths")
    result: list[RolloutLogProbPrefix] = []
    for rollout_id, values in zip(rollout_ids, rollout_log_probs, strict=True):
        tensor = normalize_log_probs_float32(values, device=device)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"rollout {rollout_id!r} contains non-finite log probabilities")
        prefix_values = tensor.to(dtype=torch.float64)
        prefix = torch.cat([torch.zeros(1, dtype=torch.float64, device=tensor.device), prefix_values.cumsum(dim=0)])
        result.append(RolloutLogProbPrefix(str(rollout_id), tensor, prefix))
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


def _stratified_window_starts(
    candidate_count: int,
    max_windows: int,
    *,
    seed: int,
    rollout_id: str,
    window_size: int,
) -> list[int]:
    if candidate_count <= max_windows:
        return list(range(candidate_count))
    starts: list[int] = []
    for slot in range(max_windows):
        lower = slot * candidate_count // max_windows
        upper = (slot + 1) * candidate_count // max_windows
        width = upper - lower
        digest = hashlib.blake2b(
            f"{seed}:{rollout_id}:{window_size}:{slot}".encode(),
            digest_size=8,
        ).digest()
        starts.append(lower + int.from_bytes(digest, "little") % width)
    return starts


def build_learnability_reference(
    prefixes: Sequence[RolloutLogProbPrefix],
    *,
    window_size: int,
    windows_per_rollout: int,
    seed: int,
    logprob_statistic: str = "mean",
) -> LearnabilityReference:
    """Sample length-matched windows, giving every eligible rollout equal mass."""

    if window_size <= 0 or windows_per_rollout <= 0:
        raise ValueError("learnability window sizes and counts must be positive")
    if logprob_statistic not in {"mean", "min"}:
        raise ValueError("learnability log-probability statistic must be mean or min")
    sampled: list[torch.Tensor] = []
    sampled_rollout_ids: list[str] = []
    sampled_starts: list[torch.Tensor] = []
    for rollout in prefixes:
        candidate_count = rollout.token_count - window_size + 1
        if candidate_count <= 0:
            continue
        starts = _stratified_window_starts(
            candidate_count,
            windows_per_rollout,
            seed=seed,
            rollout_id=rollout.rollout_id,
            window_size=window_size,
        )
        indices = torch.tensor(starts, dtype=torch.long, device=rollout.prefix_sums.device)
        if logprob_statistic == "mean":
            scores = (rollout.prefix_sums[indices + window_size] - rollout.prefix_sums[indices]) / float(window_size)
        else:
            offsets = torch.arange(window_size, dtype=torch.long, device=rollout.log_probs.device)
            scores = rollout.log_probs[indices[:, None] + offsets[None, :]].amin(dim=1).to(dtype=torch.float64)
        sampled.append(scores)
        sampled_rollout_ids.extend([rollout.rollout_id] * len(starts))
        sampled_starts.append(indices)
    if not sampled:
        device = prefixes[0].prefix_sums.device if prefixes else None
        empty = torch.empty(0, dtype=torch.float64, device=device)
        return LearnabilityReference(
            window_size=window_size,
            logprob_statistic=logprob_statistic,
            window_scores=empty,
            window_weights=empty.clone(),
            window_rollout_ids=(),
            window_starts=torch.empty(0, dtype=torch.long, device=device),
            eligible_rollouts=0,
        )
    eligible_rollouts = len(sampled)
    weights = [torch.full_like(values, 1.0 / (eligible_rollouts * values.numel())) for values in sampled]
    return LearnabilityReference(
        window_size=window_size,
        logprob_statistic=logprob_statistic,
        window_scores=torch.cat(sampled),
        window_weights=torch.cat(weights),
        window_rollout_ids=tuple(sampled_rollout_ids),
        window_starts=torch.cat(sampled_starts),
        eligible_rollouts=eligible_rollouts,
    )


def score_seed_learnability(
    seed_score: float,
    reference: LearnabilityReference,
    *,
    minimum_percentile: float,
    full_credit_percentile: float,
) -> LearnabilityScore:
    """Gate a replacement seed and linearly ramp reward credit by percentile."""

    if not math.isfinite(seed_score):
        raise ValueError("replacement seed log-probability score must be finite")
    if not 0.0 <= minimum_percentile < full_credit_percentile <= 1.0:
        raise ValueError("learnability percentiles must satisfy 0 <= minimum < full credit <= 1")
    if reference.sampled_windows == 0:
        percentile = 0.0
    else:
        percentile = float(reference.window_weights[reference.window_scores <= seed_score].sum().detach().cpu().item())
        percentile = min(max(percentile, 0.0), 1.0)
    accepted = reference.sampled_windows > 0 and percentile >= minimum_percentile
    reward_weight = min(
        max((percentile - minimum_percentile) / (full_credit_percentile - minimum_percentile), 0.0),
        1.0,
    )
    return LearnabilityScore(
        logprob_statistic=reference.logprob_statistic,
        seed_score=float(seed_score),
        percentile=percentile,
        reward_weight=reward_weight,
        accepted=accepted,
        eligible_rollouts=reference.eligible_rollouts,
        sampled_windows=reference.sampled_windows,
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
    """Parse an exact structured edit without normalizing generated text."""

    editable_solution_ids = strip_terminal_eos(solution_ids, tokenizer)
    canonical_critique_ids = strip_terminal_eos(critique_ids, tokenizer)
    solution_text = decode_exact(editable_solution_ids, tokenizer)
    critique_text = decode_exact(canonical_critique_ids, tokenizer)

    tags = (BRANCH_OPEN, BRANCH_CLOSE, NEW_CONTINUATION_OPEN, NEW_CONTINUATION_CLOSE)
    if any(critique_text.count(tag) != 1 for tag in tags):
        return _invalid("tag_count", solution_text, critique_text)

    branch_open = critique_text.index(BRANCH_OPEN)
    branch_content_start = branch_open + len(BRANCH_OPEN)
    branch_close = critique_text.index(BRANCH_CLOSE)
    new_open = critique_text.index(NEW_CONTINUATION_OPEN)
    new_content_start = new_open + len(NEW_CONTINUATION_OPEN)
    new_close = critique_text.index(NEW_CONTINUATION_CLOSE)
    if not (branch_content_start <= branch_close < new_open and new_content_start <= new_close):
        return _invalid("tag_order", solution_text, critique_text)
    if critique_text[branch_close + len(BRANCH_CLOSE) : new_open].strip():
        return _invalid("text_between_tags", solution_text, critique_text)
    if critique_text[new_close + len(NEW_CONTINUATION_CLOSE) :].strip():
        return _invalid("text_after_tags", solution_text, critique_text)

    analysis_text = critique_text[:branch_open]
    if not analysis_text.strip():
        return _invalid("empty_analysis", solution_text, critique_text)

    branch_text = critique_text[branch_content_start:branch_close]
    replacement_text = critique_text[new_content_start:new_close]
    if not branch_text or not branch_text.strip():
        return _invalid("empty_branch", solution_text, critique_text)
    if not replacement_text or not replacement_text.strip():
        return _invalid("empty_new_continuation", solution_text, critique_text)
    if _FINAL_ANSWER_PATTERN.search(replacement_text):
        return _invalid("new_continuation_final_answer", solution_text, critique_text)

    branch_ids = tokenizer.encode(branch_text, add_special_tokens=False)
    if not branch_ids or len(branch_ids) > branch_max_tokens:
        return _invalid("branch_token_cap", solution_text, critique_text)
    branch_start = solution_text.find(branch_text)
    if branch_start < 0:
        return _invalid("branch_not_found", solution_text, critique_text)
    if solution_text.find(branch_text, branch_start + 1) >= 0:
        return _invalid("branch_not_unique", solution_text, critique_text)

    branch_prefix_text = solution_text[:branch_start]
    branch_prefix_ids = [int(token) for token in tokenizer.encode(branch_prefix_text, add_special_tokens=False)]
    if decode_exact(branch_prefix_ids, tokenizer) != branch_prefix_text:
        return _invalid("branch_prefix_not_roundtrip_exact", solution_text, critique_text)
    if editable_solution_ids[: len(branch_prefix_ids)] != branch_prefix_ids:
        return _invalid("branch_not_at_token_boundary", solution_text, critique_text)
    revised_text = branch_prefix_text + replacement_text
    revised_prefix_ids = [int(token) for token in tokenizer.encode(revised_text, add_special_tokens=False)]
    if not revised_prefix_ids:
        return _invalid("empty_revised_prefix", solution_text, critique_text)
    if decode_exact(revised_prefix_ids, tokenizer) != revised_text:
        return _invalid("revision_not_roundtrip_exact", solution_text, critique_text)
    if revised_prefix_ids[: len(branch_prefix_ids)] != branch_prefix_ids:
        return _invalid("new_continuation_retokenizes_prefix", solution_text, critique_text)
    new_continuation_ids = revised_prefix_ids[len(branch_prefix_ids) :]
    if not new_continuation_ids or len(new_continuation_ids) > new_continuation_max_tokens:
        return _invalid("new_continuation_token_cap", solution_text, critique_text)
    return ParsedBranchRevision(
        valid=True,
        reason="valid",
        solution_text=solution_text,
        critique_text=critique_text,
        analysis_text=analysis_text,
        branch_text=branch_text,
        new_continuation_text=replacement_text,
        branch_start=branch_start,
        revised_text=revised_text,
        branch_prefix_ids=tuple(branch_prefix_ids),
        new_continuation_ids=tuple(new_continuation_ids),
        revised_prefix_ids=tuple(revised_prefix_ids),
    )


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
