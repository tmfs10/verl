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

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

BRANCH_OPEN = "<branch>"
BRANCH_CLOSE = "</branch>"
NEW_CONTINUATION_OPEN = "<new continuation>"
NEW_CONTINUATION_CLOSE = "</new continuation>"


@dataclass(frozen=True)
class ParsedBranchRevision:
    valid: bool
    reason: str
    solution_text: str
    critique_text: str
    branch_text: str = ""
    new_continuation_text: str = ""
    branch_start: int = -1
    revised_text: str = ""
    revised_prefix_ids: tuple[int, ...] = ()


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

    branch_text = critique_text[branch_content_start:branch_close]
    replacement_text = critique_text[new_content_start:new_close]
    if not branch_text or not branch_text.strip():
        return _invalid("empty_branch", solution_text, critique_text)
    if not replacement_text or not replacement_text.strip():
        return _invalid("empty_new_continuation", solution_text, critique_text)

    branch_ids = tokenizer.encode(branch_text, add_special_tokens=False)
    replacement_ids = tokenizer.encode(replacement_text, add_special_tokens=False)
    if not branch_ids or len(branch_ids) > branch_max_tokens:
        return _invalid("branch_token_cap", solution_text, critique_text)
    if not replacement_ids or len(replacement_ids) > new_continuation_max_tokens:
        return _invalid("new_continuation_token_cap", solution_text, critique_text)
    if solution_text.count(branch_text) != 1:
        return _invalid("branch_not_unique", solution_text, critique_text)

    branch_start = solution_text.index(branch_text)
    revised_text = solution_text[:branch_start] + replacement_text
    revised_prefix_ids = [int(token) for token in tokenizer.encode(revised_text, add_special_tokens=False)]
    if not revised_prefix_ids:
        return _invalid("empty_revised_prefix", solution_text, critique_text)
    if decode_exact(revised_prefix_ids, tokenizer) != revised_text:
        return _invalid("revision_not_roundtrip_exact", solution_text, critique_text)
    return ParsedBranchRevision(
        valid=True,
        reason="valid",
        solution_text=solution_text,
        critique_text=critique_text,
        branch_text=branch_text,
        new_continuation_text=replacement_text,
        branch_start=branch_start,
        revised_text=revised_text,
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
