# Copyright 2026
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

import torch

from verl.trainer.ppo.ray_trainer import _extract_response_tokens_and_logprobs, _validation_metric_section


class _FakeTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"tok_{token_id}" for token_id in token_ids]


def test_extract_response_tokens_and_logprobs_respects_response_mask():
    responses = torch.tensor([[11, 12, 0], [21, 22, 23]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.long)
    response_logprobs = torch.tensor([[-0.1, -0.2, -9.9], [-1.1, -9.9, -1.3]], dtype=torch.float32)

    tokens, logprobs = _extract_response_tokens_and_logprobs(
        _FakeTokenizer(),
        responses,
        response_mask,
        response_logprobs=response_logprobs,
    )

    assert tokens == [["tok_11", "tok_12"], ["tok_21", "tok_23"]]
    assert logprobs == [[-0.1, -0.2], [-1.1, -1.3]]


def test_extract_response_tokens_and_logprobs_handles_missing_logprobs():
    responses = torch.tensor([[31, 32]], dtype=torch.long)
    response_mask = torch.tensor([[1, 0]], dtype=torch.long)

    tokens, logprobs = _extract_response_tokens_and_logprobs(
        _FakeTokenizer(),
        responses,
        response_mask,
        response_logprobs=None,
    )

    assert tokens == [["tok_31"]]
    assert logprobs is None


def test_validation_metric_section_only_exports_selection_metrics():
    assert _validation_metric_section("acc", "acc", "mean@8", 8) is None
    assert _validation_metric_section("acc", "acc", "std@8", 8) is None
    assert _validation_metric_section("acc", "acc", "best@8/mean", 8) == "val-agg"
    assert _validation_metric_section("acc", "acc", "maj@8/mean", 8) == "val-agg"
    assert _validation_metric_section("acc", "acc", "worst@8/mean", 8) == "val-agg"
