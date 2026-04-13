# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.rollout.hf_rollout import HFRollout


class _DummyGenerateModule(torch.nn.Module):
    def __init__(self, generated_tokens: torch.Tensor):
        super().__init__()
        self.generated_tokens = generated_tokens
        self.last_max_new_tokens = None

    def generate(
        self,
        input_ids,
        attention_mask,
        position_ids,
        do_sample,
        max_new_tokens,
        eos_token_id,
        pad_token_id,
        generation_config,
        output_scores,
        return_dict_in_generate,
        use_cache,
    ):
        del attention_mask, position_ids, do_sample, eos_token_id, pad_token_id
        del generation_config, output_scores, return_dict_in_generate, use_cache
        self.last_max_new_tokens = max_new_tokens
        generated = self.generated_tokens.to(input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        return SimpleNamespace(sequences=torch.cat([input_ids, generated], dim=1))


def test_hf_rollout_uses_validation_response_length_override_on_cpu():
    config = OmegaConf.create(
        {
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "prompt_length": 4,
            "response_length": 5,
            "do_sample": True,
            "n": 1,
            "val_kwargs": {
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 0.0,
                "n": 1,
                "do_sample": False,
                "response_length": 3,
            },
        }
    )
    module = _DummyGenerateModule(torch.tensor([7, 8], dtype=torch.long))
    rollout = HFRollout(module, config)
    prompts = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        },
        meta_info={
            "eos_token_id": 0,
            "pad_token_id": 0,
            "do_sample": False,
            "validate": True,
            "response_length": config.val_kwargs.response_length,
        },
    )

    outputs = rollout.generate_sequences(prompts)

    assert module.last_max_new_tokens == 3
    assert outputs.batch["responses"].shape == (1, 3)
    assert outputs.batch["input_ids"].shape == (1, 7)
    assert outputs.batch["responses"][0].tolist() == [7, 8, 0]
