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

import torch

from verl.utils.dataset.rl_dataset import RLHFPromptResponseDataset
from verl.utils.model import compute_position_id_with_mask


class RawPromptResponseDataset(RLHFPromptResponseDataset):
    """RLHF prompt/response dataset without chat template or EOS appending."""

    def _build_prompt_text(self, prompt: str) -> str:
        return prompt

    def maybe_filter_out_long_total_length(self, dataframe=None):
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        response_key = self.response_key
        max_length = self.max_length

        def doc2len(doc) -> int:
            prompt = doc[prompt_key]
            response = doc[response_key]
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            return len(prompt_ids) + len(response_ids)

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= max_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts+responses longer than {max_length} tokens",
        )
        print(f"filter total-length dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        prompt = row_dict[self.prompt_key]
        response = row_dict[self.response_key]
        idx = row_dict.get("idx", item)

        prompt_text = prompt
        response_text = response

        prompt_ids_output = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"]
        prompt_attention_mask = prompt_ids_output["attention_mask"]

        response_ids_output = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"]
        response_attention_mask = response_ids_output["attention_mask"]

        # No per-sample padding/truncation; collate will pad to batch max.
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)[0]
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)[0]
        position_ids = compute_position_id_with_mask(attention_mask)

        row = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_ids[0],
            "response_mask": response_attention_mask[0],
            "idx": idx,
        }

        if self.return_raw_chat:
            row["raw_prompt"] = prompt
        if self.return_full_prompt:
            row["full_prompts"] = prompt_text

        return row
