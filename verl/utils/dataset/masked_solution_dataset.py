import copy
import os
import random
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_to_local


class MaskedSolutionChatDataset(Dataset):
    """
    Dataset that replaces a {solution} placeholder in the user message with a
    randomly masked version of the provided solution text, then applies the
    tokenizer chat template.

    Notes:
    - Ignores multimodal (image/video) content.
    - Reads from Parquet or JSON/JSONL using 🤗 datasets for efficiency.

    Config keys:
    - prompt_key: field name containing chat messages (default: "prompt").
    - solution_key / solution_field_name: field name containing the ground-truth solution string (default: "solution").
    - line_mask_field_name: field name to emit the per-line mask (default: "solution_line_mask").
    - min_masked_fraction / max_masked_fraction: range of line-mask fractions (default: both 0.3).
    - mask_seed: optional seed to make masking deterministic per item (still randomized within the range).
    - max_prompt_length: token budget after chat template (default: 1024).
    - truncation: one of {"error","left","right","middle"} (default: "error").
    - filter_overlong_prompts: whether to pre-filter by length (default: True).
    - apply_chat_template_kwargs: extra kwargs forwarded to apply_chat_template.
    - cache_dir: local cache location for source files.
    - use_shm: whether to stage in shared memory if available.
    - return_raw_chat: include pre-template messages in output if True.
    - return_full_prompt: include the rendered prompt string if True.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        # Allow solution_field_name alias for clarity with reward manager
        self.solution_key = config.get("solution_key", config.get("solution_field_name", "solution"))
        self.line_mask_field_name = config.get("line_mask_field_name", "solution_line_mask")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # Allow legacy mask_fraction for back-compat; otherwise use min/max range.
        default_mask = float(config.get("mask_fraction", 0.3))
        self.min_masked_fraction = float(config.get("min_masked_fraction", default_mask))
        self.max_masked_fraction = float(config.get("max_masked_fraction", self.min_masked_fraction))
        print('DDD', self.min_masked_fraction, self.max_masked_fraction)
        self.mask_seed = config.get("mask_seed", None)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.use_shm = config.get("use_shm", False)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        self._download()
        self._read_files()

    def _download(self):
        for i, path in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=path, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files(self):
        datasets_list = []
        for path in self.data_files:
            if path.endswith(".parquet"):
                ds = datasets.load_dataset("parquet", data_files=path, split="train")
            elif path.endswith(".jsonl") or path.endswith(".json"):
                ds = datasets.load_dataset("json", data_files=path, split="train")
            else:
                continue
            datasets_list.append(ds)

        if not datasets_list:
            # Empty dataset placeholder
            self.dataframe = datasets.Dataset.from_dict({})
        else:
            self.dataframe = datasets.concatenate_datasets(datasets_list)

        if self.filter_overlong_prompts:
            self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _replace_solution_placeholder(self, messages: list, solution_text: str) -> list:
        """Replace {solution} in user messages with provided text (no masking)."""
        msgs = copy.deepcopy(messages)
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and "{solution}" in content:
                    msg["content"] = content.replace("{solution}", solution_text)
        return msgs

    def _build_messages(self, example: dict, item_idx: Optional[int] = None) -> tuple[list, str, list[int], str]:
        """
        Build chat messages with masked solution injected and return masking metadata.

        Returns:
            messages: chat messages with {solution} replaced by masked text
            masked_solution: the masked solution string
            mask_lines: list[int] of length len(solution.split("\n")), 1 for masked line
            solution_text: the original unmasked solution
        """
        messages = copy.deepcopy(example[self.prompt_key])
        solution_text = example.get(self.solution_key, "")
        if not isinstance(solution_text, str):
            solution_text = str(solution_text)

        masked_solution, mask_lines = self._mask_solution(solution_text, item_idx=item_idx)

        # inject masked solution into user message(s)
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and "{solution}" in content:
                msg["content"] = content.replace("{solution}", masked_solution)
        return messages, masked_solution, mask_lines, solution_text

    def _mask_solution(self, solution: str, item_idx: Optional[int] = None) -> tuple[str, list[int]]:
        # Split with explicit "\n" to match reconstruction
        lines = solution.split("\n")
        # Choose a per-sample fraction in [min,max]
        rng = random.Random()
        if self.mask_seed is not None:
            try:
                base = int(self.mask_seed)
            except Exception:
                base = hash(str(self.mask_seed)) & 0xFFFFFFFF
            if item_idx is not None:
                base = (base + int(item_idx)) & 0xFFFFFFFF
            rng.seed(base)

        frac = rng.uniform(self.min_masked_fraction, self.max_masked_fraction)

        if frac <= 0:
            return solution, [0] * len(lines)
        if frac >= 1:
            return "\n".join("# <masked out>" for _ in lines), [1] * len(lines)

        masked_lines: list[str] = []
        mask_line_flags: list[int] = []
        for line in lines:
            if rng.random() < frac:
                masked_lines.append("# <masked out>")
                mask_line_flags.append(1)
            else:
                masked_lines.append(line)
                mask_line_flags.append(0)
        return "\n".join(masked_lines), mask_line_flags

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset):
        tokenizer = self.tokenizer

        def doc2len(doc) -> int:
            # Substitute unmasked solution for a stable length estimate
            messages = self._replace_solution_placeholder(doc[self.prompt_key], str(doc.get(self.solution_key, "")))
            raw_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            encoded = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            return int(encoded["input_ids"].shape[-1])

        dataframe = dataframe.filter(
            lambda d: doc2len(d) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        return dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe[idx]
        messages, masked_solution, mask_lines, solution_text = self._build_messages(row, item_idx=idx)

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        # Build output record
        out = dict(row) if isinstance(row, dict) else {}
        out["input_ids"] = input_ids[0]
        out["attention_mask"] = attention_mask[0]
        out["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )
        out["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            out["raw_prompt"] = messages
        if self.return_full_prompt:
            out["full_prompts"] = raw_prompt

        # Expose solution and masking metadata to downstream reward computation
        out[self.solution_key] = solution_text
        out["masked_solution"] = masked_solution
        out[self.line_mask_field_name] = mask_lines  # list[int] aligned to solution.split("\n")

        return out
