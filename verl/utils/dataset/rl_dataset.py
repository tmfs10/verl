# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import copy
import hashlib
import logging
import os
import json
import math
import pandas as pd
import re
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Optional, Any

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.import_utils import load_extern_object
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)


def _to_hf_dataset(df: pd.DataFrame) -> datasets.Dataset:
    """
    Convert a pandas DataFrame to a 🤗 datasets.Dataset, dropping the pandas index.
    """
    return datasets.Dataset.from_pandas(df, preserve_index=False)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.solution_key = config.get("solution_key", "ground_truth_answer")
        self.dynamic_masked_solution = bool(config.get("dynamic_masked_solution", True))
        self.min_masked_fraction = config.get("min_masked_fraction", None)
        self.max_masked_fraction = config.get("max_masked_fraction", None)
        self.mask_seed = config.get("mask_seed", None)
        self.masked_solution_token = config.get("masked_solution_token", "<|fim_middle|>")
        self.masked_solution_placeholders = tuple(
            config.get("masked_solution_placeholders", ["{solution}", "{masked_solution}"])
        )
        self.masked_solution_selection_mode = config.get("masked_solution_selection_mode", "random_fraction")
        if self.masked_solution_selection_mode not in {"random_fraction", "reward_focus_tail"}:
            raise ValueError(
                f"masked_solution_selection_mode must be one of "
                f"['random_fraction', 'reward_focus_tail'], got {self.masked_solution_selection_mode!r}"
            )
        self.masked_solution_focus_tail_percent = config.get("masked_solution_focus_tail_percent", None)
        masked_solution_focus_min_tokens = config.get("masked_solution_focus_min_tokens", 1)
        self.masked_solution_focus_min_tokens = (
            1 if masked_solution_focus_min_tokens is None else int(masked_solution_focus_min_tokens)
        )
        self._masked_solution_positions_cache: dict[str, tuple[int, ...]] = {}
        self.masked_solution_token_id = (
            self._resolve_masked_solution_token_id(self.masked_solution_token)
            if self.dynamic_masked_solution
            else None
        )

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        datasets_list = []
        skip_acc_bounds = self.config.get("skip_acc_bounds", None)
        # === [CHANGED] iterate with dataset index so we can stamp "<dataset-index>-<line-number>" ===
        for dataset_idx, path in enumerate(self.data_files):
            # Load natively into HF (Arrow-backed, memory-mapped where possible)
            if path.endswith(".parquet"):
                ds = datasets.load_dataset("parquet", data_files=path, split="train")
            elif path.endswith(".json"):
                ds = datasets.load_dataset("json", data_files=path, split="train")
            elif path.endswith(".jsonl"):
                # Use "json" loader to keep nested types Arrow-native
                ds = datasets.load_dataset("json", data_files=path, split="train")
            else:
                raise ValueError(f"Unsupported file format: {path}")

            # Optional: accuracy filter fully in Arrow space
            if skip_acc_bounds is not None and "scores" in ds.column_names:
                lb, ub = skip_acc_bounds

                def _ok(scores):
                    if not scores:
                        return True
                    accs = [s.get("acc") for s in scores if isinstance(s, dict) and "acc" in s]
                    if not accs:
                        return True
                    mean_acc = sum(accs) / len(accs)
                    return lb <= mean_acc <= ub

                ds = ds.filter(lambda ex: _ok(ex["scores"]), num_proc=self.num_workers)

            # === [CHANGED] Always override extra_info["line_number"] with "<dataset_idx>-<line_number>" ===
            # Keep everything Arrow-native; tolerate odd existing types for extra_info
            def _ensure_extra_info_with_idx(ex, idx):
                v = ex.get("extra_info")
                if v is None:
                    v = {}
                elif isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except Exception:
                        v = {}
                elif not isinstance(v, dict):
                    try:
                        v = dict(v)
                    except Exception:
                        v = {}

                # Build "<dataset-index>-<line-number>" and store as JSON-encoded string (to match existing convention)
                combined = f"{dataset_idx}-{idx}"
                v["line_number"] = json.dumps(combined)  # e.g., "\"3-42\""

                ex["extra_info"] = v
                return ex

            ds = ds.map(
                _ensure_extra_info_with_idx,
                with_indices=True,
                num_proc=self.num_workers,
                desc="Stamping extra_info.line_number with dataset-index",
            )

            datasets_list.append(ds)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(datasets_list)
        print(f"dataset len: {len(self.dataframe)}")

        total = len(self.dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        prepared_doc = dict(doc)
                        prepared_doc[prompt_key] = self._prepare_prompt_messages(doc)
                        messages = self._build_messages(prepared_doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        # Keep explicit tokenization to avoid transformers version default changes.
                        apply_kwargs.pop("tokenize", None)
                        apply_kwargs.pop("return_dict", None)
                        apply_kwargs.pop("return_tensors", None)

                        tokenized_prompt = tokenizer.apply_chat_template(
                            self._prepare_prompt_messages(doc), add_generation_prompt=True, tokenize=True, **apply_kwargs
                        )
                        return len(normalize_token_ids(tokenized_prompt))
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", **image}
        - <video>: {"type": "video", **video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = example[self.prompt_key]
        # When concatenating image and video datasets, pop will return None for image or video sample
        images = example.pop(self.image_key, None) or []
        videos = example.pop(self.video_key, None) or []

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        content_list.append({"type": "image", "image": image})
                    elif isinstance(image, dict):
                        if "bytes" in image:
                            image["image"] = Image.open(BytesIO(image["bytes"]))
                        content_list.append({"type": "image", **image})
                    else:
                        raise TypeError(f"image must be dict or PIL.Image, unsupported image type: {type(image)}")
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def _sample_mask_seed(self, example: dict, item: Optional[int] = None) -> Optional[int]:
        if self.mask_seed is None:
            return None

        extra_info = example.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except Exception:
                extra_info = {}
        elif not isinstance(extra_info, dict):
            extra_info = {}

        sample_key = extra_info.get("line_number", extra_info.get("index", item))
        seed_material = f"{self.mask_seed}:{sample_key}"
        digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _resolve_masked_solution_token_id(self, token_text: str) -> int:
        token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"masked_solution_token {token_text!r} must map to exactly one tokenizer token, got ids {token_ids}"
            )
        token_id = int(token_ids[0])
        resolved_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        if resolved_token != token_text:
            raise ValueError(
                f"masked_solution_token {token_text!r} does not round-trip to a single tokenizer token. "
                f"Resolved token is {resolved_token!r} for id {token_id}."
            )
        return token_id

    def _mask_cache_key(self, example: dict, item: Optional[int] = None) -> str:
        extra_info = example.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except Exception:
                extra_info = {}
        elif not isinstance(extra_info, dict):
            extra_info = {}

        sample_key = extra_info.get("line_number", extra_info.get("index", item))
        return str(sample_key)

    def _has_masked_solution_placeholders(self, example: dict) -> bool:
        messages = example.get(self.prompt_key, [])
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if any(placeholder in content for placeholder in self.masked_solution_placeholders):
                return True
        return False

    @staticmethod
    def _masked_solution_sentinel(item: Optional[int] = None) -> str:
        return f"<<VERL_MASKED_SOLUTION_{item if item is not None else 'sample'}>>"

    def _build_prompt_template_with_sentinel(
        self,
        example: dict,
        *,
        item: Optional[int] = None,
    ) -> tuple[str, str, int]:
        solution_text = example.get(self.solution_key, None)
        if not isinstance(solution_text, str) or not solution_text:
            raise ValueError(f"Expected non-empty {self.solution_key!r} to build masked-solution prompt.")

        messages = copy.deepcopy(example[self.prompt_key])
        sentinel = self._masked_solution_sentinel(item)

        placeholder_hits = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if sentinel in content or sentinel in solution_text:
                raise ValueError(f"Sentinel collision detected for prompt masking: {sentinel!r}")
            for placeholder in self.masked_solution_placeholders:
                count = content.count(placeholder)
                if count:
                    content = content.replace(placeholder, sentinel)
                    placeholder_hits += count
            message["content"] = content

        if placeholder_hits == 0:
            raise ValueError("No masked-solution placeholders found in prompt template.")

        apply_kwargs = dict(**self.apply_chat_template_kwargs)
        if self.tool_schemas is not None:
            apply_kwargs["tools"] = self.tool_schemas
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)

        prompt_template = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        return prompt_template, solution_text, placeholder_hits

    def _get_masked_solution_positions(self, example: dict, item: Optional[int] = None) -> set[int]:
        solution_text = example.get(self.solution_key, None)
        if (
            not isinstance(solution_text, str)
            or not solution_text
            or not self.dynamic_masked_solution
            or not self.masked_solution_placeholders
            or not self._has_masked_solution_placeholders(example)
        ):
            return set()

        if self.masked_solution_selection_mode == "reward_focus_tail":
            provided_positions = example.get("masked_solution_focus_token_indices", None)
            if provided_positions is None:
                return set()
            return {int(pos) for pos in provided_positions}

        cache_key = self._mask_cache_key(example, item)
        if cache_key in self._masked_solution_positions_cache:
            return set(self._masked_solution_positions_cache[cache_key])

        solution_token_ids = self.tokenizer.encode(solution_text, add_special_tokens=False)
        if not solution_token_ids:
            self._masked_solution_positions_cache[cache_key] = tuple()
            return set()

        positions = self._sample_masked_token_positions(
            len(solution_token_ids),
            seed=self._sample_mask_seed(example, item),
        )

        normalized_positions = tuple(sorted(int(pos) for pos in positions))
        self._masked_solution_positions_cache[cache_key] = normalized_positions
        return set(normalized_positions)

    def _build_masked_solution_text(self, solution_text: str, *, seed: Optional[int]) -> str:
        if not self.dynamic_masked_solution or not isinstance(solution_text, str) or not solution_text:
            return solution_text
        if self.masked_solution_selection_mode == "reward_focus_tail":
            raise ValueError("_build_masked_solution_text requires example-aware mask positions in reward_focus_tail mode.")
        if self.min_masked_fraction is None or self.max_masked_fraction is None:
            return solution_text

        solution_token_ids = self.tokenizer.encode(solution_text, add_special_tokens=False)
        if not solution_token_ids:
            return solution_text

        low = float(self.min_masked_fraction)
        high = float(self.max_masked_fraction)
        if high < low:
            low, high = high, low
        low = min(max(low, 0.0), 1.0)
        high = min(max(high, 0.0), 1.0)

        rng = np.random.default_rng(seed)
        mask_fraction = float(rng.uniform(low, high))
        if mask_fraction <= 0.0:
            return solution_text

        token_count = len(solution_token_ids)
        mask_count = min(token_count, max(1, int(math.ceil(token_count * mask_fraction))))
        masked_positions = set(rng.choice(token_count, size=mask_count, replace=False).tolist())
        masked_token_ids = [
            self.masked_solution_token_id if idx in masked_positions else token_id
            for idx, token_id in enumerate(solution_token_ids)
        ]
        return self.tokenizer.decode(
            masked_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _sample_masked_token_positions(
        self,
        num_solution_tokens: int,
        *,
        seed: Optional[int],
    ) -> set[int]:
        if num_solution_tokens <= 0:
            return set()

        if self.min_masked_fraction is None or self.max_masked_fraction is None:
            return set()

        low = float(self.min_masked_fraction)
        high = float(self.max_masked_fraction)
        if high < low:
            low, high = high, low
        low = min(max(low, 0.0), 1.0)
        high = min(max(high, 0.0), 1.0)

        rng = np.random.default_rng(seed)
        mask_fraction = float(rng.uniform(low, high))
        if mask_fraction <= 0.0:
            return set()

        mask_count = min(num_solution_tokens, max(1, int(math.ceil(num_solution_tokens * mask_fraction))))
        return set(rng.choice(num_solution_tokens, size=mask_count, replace=False).tolist())

    def _build_prompt_ids_override(
        self,
        example: dict,
        item: Optional[int] = None,
        *,
        masked_positions: Optional[set[int]] = None,
    ) -> Optional[list[int]]:
        if (
            self.processor is not None
            or not self.dynamic_masked_solution
            or not self.masked_solution_placeholders
            or not self._has_masked_solution_placeholders(example)
        ):
            return None

        solution_text = example.get(self.solution_key, None)
        if not isinstance(solution_text, str) or not solution_text:
            return None
        prompt_template, _, placeholder_hits = self._build_prompt_template_with_sentinel(example, item=item)
        sentinel = self._masked_solution_sentinel(item)
        template_parts = prompt_template.split(sentinel)
        if len(template_parts) != placeholder_hits + 1:
            raise ValueError(
                f"Expected {placeholder_hits} masked-solution placeholder(s), found {len(template_parts) - 1} in prompt template."
            )

        full_prompt_parts: list[str] = []
        solution_spans: list[tuple[int, int]] = []
        cursor = 0
        for part_idx, part in enumerate(template_parts):
            full_prompt_parts.append(part)
            cursor += len(part)
            if part_idx < len(template_parts) - 1:
                start = cursor
                end = start + len(solution_text)
                solution_spans.append((start, end))
                full_prompt_parts.append(solution_text)
                cursor = end
        full_prompt_text = "".join(full_prompt_parts)

        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Dynamic token-level masked-solution prompts require a fast tokenizer with offsets.")

        tokenized = self.tokenizer(
            full_prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = normalize_token_ids(tokenized["input_ids"])
        offset_mapping = tokenized["offset_mapping"]

        solution_token_positions_by_span: list[list[int]] = []
        for span_start, span_end in solution_spans:
            span_positions: list[int] = []
            for token_idx, (start, end) in enumerate(offset_mapping):
                if start == end:
                    continue
                if start < span_end and end > span_start:
                    span_positions.append(token_idx)
            solution_token_positions_by_span.append(span_positions)

        if not solution_token_positions_by_span or not all(solution_token_positions_by_span):
            raise ValueError("Failed to locate any solution tokens inside the chat-templated prompt.")

        masked_positions_within_solution = (
            set(masked_positions) if masked_positions is not None else self._get_masked_solution_positions(example, item=item)
        )
        if not masked_positions_within_solution:
            return prompt_ids

        prompt_ids = list(prompt_ids)
        for span_positions in solution_token_positions_by_span:
            if max(masked_positions_within_solution) >= len(span_positions):
                raise ValueError(
                    "Masked solution positions exceed the tokenized solution span length in the chat-templated prompt."
                )
            for rel_idx in masked_positions_within_solution:
                prompt_ids[span_positions[rel_idx]] = self.masked_solution_token_id
        return prompt_ids

    def _prepare_prompt_messages(
        self,
        example: dict,
        item: Optional[int] = None,
        *,
        masked_positions: Optional[set[int]] = None,
    ):
        messages = copy.deepcopy(example[self.prompt_key])
        if (
            not self.dynamic_masked_solution
            or not self.masked_solution_placeholders
            or not self._has_masked_solution_placeholders(example)
        ):
            return messages

        solution_text = example.get(self.solution_key, None)
        if not isinstance(solution_text, str) or not solution_text:
            return messages

        solution_token_ids = self.tokenizer.encode(solution_text, add_special_tokens=False)
        masked_positions = (
            set(masked_positions) if masked_positions is not None else self._get_masked_solution_positions(example, item=item)
        )
        if masked_positions:
            masked_token_ids = [
                self.masked_solution_token_id if idx in masked_positions else token_id
                for idx, token_id in enumerate(solution_token_ids)
            ]
            masked_solution = self.tokenizer.decode(
                masked_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            if self.masked_solution_selection_mode == "reward_focus_tail":
                # In reward_focus_tail mode, prompt masking is injected later by the trainer after it
                # computes focus positions from the live actor. Dataset-time prompt-length filtering
                # still needs a concrete prompt, so use the unmasked solution text here.
                masked_solution = solution_text
            else:
                masked_solution = self._build_masked_solution_text(
                    solution_text, seed=self._sample_mask_seed(example, item)
                )
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            for placeholder in self.masked_solution_placeholders:
                content = content.replace(placeholder, masked_solution)
            message["content"] = content
        return messages

    def materialize_masked_solution_prompt(
        self,
        example: dict,
        *,
        masked_positions: set[int],
        item: Optional[int] = None,
    ) -> tuple[list[dict], Optional[list[int]]]:
        normalized_positions = {int(pos) for pos in masked_positions}
        prepared_messages = self._prepare_prompt_messages(example, item=item, masked_positions=normalized_positions)
        prompt_ids_override = self._build_prompt_ids_override(
            example,
            item=item,
            masked_positions=normalized_positions,
        )
        return prepared_messages, prompt_ids_override

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item]
        if self.masked_solution_selection_mode == "reward_focus_tail":
            row_dict["raw_prompt"] = self._build_messages(row_dict)
        else:
            masked_solution_positions = self._get_masked_solution_positions(row_dict, item=item)
            prompt_ids_override = self._build_prompt_ids_override(row_dict, item=item)
            row_dict[self.prompt_key] = self._prepare_prompt_messages(row_dict, item=item)
            row_dict["raw_prompt"] = self._build_messages(row_dict)
            if prompt_ids_override is not None:
                row_dict["prompt_ids_override"] = prompt_ids_override

        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        if self.apply_chat_template_kwargs:
            row_dict["chat_template_kwargs"] = copy.deepcopy(dict(self.apply_chat_template_kwargs))
        return row_dict

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

    def split(self, num_splits: int):
        """
        split the dataset into num_splits sub-datasets
        Args:
            num_splits: specified number of splits
        Returns:
            List[RLHFDataset]: list of RLHFDataset splits
        Raises:
            ValueError: if num_splits is not a positive integer
        """
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        if not hasattr(self, "dataframe"):
            raise AttributeError(
                "dataframe not found in RLHFDataset\n"
                "reason: _read_files_and_tokenize() not called or Parquet file loading failed"
            )
        if self.dataframe is None:
            raise ValueError("RLHFDataset dataframe 为 None!")

        total_samples = len(self.dataframe)
        print(f"total_samples: {total_samples}")
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")

        # Calculate effective sample count after dropping remainders if needed
        if total_samples % num_splits != 0:
            total_samples = total_samples - (total_samples % num_splits)
            logging.warning(f"Dropping {len(self.dataframe) % num_splits} samples, effective samples: {total_samples}")

        split_size = total_samples // num_splits
        splits = []

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else total_samples

            split_dataframe = self.dataframe.select(range(start_idx, end_idx))

            split_dataset = RLHFDataset(
                data_files=self.data_files,
                tokenizer=self.tokenizer,
                config=self.config,
                processor=self.processor,
                max_samples=self.max_samples,
            )
            split_dataset.dataframe = split_dataframe
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files

            splits.append(split_dataset)

        return splits


def get_dataset_class(data_config: DictConfig):
    """Get RLHF dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls
