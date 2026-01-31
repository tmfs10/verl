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

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from typing import Any, Callable, Optional

import torch
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.dataset.rl_dataset import collate_fn_pad_to_batch_max
from verl.utils.fs import local_mkdir_safe
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask


def _select_kwargs_for_callable(fn: Any, provided: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(provided)

    accepted = {name for name in sig.parameters.keys()}
    return {k: v for k, v in provided.items() if k in accepted}


def _load_custom_fn(config_section: dict[str, Any], module_key: str) -> Optional[Callable[..., Any]]:
    file_path = config_section.get("path")
    if not file_path:
        return None

    function_name = config_section.get("name")
    if not function_name:
        raise ValueError(f"{module_key}.name must be set when {module_key}.path is provided")

    module = sys.modules.get(module_key, None)
    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Custom function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location(module_key, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec from '{file_path}'")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in '{module.__file__}'")

    return getattr(module, function_name)


def _normalize_prompt_update_result(result: Any) -> tuple[Optional[str], dict[str, Any]]:
    if result is None:
        return None, {}
    if isinstance(result, str):
        return result, {}
    if isinstance(result, dict):
        prompt_text = result.get("prompt") or result.get("prompt_text")
        extra = {k: v for k, v in result.items() if k not in {"prompt", "prompt_text"}}
        return prompt_text, extra
    if isinstance(result, tuple | list):
        if len(result) == 0:
            return None, {}
        if len(result) == 1:
            return result[0], {}
        return result[0], result[1] if isinstance(result[1], dict) else {}
    return str(result), {}


class RayBeamGradProjectionTrainer(RayPPOTrainer):
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            train_sampler = SequentialSampler(self.train_dataset)
        elif not isinstance(train_sampler, SequentialSampler):
            print("Warning: overriding non-sequential sampler with SequentialSampler for stable indices.")
            train_sampler = SequentialSampler(self.train_dataset)
        if collate_fn is None:
            collate_fn = default_collate_fn
        self._collate_fn = collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        sp_size = self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        dp_size = max(1, world_size // sp_size)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=dp_size,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        class _EmptyDataset:
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")

        self.val_dataset = _EmptyDataset()
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {}
        if self.config.trainer.get("ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if self.config.global_profiler.get("steps") is not None:
            wg_kwargs["profile_steps"] = self.config.global_profiler.steps
            if self.config.global_profiler.get("tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = self.config.global_profiler.global_tool_config.nsys.worker_nsight_options
        wg_kwargs["device_name"] = self.device_name

        for pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _format_prompt_text(self, prompt: Any, prompt_is_formatted: bool) -> str:
        if prompt_is_formatted:
            return str(prompt)
        if isinstance(prompt, list | tuple) and prompt and isinstance(prompt[0], dict):
            messages = prompt
            if hasattr(self.tokenizer, "apply_chat_template"):
                kwargs = self.config.data.get("apply_chat_template_kwargs", {})
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **kwargs
                )
            return str(prompt)
        if hasattr(self.train_dataset, "_build_prompt_text"):
            try:
                return self.train_dataset._build_prompt_text(str(prompt))
            except Exception:
                pass
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs = self.config.data.get("apply_chat_template_kwargs", {})
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        return str(prompt)

    def _encode_prompt(self, prompt: Any, prompt_is_formatted: bool) -> torch.Tensor:
        prompt_text = self._format_prompt_text(prompt, prompt_is_formatted)
        tokenized = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        return tokenized["input_ids"][0]

    def _encode_response(self, response: Any, append_eos: bool) -> torch.Tensor:
        if isinstance(response, torch.Tensor):
            return response
        if isinstance(response, list | tuple) and response and isinstance(response[0], int):
            return torch.tensor(response, dtype=torch.long)
        response_text = str(response)
        if append_eos and self.tokenizer.eos_token is not None:
            response_text = f"{response_text}{self.tokenizer.eos_token}"
        tokenized = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False)
        return tokenized["input_ids"][0]

    def _build_prompt_batch(self, prompt_ids_list: list[torch.Tensor]) -> DataProto:
        if not prompt_ids_list:
            raise ValueError("prompt_ids_list must be non-empty")
        max_len = max(ids.numel() for ids in prompt_ids_list)
        input_ids = []
        attention_mask = []
        for ids in prompt_ids_list:
            ids = ids.to(torch.long).cpu()
            pad_len = max_len - ids.numel()
            if pad_len > 0:
                pad = torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)
                ids_padded = torch.cat([pad, ids], dim=0)
                mask = torch.cat(
                    [torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.numel(), dtype=torch.long)], dim=0
                )
            else:
                ids_padded = ids
                mask = torch.ones(ids.numel(), dtype=torch.long)
            input_ids.append(ids_padded)
            attention_mask.append(mask)
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)
        return DataProto.from_dict(
            tensors={"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        )

    def _build_projection_batch(
        self, prompt_ids: torch.Tensor, response_ids_list: list[torch.Tensor]
    ) -> DataProto:
        data_list = []
        for response_ids in response_ids_list:
            if response_ids.numel() == 0:
                continue
            response_ids = response_ids.to(torch.long).cpu()
            input_ids = torch.cat([prompt_ids, response_ids], dim=0)
            response_mask = torch.ones(response_ids.numel(), dtype=torch.long)
            data_list.append({"input_ids": input_ids, "responses": response_ids, "response_mask": response_mask})
        if not data_list:
            raise ValueError("No valid responses to project")
        batch_dict = collate_fn_pad_to_batch_max(data_list, pad_token_id=self.pad_token_id)
        return DataProto.from_single_dict(batch_dict)

    def _compute_projections(self, batch: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_actor_dp_size"):
            dp_rank_mapping = self.actor_rollout_wg._query_dispatch_info("actor")
            self._actor_dp_size = max(dp_rank_mapping) + 1
        original_len = len(batch)
        if original_len % self._actor_dp_size != 0:
            padding_size = self._actor_dp_size - (original_len % self._actor_dp_size)
            batch.padding(padding_size=padding_size, padding_candidate="last")

        batch.meta_info.update(
            {
                "temperature": self.config.actor_rollout_ref.rollout.get("temperature", 1.0),
                "use_dynamic_bsz": False,
                "micro_batch_size": 1,
                "rademacher_k": self.rademacher_k,
                "rademacher_seed": self.rademacher_seed,
                "rademacher_chunk_size": self.rademacher_chunk_size,
                "rademacher_countsketch": self.config.trainer.get("rademacher_countsketch", False),
                "countsketch_t": self.config.trainer.get("countsketch_t", 2),
            }
        )

        output = self.actor_rollout_wg.update_actor(batch)
        projection = output.batch["projection"][:original_len].cpu()
        projection_normalized = output.batch["projection_normalized"][:original_len].cpu()
        return projection, projection_normalized

    def _project_responses(
        self,
        prompt_ids: torch.Tensor,
        response_ids_list: list[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size is None or batch_size <= 0:
            batch_size = len(response_ids_list)
        proj_list = []
        proj_norm_list = []
        for start in range(0, len(response_ids_list), batch_size):
            chunk = response_ids_list[start : start + batch_size]
            batch = self._build_projection_batch(prompt_ids, chunk)
            proj, proj_norm = self._compute_projections(batch)
            proj_list.append(proj)
            proj_norm_list.append(proj_norm)
        return torch.cat(proj_list, dim=0), torch.cat(proj_norm_list, dim=0)

    def fit(self):
        beam_cfg = self.config.trainer.get("beam_search", {})
        n_generations = int(
            beam_cfg.get(
            "n_generations", self.config.trainer.get("n_generations", self.config.actor_rollout_ref.rollout.n)
        )
        )
        beam_size = int(beam_cfg.get("beam_size", self.config.trainer.get("beam_size", 1)))
        inner_iterations = int(beam_cfg.get("inner_iterations", self.config.trainer.get("inner_iterations", 1)))
        new_prompt_key = beam_cfg.get("new_prompt_key", self.config.trainer.get("new_prompt_key", "new_prompt"))
        target_output_key = beam_cfg.get(
            "target_output_key", self.config.trainer.get("target_output_key", "target_output")
        )
        projection_batch_size = beam_cfg.get(
            "projection_batch_size", self.config.trainer.get("projection_batch_size", None)
        )
        if projection_batch_size is not None:
            projection_batch_size = int(projection_batch_size)
        use_normalized = beam_cfg.get(
            "projection_use_normalized", self.config.trainer.get("projection_use_normalized", False)
        )
        new_prompt_is_formatted = beam_cfg.get(
            "new_prompt_is_formatted", self.config.trainer.get("new_prompt_is_formatted", False)
        )
        append_eos_to_target = beam_cfg.get(
            "append_eos_to_target", self.config.trainer.get("append_eos_to_target", False)
        )

        self.rademacher_k = self.config.trainer.get("rademacher_k", None)
        if self.rademacher_k is None:
            raise ValueError("trainer.rademacher_k must be set for gradient projection")
        self.rademacher_seed = int(self.config.trainer.get("rademacher_seed", 0))
        self.rademacher_chunk_size = int(self.config.trainer.get("rademacher_chunk_size", 1_000_000))

        if n_generations <= 0:
            raise ValueError("n_generations must be > 0")
        if beam_size <= 0:
            raise ValueError("beam_size must be > 0")
        if inner_iterations <= 0:
            raise ValueError("inner_iterations must be > 0")

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

        output_dir = beam_cfg.get("output_dir", self.config.trainer.default_local_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        local_mkdir_safe(output_dir)

        prompt_update_cfg = beam_cfg.get("prompt_update_function", self.config.trainer.get("prompt_update_function", {}))
        prompt_update_fn = _load_custom_fn(prompt_update_cfg, "beam_prompt_update")
        prompt_update_kwargs = dict(prompt_update_cfg.get("prompt_kwargs", {}))
        if prompt_update_fn is None:
            print("Warning: prompt_update_function not set; prompts will be reused as-is.")

        eos_token_id = self.tokenizer.eos_token_id

        for batch_dict in self.train_dataloader:
            batch_size = batch_dict["input_ids"].shape[0]
            samples = []
            for i in range(batch_size):
                sample = {}
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        sample[key] = val[i]
                    else:
                        sample[key] = val[i]
                idx = sample.get("idx", i)
                if hasattr(idx, "item"):
                    idx = int(idx.item())
                else:
                    idx = int(idx)
                sample["idx"] = idx
                samples.append(sample)

            for sample in samples:
                if new_prompt_key not in sample:
                    raise KeyError(f"Missing '{new_prompt_key}' in data item")
                if target_output_key not in sample:
                    raise KeyError(f"Missing '{target_output_key}' in data item")

                new_prompt = sample[new_prompt_key]
                target_output = sample[target_output_key]

                input_ids = sample["input_ids"]
                attention_mask = sample.get("attention_mask", None)
                if attention_mask is not None:
                    prompt_ids = input_ids[attention_mask.bool()]
                else:
                    prompt_ids = input_ids

                old_prompt_text = None
                for key in ("full_prompts", "raw_prompt", "prompt"):
                    if key in sample:
                        old_prompt_text = sample[key]
                        break
                if old_prompt_text is None:
                    old_prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

                sample_data = {}
                for key, val in sample.items():
                    if isinstance(val, torch.Tensor):
                        sample_data[key] = val.detach().cpu()
                    else:
                        sample_data[key] = val

                new_prompt_ids = self._encode_prompt(new_prompt, prompt_is_formatted=new_prompt_is_formatted)
                target_ids = self._encode_response(target_output, append_eos=append_eos_to_target)
                target_batch = self._build_projection_batch(new_prompt_ids, [target_ids])
                target_proj, target_proj_norm = self._compute_projections(target_batch)
                target_vec = target_proj_norm[0] if use_normalized else target_proj[0]

                beam_prompts = [
                    {
                        "prompt_text": old_prompt_text,
                        "prompt_ids": prompt_ids,
                        "last_response_text": None,
                        "last_response_ids": None,
                        "distance": None,
                    }
                ]

                for iteration in range(inner_iterations):
                    prompt_ids_list = [bp["prompt_ids"] for bp in beam_prompts]
                    gen_batch = self._build_prompt_batch(prompt_ids_list)
                    gen_batch = gen_batch.repeat(repeat_times=n_generations, interleave=True)
                    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                    gen_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                    gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)

                    responses = gen_output.batch["responses"]
                    response_len = responses.size(1)
                    response_mask = gen_output.batch["response_mask"] if "response_mask" in gen_output.batch.keys() else None
                    if response_mask is None:
                        if "attention_mask" in gen_output.batch.keys():
                            response_mask = gen_output.batch["attention_mask"][:, -response_len:]
                        else:
                            response_mask = get_response_mask(responses, eos_token=eos_token_id, dtype=torch.long)

                    candidates = []
                    for i in range(len(gen_output)):
                        parent_idx = i // n_generations
                        resp_ids = responses[i]
                        mask = response_mask[i]
                        valid_len = int(mask.sum().item())
                        if valid_len <= 0:
                            continue
                        resp_ids = resp_ids[:valid_len].cpu()
                        resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                        candidates.append(
                            {
                                "parent_idx": parent_idx,
                                "response_ids": resp_ids,
                                "response_text": resp_text,
                            }
                        )

                    if not candidates:
                        print(f"Warning: no valid candidates for idx={sample['idx']} at iteration {iteration}")
                        break

                    response_ids_list = [c["response_ids"] for c in candidates]
                    proj, proj_norm = self._project_responses(
                        new_prompt_ids, response_ids_list, batch_size=projection_batch_size
                    )
                    for cand, p, pn in zip(candidates, proj, proj_norm, strict=True):
                        cand["projection"] = p
                        cand["projection_normalized"] = pn
                        vec = pn if use_normalized else p
                        cand["distance"] = torch.norm(vec - target_vec).item()

                    candidates.sort(key=lambda c: c["distance"])
                    selected = candidates[:beam_size]

                    next_beam = []
                    for rank, cand in enumerate(selected):
                        parent = beam_prompts[cand["parent_idx"]]
                        update_kwargs = {
                            "old_prompt": old_prompt_text,
                            "new_prompt": new_prompt,
                            "current_prompt": parent["prompt_text"],
                            "response_text": cand["response_text"],
                            "response_ids": cand["response_ids"],
                            "iteration": iteration,
                            "beam_rank": rank,
                            "data_item": sample_data,
                        }
                        update_kwargs.update(prompt_update_kwargs)

                        if prompt_update_fn is None:
                            prompt_text = parent["prompt_text"]
                            extra = {}
                        else:
                            filtered_kwargs = _select_kwargs_for_callable(prompt_update_fn, update_kwargs)
                            result = prompt_update_fn(**filtered_kwargs)
                            prompt_text, extra = _normalize_prompt_update_result(result)
                        if prompt_text is None:
                            continue

                        next_prompt_ids = self._encode_prompt(prompt_text, prompt_is_formatted=new_prompt_is_formatted)
                        next_beam.append(
                            {
                                "prompt_text": prompt_text,
                                "prompt_ids": next_prompt_ids,
                                "last_response_text": cand["response_text"],
                                "last_response_ids": cand["response_ids"],
                                "distance": cand["distance"],
                                "projection": cand["projection"],
                                "projection_normalized": cand["projection_normalized"],
                                "extra": extra,
                            }
                        )

                    if not next_beam:
                        print(f"Warning: prompt update produced empty beam for idx={sample['idx']}")
                        break
                    beam_prompts = next_beam

                result = {
                    "idx": sample["idx"],
                    "old_prompt": old_prompt_text,
                    "new_prompt": new_prompt,
                    "target_output": target_output,
                    "beam_size": beam_size,
                    "n_generations": n_generations,
                    "inner_iterations": inner_iterations,
                    "use_normalized_projection": use_normalized,
                    "beam": [],
                    "data_item": sample_data,
                }

                for entry in beam_prompts:
                    result["beam"].append(
                        {
                            "response_text": entry.get("last_response_text"),
                            "response_ids": entry.get("last_response_ids"),
                            "distance": entry.get("distance"),
                            "projection": entry.get("projection"),
                            "projection_normalized": entry.get("projection_normalized"),
                            "next_prompt": entry.get("prompt_text"),
                            "extra": entry.get("extra", {}),
                        }
                    )

                output_path = os.path.join(output_dir, f"beam_results_{sample['idx']}.pt")
                torch.save(result, output_path)
