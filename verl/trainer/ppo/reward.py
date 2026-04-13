# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import asyncio
import inspect
import multiprocessing
import warnings
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

import ray
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from verl.experimental.reward_loop.reward_manager.base import RawRewardFn, RewardManagerBase
    from verl.trainer.config.config import ModuleConfig
    from verl.workers.config.reward import RewardManagerConfig


def _select_kwargs_for_callable(fn: Any, provided: dict[str, Any]) -> dict[str, Any]:
    """Return only kwargs accepted by callable `fn`.

    - If `fn` accepts var-keywords (i.e., **kwargs), return all provided.
    - Otherwise, filter to names present in the signature.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins or callables without signature
        return {}

    # If accepts **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(provided)

    accepted = {name for name in sig.parameters.keys()}
    return {k: v for k, v in provided.items() if k in accepted}


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_manager(config: DictConfig, tokenizer: Any, **reward_kwargs: Any) -> RewardManagerBase:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    reward_manager_cfg: RewardManagerConfig = config.reward.reward_manager
    reward_manager_cls: type[RewardManagerBase]
    if reward_manager_cfg.source == "register":
        from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls

        try:
            reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)
        except ValueError:
            from verl.workers.reward_manager.registry import get_reward_manager_cls as get_legacy_reward_manager_cls

            reward_manager_cls = get_legacy_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_manager_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        reward_manager_cls_name = reward_manager_cfg.name
        reward_manager_cls = cast(
            "type[RewardManagerBase]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cls_name),
        )

    if compute_score is None:
        sandbox_config = config.reward.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    init_kwargs = dict(reward_kwargs)
    try:
        init_sig = inspect.signature(reward_manager_cls.__init__)
    except (TypeError, ValueError):
        init_sig = None

    if init_sig is not None and "num_examine" in init_sig.parameters and "num_examine" not in init_kwargs:
        init_kwargs["num_examine"] = 0

    ctor_kwargs = {
        "config": config,
        "tokenizer": tokenizer,
        "compute_score": final_compute_score,
        **init_kwargs,
    }
    if init_sig is not None:
        ctor_kwargs = _select_kwargs_for_callable(reward_manager_cls.__init__, ctor_kwargs)

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(**ctor_kwargs)

def extract_reward(batch: DataProto):
    """
    Extract reward tensor and extra info from batch data.
    """
    reward_tensor = batch.batch["rm_scores"]
    reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
    reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
    return reward_tensor, reward_extra_infos_dict


def _compute_reward_via_run_single(data: DataProto, reward_fn: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compatibility path for experimental reward managers without __call__."""
    if "rm_scores" in data.batch.keys():
        reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        return data.batch["rm_scores"], reward_extra_info

    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    reward_extra_info = defaultdict(list)

    async def process_batch():
        tasks = []
        for idx in range(len(data)):
            tasks.append(reward_fn.run_single(data[idx : idx + 1]))
        return await asyncio.gather(*tasks)

    results = reward_fn.loop.run_until_complete(process_batch())

    for idx, result in enumerate(results):
        data_item = data[idx]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

        reward_tensor[idx, valid_response_length - 1] = result["reward_score"]
        if "reward_extra_info" in result:
            for key, value in result["reward_extra_info"].items():
                reward_extra_info[key].append(value)

    return reward_tensor, reward_extra_info


def compute_reward(data: DataProto, reward_fn: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    """
    if not callable(reward_fn) and hasattr(reward_fn, "run_single"):
        return _compute_reward_via_run_single(data, reward_fn)

    # Determine the effective callable to inspect
    call_target = reward_fn
    if not inspect.isfunction(reward_fn) and hasattr(reward_fn, "__call__"):
        call_target = reward_fn.__call__

    filtered_kwargs = _select_kwargs_for_callable(call_target, kwargs)

    try:
        reward_result = reward_fn(data, return_dict=True, **filtered_kwargs)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        # Fallback to legacy call without return_dict
        fallback_kwargs = filtered_kwargs
        reward_tensor = reward_fn(data, **fallback_kwargs)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None, **kwargs):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(config, tokenizer, **config.reward.get("reward_kwargs", {}))

    return compute_reward(data, reward_fn, **kwargs)
