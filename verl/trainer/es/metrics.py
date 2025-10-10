from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from verl import DataProto


def _response_info(data: DataProto) -> dict[str, torch.Tensor]:
    resp_len = data.batch["responses"].shape[-1]
    prompt_mask = data.batch["attention_mask"][:, :-resp_len]
    response_mask = data.batch["attention_mask"][:, -resp_len:]
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()
    return {
        "prompt_length": prompt_length,
        "response_length": response_length,
        "max_prompt_length": torch.tensor(prompt_mask.size(-1), device=prompt_mask.device, dtype=torch.float32),
        "max_response_length": torch.tensor(resp_len, device=prompt_mask.device, dtype=torch.float32),
    }


def compute_rollout_reward_timing_metrics(
    data: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_info: dict[str, Any] | None,
    timing_raw: dict[str, float],
) -> dict[str, float]:
    info = _response_info(data)
    prompt_len = info["prompt_length"]
    response_len = info["response_length"]
    max_prompt = info["max_prompt_length"].item()
    max_response = info["max_response_length"].item()

    # Aborted if response length is zero
    aborted_mask = (response_len == 0)
    # Scalar per-sample rewards: sum over response tokens
    sample_rewards = reward_tensor.sum(-1).float()

    # Rollout metrics
    m: dict[str, float] = {}
    m["response_length/mean"] = float(response_len.mean().item())
    m["response_length/max"] = float(response_len.max().item())
    m["response_length/min"] = float(response_len.min().item())
    m["response_length/clip_ratio"] = float((response_len == max_response).float().mean().item())
    m["response/aborted_ratio"] = float(aborted_mask.float().mean().item())

    m["prompt_length/mean"] = float(prompt_len.mean().item())
    m["prompt_length/max"] = float(prompt_len.max().item())
    m["prompt_length/min"] = float(prompt_len.min().item())
    m["prompt_length/clip_ratio"] = float((prompt_len == max_prompt).float().mean().item())

    # Reward metrics
    if sample_rewards.numel() > 0:
        m["reward/mean"] = float(sample_rewards.mean().item())
        m["reward/max"] = float(sample_rewards.max().item())
        m["reward/min"] = float(sample_rewards.min().item())

    # Reward extra info
    if reward_extra_info:
        for k, v in reward_extra_info.items():
            key = f"reward/{k}"
            if isinstance(v, (list, tuple)):
                arr = np.array(v, dtype=object)
                # Only log numeric arrays
                if arr.dtype != object:
                    m[f"{key}/mean"] = float(np.mean(arr))
            elif isinstance(v, (int, float, np.floating, np.integer)):
                m[key] = float(v)
            # else: skip non-numeric

    # Timing metrics (s) and per-token (ms)
    m["timing_s/gen"] = float(timing_raw.get("gen", 0.0))
    m["timing_s/step"] = float(timing_raw.get("step", m["timing_s/gen"]))
    total_resp_tokens = float(response_len.sum().item()) + 1e-8
    m["timing_per_token_ms/gen"] = 1000.0 * m["timing_s/gen"] / total_resp_tokens

    # Meta counters for driver-side aggregation
    m["meta/n_samples"] = float(response_len.numel())
    m["meta/total_response_tokens"] = float(response_len.sum().item())

    return m

