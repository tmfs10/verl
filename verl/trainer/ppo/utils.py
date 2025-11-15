# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import warnings
from enum import Enum
from collections import defaultdict

from omegaconf import DictConfig

import numpy as _np
import torch

from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


def need_reference_policy(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need ref policy."""
    return Role.RefPolicy in role_worker_mapping


def need_reward_model(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need reward model."""
    return Role.RewardModel in role_worker_mapping


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False


def compute_group_loss_weights(
    uids,
    acc_arr,
    response_mask: torch.Tensor,
    *,
    token_mean: bool = False,
    skip_zero: bool = False,
) -> torch.Tensor:
    """
    Compute per-sample loss weights to balance contributions of positives and negatives within groups.

    Groups are defined by `uid`. For each group containing both positives (acc==1) and negatives (acc==0),
    assign weights w_pos, w_neg so that total weighted contribution from positives equals that from negatives.

    - If `token_mean` is True, uses total response token counts (sum of response_mask) per sequence as counts.
    - If `skip_zero` is True, groups that are all-positive or all-negative get zero weights (dropped).

    Args:
        uids: Iterable of group identifiers of length B.
        acc_arr: Iterable/array of 0.0/1.0 of length B.
        response_mask: (B, T) torch tensor.
        token_mean: Whether to balance by token counts instead of sample counts.
        skip_zero: If True, zero weight for mono-class groups.

    Returns:
        (B,) torch tensor of weights on response_mask.device/dtype.
    """
    assert response_mask.dim() == 2, "response_mask must be (B, T)"
    device = response_mask.device
    dtype = response_mask.dtype

    # Normalize inputs
    uid_list = [str(u) for u in list(uids)]
    acc_np = _np.asarray(acc_arr)
    assert _np.all((acc_np == 0.0) | (acc_np == 1.0)), "acc array must be 0.0 or 1.0"

    B = len(uid_list)
    assert B == response_mask.size(0), "uids/acc length must match batch size"

    weights = _np.ones((B,), dtype=_np.float32)

    gid2idx: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(uid_list):
        gid2idx[g].append(i)

    # Precompute response lengths if needed
    resp_lens = None
    if token_mean:
        resp_lens = response_mask.sum(dim=-1).detach().to(dtype=torch.float32).cpu().numpy()

    for g, idxs in gid2idx.items():
        if not idxs:
            continue
        g_acc = acc_np[idxs]
        idxs0 = [ii for j, ii in enumerate(idxs) if g_acc[j] == 0.0]
        idxs1 = [ii for j, ii in enumerate(idxs) if g_acc[j] == 1.0]

        if token_mean:
            n1 = float(_np.sum(resp_lens[idxs1])) if len(idxs1) > 0 else 0.0
            n0 = float(_np.sum(resp_lens[idxs0])) if len(idxs0) > 0 else 0.0
        else:
            n1 = float(len(idxs1))
            n0 = float(len(idxs0))

        if n1 > 0 and n0 > 0:
            # Use symmetric weighting so sums match, same approach for critic/actor
            # Base ratios
            w_pos = (n1 + n0) / (2.0 * max(n1, 1.0))
            w_neg = (n1 / max(n0, 1.0)) * w_pos
            for j, ii in enumerate(idxs):
                weights[ii] = w_pos if g_acc[j] == 1.0 else w_neg
        else:
            if skip_zero:
                for ii in idxs:
                    weights[ii] = 0.0

    return torch.tensor(weights, dtype=dtype, device=device)
