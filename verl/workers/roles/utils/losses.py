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
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_critic_diff_penalty,
    compute_rmauc_loss,
    compute_value_loss,
    get_policy_loss_fn,
    kl_penalty,
    normalize_rmauc_loss,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig, CriticConfig

_rmauc_state = {"vf_ema": None, "rmauc_ema": None}


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.LEFT_RIGHT)

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        cu_seqlens = log_prob.offsets()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
        loss_mask_flatten[cu_seqlens[1:] - 1] = 0
        loss = -masked_mean(log_prob_flatten, loss_mask_flatten)
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_mean(log_prob, response_mask)

    return loss, {"loss": loss.detach().item()}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
    )

    metrics.update(
        {
            "pg_loss": pg_loss.detach().item(),
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
    )
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    vpreds = model_output["values"]
    values = data["values"]

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)
    penalty_cfg = tu.get_non_tensor_data(data=data, key="critic_diff_penalty", default={})
    use_rmauc_loss = tu.get_non_tensor_data(data=data, key="use_rmauc_loss", default=False)
    rmauc_cfg = tu.get_non_tensor_data(data=data, key="rmauc_cfg", default={})

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )
    coeff = float(penalty_cfg.get("coeff", 0.0) or 0.0)
    if use_rmauc_loss:
        acc = data["acc"]
        acc = acc.to(vpreds.device, vpreds.dtype).view(-1)
        rmauc = compute_rmauc_loss(values=vpreds, acc=acc, response_mask=response_mask)
        if rmauc_cfg.get("normalize", False):
            global _rmauc_state
            rmauc_norm, _rmauc_state, rmauc_scale = normalize_rmauc_loss(
                rmauc_loss=rmauc, vf_loss=vf_loss, state=_rmauc_state, beta=float(rmauc_cfg.get("ema_beta", 0.9))
            )
        else:
            rmauc_norm = rmauc
            rmauc_scale = 1.0
        base_loss = 2.0 - rmauc_norm
    else:
        rmauc = None
        rmauc_scale = 1.0
        base_loss = vf_loss

    diff_loss, num_pairs = None, 0
    if coeff > 0:
        diff_loss, num_pairs, _ = compute_critic_diff_penalty(vpreds=vpreds, response_mask=response_mask)
        total_loss = (1.0 - coeff) * base_loss + coeff * diff_loss
    else:
        total_loss = base_loss

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )
    if use_rmauc_loss and rmauc is not None:
        metrics.update({"critic/rmauc": rmauc.detach().item(), "critic/rmauc_scale": rmauc_scale})
    if coeff > 0 and diff_loss is not None:
        metrics.update(
            {
                "critic/value_diff_mse": diff_loss.detach().item(),
                "critic/value_diff_penalty": (coeff * diff_loss).detach().item(),
                "critic/value_diff_pairs": num_pairs,
            }
        )

    return total_loss, metrics
