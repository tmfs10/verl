"""
Conditional log-probability reward manager.

Computes the log-probability of a ground-truth response conditioned on the
prompt concatenated with the generated response from rollout. The scalar
reward per sample is the reduction (mean or sum) of token log-probs for the
ground-truth continuation, and is placed on the last token of the generated
response to match the expected token-level reward tensor shape.
"""

from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.torch_functional import postprocess_data, get_response_mask
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("conditional_logprob")
class ConditionalLogProbRewardManager(AbstractRewardManager):
    """
    Reward = log P(gt_response | prompt, generated_response) under the actor.

    - Expects the batch to contain per-sample ground-truth response text under
      `ground_truth_response` in `non_tensor_batch`. As a fallback, it will try
      `reward_model.ground_truth` if present.
    - Requires `actor_wg` to be provided at call time (the actor worker group
      that exposes `compute_log_prob`).
    - Emits token-level rewards aligned to the generated responses by assigning
      the scalar reward to the last valid token of each generated response.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,  # Unused, kept for signature compatibility
        reward_fn_key: str = "data_source",
        reduction: str = "mean",  # "mean" or "sum"
        gt_field_name: str = "ground_truth_response",
        max_gt_len: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        assert reduction in ("mean", "sum"), f"Unsupported reduction: {reduction}"
        self.reduction = reduction
        self.gt_field_name = gt_field_name
        self.max_gt_len = max_gt_len

    def _get_ground_truth_text(self, data_item) -> Optional[str]:
        # Primary: explicit ground_truth_response
        gt = data_item.non_tensor_batch.get(self.gt_field_name, None)
        if gt is not None:
            return gt
        # Fallback: reward_model.ground_truth if present
        rm = data_item.non_tensor_batch.get("reward_model", None)
        if isinstance(rm, dict):
            return rm.get("ground_truth", None)
        return None

    def __call__(self, data: DataProto, return_dict: bool = False, **kwargs: Any):
        actor_wg = kwargs.get("actor_wg", None)
        if actor_wg is None:
            raise ValueError("ConditionalLogProbRewardManager requires 'actor_wg' kwarg (actor worker group)")

        # If there is rm score, return it directly for compatibility
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        prompts = data.batch["prompts"]
        input_seq = data.batch["input_ids"]  # prompts + generated responses
        prompt_pos = data.batch["position_ids"]
        prompt_attn = data.batch["attention_mask"]
        responses_gen = data.batch["responses"]

        device = prompts.device
        dtype = prompt_attn.dtype
        pad_token_id = (
            data.meta_info.get("pad_token_id")
            if isinstance(data.meta_info, dict)
            else (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        )

        # Tokenize ground-truth responses and determine a common length
        gt_token_lists: list[torch.Tensor] = []
        for i in range(len(data)):
            gt_text = self._get_ground_truth_text(data[i])
            if gt_text is None or len(gt_text) == 0:
                gt_ids = torch.empty((0,), dtype=torch.long)
            else:
                enc = self.tokenizer(gt_text, return_tensors="pt", add_special_tokens=False)
                gt_ids = enc["input_ids"].view(-1)
            gt_token_lists.append(gt_ids)

        if self.max_gt_len is None:
            max_gt_len = max((t.size(0) for t in gt_token_lists), default=0)
        else:
            max_gt_len = self.max_gt_len

        # Build batched GT response ids and masks with right padding
        gt_ids_batched: list[torch.Tensor] = []
        for ids in gt_token_lists:
            ids = ids.unsqueeze(0)  # [1, L]
            # Pad/truncate to max_gt_len using postprocess utility
            ids_pad, attn_pad = postprocess_data(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_length=max_gt_len if max_gt_len > 0 else 1,
                pad_token_id=pad_token_id,
                left_pad=False,
                truncation="right",
            )
            gt_ids_batched.append(ids_pad[0])

        if len(gt_ids_batched) == 0:
            # No GT provided; return zero rewards
            reward_tensor = torch.zeros_like(responses_gen, dtype=torch.float32)
            return {"reward_tensor": reward_tensor} if return_dict else reward_tensor

        gt_ids = torch.stack(gt_ids_batched, dim=0).to(device)

        # Build attention mask for GT continuation (stop at eos if found)
        gt_resp_mask = get_response_mask(response_id=gt_ids, eos_token=self.tokenizer.eos_token_id, dtype=dtype)

        # New concatenated input: (prompts + generated) as prompts, gt as responses
        prompts_rep = input_seq  # already cat(prompts, generated)
        prompt_pos_rep = prompt_pos
        prompt_attn_rep = prompt_attn

        seq_concat = torch.cat([prompts_rep, gt_ids], dim=-1)

        # Extend position ids for GT continuation
        gt_len = gt_ids.size(1)
        delta_pos = torch.arange(1, gt_len + 1, device=prompt_pos_rep.device)
        delta_pos = delta_pos.unsqueeze(0).expand(seq_concat.size(0), -1)
        if prompt_pos_rep.dim() == 3:  # mrope format
            delta_pos = delta_pos.view(seq_concat.size(0), 1, -1).expand(seq_concat.size(0), 3, -1)
        gt_position_ids = prompt_pos_rep[..., -1:] + delta_pos
        position_ids = torch.cat([prompt_pos_rep, gt_position_ids], dim=-1)

        attention_mask = torch.cat([prompt_attn_rep, gt_resp_mask], dim=-1)

        # Build DataProto for log-prob computation of GT continuation
        gt_batch = DataProto.from_dict(
            tensors={
                "prompts": prompts_rep,
                "responses": gt_ids,
                "input_ids": seq_concat,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

        # Compute token-level log-probs for GT under actor
        logprob_out = actor_wg.compute_log_prob(gt_batch)
        gt_log_probs = logprob_out.batch["old_log_probs"]  # [B, gt_len]

        # Aggregate per-sample reward from token-level log-probs (mask invalid padding)
        valid_mask = gt_resp_mask.to(gt_log_probs.dtype)
        token_sums = (gt_log_probs * valid_mask).sum(dim=-1)
        valid_counts = valid_mask.sum(dim=-1).clamp_min(1.0)
        if self.reduction == "mean":
            rewards_scalar = token_sums / valid_counts
        else:
            rewards_scalar = token_sums

        # Map scalar reward onto the generated response token grid at last valid token
        reward_tensor = torch.zeros_like(responses_gen, dtype=torch.float32)
        prompt_len = prompts.size(-1)
        gen_valid_lens = attention_mask[:, prompt_len:].sum(dim=-1) - gt_resp_mask.sum(dim=-1)
        # gen_valid_lens is the length of generated response (before GT)
        for i in range(reward_tensor.size(0)):
            L = int(gen_valid_lens[i].item())
            if L > 0:
                reward_tensor[i, L - 1] = rewards_scalar[i].to(reward_tensor.dtype)

        if self.num_examine > 0:
            # Print a few examples for debugging
            already_printed = 0
            seqs = self.tokenizer.batch_decode(responses_gen, skip_special_tokens=True)
            for i in range(min(self.num_examine, len(seqs))):
                gt_text = self._get_ground_truth_text(data[i])
                print("[generated]", seqs[i])
                print("[ground_truth_response]", gt_text)
                print("[cond_logprob_reward]", float(rewards_scalar[i].item()))
                already_printed += 1
                if already_printed >= self.num_examine:
                    break

        if return_dict:
            extra = {"cond_logprob": rewards_scalar.detach().cpu().tolist()}
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor

