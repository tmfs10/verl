"""
Hybrid conditional reward manager with masked-solution log-prob and optional batch score.

Computes:
- mean log P(masked-out tokens in solution | prompt, process_response_for_logprob(prompt, generated_response))
- optionally also computes external scores via `compute_score` (same interface as batch.py)
- combines the two via a user-provided `combine_with_vr` to produce final per-sample scores

Emits a reward tensor aligned to the generated responses: the scalar final score is placed
on the last valid generated token, matching the expected training shape.
"""

from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import (
    postprocess_data,
    get_response_mask,
    pad_sequence_to_length,
)
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


ResultCombiner = Callable[[list[dict | float], list[float]], list[dict]]


@register("conditional_masked_hybrid")
class ConditionalMaskedHybridRewardManager(AbstractRewardManager):
    """
    Reward = batch_scores + mean_logP_masked(solution | prompt, processed_response).

    - `process_response_for_logprob`: function(prompt_str, generated_response_str) -> new_generated_response_str
    - Masked tokens are identified from dataset metadata: per-line mask for the solution;
      we expand it to token-level by tokenizing each line (with newlines preserved).
    - `compute_score`: optional external scoring function (same signature as in BatchRewardManager).
    - `combine_with_vr`: bool, whether to combine the batch scores and the masked log-prob means.
      If not provided, defaults to adding the two (final_score = score + masked_mean) and attaching
      `masked_cond_logprob` for traceability.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score: RawRewardFn | None = None,
        reward_fn_key: str = "data_source",
        process_response_for_logprob: Optional[Callable[[str, str], str]] = None,
        combine_with_vr: bool = False,
        solution_field_name: str = "solution",
        line_mask_field_name: str = "solution_line_mask",
        reduction: str = "mean",  # only "mean" used for masked tokens aggregation
        normalize_logprob: bool = False,
        **reward_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.process_response_for_logprob = process_response_for_logprob
        self.combine_with_vr = combine_with_vr
        self.solution_field_name = solution_field_name
        self.line_mask_field_name = line_mask_field_name
        assert reduction in ("mean",), "Only 'mean' aggregation is supported for masked tokens"
        self.reduction = reduction
        self.normalize_logprob = normalize_logprob
        self.reward_kwargs = reward_kwargs

        print(
            f"Logprob args: "
            f"normalize_logprob={self.normalize_logprob}, combine_with_vr={self.combine_with_vr}, "
            f"num_examine={self.num_examine}, solution_key={self.solution_field_name}"
        )

    def _to_py_list(self, arr_like):
        if isinstance(arr_like, (list, tuple)):
            return list(arr_like)
        if isinstance(arr_like, np.ndarray):
            return arr_like.tolist()
        return list(arr_like)

    def _decode_prompts_and_responses(self, data: DataProto):
        tokenizer = self.tokenizer
        prompt_ids = data.batch["prompts"]  # [B, P]
        response_ids = data.batch["responses"]  # [B, R]
        attention_mask = data.batch["attention_mask"]  # [B, P+R]

        B, P = prompt_ids.shape
        valid_response_lengths = attention_mask[:, P:].sum(dim=-1)

        prompt_strs, response_strs = [], []
        for i in range(B):
            prompt_str = tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            valid_len = int(valid_response_lengths[i].item())
            resp_str = tokenizer.decode(response_ids[i][:valid_len], skip_special_tokens=True)
            prompt_strs.append(prompt_str)
            response_strs.append(resp_str)
        return prompt_strs, response_strs, valid_response_lengths

    def _compose_new_response_and_split(
        self,
        processed_resp_strs: list[str],
        solution_strs: list[str],
        line_masks: list[list[int]],
        max_response_len: Optional[int],
    ):
        """Compose new_response = processed_response + "\n" + solution with constraints, then split.

        Rules:
        - Ensure processed_response ends with "</think>" (assert).
        - Insert exactly one newline between processed_response and solution.
        - If total tokens of new_response exceed max_response_len, mark overflow; caller will skip logprob and assign the lowest score for that prompt group.
        - Otherwise split the new_response tokens into ctx_ext_tokens (prefix) and kept solution tokens (suffix).
        - Return per-sample tensors for ctx_ext_ids, gt_ids (kept solution), masked_token_mask for gt, and overflow flags.
        """
        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        newline_tokens = tokenizer("\n", return_tensors="pt", add_special_tokens=False)["input_ids"].view(-1)

        ctx_ext_list: list[torch.Tensor] = []
        gt_ids_list: list[torch.Tensor] = []
        masked_token_mask_list: list[torch.Tensor] = []
        overflow_list: list[bool] = []

        for proc_str, sol_str, lm in zip(processed_resp_strs, solution_strs, line_masks, strict=True):
            # 1) if processed_resp does not end with </think>, mark overflow; logprob will be ignored later
            missing_think = not proc_str.endswith("</think>")

            # 2) tokenize parts
            proc_ids = tokenizer(proc_str, return_tensors="pt", add_special_tokens=False)["input_ids"].view(-1)
            sol_ids_full = tokenizer(sol_str, return_tensors="pt", add_special_tokens=False)["input_ids"].view(-1)

            # 3) build new_response tokens with exactly one newline between
            new_resp_tokens = torch.cat([proc_ids, newline_tokens, sol_ids_full], dim=0)

            overflow_flag = missing_think or (max_response_len is not None and new_resp_tokens.size(0) > max_response_len)

            if overflow_flag and max_response_len is not None and new_resp_tokens.size(0) > max_response_len:
                # Truncate to fit shape; reward will still be sentinel later.
                new_resp_tokens = new_resp_tokens[-max_response_len:]

            # 6) compute how many solution tokens are kept (suffix overlap)
            L_sol = sol_ids_full.size(0)
            kept_sol_len = min(L_sol, new_resp_tokens.size(0))
            kept_sol_ids = sol_ids_full[-kept_sol_len:]
            kept_ctx_ext_len = new_resp_tokens.size(0) - kept_sol_len
            kept_ctx_ext_ids = new_resp_tokens[:kept_ctx_ext_len]

            # 7) expand line mask to token-level mask for full solution, then take suffix
            _, sol_token_mask_full = self._tokenize_solution_with_line_mask(sol_str, lm)
            kept_mask = sol_token_mask_full[-kept_sol_len:] if kept_sol_len > 0 else torch.empty((0,), dtype=torch.long)

            # If overflow, ensure tensors are non-empty with zero mask
            if overflow_flag:
                if kept_ctx_ext_ids.numel() == 0:
                    kept_ctx_ext_ids = torch.zeros((1,), dtype=torch.long)
                if kept_sol_ids.numel() == 0:
                    kept_sol_ids = torch.zeros((1,), dtype=torch.long)
                if kept_mask.numel() == 0:
                    kept_mask = torch.zeros((1,), dtype=torch.long)

            ctx_ext_list.append(kept_ctx_ext_ids)
            gt_ids_list.append(kept_sol_ids)
            masked_token_mask_list.append(kept_mask)
            overflow_list.append(overflow_flag)

        return ctx_ext_list, gt_ids_list, masked_token_mask_list, overflow_list

    def _tokenize_solution_with_line_mask(self, solution_text: str, line_mask: list[int]):
        """Tokenize solution with line-level mask expanded to token-level.

        We tokenize each line separately, appending a trailing "\n" for all but the last line so the
        concatenation matches the original string exactly.
        """
        tokenizer = self.tokenizer
        lines = solution_text.split("\n")
        token_ids_parts = []
        token_mask_parts = []
        for i, line in enumerate(lines):
            seg = line if i == len(lines) - 1 else (line + "\n")
            enc = tokenizer(seg, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].view(-1)
            token_ids_parts.append(ids)
            mask_val = 1 if (i < len(line_mask) and int(line_mask[i]) == 1) else 0
            token_mask_parts.append(torch.full((ids.size(0),), mask_val, dtype=torch.long))

        if len(token_ids_parts) == 0:
            return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
        return torch.cat(token_ids_parts, dim=0), torch.cat(token_mask_parts, dim=0)

    def _build_gt_and_masks(
        self,
        solutions: list[str],
        line_masks: list[list[int]],
        device: torch.device,
        pad_token_id: int,
        eos_token_id: int | list[int],
    ):
        """Tokenize solutions and build padded GT ids and corresponding token-level masks.

        Returns:
            gt_ids: [B, L]
            gt_attn_mask: [B, L] (1 for real tokens, 0 for pad)
            masked_token_mask: [B, L] (1 for masked tokens, 0 otherwise)
            gt_resp_mask_eos: [B, L] (1 until first eos, 0 after)
        """
        B = len(solutions)
        gt_ids_list = []
        tok_mask_list = []
        for s, lm in zip(solutions, line_masks, strict=True):
            ids, tmask = self._tokenize_solution_with_line_mask(s, lm)
            gt_ids_list.append(ids)
            tok_mask_list.append(tmask)

        if self.max_gt_len is None:
            max_len = max((x.size(0) for x in gt_ids_list), default=0)
        else:
            max_len = self.max_gt_len
        max_len = max(1, max_len)

        padded_ids = []
        padded_attn = []
        padded_mask = []
        for ids, tmask in zip(gt_ids_list, tok_mask_list, strict=True):
            ids = ids.unsqueeze(0)
            attn = torch.ones_like(ids)
            ids_pad, attn_pad = postprocess_data(
                input_ids=ids,
                attention_mask=attn,
                max_length=max_len,
                pad_token_id=pad_token_id,
                left_pad=False,
                truncation="right",
            )
            # pad token mask to the same length
            mask_pad = pad_sequence_to_length(tmask.unsqueeze(0), max_len, 0, left_pad=False)
            padded_ids.append(ids_pad[0])
            padded_attn.append(attn_pad[0])
            padded_mask.append(mask_pad[0])

        gt_ids = torch.stack(padded_ids, dim=0).to(device)
        gt_attn_mask = torch.stack(padded_attn, dim=0).to(device)
        masked_token_mask = torch.stack(padded_mask, dim=0).to(device)

        # stop after eos if present
        gt_resp_mask_eos = get_response_mask(response_id=gt_ids, eos_token=eos_token_id, dtype=gt_attn_mask.dtype)
        return gt_ids, gt_attn_mask, masked_token_mask, gt_resp_mask_eos

    def _default_combine(self, base_scores: list[dict | float], masked_means: list[float]) -> list[dict]:
        out: list[dict] = []
        for base, m in zip(base_scores, masked_means, strict=True):
            if isinstance(base, dict):
                d = dict(base)
                d["masked_cond_logprob"] = float(m)
                if "score" in d and isinstance(d["score"], (int, float)):
                    d["score"] = float(d["score"]) + float(m)
                else:
                    d["score"] = float(m)
            else:
                d = {"score": float(base) + float(m), "masked_cond_logprob": float(m)}
            out.append(d)
        return out

    def _compute_batch_scores(self, data: DataProto) -> list[dict | float]:
        if self.compute_score is None:
            # default: zero base score
            return [0.0 for _ in range(len(data))]

        tokenizer = self.tokenizer
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        P = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, P:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_resp_ids = response_ids[i][:valid_len]
            responses_str.append(tokenizer.decode(valid_resp_ids, skip_special_tokens=True))

        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False, **kwargs: Any):
        actor_wg = kwargs.get("actor_wg", None)
        if actor_wg is None:
            raise ValueError("ConditionalMaskedHybridRewardManager requires 'actor_wg' kwarg (actor worker group)")

        # If there is rm score, return it directly for compatibility
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        device = data.batch["prompts"].device
        dtype = data.batch["attention_mask"].dtype
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Validation: skip logprob computation, use only external/verification score
        if data.meta_info.get("validate", False):
            base_scores = self._compute_batch_scores(data)
            masked_means = [0.0 for _ in range(len(base_scores))]
            final_results = base_scores

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)
            prompt_len = data.batch["prompts"].shape[-1]
            valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

            rewards = []
            for i in range(len(data)):
                length = int(valid_response_lengths[i].item())
                res = final_results[i]
                if isinstance(res, dict):
                    reward_val = res.get("score", 0.0)
                    reward = float(0.0 if reward_val is None else reward_val)
                    for k, v in res.items():
                        reward_extra_info[k].append(0.0 if v is None else v)
                else:
                    reward = float(0.0 if res is None else res)
                rewards.append(reward)
                if length > 0:
                    reward_tensor[i, length - 1] = reward

            data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=device)
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info} if return_dict else reward_tensor

        # 1) Decode strings and optionally process response for conditional log-prob context
        prompt_strs, gen_resp_strs, valid_response_lengths = self._decode_prompts_and_responses(data)
        if self.process_response_for_logprob is not None:
            proc_resp_strs = [self.process_response_for_logprob(p, r) for p, r in zip(prompt_strs, gen_resp_strs, strict=True)]
        else:
            proc_resp_strs = gen_resp_strs

        # 2) Solutions and line masks from dataset
        solutions_arr = data.non_tensor_batch.get(self.solution_field_name, None)
        if solutions_arr is None:
            raise KeyError(
                f"'{self.solution_field_name}' not found in non_tensor_batch; ensure dataset provides the original solution."
            )
        solutions = [str(x) if x is not None else "" for x in self._to_py_list(solutions_arr)]
        line_masks_arr = data.non_tensor_batch.get(self.line_mask_field_name, None)
        if line_masks_arr is None:
            raise KeyError(
                f"'{self.line_mask_field_name}' not found in non_tensor_batch; ensure the dataset provides line masks."
            )
        # Each entry should be a list[int]; if it's an array/tuple, coerce to list
        line_masks = []
        for lm in self._to_py_list(line_masks_arr):
            if isinstance(lm, (list, tuple)):
                line_masks.append(list(lm))
            else:
                # numpy arrays or other sequences
                try:
                    line_masks.append(list(lm))
                except Exception:
                    # fallback: treat as scalar mask disabled
                    line_masks.append([0])

        # 3) Compose new_response with truncation and split into ctx-ext and kept solution
        max_resp_len = data.batch["responses"].shape[-1]
        ctx_ext_list, gt_ids_list, masked_token_mask_list, overflow_list = self._compose_new_response_and_split(
            processed_resp_strs=proc_resp_strs,
            solution_strs=solutions,
            line_masks=line_masks,
            max_response_len=max_resp_len,
        )

        data_sources = self._to_py_list(data.non_tensor_batch[self.reward_fn_key])
        valid_indices = list(range(len(data)))  # include all; overflow handled later

        # 4) Build combined response (ctx-ext + GT) and pad once, bounded by max_response_len
        prompt_ids = data.batch["prompts"]  # [B, P]
        prompt_attn = data.batch["attention_mask"][:, : prompt_ids.shape[-1]]
        pad_id = pad_token_id

        ctx_lens = [t.numel() for t in ctx_ext_list]
        gt_lens = [t.numel() for t in gt_ids_list]
        max_gt_len = max(gt_lens, default=1)
        max_ctx_plus_gt = max((c + g for c, g in zip(ctx_lens, gt_lens, strict=True)), default=0)
        # Safety: cap at the response budget (already enforced in _compose, but double-checked)
        max_ctx_plus_gt = min(max_ctx_plus_gt, max_resp_len)
        max_ctx_plus_gt = max(1, max_ctx_plus_gt)

        resp_ids_list = []
        resp_attn_list = []
        masked_token_mask_batched = []
        gt_resp_mask_eos_list = []

        for i in valid_indices:
            ctx = ctx_ext_list[i].to(device)
            gt = gt_ids_list[i].to(device)
            ctx_len = ctx.numel()
            gt_len = gt.numel()

            # Response tokens = ctx extension followed by kept GT tokens
            resp_ids = torch.cat([ctx, gt], dim=-1)
            resp_ids = pad_sequence_to_length(resp_ids.unsqueeze(0), max_ctx_plus_gt, pad_id, left_pad=False)[0]
            resp_ids_list.append(resp_ids)

            # Build attention over response tokens (ctx ones, GT until eos, zero for pad)
            resp_attn = torch.zeros((max_ctx_plus_gt,), dtype=prompt_attn.dtype, device=device)
            if ctx_len > 0:
                resp_attn[:ctx_len] = 1
            gt_mask_local = torch.zeros((max_ctx_plus_gt,), dtype=prompt_attn.dtype, device=device)
            if gt_len > 0:
                gt_eos_mask = get_response_mask(
                    response_id=gt.unsqueeze(0), eos_token=tokenizer.eos_token_id, dtype=prompt_attn.dtype
                )[0]
                gt_mask_local[ctx_len : ctx_len + gt_len] = gt_eos_mask
                resp_attn[ctx_len : ctx_len + gt_len] = gt_eos_mask
            resp_attn_list.append(resp_attn)
            gt_resp_mask_eos_list.append(gt_mask_local)

            # Masked-token mask aligned to the GT segment positions
            m = masked_token_mask_list[i]
            mask_vec = torch.zeros((max_ctx_plus_gt,), dtype=torch.long, device=device)
            if m.numel() > 0:
                mask_vec[ctx_len : ctx_len + m.numel()] = m.to(device)
            masked_token_mask_batched.append(mask_vec)

        responses = torch.stack(resp_ids_list, dim=0)
        masked_token_mask = torch.stack(masked_token_mask_batched, dim=0).to(device)
        gt_resp_mask_eos = torch.stack(gt_resp_mask_eos_list, dim=0)

        # 6) Concatenate prompts with padded responses for model input
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_attn, torch.stack(resp_attn_list, dim=0)], dim=-1)
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)

        # 6) Compute token-level log-probs for GT under actor
        gt_batch = DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,  # not used by log_prob but kept for consistency
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        seq_len_total = input_ids.shape[-1]
        existing_max_token_len = gt_batch.meta_info.get("max_token_len", None)
        if existing_max_token_len is None or existing_max_token_len < seq_len_total:
            gt_batch.meta_info["max_token_len"] = seq_len_total

        # Truncate sequences to actor's max_token_len if set
        max_token_len = None
        try:
            max_token_len = actor_wg.config.actor.ppo_infer_max_token_len_per_gpu
        except Exception:
            max_token_len = None

        if max_token_len is not None and gt_batch.batch["input_ids"].shape[-1] > max_token_len:
            gt_batch.batch["input_ids"] = gt_batch.batch["input_ids"][:, -max_token_len:]
            gt_batch.batch["attention_mask"] = gt_batch.batch["attention_mask"][:, -max_token_len:]
            gt_batch.batch["position_ids"] = gt_batch.batch["position_ids"][:, -max_token_len:]
            seq_len_total = max_token_len

        logprob_out = actor_wg.compute_log_prob(gt_batch)
        gt_log_probs_full = logprob_out.batch["old_log_probs"]  # [B, gt_len]

        # 7) Aggregate mean log-prob over masked tokens (and valid gt mask)
        masked = masked_token_mask.to(gt_log_probs_full.dtype) * gt_resp_mask_eos.to(gt_log_probs_full.dtype)
        token_sums = (gt_log_probs_full * masked).sum(dim=-1)
        counts = masked.sum(dim=-1).clamp_min(1.0)
        masked_means_tensor_valid = token_sums / counts

        masked_means_full = [v for v in masked_means_tensor_valid]

        # Collect per-sample masked means with overflow handling
        data_sources = self._to_py_list(data.non_tensor_batch[self.reward_fn_key])

        # Precompute group and global minima from valid samples
        group_to_vals: dict[str, list[torch.Tensor]] = defaultdict(list)
        for idx, val in enumerate(masked_means_full):
            group_to_vals[str(data_sources[idx])].append(val)
        global_min = torch.min(masked_means_tensor_valid) if len(masked_means_tensor_valid) > 0 else None

        # For overflowed samples: use group min if available; else global min; else 0.
        zero_tensor = torch.tensor(0.0, device=device, dtype=gt_log_probs_full.dtype if 'gt_log_probs_full' in locals() else torch.float32)
        for idx, ov in enumerate(overflow_list):
            if not ov:
                continue
            group = str(data_sources[idx])
            if group_to_vals.get(group):
                fill_val = min(group_to_vals[group])
            elif global_min is not None:
                fill_val = global_min
            else:
                fill_val = zero_tensor
            masked_means_full[idx] = fill_val

        # Optional min-max normalization to [0,1] per group
        if self.normalize_logprob:
            for group in set(map(str, data_sources)):
                indices = [i for i, g in enumerate(data_sources) if str(g) == group]
                vals = torch.stack([masked_means_full[i] for i in indices])
                vmin = vals.min()
                vmax = vals.max()
                if torch.isclose(vmin, vmax):
                    scaled = torch.zeros_like(vals)
                else:
                    scaled = (vals - vmin) / (vmax - vmin)
                for dst, sval in zip(indices, scaled, strict=True):
                    masked_means_full[dst] = sval

        masked_means = [v.detach().cpu().item() for v in masked_means_full]

        # 8) Optional external batch scores
        base_scores = self._compute_batch_scores(data)

        # 9) Combine
        if self.combine_with_vr:
            final_results = self._default_combine(base_scores, masked_means)
        else:
            final_results = base_scores

        # 10) Build reward tensor aligned to original generated responses
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_len = data.batch["prompts"].shape[-1]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        rewards = []
        already_printed: dict[str, Any] = {}
        for i in range(len(data)):
            length = int(valid_response_lengths[i].item())
            res = final_results[i]
            if isinstance(res, dict):
                reward = float(res.get("score", 0.0))
                for k, v in res.items():
                    reward_extra_info[k].append(v)
            else:
                reward = float(res)
            rewards.append(reward)
            if length > 0:
                reward_tensor[i, length - 1] = reward

            # optional logging
            data_source = data_sources[i]
            if self.num_examine > 0 and already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:length], skip_special_tokens=True
                )
                print("[prompt]", prompt_strs[i])
                print("[generated]", response_str)
                print("[processed_for_logprob]", proc_resp_strs[i])
                print("[solution]", solutions[i])
                print("[masked_mean_logP]", masked_means[i])
                print("[final_score]", reward)
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
