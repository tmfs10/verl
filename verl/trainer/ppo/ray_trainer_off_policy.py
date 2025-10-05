
"""
Off-policy trainer: consumes pre-generated rollouts from batch_dict["outputs"].
"""

import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer as RayTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import postprocess_data, get_response_mask
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.rollout_skip import RolloutSkip


class RayTrainerOffPolicy(RayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_offpolicy_outputs(self, batch: DataProto) -> DataProto:
        """
        Build generated outputs DataProto from precomputed output strings in batch.non_tensor_batch["outputs"].
        Returns DataProto with keys: prompts, responses, input_ids, attention_mask, position_ids.
        """
        assert "outputs" in batch.non_tensor_batch, "Expected 'outputs' in batch for off-policy training"

        prompts = batch.batch["input_ids"]
        prompt_attention = batch.batch["attention_mask"]
        prompt_position = batch.batch["position_ids"]
        batch_size = prompts.size(0)

        outputs_arr = batch.non_tensor_batch["outputs"]
        # outputs_arr is usually a numpy array of dtype object, where each element is a list[str] or str
        per_sample_counts: list[int] = []
        flattened_responses: list[torch.Tensor] = []
        repeated_prompts: list[torch.Tensor] = []
        repeated_prompt_attention: list[torch.Tensor] = []
        repeated_prompt_position: list[torch.Tensor] = []
        response_texts: list[str] = []
        flattened_scores: list = []  # keep as py objects (float or dict)

        response_length = self.config.actor_rollout_ref.rollout.response_length
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for i in range(batch_size):
            outs_i = outputs_arr[i]
            if isinstance(outs_i, str):
                outs_i = [outs_i]
            elif isinstance(outs_i, (list, tuple, np.ndarray)):
                outs_i = list(outs_i)
            else:
                raise TypeError(f"Unsupported outputs type at index {i}: {type(outs_i)}")

            per_sample_counts.append(len(outs_i))
            # optional scores (one per output)
            scores_i = None
            if "scores" in batch.non_tensor_batch:
                scores_i = batch.non_tensor_batch["scores"][i]
                if isinstance(scores_i, np.ndarray):
                    scores_i = scores_i.tolist()
                # ensure list alignment if present
                if isinstance(scores_i, (list, tuple)):
                    if len(scores_i) != len(outs_i):
                        raise ValueError(
                            f"scores length {len(scores_i)} does not match outputs length {len(outs_i)} for index {i}"
                        )
                else:
                    # allow scalar when only one output
                    if len(outs_i) != 1:
                        raise ValueError(
                            f"Expected a list of scores for multiple outputs at index {i}, got scalar of type {type(scores_i)}"
                        )

            for j, out_str in enumerate(outs_i):
                # tokenize response only; no special tokens
                model_inputs = self.tokenizer(out_str, return_tensors="pt", add_special_tokens=False)
                resp_ids = model_inputs.pop("input_ids")
                resp_attn = model_inputs.pop("attention_mask")
                # right-pad / truncate to response_length
                resp_ids, resp_attn = postprocess_data(
                    input_ids=resp_ids,
                    attention_mask=resp_attn,
                    max_length=response_length,
                    pad_token_id=pad_token_id,
                    left_pad=False,
                    truncation="right",
                )
                flattened_responses.append(resp_ids[0])
                repeated_prompts.append(prompts[i])
                repeated_prompt_attention.append(prompt_attention[i])
                repeated_prompt_position.append(prompt_position[i])
                response_texts.append(out_str)
                if scores_i is not None:
                    if isinstance(scores_i, (list, tuple)):
                        flattened_scores.append(scores_i[j])
                    else:
                        flattened_scores.append(scores_i)

        if len(flattened_responses) == 0:
            raise ValueError("No responses provided in 'outputs'")

        device = prompts.device
        dtype = prompt_attention.dtype

        prompts_rep = torch.stack(repeated_prompts, dim=0).to(device)
        prompt_attn_rep = torch.stack(repeated_prompt_attention, dim=0).to(device)
        prompt_pos_rep = torch.stack(repeated_prompt_position, dim=0).to(device)
        responses = torch.stack(flattened_responses, dim=0).to(device)

        # build concatenated input_ids, attention_mask, position_ids (mirror vllm_rollout_spmd)
        seq = torch.cat([prompts_rep, responses], dim=-1)

        resp_len = responses.size(1)
        delta_position_id = torch.arange(1, resp_len + 1, device=prompt_pos_rep.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(seq.size(0), -1)
        if prompt_pos_rep.dim() == 3:  # mrope (e.g., qwen2vl)
            delta_position_id = delta_position_id.view(seq.size(0), 1, -1).expand(seq.size(0), 3, -1)
        response_position_ids = prompt_pos_rep[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_pos_rep, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=self.tokenizer.eos_token_id, dtype=dtype
        )
        attention_mask = torch.cat([prompt_attn_rep, response_attention_mask], dim=-1)

        # Build non-tensor batch by repeating per-sample metadata
        new_non_tensor = {}
        for key, arr in batch.non_tensor_batch.items():
            if key == "outputs":
                # Flattened response strings for convenience
                new_non_tensor["response_str"] = np.array(response_texts, dtype=object)
                continue
            if key == "scores" and len(flattened_scores) > 0:
                # Align scores to flattened responses (one per response)
                new_non_tensor["scores"] = np.array(flattened_scores, dtype=object)
                continue
            # Repeat each element according to per_sample_counts
            repeated_list = []
            for i in range(batch_size):
                val = arr[i]
                for _ in range(per_sample_counts[i]):
                    repeated_list.append(val)
            new_non_tensor[key] = np.array(repeated_list, dtype=object)

        gen_batch = TensorDict(
            {
                "prompts": prompts_rep,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=seq.size(0),
        )

        return DataProto(batch=gen_batch, non_tensor_batch=new_non_tensor)

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Off-policy: build gen_batch_output from provided outputs
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self._build_offpolicy_outputs(batch)

                    # Replace the original repeat+union with the flattened off-policy outputs
                    batch = gen_batch_output

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_extra_infos_dict: dict = {}
                        used_async_reward = False
                        # Prefer precomputed rewards if provided alongside outputs
                        if "scores" in batch.non_tensor_batch:
                            # Build token-level reward tensor directly from provided scores
                            prompts = batch.batch["prompts"]
                            responses = batch.batch["responses"]
                            attention_mask = batch.batch["attention_mask"]
                            prompt_len = prompts.shape[-1]
                            valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

                            reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
                            rewards_scalar: list[float] = []
                            scores_list = batch.non_tensor_batch["scores"]

                            # Ensure iterable alignment
                            assert len(scores_list) == len(responses), (
                                f"scores length {len(scores_list)} must match number of responses {len(responses)}"
                            )

                            for i in range(len(responses)):
                                length = int(valid_response_lengths[i].item())
                                score_item = scores_list[i]
                                if isinstance(score_item, dict):
                                    reward_value = score_item.get("score")
                                    # collect extra infos mirroring batch reward manager
                                    for k, v in score_item.items():
                                        reward_extra_infos_dict.setdefault(k, []).append(v)
                                else:
                                    reward_value = score_item

                                rewards_scalar.append(float(reward_value))
                                if length > 0:
                                    reward_tensor[i, length - 1] = float(reward_value)

                            # expose acc following reward manager convention
                            batch.batch["acc"] = torch.tensor(
                                rewards_scalar, dtype=torch.float32, device=prompts.device
                            )
                        elif self.config.reward_model.launch_reward_fn_async:
                            used_async_reward = True
                            future_reward = compute_reward_async.remote(
                                data=batch, reward_fn=self.reward_fn, actor_wg=self.actor_rollout_wg
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn, actor_wg=self.actor_rollout_wg
                            )

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        metrics.update({"actor/entropy": entropy_agg.detach().item()})
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if 'used_async_reward' in locals() and used_async_reward:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
