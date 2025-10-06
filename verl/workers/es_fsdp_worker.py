import datetime
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from verl.single_controller.base import Worker
from verl.trainer.ppo.reward import compute_reward, get_custom_reward_fn, load_reward_manager
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig
from verl.utils.config import omega_conf_to_dataclass

from verl.single_controller.base.decorator import Dispatch, register

from trainer.es.config import ESAlgoConfig, ESConfig, ESDataConfig, ESRewardConfig
from trainer.es.noise import iter_flat_params, stateless_normal_like
from trainer.es.rollout import HFGenerationArgs, hf_generate_batch, tokenize_batch_chat


def _init_dist_if_needed(timeout_s: int = 600):
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_s),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )


def _create_device_mesh():
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=["fsdp"])


def _make_model_fsdp(model_cfg: HFModelConfig, device_mesh) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM
    from verl.utils.fsdp_utils import get_init_weight_context_manager, get_fsdp_wrap_policy, init_fn

    hf_config = model_cfg.hf_config

    with get_init_weight_context_manager(use_meta_tensor=not hf_config.tie_word_embeddings, mesh=device_mesh)():
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.local_path,
            config=hf_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_cfg.trust_remote_code,
            attn_implementation=getattr(hf_config, "attn_implementation", "flash_attention_2"),
        )

    auto_wrap_policy = get_fsdp_wrap_policy(
        model,
        config={"min_num_params": 0},
        is_lora=(getattr(model_cfg, "lora_rank", 0) or 0) > 0,
    )

    fsdp_model = FSDP(
        model,
        param_init_fn=init_fn,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.device(get_device_name(), dist.get_rank() % torch.cuda.device_count())
        if get_device_name() == "cuda" and torch.cuda.is_available()
        else None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=None,
        sync_module_states=True,
        device_mesh=device_mesh,
        forward_prefetch=False,
    )
    return fsdp_model


@dataclass
class RolloutState:
    chat_list: list
    ground_truth: list | None
    data_source: list | None


class ESFSDPWorker(Worker):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        _init_dist_if_needed(timeout_s=600)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device_mesh = _create_device_mesh()

        # resolve config dataclasses
        self.model_cfg: HFModelConfig = omega_conf_to_dataclass(config.model)
        self.rollout_cfg: RolloutConfig = omega_conf_to_dataclass(config.rollout)
        self.algo: ESAlgoConfig = omega_conf_to_dataclass(config.algorithm)
        self.data_cfg: ESDataConfig = omega_conf_to_dataclass(config.data)
        self.reward_cfg: ESRewardConfig = omega_conf_to_dataclass(config.reward_model)

        # Tokenizer and padding setup
        self.tokenizer = self.model_cfg.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model + optimizer
        self.model = _make_model_fsdp(self.model_cfg, self.device_mesh)
        self.model.train(False)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.algo.lr, weight_decay=self.algo.weight_decay)

        # Data and reward
        self.rollout_state = self._load_dataset()
        self.reward_fn = self._build_reward_fn()

        # Generation args
        self.gen_args = HFGenerationArgs(
            do_sample=self.rollout_cfg.do_sample,
            temperature=float(self.rollout_cfg.temperature),
            top_p=float(self.rollout_cfg.top_p),
            top_k=int(self.rollout_cfg.top_k),
            response_length=int(self.rollout_cfg.response_length),
        )

    def _load_dataset(self) -> RolloutState:
        df = pd.read_parquet(self.data_cfg.path)
        chat_list = df[self.data_cfg.prompt_key].tolist()
        chat_list = [x.tolist() if hasattr(x, "tolist") else x for x in chat_list]
        ground_truth = None
        if self.data_cfg.ground_truth_key and self.data_cfg.ground_truth_key in df.columns:
            ground_truth = df[self.data_cfg.ground_truth_key].tolist()
        data_source = None
        if self.data_cfg.reward_fn_key in df.columns:
            data_source = df[self.data_cfg.reward_fn_key].tolist()
        return RolloutState(chat_list=chat_list, ground_truth=ground_truth, data_source=data_source)

    def _build_reward_fn(self):
        cfg = OmegaConf.create(
            {
                "custom_reward_function": self.config.custom_reward_function,
                "reward_model": {
                    "reward_manager": self.reward_cfg.reward_manager,
                    "reward_kwargs": self.reward_cfg.reward_kwargs,
                },
                "data": {"reward_fn_key": self.data_cfg.reward_fn_key},
            }
        )
        reward_fn = get_custom_reward_fn(cfg)
        if reward_fn is not None:
            return reward_fn
        return load_reward_manager(cfg, tokenizer=self.tokenizer, num_examine=0, **self.reward_cfg.reward_kwargs)

    def _iter_assigned_dirs(self, total_dirs: int) -> Iterable[int]:
        for idx in range(self.rank, total_dirs, self.world_size):
            yield idx

    def _apply_perturbation(self, dir_seed: int, scale: float):
        with torch.no_grad():
            for key, p in iter_flat_params(self.model):
                eps = stateless_normal_like(p, dir_seed=dir_seed, key=key)
                p.add_(scale * eps)

    def _eval_return(self) -> float:
        bsz = self.algo.rollout_batch_size or 1
        total = len(self.rollout_state.chat_list)
        indices = list(range(self.rank, total, self.world_size))[:bsz]
        if len(indices) == 0:
            return 0.0

        chats = [self.rollout_state.chat_list[i] for i in indices]
        input_ids, attn, pos = tokenize_batch_chat(
            tokenizer=self.tokenizer,
            chat_list=chats,
            prompt_length=int(self.rollout_cfg.prompt_length),
            apply_chat_template_kwargs=self.data_cfg.apply_chat_template_kwargs,
        )

        data = hf_generate_batch(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=pos,
            gen_args=self.gen_args,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        non_tensor_batch = {}
        if self.rollout_state.ground_truth is not None:
            non_tensor_batch.setdefault("reward_model", {})
            non_tensor_batch["reward_model"]["ground_truth"] = [self.rollout_state.ground_truth[i] for i in indices]
        if self.rollout_state.data_source is not None:
            non_tensor_batch[self.data_cfg.reward_fn_key] = [self.rollout_state.data_source[i] for i in indices]
        data.non_tensor_batch.update(non_tensor_batch)

        reward_tensor, _ = compute_reward(data, self.reward_fn)
        returns = reward_tensor.sum(dim=1).mean().item()
        return float(returns)

    def _gather_dir_returns(self, local_entries: List[Tuple[int, float, float]]):
        gathered: List[List[Tuple[int, float, float]]] = [None for _ in range(self.world_size)]  # type: ignore
        dist.all_gather_object(gathered, local_entries)
        merged: List[Tuple[int, float, float]] = []
        for lst in gathered:
            if lst:
                merged.extend(lst)
        merged.sort(key=lambda x: x[0])
        return merged

    def _compute_advantages(self, dir_stats: List[Tuple[int, float, float]]):
        import numpy as np

        r_plus = np.array([x[1] for x in dir_stats], dtype=float)
        r_minus = np.array([x[2] for x in dir_stats], dtype=float)
        adv = r_plus - r_minus if self.algo.antithetic else r_plus

        mode = (self.algo.normalize or "none").lower()
        if mode == "none":
            pass
        elif mode == "standardize":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        elif mode == "centered_rank":
            ranks = adv.argsort().argsort().astype(float)
            ranks = (ranks + 1) / (len(ranks) + 1)
            adv = ranks - 0.5
        else:
            raise ValueError(f"Unknown normalize mode: {self.algo.normalize}")
        return adv

    def _accumulate_param_space_grad(self, seeds: List[int], coeffs: List[float]):
        for _, p in iter_flat_params(self.model):
            p.grad = None

        inv_scale = 1.0 / (float(len(seeds)) * float(self.algo.sigma))
        with torch.no_grad():
            for seed, c in zip(seeds, coeffs):
                coeff = float(c) * inv_scale
                for key, p in iter_flat_params(self.model):
                    eps = stateless_normal_like(p, dir_seed=int(seed), key=key)
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(coeff * eps)

        if self.algo.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.algo.grad_clip_norm)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_es(self):
        torch.manual_seed(int(self.algo.seed))
        total_dirs = int(self.algo.n_directions)

        for it in range(int(self.algo.total_iters)):
            local_dir_stats: List[Tuple[int, float, float]] = []
            for dir_idx in self._iter_assigned_dirs(total_dirs):
                self._apply_perturbation(dir_seed=dir_idx, scale=float(self.algo.sigma))
                r_plus = self._eval_return()
                self._apply_perturbation(dir_seed=dir_idx, scale=-float(self.algo.sigma))

                if self.algo.antithetic:
                    self._apply_perturbation(dir_seed=dir_idx, scale=-float(self.algo.sigma))
                    r_minus = self._eval_return()
                    self._apply_perturbation(dir_seed=dir_idx, scale=float(self.algo.sigma))
                else:
                    r_minus = 0.0

                local_dir_stats.append((dir_idx, float(r_plus), float(r_minus)))

            dir_stats = self._gather_dir_returns(local_dir_stats)
            adv = self._compute_advantages(dir_stats)
            seeds = [idx for idx, _, _ in dir_stats]
            coeffs = adv.tolist()

            self._accumulate_param_space_grad(seeds, coeffs)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.rank == 0:
                avg_r = sum([(p + max(0.0, m)) for _, p, m in dir_stats]) / max(1, len(dir_stats))
                print(
                    f"[ES:ray] iter={it} dirs={total_dirs} sigma={self.algo.sigma:.3g} lr={self.algo.lr:.3g} avg_r={avg_r:.4f}"
                )
