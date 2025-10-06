from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig


@dataclass
class ESAlgoConfig(BaseConfig):
    total_iters: int = 1000
    n_directions: int = 64
    sigma: float = 0.02
    lr: float = 0.01
    weight_decay: float = 0.0
    antithetic: bool = True
    normalize: str = "standardize"  # one of: none, standardize, centered_rank
    seed: int = 0

    # number of prompts per rollout evaluation
    rollout_batch_size: int = 64
    # number of batches per direction evaluation (optional)
    rollout_batches_per_iter: int = 1

    # gradient clip for stability (applied to parameter-space gradient estimate)
    grad_clip_norm: Optional[float] = None


@dataclass
class ESDataConfig(BaseConfig):
    path: str = MISSING  # parquet file path containing prompts (chat template list) and optional ground truth
    prompt_key: str = "prompt"  # column with chat template (list of messages)
    reward_fn_key: str = "data_source"  # key to differentiate reward routing
    batch_size: Optional[int] = None  # alias to algo.rollout_batch_size if provided

    # Optional: if rewards require ground truth, specify the column
    ground_truth_key: Optional[str] = None
    # Additional kwargs passed to tokenizer.apply_chat_template
    apply_chat_template_kwargs: dict = field(default_factory=dict)


@dataclass
class ESRewardConfig(BaseConfig):
    enable: bool = True
    # support the same layout as PPO reward section (e.g. reward_manager name and kwargs)
    reward_manager: str = "naive"
    reward_kwargs: dict = field(default_factory=dict)


@dataclass
class ESConfig(BaseConfig):
    # Distributed setup
    nnodes: int = 1
    n_gpus_per_node: int = 1
    device: str = "cuda"

    # Components
    model: HFModelConfig = field(default_factory=HFModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    algorithm: ESAlgoConfig = field(default_factory=ESAlgoConfig)
    data: ESDataConfig = field(default_factory=ESDataConfig)
    reward_model: ESRewardConfig = field(default_factory=ESRewardConfig)

    # Custom reward entry (optional)
    custom_reward_function: dict = field(default_factory=dict)

