# Copyright 2025 Individual Contributor

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.tracking import Tracking
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="es_trainer", version_base=None)
def main(config):
    run_es(config)


def run_es(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    OmegaConf.resolve(config)
    from workers.es_fsdp_worker import ESFSDPWorker

    # Build worker group across GPUs
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ESFSDPWorker), config=config)
    resource_pool = RayResourcePool(
        process_on_nodes=[config.n_gpus_per_node] * config.nnodes,
        use_gpu=True,
        name_prefix="es_global_pool",
        max_colocate_count=1,
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.device,
        ray_wait_register_center_timeout=config.get("ray_kwargs", {}).get("ray_wait_register_center_timeout", 300),
    )

    # Logger similar to PPO's Tracking
    trainer_cfg = config.get("trainer", {})
    logger = Tracking(
        project_name=trainer_cfg.get("project_name", "verl_es"),
        experiment_name=trainer_cfg.get("experiment_name", "es_run"),
        default_backend=trainer_cfg.get("logger", ["console"]),
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Optional validation before training
    val_cfg = config.get("trainer", {})
    if val_cfg.get("val_before_train", True):
        per_rank_val = wg.validate()
        val_agg = _aggregate_rank_metrics(per_rank_val, wg.world_size)
        # prefix with val/
        val_logged = {f"val/{k}": v for k, v in val_agg.items() if not k.startswith("meta/")}
        logger.log(data=val_logged, step=0)
        if val_cfg.get("val_only", False):
            print("Validation only run completed.")
            return

    total_iters = int(config.algorithm.total_iters)
    for it in range(total_iters):
        # run one ES iteration remotely; returns a list of metrics dicts (per rank)
        per_rank = wg.es_step(step_idx=it)
        agg = _aggregate_rank_metrics(per_rank, wg.world_size)

        # add training step
        agg["training/global_step"] = it + 1
        logger.log(data=agg, step=it + 1)

        # Validation during training
        test_freq = val_cfg.get("test_freq", -1)
        if test_freq > 0 and ((it + 1) % test_freq == 0 or (it + 1) == total_iters):
            per_rank_val = wg.validate()
            val_agg = _aggregate_rank_metrics(per_rank_val, wg.world_size)
            val_logged = {f"val/{k}": v for k, v in val_agg.items() if not k.startswith("meta/")}
            logger.log(data=val_logged, step=it + 1)

    print("ES training completed.")


def _aggregate_rank_metrics(per_rank: list[dict], world: int) -> dict:
    agg = {}
    sum_weights = 0.0
    for m in per_rank:
        n = float(m.get("meta/n_samples", 0.0))
        sum_weights += n
        for k, v in m.items():
            if k.endswith("/max"):
                agg[k] = max(agg.get(k, float("-inf")), float(v))
            elif k.endswith("/min"):
                agg[k] = min(agg.get(k, float("inf")), float(v))
            elif k in ("timing_s/step",):
                agg[k] = max(agg.get(k, 0.0), float(v))
            elif k.startswith("timing_s/") or k.startswith("timing_per_token_ms/"):
                agg.setdefault(f"__sum__/{k}", 0.0)
                agg[f"__sum__/{k}"] += float(v)
            elif k.startswith("meta/"):
                agg[k] = agg.get(k, 0.0) + float(v)
            elif k.endswith("/mean") or k.startswith("reward/") or k.startswith("response_length/") or k.startswith("prompt_length/") or k == "response/aborted_ratio":
                agg.setdefault(f"__wsum__/{k}", 0.0)
                agg[f"__wsum__/{k}"] += float(v) * n
            else:
                agg[k] = float(v)

    for k in list(agg.keys()):
        if k.startswith("__sum__/timing_s/") or k.startswith("__sum__/timing_per_token_ms/"):
            name = k.removeprefix("__sum__/")
            agg[name] = agg[k] / world
            del agg[k]
        if k.startswith("__wsum__/"):
            name = k.removeprefix("__wsum__/")
            if sum_weights > 0:
                agg[name] = agg[k] / sum_weights
            else:
                agg[name] = 0.0
            del agg[k]
    return agg


if __name__ == "__main__":
    main()
