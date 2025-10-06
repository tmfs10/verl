# Copyright 2025 Individual Contributor

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


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

    # Execute ES across all ranks
    print("Starting ES training via Ray...")
    wg.run_es()


if __name__ == "__main__":
    main()

