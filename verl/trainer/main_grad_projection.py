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

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_grad_projection_trainer import RayGradProjectionTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local


@hydra.main(config_path="config", config_name="ppo_grad_projection", version_base=None)
def main(config):
    from pprint import pprint

    print("Loaded config (resolved):")
    pprint(OmegaConf.to_container(config, resolve=True))
    run_grad_projection(config)


def run_grad_projection(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_worker(self, config):
        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("Gradient projection worker only supports FSDP strategies for now.")

        from verl.workers.grad_projection_workers import FSDPGradProjectionWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(FSDPGradProjectionWorker)
        return RayWorkerGroup

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        # Avoid optimizer initialization since we only need gradients.
        with open_dict(config):
            config.actor_rollout_ref.actor.optim = None
            # Ensure deprecated global micro-batch size doesn't override per-GPU setting.
            if (
                config.actor_rollout_ref.actor.get("ppo_micro_batch_size", None) is not None
                and config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", None) is not None
            ):
                config.actor_rollout_ref.actor.ppo_micro_batch_size = None

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_tokenizer
        from verl.utils.dataset.raw_prompt_response_dataset import RawPromptResponseDataset
        from verl.utils.dataset.rl_dataset import collate_fn_pad_to_batch_max
        from verl.trainer.main_ppo import create_rl_sampler

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        train_dataset = RawPromptResponseDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=None,
            config=config.data,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        def collate_fn(data_list):
            return collate_fn_pad_to_batch_max(data_list, pad_token_id=pad_token_id)

        ray_worker_group_cls = self.add_actor_worker(config)
        resource_pool_manager = self.init_resource_pool_mgr(config)

        trainer = RayGradProjectionTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=None,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=None,
            val_reward_fn=None,
            train_dataset=train_dataset,
            val_dataset=None,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
