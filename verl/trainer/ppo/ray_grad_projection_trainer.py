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

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role


class RayGradProjectionTrainer(RayPPOTrainer):
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
        from torch.utils.data import SequentialSampler

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            train_sampler = SequentialSampler(self.train_dataset)
        elif not isinstance(train_sampler, SequentialSampler):
            print("Warning: overriding non-sequential sampler with SequentialSampler for stable indices.")
            train_sampler = SequentialSampler(self.train_dataset)
        if collate_fn is None:
            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        # Hard-code single-sample batches to avoid any padding requirements.
        sp_size = self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        dp_size = max(1, world_size // sp_size)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=dp_size,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if self.val_dataset is None:
            class _EmptyDataset:
                def __len__(self):
                    return 0

                def __getitem__(self, idx):
                    raise IndexError("Empty dataset")

            self.val_dataset = _EmptyDataset()
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=1,
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=1,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate_fn,
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        self.resource_pool_to_cls[resource_pool]["actor"] = actor_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_wg = all_wg["actor"]
        self.actor_wg.init_model()

    def fit(self):
        import os
        import re
        import torch

        from verl.utils.fs import local_mkdir_safe

        rademacher_k = self.config.trainer.get("rademacher_k", None)
        if rademacher_k is None:
            raise ValueError("trainer.rademacher_k must be set for gradient projection")
        rademacher_flush_every = self.config.trainer.get("rademacher_flush_every", 0) or 0
        if rademacher_flush_every < 0:
            raise ValueError("trainer.rademacher_flush_every must be >= 0")

        output_dir = self.config.trainer.default_local_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        local_mkdir_safe(output_dir)
        max_saved_idx = -1
        existing_partial_paths = []
        for filename in os.listdir(output_dir):
            match = re.match(r"grads_(\d+)\.pt$", filename)
            if match:
                max_saved_idx = max(max_saved_idx, int(match.group(1)))
                existing_partial_paths.append(os.path.join(output_dir, filename))
        resume_from_idx = max_saved_idx + 1

        self.global_steps = 0
        total_items = len(self.train_dataset)
        processed = resume_from_idx
        self.projections = []
        all_projections = []

        if existing_partial_paths:
            existing_partial_paths.sort(key=lambda path: int(re.search(r"grads_(\d+)\.pt$", path).group(1)))
            for path in existing_partial_paths:
                all_projections.extend(torch.load(path, map_location="cpu"))

        for batch_dict in self.train_dataloader:
            if not hasattr(self, "_actor_dp_size"):
                dp_rank_mapping = self.actor_wg._query_dispatch_info("actor")
                self._actor_dp_size = max(dp_rank_mapping) + 1

            batch_size = batch_dict["input_ids"].shape[0]
            assert batch_size <= self._actor_dp_size, (
                f"Batch size must be <= dp_size ({self._actor_dp_size}), got {batch_size}"
            )
            batch = DataProto.from_single_dict(batch_dict)
            original_len = len(batch)
            if original_len % self._actor_dp_size != 0:
                padding_size = self._actor_dp_size - (original_len % self._actor_dp_size)
                batch.padding(padding_size=padding_size, padding_candidate="last")
            batch.meta_info.update(
                {
                    "temperature": self.config.actor_rollout_ref.rollout.get("temperature", 1.0),
                    "use_dynamic_bsz": False,
                    "micro_batch_size": 1,
                    "rademacher_k": rademacher_k,
                    "rademacher_seed": self.config.trainer.get("rademacher_seed", 0),
                    "rademacher_chunk_size": self.config.trainer.get("rademacher_chunk_size", 1_000_000),
                }
            )

            output = self.actor_wg.update_actor(batch)
            projection = output.batch["projection"][:original_len].cpu()
            idxs = batch.non_tensor_batch["idx"][:original_len]

            if hasattr(idxs, "tolist"):
                idxs = idxs.tolist()
            else:
                idxs = list(idxs)

            start_pos = 0
            if resume_from_idx > 0:
                while start_pos < len(idxs) and idxs[start_pos] < resume_from_idx:
                    start_pos += 1
                if start_pos >= len(idxs):
                    print(f"Processed {processed}/{total_items}")
                    continue

            if start_pos > 0:
                idxs = idxs[start_pos:]
                projection = projection[start_pos:]

            for idx, proj in zip(idxs, projection):
                if hasattr(idx, "item"):
                    idx = int(idx.item())
                else:
                    idx = int(idx)
                proj = proj.cpu()
                self.projections.append((idx, proj))
                all_projections.append((idx, proj))
            processed += len(idxs)
            print(f"Processed {processed}/{total_items}")

            if rademacher_flush_every > 0 and len(self.projections) >= rademacher_flush_every:
                last_idx = self.projections[-1][0]
                torch.save(self.projections, os.path.join(output_dir, f"grads_{last_idx}.pt"))
                self.projections = []

        if rademacher_flush_every > 0 and self.projections:
            last_idx = self.projections[-1][0]
            torch.save(self.projections, os.path.join(output_dir, f"grads_{last_idx}.pt"))
            self.projections = []

        torch.save(all_projections, os.path.join(output_dir, "grads.pt"))
        for filename in os.listdir(output_dir):
            if re.match(r"grads_(\d+)\.pt$", filename):
                os.remove(os.path.join(output_dir, filename))
