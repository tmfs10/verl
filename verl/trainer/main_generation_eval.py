
"""
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray
import json
import datasets
import time
import shutil
import torch

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint
from collections import defaultdict

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.dataset.rl_dataset import RLHFDataset, _to_hf_dataset

def format_metrics_line(iteration, num_prompts, n_samples, metrics):
    """Format metrics as a single line string for logging."""
    parts = [f"Iter {iteration:3d}"]
    
    # Batch info
    parts.append(f"batch={num_prompts}x{n_samples}={num_prompts * n_samples}")
    
    # Response lengths
    parts.append(f"resp_len[mean={metrics['response_length/mean']:.1f}, min={metrics['response_length/min']:.0f}, max={metrics['response_length/max']:.0f}]")
    
    # Rewards
    parts.append(f"reward[mean={metrics['reward/mean']:.4f}, min={metrics['reward/min']:.4f}, max={metrics['reward/max']:.4f}]")
    
    # Add extra reward metrics if they exist
    extra_reward_metrics = []
    for key in sorted(metrics.keys()):
        if key.startswith('reward/') and not any(key.endswith(suffix) for suffix in ['/mean', '/max', '/min']):
            # This is a base key like 'reward/accuracy'
            base_key = key
            if f'{base_key}/mean' in metrics:
                extra_reward_metrics.append(f"{base_key.split('/')[-1]}={metrics[f'{base_key}/mean']:.3f}")
    if extra_reward_metrics:
        parts.append(f"extra[{', '.join(extra_reward_metrics)}]")
    
    # Timing
    parts.append(f"time[gen={metrics['timing/generation_s']:.2f}s, score={metrics['timing/scoring_s']:.2f}s, total={metrics['timing/total_s']:.2f}s]")
    
    # Performance
    parts.append(f"perf[{metrics['perf/tokens_per_sec']:.1f} tok/s, {metrics['perf/tokens_per_sec_per_gpu']:.1f} tok/s/gpu]")
    
    return " | ".join(parts)

class GenerateDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, processor, config, exclude_indices_file, rank, world_size):
        self.exclude_indices_file = exclude_indices_file
        self.rank = rank
        self.world_size = world_size
        super().__init__(data_files, tokenizer, config, processor)
    
    def _read_files_and_tokenize(self):
        if os.path.exists(self.exclude_indices_file):
            with open(self.exclude_indices_file, 'r') as f:
                exclude_indices = set(int(line.strip()) for line in f)
        else:
            exclude_indices = set()

        print(f'Skipping {len(exclude_indices)} indices')
        
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            if parquet_file.endswith('.parquet'):
                # read parquet files and cache
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith('.jsonl'):
                d = []
                with open(parquet_file, 'r') as f:
                    num_lines = 0
                    for i, line in enumerate(f):
                        num_lines += 1
                        if (i % self.world_size != self.rank) or (i in exclude_indices):
                            continue
                        try:
                            line = json.loads(line)
                            if 'extra_info' not in line:
                                line['extra_info'] = {}
                            line['extra_info']['line_number'] = json.dumps(i)
                            d.append(line)
                        except:
                            pass
                print(f'Read {len(d)}/{num_lines} lines for {parquet_file}')
                dataframe = pd.DataFrame(d)
                dataframe = _to_hf_dataset(dataframe)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)


@hydra.main(config_path="config", config_name="generate_and_eval", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    os.makedirs(config.data.output_path, exist_ok=True)
    indices_done_path = os.path.join(config.data.output_path, f'rank_{config.rank}_indices_done.txt')

    if not config.resume:
        shutil.rmtree(config.data.output_path, ignore_errors=True)

    from verl.utils import hf_processor
    processor = hf_processor(local_path, trust_remote_code=True, use_fast=True)
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_sampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    dataset = GenerateDataset(data_files=[config.data.path], tokenizer=tokenizer, processor=processor, config=config.data, exclude_indices_file=indices_done_path, rank=config.rank, world_size=config.world_size)
    dataloader = StatefulDataLoader(dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle, num_workers=config.data.dataloader.num_workers, collate_fn=collate_fn)

    config.data.output_path = os.path.join(config.data.output_path, f'rank_{config.rank}.jsonl')

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    compute_score = get_custom_reward_fn(config)
    mode = "a" if config.resume else "w"

    with open(config.data.output_path, mode) as f, open(indices_done_path, mode) as f_done:
        iteration = 0
        for batch_dict in dataloader:
            iteration += 1
            timing = {}
            metrics = {}
            timing['iteration_start'] = time.time()
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            num_prompts = len(batch)
            valid_prompt_lengths = batch.batch['attention_mask'].sum(dim=-1)
            assert valid_prompt_lengths.min().item() > 0, f"valid_prompt_lengths.min().item() == {valid_prompt_lengths.min().item()}"
            input_texts = [tokenizer.decode(ids[-valid_prompt_lengths[i]:], skip_special_tokens=False) for i, ids in enumerate(batch.batch['input_ids'])]

            batch = batch.repeat(repeat_times=config.data.n_samples, interleave=True)
            data_gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

            data_gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": config.rollout.do_sample,
                "validate": True,
                "global_steps": 0,
            }

            timing['generation_start'] = time.time()
            data_gen_batch = wg.generate_sequences(data_gen_batch)
            timing['generation_end'] = time.time()
            timing['generation_duration'] = timing['generation_end'] - timing['generation_start']

            output_ids = data_gen_batch.batch["responses"]
            responses_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            batch = batch.union(data_gen_batch)
            ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in batch]
            extras = batch.non_tensor_batch.get("extra_info", [None] * len(batch))
            
            timing['scoring_start'] = time.time()
            # Call compute_score - returns a list of dicts, one per response
            score_result = compute_score(
                data_sources=None,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                **config.reward_model.get("reward_kwargs", {}),
            )
            timing['scoring_end'] = time.time()
            timing['scoring_duration'] = timing['scoring_end'] - timing['scoring_start']
            
            # Handle score result - list of dicts
            scores = []
            reward_extra_infos = defaultdict(list)
            
            if isinstance(score_result, list) and len(score_result) > 0 and isinstance(score_result[0], dict):
                # score_result is a list of dicts
                # Find common keys across all dicts
                all_keys = set()
                for item in score_result:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                
                # Extract scores and other numeric fields
                for item in score_result:
                    # Get the score value (might be under 'score', 'reward', or other keys)
                    if 'score' in item:
                        scores.append(item['score'])
                    elif 'reward' in item:
                        scores.append(item['reward'])
                    else:
                        # Default to 0 if no score field found
                        scores.append(0.0)
                    
                    # Collect other numeric fields
                    for key in all_keys:
                        if key not in ['score', 'reward'] and key in item:
                            value = item[key]
                            if isinstance(value, (int, float)):
                                reward_extra_infos[key].append(value)
            
            # Compute response length statistics
            response_lengths = []
            for i in range(len(batch)):
                response_mask = batch.batch["response_mask"][i] if "response_mask" in batch.batch else batch.batch["attention_mask"][i][-output_ids.shape[-1]:]
                response_length = response_mask.sum().item() if torch.is_tensor(response_mask) else response_mask.sum()
                response_lengths.append(response_length)
            
            # Calculate metrics
            metrics['response_length/mean'] = np.mean(response_lengths)
            metrics['response_length/max'] = np.max(response_lengths)
            metrics['response_length/min'] = np.min(response_lengths)

            if len(scores) > 0:
                metrics['reward/mean'] = np.mean(scores)
                metrics['reward/max'] = np.max(scores)
                metrics['reward/min'] = np.min(scores)
            
            # Add any extra reward metrics
            for key, values in reward_extra_infos.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    metrics[f'reward/{key}/mean'] = np.mean(values)
                    metrics[f'reward/{key}/max'] = np.max(values)
                    metrics[f'reward/{key}/min'] = np.min(values)
            
            # Add timing metrics
            metrics['timing/generation_s'] = timing['generation_duration']
            metrics['timing/scoring_s'] = timing['scoring_duration']
            metrics['timing/total_s'] = time.time() - timing['iteration_start']
            
            # Calculate throughput
            total_tokens = sum(response_lengths)
            n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
            metrics['perf/tokens_per_sec'] = total_tokens / timing['generation_duration']
            metrics['perf/tokens_per_sec_per_gpu'] = metrics['perf/tokens_per_sec'] / n_gpus
            metrics['perf/total_tokens'] = total_tokens
            
            # Print metrics for this iteration
            metrics_line = format_metrics_line(iteration, num_prompts, config.data.n_samples, metrics)

            assert len(batch) == num_prompts * config.data.n_samples, f"len(data) == {len(batch)} != {num_prompts * config.data.n_samples}"
            for i in range(num_prompts):
                o = {'prompt': input_texts[i], 'responses': [], 'scores': []}
                assert batch.batch['attention_mask'].shape[-1] == config.data.max_prompt_length + config.data.max_response_length, f"data['attention_mask'].shape[-1] == {batch.batch['attention_mask'].shape[-1]} != {config.data.max_prompt_length + config.data.max_response_length}"
                for n_sample in range(config.data.n_samples):
                    data_item = batch[i * config.data.n_samples + n_sample]
                    valid_response_length = data_item.batch["attention_mask"][config.data.max_prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                    o['responses'].append(response_str)
                    o['scores'].append(scores[i * config.data.n_samples + n_sample])
                print(json.dumps(o), file=f)
                line_number = json.loads(batch.non_tensor_batch['extra_info'][i * config.data.n_samples]['line_number'])
                f_done.write(f"{line_number}\n")

            print(metrics_line)

if __name__ == "__main__":
    main()
