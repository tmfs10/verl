
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
import copy

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
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.dataset.rl_dataset import RLHFDataset, _to_hf_dataset


def _jsonable(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prompt_record_from_batch(batch, row_idx: int) -> dict:
    record = {}
    for key, values in batch.non_tensor_batch.items():
        if key in {"uid"}:
            continue
        try:
            record[key] = _jsonable(values[row_idx])
        except Exception:
            pass
    return record


def _data_files_from_config(data_config) -> list[str]:
    value = data_config.get("path", None)
    if value is None:
        value = data_config.get("train_files", None)
    if value is None:
        raise ValueError("Generation eval requires either data.path or data.train_files")
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        files = [str(item) for item in value]
        if not files:
            raise ValueError("Generation eval received an empty data file list")
        return files
    raise TypeError(f"Unsupported generation data file config type: {type(value).__name__}")


def _get_generation_custom_reward_fn(config):
    if config.get("reward", None) is not None:
        return get_custom_reward_fn(config)

    reward_fn_config = config.get("custom_reward_function", None)
    if not reward_fn_config:
        return None
    if OmegaConf.is_config(reward_fn_config):
        reward_fn_config = OmegaConf.to_container(reward_fn_config, resolve=True)
    compat_config = OmegaConf.create({"reward": {"custom_reward_function": reward_fn_config}})
    return get_custom_reward_fn(compat_config)


def _resume_line_number(row: dict, row_idx: int) -> int:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and "line_number" in extra_info:
        try:
            return int(json.loads(str(extra_info["line_number"])))
        except (TypeError, ValueError, json.JSONDecodeError):
            return int(extra_info["line_number"])
    return row_idx

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

    def _prepare_prompt_messages(self, example: dict, item=None, *, masked_positions=None):
        return copy.deepcopy(example[self.prompt_key])

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        messages = row_dict.get("raw_prompt") or self._build_messages(row_dict)
        apply_kwargs = dict(**self.apply_chat_template_kwargs)
        if self.tool_schemas is not None:
            apply_kwargs["tools"] = self.tool_schemas
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **apply_kwargs,
        )
        prompt_ids = normalize_token_ids(prompt_ids)
        if len(prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                prompt_ids = prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                prompt_ids = prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise ValueError(f"Prompt length {len(prompt_ids)} exceeds max_prompt_length {self.max_prompt_length}")
            else:
                raise ValueError(f"Unsupported generation prompt truncation mode: {self.truncation}")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for generation eval.")

        input_ids = torch.full((self.max_prompt_length,), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((self.max_prompt_length,), dtype=torch.long)
        token_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        if len(prompt_ids) > 0:
            input_ids[-len(prompt_ids) :] = token_tensor
            attention_mask[-len(prompt_ids) :] = 1
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = compute_position_id_with_mask(attention_mask)
        return row_dict
    
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
                dataframe = dataframe.filter(
                    lambda row, idx: (idx % self.world_size == self.rank) and (_resume_line_number(row, idx) not in exclude_indices),
                    with_indices=True,
                    desc=f"Sharding/resume filtering {parquet_file}",
                )
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
    seed = config.rollout.seed
    indices_done_path = os.path.join(config.data.output_path, f'rank_{config.rank}_{seed}_indices_done.txt')

    if not config.resume:
        shutil.rmtree(config.data.output_path, ignore_errors=True)
    os.makedirs(config.data.output_path, exist_ok=True)

    assert config.data.batch_size % (config.trainer.n_gpus_per_node * config.trainer.nnodes) == 0, f"batch_size {config.data.batch_size} must be divisible by n_gpus_per_node {config.trainer.n_gpus_per_node} * nnodes {config.trainer.nnodes}"

    from verl.utils import hf_processor
    processor = hf_processor(local_path, trust_remote_code=True, use_fast=True)
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_sampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    dataset = GenerateDataset(data_files=_data_files_from_config(config.data), tokenizer=tokenizer, processor=processor, config=config.data, exclude_indices_file=indices_done_path, rank=config.rank, world_size=config.world_size)
    dataloader = StatefulDataLoader(dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle, num_workers=config.data.dataloader.num_workers, collate_fn=collate_fn)

    config.data.output_path = os.path.join(config.data.output_path, f'rank_{config.rank}_{seed}.jsonl')

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_score = _get_generation_custom_reward_fn(config)
    if compute_score is None:
        raise ValueError("Generation eval requires a custom reward function.")

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    mode = "a" if config.resume else "w"

    with open(config.data.output_path, mode) as f, open(indices_done_path, mode) as f_done:
        iteration = 0
        for i_batch, batch_dict in enumerate(dataloader):
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
            print(f'Generated batch: {i_batch}')
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
            elif isinstance(score_result, list):
                scores = [float(item) for item in score_result]
            elif score_result is not None:
                scores = [float(score_result)] * len(responses_str)
            
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
                prompt_record = _prompt_record_from_batch(batch, i * config.data.n_samples)
                o = {
                    'prompt': input_texts[i],
                    'responses': [],
                    'scores': [],
                    'prompt_record': prompt_record,
                    'data_source': prompt_record.get('data_source'),
                    'extra_info': prompt_record.get('extra_info'),
                    'reward_model': prompt_record.get('reward_model'),
                }
                assert batch.batch['attention_mask'].shape[-1] == config.data.max_prompt_length + config.data.max_response_length, f"data['attention_mask'].shape[-1] == {batch.batch['attention_mask'].shape[-1]} != {config.data.max_prompt_length + config.data.max_response_length}"
                for n_sample in range(config.data.n_samples):
                    data_item = batch[i * config.data.n_samples + n_sample]
                    valid_response_length = data_item.batch["attention_mask"][config.data.max_prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                    o['responses'].append(response_str)
                    o['scores'].append(scores[i * config.data.n_samples + n_sample])
                print(json.dumps(o), file=f)
                line_number = _resume_line_number(
                    {"extra_info": batch.non_tensor_batch['extra_info'][i * config.data.n_samples]},
                    i_batch * num_prompts + i,
                )
                f_done.write(f"{line_number}\n")

            print(metrics_line)

if __name__ == "__main__":
    main()
