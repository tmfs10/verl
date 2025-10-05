import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
import hydra
import os

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="generate_and_eval", version_base=None)
def main(config):
    dataset = []
    output_eval_dir = config.data.output_path
    os.makedirs(output_eval_dir, exist_ok=True)
    output_filepaths = [f for f in os.listdir(output_eval_dir) if f.endswith(".jsonl")]
    max_line_number = max([int(f.split(".")[0][len("eval_"):]) for f in output_filepaths]) if output_filepaths else -1
    rank = config.get("rank", 0)
    world_size = config.get("world_size", 1)
    num_lines = 0
    with open(config.data.val_files, "r") as f:
        for i, line in enumerate(f):
            num_lines += 1
    rank_batch_size = int(math.ceil(num_lines / world_size))
    rank_batches = [[rank_batch_size * local_rank, min(rank_batch_size * (local_rank + 1), num_lines)] for local_rank in range(world_size)]
    with open(config.data.val_files, "r") as f:
        for i, line in enumerate(f):
            if i < rank_batches[rank][0] or i >= rank_batches[rank][1]:
                continue
            data = json.loads(line)
            line_number = data.get('extra_info', {}).get('line_number', None)
            line_number = json.loads(line_number) if line_number is not None else None
            if line_number is None:
                line_number = i
                data['extra_info']['line_number'] = json.dumps(line_number)
            if line_number <= max_line_number:
                continue
            dataset.append(data)
    print(f"Processing {len(dataset)} items. Starting from line {max_line_number+1}")

    batch_size = config.data.batch_size
    compute_score = get_custom_reward_fn(config)
    for i in range(0, len(dataset), batch_size):
        print(f"Processing batch {i} of size {batch_size}")
        batch = dataset[i:i+batch_size]
        ground_truths = [item["reward_model"]["ground_truth"] for item in batch]
        extras = [item["extra_info"] for item in batch]
        max_line_number = max([json.loads(item["extra_info"]["line_number"]) for item in batch])
        outputs = [item.pop("outputs") for item in batch]
        num_outputs = [len(output) for output in outputs]
        total_outputs = sum(num_outputs)
        ground_truths = [g for i, g in enumerate(ground_truths) for _ in range(num_outputs[i])]
        extras = [e for i, e in enumerate(extras) for _ in range(num_outputs[i])]
        responses_str = [o for output in outputs for o in output]
    
        score_result = compute_score(
            data_sources=None,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **config.reward_model.get("reward_kwargs", {}),
        )
        print(f"Computed {len(score_result)} scores")

        assert len(score_result) == total_outputs, f"Expected {total_outputs} scores, got {len(score_result)}"

        j = 0
        output_filepath = os.path.join(output_eval_dir, f"eval_{max_line_number}.jsonl")
        print(f"Writing to {output_filepath}")
        with open(output_filepath, "w") as f:
            for i, d in enumerate(batch):
                responses = responses_str[j:j+num_outputs[i]]
                d['response_strs'] = responses
                d['scores'] = score_result[j:j+num_outputs[i]]
                j += num_outputs[i]
                print(json.dumps(d), file=f)

    # create done file at the end
    done_file = os.path.join(output_eval_dir, f"done_{rank}.txt")
    with open(done_file, "w") as f:
        f.write(f"Done processing {max_line_number} lines")
    print(f"Done processing {max_line_number} lines")

if __name__ == "__main__":
    main()
