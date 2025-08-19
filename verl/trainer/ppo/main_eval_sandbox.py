import json
from collections import defaultdict

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    dataset = []
    with open(config.data.path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    ground_truths = [item["reward_model"]["ground_truth"] for item in dataset]
    extras = [item["extra_info"] for item in dataset]
    outputs = [item.pop("outputs") for item in dataset]
    num_outputs = [len(output) for output in outputs]
    total_outputs = sum(num_outputs)
    ground_truths = [g for g in ground_truths for _ in range(num_outputs)]
    extras = [e for e in extras for _ in range(num_outputs)]
    responses_str = [o for output in outputs for o in output]

    compute_score = get_custom_reward_fn(config)
    score_result = compute_score(
        data_sources=None,
        solution_strs=responses_str,
        ground_truths=ground_truths,
        extra_infos=extras,
        **config.reward_model.get("reward_kwargs", {}),
    )

    assert len(score_result) == total_outputs, f"Expected {total_outputs} scores, got {len(score_result)}"

    with open(config.data.output_path, "w") as f:
        j = 0
        for i, d in enumerate(dataset):
            for _ in range(num_outputs[i]):
                d2 = d.copy()
                d2['response_str'] = responses_str[j]
                assert type(score_result[j]) == dict, f"Expected dict, got {type(score_result[j])}"
                d2.update(score_result[j])
                print(json.dumps(d2), file=f)
                j += 1
