"""
Generate and score prompts with standalone async rollout servers.

This entrypoint preserves the JSONL contract used by ``main_generation_eval``
while avoiding its synchronous ``generate_sequences`` worker call. It is meant
for offline generation/evaluation jobs where vLLM HTTP replicas can load the
base model directly and stream completed prompt rows to disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any

import aiohttp
import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import ListConfig, OmegaConf

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.rollout.replica import get_rollout_replica_class


def _jsonable(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    if isinstance(value, (list, tuple, ListConfig)):
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


def _parse_line_number(value) -> int:
    if isinstance(value, str):
        try:
            return int(json.loads(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return int(value)
    return int(value)


def _resume_line_number(row: dict[str, Any], row_idx: int) -> int:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = None
    if isinstance(extra_info, dict) and "line_number" in extra_info:
        return _parse_line_number(extra_info["line_number"])
    return row_idx


def _as_messages(value: Any) -> list[dict[str, Any]]:
    messages = _jsonable(value)
    if not isinstance(messages, list):
        raise TypeError(f"Expected messages list, got {type(messages).__name__}")
    return messages


def _score_values(score_result, count: int) -> tuple[list[float], dict[str, list[float]]]:
    scores: list[float] = []
    reward_extra_infos: dict[str, list[float]] = defaultdict(list)
    if isinstance(score_result, list) and score_result and isinstance(score_result[0], dict):
        keys = set()
        for item in score_result:
            if isinstance(item, dict):
                keys.update(item.keys())
        for item in score_result:
            if "score" in item:
                scores.append(float(item["score"]))
            elif "reward" in item:
                scores.append(float(item["reward"]))
            else:
                scores.append(0.0)
            for key in keys:
                value = item.get(key) if isinstance(item, dict) else None
                if key not in {"score", "reward"} and isinstance(value, (int, float)):
                    reward_extra_infos[key].append(float(value))
    elif isinstance(score_result, list):
        scores = [float(item) for item in score_result]
    elif score_result is not None:
        scores = [float(score_result)] * count
    if len(scores) != count:
        raise ValueError(f"Reward function returned {len(scores)} scores for {count} responses")
    return scores, reward_extra_infos


def _read_done(done_path: Path) -> set[int]:
    if not done_path.exists():
        return set()
    done: set[int] = set()
    with done_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(line))
    return done


def _load_records(config, tokenizer, done_path: Path) -> list[dict[str, Any]]:
    done = _read_done(done_path)
    print(f"Skipping {len(done)} completed indices")
    records: list[dict[str, Any]] = []
    skipped_long = 0
    row_idx = 0
    for data_file in _data_files_from_config(config.data):
        if data_file.endswith(".parquet"):
            frame = pd.read_parquet(data_file)
            rows = (row.to_dict() for _, row in frame.iterrows())
        elif data_file.endswith(".jsonl"):
            def _jsonl_rows(path=data_file):
                with open(path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        row.setdefault("extra_info", {})
                        if isinstance(row["extra_info"], dict):
                            row["extra_info"].setdefault("line_number", json.dumps(line_number))
                        yield row

            rows = _jsonl_rows()
        else:
            raise ValueError(f"Unsupported generation data file: {data_file}")

        for row in rows:
            row = _jsonable(row)
            line_number = _resume_line_number(row, row_idx)
            if row_idx % int(config.world_size) != int(config.rank) or line_number in done:
                row_idx += 1
                continue
            messages = _as_messages(row[config.data.prompt_key])
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if len(prompt_ids) > int(config.data.max_prompt_length):
                skipped_long += 1
                row_idx += 1
                continue
            records.append(
                {
                    "row_idx": row_idx,
                    "line_number": line_number,
                    "row": row,
                    "messages": messages,
                    "prompt_text": tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    ),
                }
            )
            row_idx += 1
        print(f"Read {len(records)} usable records so far from {data_file}")

    if skipped_long:
        print(f"Skipped {skipped_long} records longer than {config.data.max_prompt_length} prompt tokens")
    print(f"dataset len: {len(records)}")
    return records


async def _wait_for_server(server_address: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    client_timeout = aiohttp.ClientTimeout(total=10)
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.get(f"http://{server_address}/health") as response:
                    if response.status < 500:
                        return
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = exc
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server {server_address}") from last_error


async def _submit_request(
    session: aiohttp.ClientSession,
    server_address: str,
    payload: dict[str, Any],
    *,
    max_attempts: int = 4,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            async with session.post(
                url=f"http://{server_address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123"},
                json=payload,
            ) as response:
                data = await response.json()
                if response.status >= 400 or "error" in data:
                    raise RuntimeError(f"vLLM request failed with status {response.status}: {data}")
                return str(data["choices"][0]["message"]["content"])
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = exc
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"vLLM request failed after {max_attempts} attempts on {server_address}") from last_error


async def _generate_on_server(
    server_address: str,
    model_path: str,
    n_samples: int,
    sampling_params: dict[str, Any],
    records: list[dict[str, Any]],
    offset: int,
) -> list[tuple[int, int, str]]:
    timeout = aiohttp.ClientTimeout(total=None)
    semaphore = asyncio.Semaphore(1)

    async def _bounded_submit(payload: dict[str, Any]) -> str:
        async with semaphore:
            return await _submit_request(session, server_address, payload)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        metadata = []
        for local_idx, record in enumerate(records):
            for sample_idx in range(n_samples):
                payload = {"model": model_path, "messages": record["messages"], **sampling_params}
                tasks.append(_bounded_submit(payload))
                metadata.append((offset + local_idx, sample_idx))
        outputs = await asyncio.gather(*tasks)
    return [(prompt_idx, sample_idx, output) for (prompt_idx, sample_idx), output in zip(metadata, outputs, strict=True)]


async def _generate_batch(
    server_addresses: list[str],
    model_path: str,
    n_samples: int,
    sampling_params: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[list[str]]:
    chunks: list[list[dict[str, Any]]] = []
    offsets: list[int] = []
    start = 0
    for chunk in np.array_split(np.arange(len(records)), len(server_addresses)):
        indices = [int(idx) for idx in chunk.tolist()]
        chunks.append([records[idx] for idx in indices])
        offsets.append(start)
        start += len(indices)

    results = await asyncio.gather(
        *[
            _generate_on_server(server_addresses[idx], model_path, n_samples, sampling_params, chunk, offsets[idx])
            for idx, chunk in enumerate(chunks)
            if chunk
        ]
    )
    responses = [[""] * n_samples for _ in records]
    for chunk_result in results:
        for prompt_idx, sample_idx, response in chunk_result:
            responses[prompt_idx][sample_idx] = response
    return responses


async def _start_servers(config) -> tuple[list[str], list[Any]]:
    tp = int(config.rollout.tensor_model_parallel_size)
    dp = int(config.rollout.get("data_parallel_size", 1))
    pp = int(config.rollout.get("pipeline_model_parallel_size", 1))
    replica_world_size = tp * dp * pp
    total_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    if total_gpus % replica_world_size != 0:
        raise ValueError(f"Total GPUs {total_gpus} is not divisible by rollout world size {replica_world_size}")

    rollout_server_class = get_rollout_replica_class(config.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=config.rollout,
            model_config=config.model,
            gpus_per_node=int(config.trainer.n_gpus_per_node),
        )
        for replica_rank in range(total_gpus // replica_world_size)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])
    server_addresses = [server.server_address for server in rollout_servers]
    print(f"Started {len(server_addresses)} rollout server(s): {server_addresses}")
    await asyncio.gather(*[_wait_for_server(address) for address in server_addresses])
    print("All rollout server health checks passed")
    return server_addresses, rollout_servers


@hydra.main(config_path="config", config_name="generate_and_eval", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}},
            num_cpus=config.ray_init.num_cpus,
        )
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    model_path = str(config.model.get("path") or config.actor_rollout_ref.model.path)
    if not model_path:
        raise ValueError("Generation eval server requires model.path or actor_rollout_ref.model.path")
    OmegaConf.set_struct(config.model, False)
    config.model.path = model_path

    compute_score = _get_generation_custom_reward_fn(config)
    if compute_score is None:
        raise ValueError("Generation eval requires a custom reward function.")

    output_root = Path(str(config.data.output_path))
    if output_root.suffix == ".jsonl":
        output_jsonl = output_root
        output_root = output_root.parent
    else:
        output_jsonl = output_root / f"rank_{config.rank}_{config.rollout.seed}.jsonl"
    done_path = output_root / f"rank_{config.rank}_{config.rollout.seed}_indices_done.txt"
    if not config.resume:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = hf_tokenizer(copy_to_local(model_path), trust_remote_code=True)
    records = _load_records(config, tokenizer, done_path)
    if not records:
        output_jsonl.touch(exist_ok=True)
        done_path.touch(exist_ok=True)
        print("No records to generate")
        return

    sampling_params = {
        "temperature": float(config.rollout.temperature),
        "top_p": float(config.rollout.top_p),
        "max_tokens": int(config.rollout.response_length),
    }
    if int(config.rollout.top_k) >= 0:
        sampling_params["top_k"] = int(config.rollout.top_k)

    if float(config.rollout.temperature) == 0.0:
        if int(config.data.n_samples) != 1:
            raise ValueError("When temperature=0, n_samples must be 1.")

    server_addresses, rollout_servers = asyncio.run(_start_servers(config))
    mode = "a" if config.resume else "w"
    batch_size = int(config.data.batch_size)
    n_samples = int(config.data.n_samples)
    total = len(records)
    with output_jsonl.open(mode, encoding="utf-8") as out_f, done_path.open(mode, encoding="utf-8") as done_f:
        for batch_start in range(0, total, batch_size):
            batch_records = records[batch_start : batch_start + batch_size]
            t0 = time.time()
            responses = asyncio.run(
                _generate_batch(server_addresses, model_path, n_samples, sampling_params, batch_records)
            )
            t1 = time.time()
            flat_responses = [response for per_prompt in responses for response in per_prompt]
            ground_truths = []
            extras = []
            for record in batch_records:
                row = record["row"]
                reward_model = row.get("reward_model") or {}
                if isinstance(reward_model, str):
                    try:
                        reward_model = json.loads(reward_model)
                    except json.JSONDecodeError:
                        reward_model = {}
                ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
                for _ in range(n_samples):
                    ground_truths.append(ground_truth)
                    extras.append(row.get("extra_info"))
            score_result = compute_score(
                data_sources=None,
                solution_strs=flat_responses,
                ground_truths=ground_truths,
                extra_infos=extras,
                **config.reward_model.get("reward_kwargs", {}),
            )
            scores, _reward_extra_infos = _score_values(score_result, len(flat_responses))
            t2 = time.time()

            for idx, record in enumerate(batch_records):
                row = record["row"]
                scores_slice = scores[idx * n_samples : (idx + 1) * n_samples]
                prompt_record = {
                    key: value
                    for key, value in row.items()
                    if key not in {"uid"}
                }
                output_row = {
                    "prompt": record["prompt_text"],
                    "responses": responses[idx],
                    "scores": scores_slice,
                    "prompt_record": prompt_record,
                    "data_source": row.get("data_source"),
                    "extra_info": row.get("extra_info"),
                    "reward_model": row.get("reward_model"),
                }
                out_f.write(json.dumps(_jsonable(output_row), ensure_ascii=False) + "\n")
                done_f.write(f"{record['line_number']}\n")
            out_f.flush()
            done_f.flush()
            print(
                "Generated rows "
                f"{batch_start + len(batch_records)}/{total}; "
                f"generation={t1 - t0:.2f}s scoring={t2 - t1:.2f}s"
            )
    # Keep a live reference to the Ray server actors until generation is complete.
    _ = rollout_servers


if __name__ == "__main__":
    main()
