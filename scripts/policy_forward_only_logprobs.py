#!/usr/bin/env python3
"""Run policy-model-only VERL forward logprob scoring.

This entrypoint intentionally uses the newer TrainingWorker engine path with
engine.forward_only=True and optimizer_config=None. It initializes the policy
model, skips optimizer / lr-scheduler construction, and runs infer_batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_runtime_imports() -> None:
    """Import GPU/distributed dependencies after argparse handles --help."""

    global AutoTokenizer
    global CheckpointConfig
    global DataProto
    global FSDPEngineConfig
    global HFModelConfig
    global McoreEngineConfig
    global RayClassWithInitArgs
    global RayResourcePool
    global RayWorkerGroup
    global TrainingWorker
    global TrainingWorkerConfig
    global HF_MODULES_CACHE
    global compute_position_id_with_mask
    global left_right_2_no_padding
    global no_padding_2_padding
    global ray
    global torch
    global tu

    import ray as ray_module
    import torch as torch_module
    from transformers import AutoTokenizer as auto_tokenizer_cls
    from transformers.utils import HF_MODULES_CACHE as hf_modules_cache

    from verl import DataProto as data_proto_cls
    from verl.single_controller.ray import (
        RayClassWithInitArgs as ray_class_with_init_args_cls,
        RayResourcePool as ray_resource_pool_cls,
        RayWorkerGroup as ray_worker_group_cls,
    )
    from verl.trainer.config import CheckpointConfig as checkpoint_config_cls
    from verl.utils import tensordict_utils as tensordict_utils_module
    from verl.utils.model import compute_position_id_with_mask as compute_position_id_with_mask_fn
    from verl.workers.config import (
        FSDPEngineConfig as fsdp_engine_config_cls,
        HFModelConfig as hf_model_config_cls,
        McoreEngineConfig as mcore_engine_config_cls,
        TrainingWorkerConfig as training_worker_config_cls,
    )
    from verl.workers.engine_workers import TrainingWorker as training_worker_cls
    from verl.workers.utils.padding import (
        left_right_2_no_padding as left_right_2_no_padding_fn,
        no_padding_2_padding as no_padding_2_padding_fn,
    )

    ray = ray_module
    torch = torch_module
    AutoTokenizer = auto_tokenizer_cls
    DataProto = data_proto_cls
    RayClassWithInitArgs = ray_class_with_init_args_cls
    RayResourcePool = ray_resource_pool_cls
    RayWorkerGroup = ray_worker_group_cls
    CheckpointConfig = checkpoint_config_cls
    tu = tensordict_utils_module
    compute_position_id_with_mask = compute_position_id_with_mask_fn
    FSDPEngineConfig = fsdp_engine_config_cls
    HFModelConfig = hf_model_config_cls
    McoreEngineConfig = mcore_engine_config_cls
    TrainingWorkerConfig = training_worker_config_cls
    TrainingWorker = training_worker_cls
    HF_MODULES_CACHE = hf_modules_cache
    left_right_2_no_padding = left_right_2_no_padding_fn
    no_padding_2_padding = no_padding_2_padding_fn


def parse_json_arg(value: str | None, *, arg_name: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return parsed


def parse_process_on_nodes(value: str | None) -> list[int]:
    if value:
        nodes = [int(part) for part in value.split(",") if part.strip()]
    else:
        nodes = [torch.cuda.device_count()]
    if not nodes or any(node <= 0 for node in nodes):
        raise ValueError(
            "No GPUs were detected. Pass --process-on-nodes, e.g. --process-on-nodes 8 or 8,8 for a Ray cluster."
        )
    return nodes


def ensure_hf_dynamic_module_path() -> dict[str, str]:
    """Make trust-remote-code modules importable inside Ray actors."""

    modules_cache = os.environ.get("HF_MODULES_CACHE") or HF_MODULES_CACHE
    Path(modules_cache).mkdir(parents=True, exist_ok=True)
    if modules_cache not in sys.path:
        sys.path.insert(0, modules_cache)
    pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in pythonpath.split(os.pathsep) if part]
    if modules_cache not in pythonpath_parts:
        pythonpath = os.pathsep.join([modules_cache, *pythonpath_parts])
        os.environ["PYTHONPATH"] = pythonpath
    os.environ["HF_MODULES_CACHE"] = modules_cache
    return {
        "HF_MODULES_CACHE": modules_cache,
        "PYTHONPATH": os.environ.get("PYTHONPATH", modules_cache),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
    }


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def iter_input_rows(args: argparse.Namespace, *, skip_rows: int = 0):
    if args.input_jsonl:
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx < skip_rows:
                    continue
                if not line.strip():
                    continue
                row = json.loads(line)
                if args.text_field not in row:
                    raise KeyError(f"Input row {line_idx} is missing text field {args.text_field!r}")
                yield line_idx, row[args.text_field], row
    else:
        for line_idx, text in enumerate(args.text):
            if line_idx < skip_rows:
                continue
            yield line_idx, text, {"text": text}


def batched(iterator, batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_worker_config(args: argparse.Namespace) -> TrainingWorkerConfig:
    override_config = parse_json_arg(args.override_config_json, arg_name="--override-config-json")
    model_config = HFModelConfig(
        path=args.model,
        tokenizer_path=args.tokenizer or args.model,
        load_tokenizer=True,
        trust_remote_code=args.trust_remote_code,
        external_lib=args.external_lib,
        override_config=override_config,
        enable_gradient_checkpointing=False,
        enable_activation_offload=False,
        use_remove_padding=args.use_remove_padding,
        use_fused_kernels=args.use_fused_kernels,
        use_liger=args.use_liger,
    )

    infer_max_token_len_per_gpu = args.infer_max_token_len_per_gpu
    if infer_max_token_len_per_gpu is None:
        infer_max_token_len_per_gpu = args.max_length * max(args.batch_size, 1)

    common_engine_kwargs = dict(
        forward_only=True,
        param_offload=args.param_offload,
        optimizer_offload=False,
        grad_offload=False,
        dtype=args.dtype,
        use_dynamic_bsz=True,
        use_remove_padding=args.use_remove_padding,
        infer_max_token_len_per_gpu=infer_max_token_len_per_gpu,
        infer_micro_batch_size_per_gpu=args.infer_micro_batch_size_per_gpu,
        use_fused_kernels=args.use_fused_kernels,
    )

    if args.backend in {"fsdp", "fsdp2"}:
        engine_config = FSDPEngineConfig(
            strategy=args.backend,
            fsdp_size=args.fsdp_size,
            ulysses_sequence_parallel_size=args.ulysses_sequence_parallel_size,
            model_dtype=args.dtype,
            use_torch_compile=args.use_torch_compile,
            **common_engine_kwargs,
        )
    elif args.backend == "megatron":
        override_transformer_config = parse_json_arg(
            args.override_transformer_config_json, arg_name="--override-transformer-config-json"
        )
        override_mcore_model_config = parse_json_arg(
            args.override_mcore_model_config_json, arg_name="--override-mcore-model-config-json"
        )
        engine_config = McoreEngineConfig(
            use_mbridge=True,
            vanilla_mbridge=args.vanilla_mbridge,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
            sequence_parallel=args.sequence_parallel,
            override_transformer_config=override_transformer_config,
            override_mcore_model_config=override_mcore_model_config,
            **common_engine_kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    return TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=None,
        checkpoint_config=CheckpointConfig(save_contents=[], load_contents=["model"]),
    )


def tokenize_batch(tokenizer, texts: list[str], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        add_special_tokens=not args.no_special_tokens,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    valid_lengths = attention_mask.sum(dim=1)
    if torch.any(valid_lengths < 2):
        bad = torch.nonzero(valid_lengths < 2, as_tuple=False).flatten().tolist()
        raise ValueError(f"Rows at batch offsets {bad} have fewer than 2 tokens after tokenization")

    position_ids = compute_position_id_with_mask(attention_mask)
    prompts = input_ids[:, :1]
    responses = input_ids[:, 1:]
    response_mask = attention_mask[:, 1:]
    global_token_num = valid_lengths.tolist()

    return {
        "input_ids": input_ids,
        "prompts": prompts,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
        "global_token_num": global_token_num,
    }


def infer_logprobs(worker_group: RayWorkerGroup, tokenized: dict[str, torch.Tensor]):
    data = DataProto.from_single_dict(
        {
            "input_ids": tokenized["input_ids"],
            "prompts": tokenized["prompts"],
            "attention_mask": tokenized["attention_mask"],
            "position_ids": tokenized["position_ids"],
            "responses": tokenized["responses"],
            "response_mask": tokenized["response_mask"],
        },
        meta_info={
            "temperature": 1.0,
            "global_token_num": tokenized["global_token_num"],
            "compute_loss": False,
        },
    )
    data_td = left_right_2_no_padding(data.to_tensordict())
    output = worker_group.infer_batch(data_td).get()
    logprobs_unpad = tu.get(output, "log_probs").cpu()
    return no_padding_2_padding(logprobs_unpad, data_td)


def write_batch_outputs(
    *,
    out_f,
    tokenizer,
    batch_items,
    tokenized: dict[str, torch.Tensor],
    logprobs: torch.Tensor,
):
    attention_mask = tokenized["attention_mask"]
    input_ids = tokenized["input_ids"]
    for batch_idx, (source_line_number, text, source_row) in enumerate(batch_items):
        seq_len = int(attention_mask[batch_idx].sum().item())
        ids = input_ids[batch_idx, :seq_len].tolist()
        scored_ids = ids[1:]
        scored_logprobs = logprobs[batch_idx, : len(scored_ids)].tolist()
        output_row = {
            "source_line_number": source_line_number,
            "text": text,
            "input_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "scored_token_ids": scored_ids,
            "scored_tokens": tokenizer.convert_ids_to_tokens(scored_ids),
            "logprobs": [float(x) for x in scored_logprobs],
            "source": source_row,
        }
        out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    out_f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Policy model path.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--backend", choices=["fsdp", "fsdp2", "megatron"], default="megatron")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--external-lib", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--override-config-json", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--ray-address", default=None, help="Ray address, e.g. auto. Defaults to local Ray.")
    parser.add_argument("--process-on-nodes", default=None, help="Comma-separated GPU worker counts, e.g. 8 or 8,8.")
    parser.add_argument("--fsdp-size", type=int, default=-1)
    parser.add_argument("--ulysses-sequence-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--virtual-pipeline-model-parallel-size", type=int, default=None)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--expert-model-parallel-size", type=int, default=1)
    parser.add_argument("--expert-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--sequence-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vanilla-mbridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-remove-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fused-kernels", action="store_true")
    parser.add_argument("--use-liger", action="store_true")
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--param-offload", action="store_true")
    parser.add_argument("--infer-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--infer-micro-batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--override-transformer-config-json", default=None)
    parser.add_argument("--override-mcore-model-config-json", default=None)
    args = parser.parse_args()

    if not args.input_jsonl and not args.text:
        raise ValueError("Provide --input-jsonl or at least one --text")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    load_runtime_imports()
    ray_env_vars = ensure_hf_dynamic_module_path()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = count_jsonl_rows(output_path) if args.resume else 0

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    process_on_nodes = parse_process_on_nodes(args.process_on_nodes)
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    try:
        config = build_worker_config(args)
        assert config.optimizer_config is None
        assert config.engine_config.forward_only

        ray_cls = RayClassWithInitArgs(cls=ray.remote(runtime_env={"env_vars": ray_env_vars})(TrainingWorker), config=config)
        resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, max_colocate_count=1)
        worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)

        print(
            "Initializing forward-only policy worker "
            f"backend={args.backend} model={args.model} process_on_nodes={process_on_nodes}",
            flush=True,
        )
        worker_group.reset()

        processed = completed
        rows = iter_input_rows(args, skip_rows=completed)
        mode = "a" if args.resume else "w"
        with output_path.open(mode, encoding="utf-8") as out_f:
            for batch_items in batched(rows, args.batch_size):
                texts = [item[1] for item in batch_items]
                tokenized = tokenize_batch(tokenizer, texts, args)
                logprobs = infer_logprobs(worker_group, tokenized)
                write_batch_outputs(
                    out_f=out_f,
                    tokenizer=tokenizer,
                    batch_items=batch_items,
                    tokenized=tokenized,
                    logprobs=logprobs,
                )
                processed += len(batch_items)
                if processed % args.progress_every == 0:
                    print(f"Processed {processed} rows", flush=True)
        print(f"Done. Wrote {processed - completed} new rows to {output_path}", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
