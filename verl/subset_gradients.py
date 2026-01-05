#!/usr/bin/env python3
"""
Compute either (a) a random K-sized subset of parameter gradients or (b) a
Rademacher random projection to K dimensions for a Hugging Face *causal* LM on a
JSONL dataset. Designed for multi-GPU nodes via torch.distributed/torchrun with
streaming .pt outputs to avoid memory blow-ups.

Each JSONL line must contain: {"input": str, "output": str} plus optionally:
- {"idx": int} OR {"line_number": int}. If neither is present, the JSONL file
  line number is used as idx.  # CHANGED

Usage examples (subset = default, proj = Rademacher projection):
1) Single GPU, subset
   python subset_gradients.py --jsonl data.jsonl --model /path/to/model --k 10000 --mode subset --out-prefix /tmp/grads/run1

2) Multi-GPU subset (data shards split by rank)
   torchrun --nproc_per_node=4 subset_gradients.py --jsonl data.jsonl --model /path/to/model --k 10000 --mode subset --out-prefix /tmp/grads/run1

3) Single process, model sharded via device_map (uses ALL visible GPUs)
   python subset_gradients.py --jsonl data.jsonl --model big_model --k 10000 --mode subset --out-prefix /tmp/grads/run1 --device-map auto

4) Multi-worker + model sharding (recommended when model needs >1 GPU AND you want throughput)
   # example: 8 GPUs total, model needs 2 GPUs => 4 workers
   # launch each worker with a disjoint CUDA_VISIBLE_DEVICES subset and set --shard-id/--num-shards
   CUDA_VISIBLE_DEVICES=0,1 python subset_gradients.py ... --device-map auto --num-shards 4 --shard-id 0
   CUDA_VISIBLE_DEVICES=2,3 python subset_gradients.py ... --device-map auto --num-shards 4 --shard-id 1
   CUDA_VISIBLE_DEVICES=4,5 python subset_gradients.py ... --device-map auto --num-shards 4 --shard-id 2
   CUDA_VISIBLE_DEVICES=6,7 python subset_gradients.py ... --device-map auto --num-shards 4 --shard-id 3

Notes:
- Outputs are written as small shard files named "<out-prefix>_rank{R}_part{N}.pt" plus:
    subset mode -> "<out-prefix>_indices.pt" storing V
    proj   mode -> "<out-prefix>_projection_meta.pt" storing metadata (k, seed, total_params)
- If you use --shard-id/--num-shards (manual sharding), no torch.distributed is needed.
- If you use torchrun, sharding defaults to (rank, world_size).
- Combined sequences are truncated to --max-length (left-truncation: keep last tokens).
- CHANGED: If the *input field alone* exceeds --max-length tokens, that sample is skipped (no truncation).
- CHANGED: Removed any logits/loss chunking; use standard HF loss path.
- pad token defaults to eos if absent.
"""
import argparse
import json
import os
import uuid
from itertools import accumulate
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class JsonlShardDataset(IterableDataset):
    """Shards a JSONL file across shards by line number modulo num_shards."""

    def __init__(self, path: str, shard_id: int, num_shards: int) -> None:
        self.path = path
        self.shard_id = shard_id
        self.num_shards = num_shards

    def __iter__(self) -> Iterable[Dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):  # 0-based line number  # CHANGED
                # If you want 1-based line numbers instead, use:
                # for i, line in enumerate(f, start=1):
                if i % self.num_shards != self.shard_id:
                    continue
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    # Always attach the global file line number for fallback idx.  # CHANGED
                    obj.setdefault("__line_number", i)  # CHANGED
                yield obj


def strip_distributed_env_vars() -> None:
    """
    Remove common torchrun/elastic env vars so libraries don't try to auto-init distributed.
    Safe for manual-shard mode where each worker is independent.
    """
    keys = [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
    ]
    for k in keys:
        os.environ.pop(k, None)


def init_distributed_if_needed() -> Tuple[int, int, int, bool]:
    """Return (rank, local_rank, world_size, using_dist)."""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, local_rank, world_size, True

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, local_rank, world_size, True

    return 0, 0, 1, False


def atomic_torch_save(obj, path: str) -> None:
    """
    Atomic write to avoid partial/corrupt files with multiple workers.

    CHANGED: PyTorch zip serializer can reject hidden-dot temp filenames.
    Use '<path>.tmp.<pid>.<uuid>' instead of '.tmp_xxx'.
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"  # CHANGED
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def select_param_indices(params: List[torch.nn.Parameter], k: int, seed: int) -> torch.Tensor:
    total_params = sum(p.numel() for p in params)
    if k > total_params:
        raise ValueError(f"Requested k={k} but model only has {total_params} parameters")
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(total_params, generator=g)[:k]


def build_index_map(params: List[torch.nn.Parameter], indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map flattened indices to (param_idx, inner_idx) tensors for fast lookup."""
    sizes = [p.numel() for p in params]
    boundaries = torch.tensor([0] + list(accumulate(sizes)), dtype=torch.long)
    param_idx = torch.searchsorted(boundaries[1:], indices, right=False)
    inner_idx = indices - boundaries[param_idx]
    return param_idx, inner_idx


def count_shard_lines(path: str, shard_id: int, num_shards: int) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f):
            if i % num_shards == shard_id:
                total += 1
    return total


def get_model_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    """For device_map-sharded models, inputs must go to the model's first param device."""
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return fallback


def parse_device_map_arg(
    s: Optional[str],
) -> Optional[Union[str, int, Dict[str, Union[int, str, torch.device]], torch.device]]:
    if s is None:
        return None
    s_strip = str(s).strip()
    if s_strip.isdigit():
        return int(s_strip)
    return s_strip


def compute_accelerate_device_map_auto(model_id: str) -> Dict[str, Union[int, str, torch.device]]:
    """
    Implement device_map='auto' via Accelerate and return an explicit device_map dict.
    This avoids passing the string 'auto' into from_pretrained in environments where that
    triggers unwanted distributed/TP behavior.
    """
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import get_balanced_memory, infer_auto_device_map
    except Exception as e:
        raise RuntimeError(
            "device_map='auto' requested but accelerate is not available/importable. "
            "Either install accelerate, or omit --device-map."
        ) from e

    config = AutoConfig.from_pretrained(model_id)
    estimate_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config)

    # Silence accelerate warning when possible
    if hasattr(empty_model, "tie_weights"):
        try:
            empty_model.tie_weights()
        except Exception:
            pass

    no_split = getattr(empty_model, "_no_split_modules", None)
    max_memory = get_balanced_memory(empty_model, dtype=estimate_dtype)

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=no_split,
        dtype=estimate_dtype,
    )
    del empty_model
    return device_map


def prepare_inputs_causal(tokenizer, sample: Dict, device: torch.device, max_length: int) -> Optional[Dict]:
    prompt_ids = tokenizer(
        sample["input"],
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
        truncation=False,
    )["input_ids"]

    # CHANGED: Skip sample if *input alone* exceeds max_length tokens.
    if len(prompt_ids) > max_length:
        return None

    target_ids = tokenizer(
        sample["output"],
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
        truncation=False,
    )["input_ids"]

    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        prompt_len = max(0, len(input_ids) - len(target_ids))
    else:
        prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + target_ids[-len(input_ids) + prompt_len :]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, device=device)
    return {"input_ids": input_tensor, "attention_mask": attention_mask, "labels": labels_tensor}


def extract_subset(params: List[torch.nn.Parameter], param_idx: torch.Tensor, inner_idx: torch.Tensor) -> torch.Tensor:
    """
    Vectorized extraction (group by param) to avoid K Python-level GPU syncs.
    Returns float32 CPU tensor of shape [K].
    """
    k = int(param_idx.numel())
    out = torch.zeros(k, dtype=torch.float32, device="cpu")

    order = torch.argsort(param_idx)
    p_sorted = param_idx[order]
    i_sorted = inner_idx[order]

    if p_sorted.numel() == 0:
        return out

    unique_p, counts = torch.unique_consecutive(p_sorted, return_counts=True)
    start = 0
    for p_i, c in zip(unique_p.tolist(), counts.tolist()):
        sl = slice(start, start + c)
        positions = order[sl]
        inner = i_sorted[sl]
        grad = params[p_i].grad
        if grad is not None:
            gflat = grad.detach().reshape(-1)
            inner_dev = inner.to(device=gflat.device, dtype=torch.long)
            vals = gflat.index_select(0, inner_dev)
            out[positions] = vals.detach().to(torch.float32).cpu()
        start += c

    return out


def int_to_signed_int64(x: int) -> int:
    """
    Wrap an arbitrary Python int into signed 64-bit range.
    This matches uint64 modulo 2^64, then reinterprets as int64.
    """
    mask64 = (1 << 64) - 1
    signbit = 1 << 63
    x = x & mask64
    if x >= signbit:
        x -= (1 << 64)
    return int(x)


def project_rademacher(params: List[torch.nn.Parameter], k: int, seed: int, scale: bool = True) -> torch.Tensor:
    """
    Memory-friendly Rademacher projection using a deterministic signed hashing scheme.

    Ensure all scalar constants are representable as int64 by wrapping into signed 64-bit.
    """
    proj = torch.zeros(k, dtype=torch.float32, device="cpu")
    offset = 0

    MULT = int_to_signed_int64(6364136223846793005)
    INC = int_to_signed_int64(1442695040888963407 * (seed + 1))

    for p in params:
        grad = p.grad
        n = p.numel()
        if grad is None:
            offset += n
            continue

        g_flat = grad.detach().reshape(-1)
        device = g_flat.device

        mult_t = torch.tensor(MULT, device=device, dtype=torch.int64)
        inc_t = torch.tensor(INC, device=device, dtype=torch.int64)

        idx = torch.arange(n, device=device, dtype=torch.int64) + offset
        hashed = idx * mult_t + inc_t

        bucket = torch.remainder(hashed, k).to(torch.int64)
        sign = torch.where((hashed & 1) == 0, 1.0, -1.0).to(g_flat.dtype)

        contrib = g_flat * sign
        proj_chunk = torch.zeros(k, device=device, dtype=g_flat.dtype)
        proj_chunk.scatter_add_(0, bucket, contrib)
        proj += proj_chunk.detach().cpu().to(torch.float32)

        offset += n

    if scale:
        proj /= k**0.5
    return proj


def save_shard(buffer: List[Dict], out_prefix: str, rank: int, part: int) -> None:
    path = f"{out_prefix}_rank{rank}_part{part}.pt"
    atomic_torch_save(buffer, path)  # atomic shard saves


def get_sample_idx(sample: Dict) -> int:
    """
    CHANGED: Choose idx with priority:
      1) sample['idx'] if present
      2) sample['line_number'] if present
      3) sample['__line_number'] (global JSONL line number injected by dataset)
    """
    if isinstance(sample, dict):
        if "idx" in sample and sample["idx"] is not None:
            try:
                return int(sample["idx"])
            except Exception as e:
                raise ValueError(f"Could not convert sample['idx']={sample.get('idx')!r} to int") from e

        if "line_number" in sample and sample["line_number"] is not None:
            try:
                return int(sample["line_number"])
            except Exception as e:
                raise ValueError(f"Could not convert sample['line_number']={sample.get('line_number')!r} to int") from e

        if "__line_number" in sample and sample["__line_number"] is not None:
            try:
                return int(sample["__line_number"])
            except Exception as e:
                raise ValueError(f"Could not convert sample['__line_number']={sample.get('__line_number')!r} to int") from e

    raise KeyError("Sample is missing idx/line_number and has no injected __line_number")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract random subset of parameter gradients (causal LM only)")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file with input/output fields (idx optional)")
    parser.add_argument("--model", required=True, help="Local HF model directory or model id")
    parser.add_argument("--k", type=int, required=True, help="Number of dimensions / indices (K)")
    parser.add_argument("--out-prefix", required=True, help="Prefix for .pt outputs (no extension)")
    parser.add_argument(
        "--mode",
        choices=["subset", "proj"],
        default="proj",
        help="subset = save K parameter entries; proj = Rademacher projection to K dims",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Samples per batch (default: 1)")
    parser.add_argument("--flush-every", type=int, default=64, help="Save after this many samples per rank")
    parser.add_argument("--seed", type=int, default=38293, help="Random seed for subset selection")
    parser.add_argument("--max-length", type=int, default=32768, help="Max total sequence length")
    parser.add_argument("--device-map", default=None, help="Device map: omit for single-GPU; 'auto' to shard across visible GPUs")

    parser.add_argument("--shard-id", type=int, default=None, help="Manual shard id (overrides torch.distributed rank/world_size)")
    parser.add_argument("--num-shards", type=int, default=None, help="Manual num shards (overrides torch.distributed rank/world_size)")

    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        raise FileNotFoundError(
            f"--jsonl path does not exist inside this container: {args.jsonl}\n"
            "If /data isn't present, pass the /lustre/... path."
        )

    manual_sharding = (args.shard_id is not None) or (args.num_shards is not None)

    if manual_sharding:
        if args.shard_id is None or args.num_shards is None:
            raise ValueError("If you set --shard-id you must also set --num-shards (and vice versa).")
        strip_distributed_env_vars()
        rank = int(args.shard_id)
        world_size = int(args.num_shards)
        local_rank = 0
        using_dist = False
    else:
        rank, local_rank, world_size, using_dist = init_distributed_if_needed()

    torch.manual_seed(args.seed + rank)

    default_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map_arg = parse_device_map_arg(args.device_map)

    # Treat device_map="auto" as unnecessary for 1 visible GPU
    if device_map_arg == "auto" and torch.cuda.is_available() and torch.cuda.device_count() <= 1:
        device_map_arg = None

    if device_map_arg == "auto":
        device_map_dict = compute_accelerate_device_map_auto(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=device_map_dict)
    elif device_map_arg is not None:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=device_map_arg)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
        model.to(default_device)

    # Training-style settings: avoid cache (saves memory and is correct for backward)
    if hasattr(model, "config"):
        model.config.use_cache = False

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)

    input_device = get_model_input_device(model, default_device)

    if args.mode == "subset":
        indices = select_param_indices(params, args.k, args.seed)

        if rank == 0:
            idx_path = f"{args.out_prefix}_indices.pt"
            if not os.path.exists(idx_path):
                atomic_torch_save(indices, idx_path)

        if using_dist and dist.is_initialized():
            bcast_tensor = indices.to(default_device if torch.cuda.is_available() else "cpu")
            dist.broadcast(bcast_tensor, src=0)
            indices = bcast_tensor.cpu()

        param_idx, inner_idx = build_index_map(params, indices)
    else:
        if rank == 0:
            meta_path = f"{args.out_prefix}_projection_meta.pt"
            if not os.path.exists(meta_path):
                atomic_torch_save(
                    {"mode": "proj", "k": args.k, "seed": args.seed, "total_params": total_params},
                    meta_path,
                )
        param_idx = inner_idx = None

    dataset = JsonlShardDataset(args.jsonl, rank, world_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: x)

    shard_len = count_shard_lines(args.jsonl, rank, world_size)
    pbar = tqdm(total=shard_len, position=rank, desc=f"rank {rank}")

    buffer: List[Dict] = []
    part = 0
    skipped = 0

    for batch in loader:
        if not isinstance(batch, list):
            raise RuntimeError("Unexpected batch type from DataLoader")

        for sample in batch:
            model.zero_grad(set_to_none=True)

            model_inputs = prepare_inputs_causal(tokenizer, sample, input_device, args.max_length)

            if model_inputs is None:
                skipped += 1
                model.zero_grad(set_to_none=True)
                pbar.update(1)
                continue

            outputs = model(**model_inputs)
            loss = outputs.loss
            loss.backward()

            sample_idx = get_sample_idx(sample)  # CHANGED

            if args.mode == "subset":
                values = extract_subset(params, param_idx, inner_idx)
                buffer.append({"idx": sample_idx, "grad_subset": values})  # CHANGED
            else:
                proj = project_rademacher(params, args.k, args.seed)
                buffer.append({"idx": sample_idx, "grad_proj": proj})  # CHANGED

            if len(buffer) >= args.flush_every:
                save_shard(buffer, args.out_prefix, rank, part)
                buffer.clear()
                part += 1

            model.zero_grad(set_to_none=True)
            pbar.update(1)

    if buffer:
        save_shard(buffer, args.out_prefix, rank, part)

    pbar.close()

    if using_dist and dist.is_initialized():
        dist.barrier()

    if skipped > 0:
        print(f"[rank {rank}] Skipped {skipped} samples where input tokens > max_length={args.max_length}")

    if rank == 0:
        if args.mode == "subset":
            print("Done. Gradient subset indices stored in", f"{args.out_prefix}_indices.pt")
        else:
            print("Done. Projection metadata stored in", f"{args.out_prefix}_projection_meta.pt")


if __name__ == "__main__":
    main()
