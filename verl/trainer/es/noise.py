import math
from typing import Iterator

import torch


def _mix_seed(a: int, b: int) -> int:
    # 64-bit mix of two integers for deterministic per-parameter seeding
    x = (a ^ (b + 0x9E3779B97F4A7C15 + ((a << 6) & ((1 << 64) - 1)) + (a >> 2))) & ((1 << 64) - 1)
    return x


def stateless_normal_like(param: torch.Tensor, dir_seed: int, key: int) -> torch.Tensor:
    """
    Generate a standard normal tensor with the same shape/device/dtype as `param`,
    deterministically from (dir_seed, key) without storing RNG state.

    This is shard-local: it only depends on local parameter storage and does not
    assume any global concatenation ordering.
    """
    gen = torch.Generator(device=param.device)
    gen.manual_seed(_mix_seed(int(dir_seed), int(key)))
    return torch.randn_like(param, generator=gen, dtype=param.dtype, device=param.device)


def iter_flat_params(module: torch.nn.Module) -> Iterator[tuple[int, torch.nn.Parameter]]:
    """Iterate parameters in a stable order with an integer index key per param.

    The key can be used as part of the RNG seed to deterministically regenerate noise.
    """
    for i, p in enumerate(module.parameters()):
        if p.requires_grad:
            yield i, p

