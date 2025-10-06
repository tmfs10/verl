from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from tensordict import TensorDict
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


@dataclass
class HFGenerationArgs:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    response_length: int


def tokenize_batch_chat(
    tokenizer,
    chat_list: list[list[dict[str, Any]]],
    prompt_length: int,
    apply_chat_template_kwargs: dict,
):
    inputs = tokenizer.apply_chat_template(
        chat_list,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        max_length=prompt_length,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
        **(apply_chat_template_kwargs or {}),
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = compute_position_id_with_mask(attention_mask)
    return input_ids, attention_mask, position_ids


@torch.no_grad()
def hf_generate_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    gen_args: HFGenerationArgs,
    eos_token_id: int,
    pad_token_id: int,
) -> DataProto:
    # Build generation config
    kwargs = {
        "do_sample": gen_args.do_sample,
        "num_beams": 1,
        "top_p": float(gen_args.top_p),
        "top_k": int(max(0, gen_args.top_k)),
        "temperature": float(gen_args.temperature),
        "num_return_sequences": 1,
    }
    generation_config = GenerationConfig(**kwargs)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)

    prompt_length = input_ids.size(1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        do_sample=gen_args.do_sample,
        max_new_tokens=int(gen_args.response_length),
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        generation_config=generation_config,
        output_scores=False,
        return_dict_in_generate=True,
        use_cache=True,
    )

    sequences = outputs.sequences
    generated_batch_size = sequences.size(0)
    # Pad to fixed response_length if EOS encountered early
    sequence_length = prompt_length + int(gen_args.response_length)
    if sequences.shape[1] < sequence_length:
        delta = sequence_length - sequences.shape[1]
        pad = torch.full((generated_batch_size, delta), pad_token_id, device=sequences.device, dtype=sequences.dtype)
        sequences = torch.cat((sequences, pad), dim=1)

    prompt = sequences[:, :prompt_length]
    response = sequences[:, prompt_length:]

    response_length = response.size(1)
    delta_position = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0)
    response_position_ids = position_ids[:, -1:] + delta_position
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

    # Construct response mask: attend until EOS, else up to response_length
    eos = torch.tensor([eos_token_id], device=response.device, dtype=response.dtype)
    is_eos = (response == eos)
    # first EOS per row, else full length
    eos_pos = torch.argmax(is_eos.to(torch.int32), dim=1)
    has_eos = is_eos.any(dim=1)
    lengths = torch.where(has_eos, eos_pos + 1, torch.full_like(eos_pos, response_length))
    # build attention mask for response
    arange_ids = torch.arange(response_length, device=response.device).unsqueeze(0)
    response_mask = (arange_ids < lengths.unsqueeze(1)).to(attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    batch = TensorDict(
        {
            "prompts": prompt,
            "responses": response,
            "input_ids": sequences,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=generated_batch_size,
    )

    return DataProto(batch=batch)

