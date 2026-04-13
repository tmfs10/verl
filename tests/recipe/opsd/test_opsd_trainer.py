import torch
import numpy as np

from recipe.opsd.opsd_trainer import RayOPSDTrainer
from verl.protocol import DataProto


def _make_trainer(distill_max_response_tokens):
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)

    def _opsd_get(key, default):
        if key == "distill_max_response_tokens":
            return distill_max_response_tokens
        return default

    trainer._opsd_get = _opsd_get
    return trainer


def test_apply_distill_response_cap_is_noop_when_unset():
    trainer = _make_trainer(None)
    distill_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    capped = trainer._apply_distill_response_cap(distill_mask)

    assert torch.equal(capped, distill_mask)


def test_apply_distill_response_cap_keeps_only_prefix_tokens():
    trainer = _make_trainer(3)
    distill_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    capped = trainer._apply_distill_response_cap(distill_mask)

    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(capped, expected)


def test_reference_solution_teacher_message_matches_paper_prompt_shape():
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)

    message = trainer._build_reference_solution_teacher_message(
        "What is 1+1?",
        "We add the numbers and get 2.",
    )

    assert message.startswith("Problem: What is 1+1?")
    assert "=== Reference Solution Begin ===" in message
    assert "We add the numbers and get 2." in message
    assert "Please reason step by step, and put your final answer within \\boxed{}." in message


def test_get_gen_batch_preserves_opsd_teacher_fields():
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)
    batch = DataProto.from_single_dict(
        {
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": "openthoughts_math",
            "reward_model": {"ground_truth": '"2"'},
            "extra_info": {},
            "uid": "u1",
            "prompt_group_id": "g1",
            "ground_truth_answer": "The answer is 2.",
            "problem": "What is 1+1?",
            "raw_prompt": [{"role": "user", "content": "Problem: What is 1+1?"}],
        }
    )

    gen_batch = trainer._get_gen_batch(batch)

    assert gen_batch.non_tensor_batch["problem"][0] == "What is 1+1?"
    assert batch.non_tensor_batch["problem"][0] == "What is 1+1?"
    assert gen_batch.non_tensor_batch["ground_truth_answer"][0] == "The answer is 2."
