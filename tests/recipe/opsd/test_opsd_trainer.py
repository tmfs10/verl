import torch
import numpy as np
from omegaconf import OmegaConf

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


def test_actor_layout_hash_is_exact_and_per_sample():
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        }
    )

    before = RayOPSDTrainer._actor_layout_sha256(batch)
    unchanged = RayOPSDTrainer._actor_layout_sha256(batch)
    batch.batch["input_ids"][1, 2] = 5
    changed = RayOPSDTrainer._actor_layout_sha256(batch)

    assert before == unchanged
    assert before[0] == changed[0]
    assert before[1] != changed[1]


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
    trainer.config = OmegaConf.create(
        {"algorithm": {"opsd": {"enable": True, "ground_truth_field": "ground_truth_answer"}}}
    )
    raw_prompt = np.empty(1, dtype=object)
    raw_prompt[0] = [{"role": "user", "content": "Problem: What is 1+1?"}]
    batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.tensor([0], dtype=torch.uint8)},
        non_tensors={
            "data_source": np.asarray(["openthoughts_math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": '"2"'}], dtype=object),
            "extra_info": np.asarray([{}], dtype=object),
            "uid": np.asarray(["u1"], dtype=object),
            "prompt_group_id": np.asarray(["g1"], dtype=object),
            "ground_truth_answer": np.asarray(["The answer is 2."], dtype=object),
            "problem": np.asarray(["What is 1+1?"], dtype=object),
            "raw_prompt": raw_prompt,
        },
    )

    gen_batch = trainer._get_gen_batch(batch)

    assert gen_batch.non_tensor_batch["problem"][0] == "What is 1+1?"
    assert batch.non_tensor_batch["problem"][0] == "What is 1+1?"
    assert gen_batch.non_tensor_batch["ground_truth_answer"][0] == "The answer is 2."


def test_get_ground_truth_answer_supports_nested_non_tensor_fields():
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)
    trainer._opsd_get = lambda key, default: "reward_model.ground_truth" if key == "ground_truth_field" else default
    batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.tensor([0], dtype=torch.uint8)},
        non_tensors={
            "reward_model": np.asarray([{"ground_truth": "2"}], dtype=object),
        },
    )

    assert trainer._get_ground_truth_answer(batch, 0) == "2"


def test_steering_teacher_inputs_are_identical_and_never_include_ground_truth():
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)
    settings = {
        "steering": {"source_mode": "caa", "correct_rollout_aggregation": "all"},
        "sdpo_distill_only_failed": False,
        "distill_max_response_tokens": None,
    }
    trainer._opsd_get = lambda key, default: settings.get(key, default)
    input_ids = torch.tensor(
        [
            [0, 10, 11, 20, 21],
            [0, 10, 11, 22, 23],
            [0, 30, 31, 40, 41],
            [0, 30, 31, 42, 43],
        ]
    )
    attention = torch.tensor([[0, 1, 1, 1, 1]]).repeat(4, 1)
    positions = torch.tensor([[0, 0, 1, 2, 3]]).repeat(4, 1)
    responses = input_ids[:, -2:]
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention,
            "position_ids": positions,
            "responses": responses,
        },
        non_tensors={
            "uid": np.asarray(["a", "a", "b", "b"], dtype=object),
            "acc": np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "solution": np.asarray(["SECRET_GT"] * 4, dtype=object),
            "Answer": np.asarray(["SECRET_ANSWER"] * 4, dtype=object),
        },
    )
    response_mask = torch.ones(4, 2)

    teacher_fields, _ = trainer._build_sdpo_steering_fields(batch, response_mask)

    assert torch.equal(teacher_fields.batch["teacher_input_ids"], input_ids)
    assert torch.equal(teacher_fields.batch["teacher_attention_mask"], attention)
    assert torch.equal(teacher_fields.batch["teacher_position_ids"], positions)
    assert torch.equal(
        teacher_fields.batch["opsd_distill_mask"],
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    # Every target in the mixed group gets the same CAA direction, regardless
    # of whether that target rollout itself succeeded or failed.
    assert torch.equal(
        teacher_fields.batch["steering_source_indices"][0],
        teacher_fields.batch["steering_source_indices"][1],
    )
    assert torch.equal(
        teacher_fields.batch["steering_source_signs"][0],
        teacher_fields.batch["steering_source_signs"][1],
    )
    assert torch.equal(
        teacher_fields.batch["steering_source_candidate_mask"][0],
        teacher_fields.batch["steering_source_candidate_mask"][1],
    )
    # Group b is all-correct and therefore has no CAA candidates.
    assert teacher_fields.batch["steering_source_candidate_mask"][2:].sum().item() == 0


def test_global_batch_steering_uses_all_outcomes_and_identical_teacher_inputs():
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)
    settings = {
        "steering": {
            "source_mode": "caa",
            "caa_scope": "global_batch",
            "correct_rollout_aggregation": "all",
            "gap_diagnostics": {
                "enabled": True,
                "crossfit_enabled": True,
                "fold_seed": 1234,
            },
        },
        "sdpo_distill_only_failed": False,
        "distill_max_response_tokens": None,
    }
    trainer._opsd_get = lambda key, default: settings.get(key, default)
    input_ids = torch.tensor(
        [
            [0, 10, 11, 20, 21],
            [0, 10, 11, 22, 23],
            [0, 30, 31, 40, 41],
            [0, 30, 31, 42, 43],
        ]
    )
    attention = torch.tensor([[0, 1, 1, 1, 1]]).repeat(4, 1)
    positions = torch.tensor([[0, 0, 1, 2, 3]]).repeat(4, 1)
    responses = input_ids[:, -2:]
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention,
            "position_ids": positions,
            "responses": responses,
        },
        non_tensors={
            "uid": np.asarray(["a", "a", "b", "b"], dtype=object),
            "acc": np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "solution": np.asarray(["SECRET_GT"] * 4, dtype=object),
        },
    )
    response_mask = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )

    teacher_fields, _ = trainer._build_sdpo_steering_fields(batch, response_mask)

    assert torch.equal(teacher_fields.batch["teacher_input_ids"], input_ids)
    assert torch.equal(teacher_fields.batch["teacher_attention_mask"], attention)
    assert torch.equal(teacher_fields.batch["teacher_position_ids"], positions)
    assert torch.equal(
        teacher_fields.batch["steering_source_outcome_sign"],
        torch.tensor([-1.0, 1.0, 1.0, 1.0]),
    )
    assert torch.equal(teacher_fields.batch["opsd_distill_mask"], response_mask)
    assert teacher_fields.batch["opsd_gap_prompt_group"].tolist() == [0, 0, 1, 1]
    assert teacher_fields.batch["opsd_gap_prompt_count"].tolist() == [2, 2, 2, 2]
    assert (
        teacher_fields.batch["opsd_gap_crossfit_fold"][0]
        == teacher_fields.batch["opsd_gap_crossfit_fold"][1]
    )
    assert (
        teacher_fields.batch["opsd_gap_crossfit_fold"][2]
        == teacher_fields.batch["opsd_gap_crossfit_fold"][3]
    )
    assert "steering_source_input_ids" not in teacher_fields.batch
    assert "steering_source_candidate_mask" not in teacher_fields.batch


def test_global_batch_steering_disables_distillation_without_both_classes():
    trainer = RayOPSDTrainer.__new__(RayOPSDTrainer)
    settings = {
        "steering": {"source_mode": "caa", "caa_scope": "global_batch"},
        "sdpo_distill_only_failed": False,
        "distill_max_response_tokens": None,
    }
    trainer._opsd_get = lambda key, default: settings.get(key, default)
    input_ids = torch.tensor([[10, 11, 20, 21], [10, 11, 22, 23]])
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(4).repeat(2, 1),
            "responses": input_ids[:, -2:],
        },
        non_tensors={
            "uid": np.asarray(["a", "a"], dtype=object),
            "acc": np.asarray([1.0, 1.0], dtype=np.float32),
        },
    )

    teacher_fields, _ = trainer._build_sdpo_steering_fields(batch, torch.ones(2, 2))

    assert teacher_fields.batch["opsd_distill_mask"].sum().item() == 0
