from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from recipe.opsd.dp_actor import OPSDDataParallelPPOActor
from verl.protocol import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class _TinyModule(torch.nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[weight]], dtype=torch.float32))


def _make_actor(teacher_model: str) -> OPSDDataParallelPPOActor:
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)
    actor.config = OmegaConf.create(
        {
            "opsd_teacher_model": teacher_model,
            "opsd_teacher_ema_rate": 0.25,
            "grad_clip": 1.0,
        }
    )
    actor.actor_module = _TinyModule(weight=1.0)
    actor.teacher_actor_module = _TinyModule(weight=-3.0)
    actor._teacher_initialized = teacher_model in {"fixed", "separate"}
    actor.teacher_optimizer = None
    actor.teacher_lr_scheduler = None
    actor.scaler = None
    return actor


def _weight(module: torch.nn.Module) -> torch.Tensor:
    return next(module.parameters()).detach().clone()


def test_valid_logprob_stats_excludes_nonfinite_values_outside_response_mask():
    log_probs = torch.tensor(
        [
            [-1.0, float("nan"), float("inf")],
            [float("nan"), float("-inf"), float("nan")],
        ]
    )
    response_mask = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.float32)

    stats = OPSDDataParallelPPOActor._valid_logprob_stats(
        log_probs, response_mask, "actor/test"
    )

    assert stats == {
        "actor/test_token_logprob_mean": -1.0,
        "actor/test_token_logprob_std": 0.0,
        "actor/test_seq_logprob_mean": -1.0,
        "actor/test_seq_logprob_std": 0.0,
        "actor/test_valid_token_count": 1.0,
        "actor/test_valid_sequence_count": 1.0,
    }


def test_valid_logprob_stats_empty_mask_is_finite_and_explicitly_counted():
    stats = OPSDDataParallelPPOActor._valid_logprob_stats(
        torch.full((1, 2), float("nan")),
        torch.zeros(1, 2),
        "actor/test",
    )

    assert stats == {
        "actor/test_token_logprob_mean": 0.0,
        "actor/test_token_logprob_std": 0.0,
        "actor/test_seq_logprob_mean": 0.0,
        "actor/test_seq_logprob_std": 0.0,
        "actor/test_valid_token_count": 0.0,
        "actor/test_valid_sequence_count": 0.0,
    }


def test_masked_mean_item_excludes_nonfinite_values_outside_mask():
    value = OPSDDataParallelPPOActor._masked_mean_item(
        torch.tensor([[-2.0, float("nan"), float("inf")]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )

    assert value == -2.0


def test_fixed_teacher_preserves_its_external_weights():
    actor = _make_actor("fixed")

    teacher_module = actor._get_distill_teacher_module({"teacher_model": "fixed", "mode": "opsd"})

    assert teacher_module is actor.teacher_actor_module
    assert actor._teacher_initialized is True
    assert torch.allclose(_weight(actor.teacher_actor_module), torch.tensor([[-3.0]]))

    with torch.no_grad():
        next(actor.actor_module.parameters()).add_(5.0)

    teacher_module = actor._get_distill_teacher_module({"teacher_model": "fixed", "mode": "opsd"})

    assert teacher_module is actor.teacher_actor_module
    assert not torch.allclose(_weight(actor.teacher_actor_module), _weight(actor.actor_module))
    assert torch.allclose(_weight(actor.teacher_actor_module), torch.tensor([[-3.0]]))


def test_separate_teacher_preserves_independent_parameters():
    actor = _make_actor("separate")

    teacher_module = actor._get_distill_teacher_module(
        {"teacher_model": "separate", "mode": "opsd_rlvr"}
    )

    assert teacher_module is actor.teacher_actor_module
    assert next(teacher_module.parameters()) is not next(actor.actor_module.parameters())
    assert torch.allclose(_weight(teacher_module), torch.tensor([[-3.0]]))


def test_ema_teacher_updates_after_optimizer_step():
    actor = _make_actor("ema")
    actor._get_distill_teacher_module({"teacher_model": "ema", "mode": "opsd"})

    with torch.no_grad():
        next(actor.actor_module.parameters()).fill_(5.0)

    with patch.object(DataParallelPPOActor, "_optimizer_step", return_value=torch.tensor(1.0)):
        grad_norm = actor._optimizer_step()

    assert torch.equal(grad_norm, torch.tensor(1.0))
    assert torch.allclose(_weight(actor.teacher_actor_module), torch.tensor([[2.0]]))


def test_separate_teacher_reports_skipped_nonfinite_optimizer_step():
    actor = _make_actor("separate")
    actor.teacher_optimizer = torch.optim.AdamW(actor.teacher_actor_module.parameters(), lr=1e-3)
    parameter = next(actor.teacher_actor_module.parameters())
    parameter.grad = torch.full_like(parameter, float("nan"))
    before = parameter.detach().clone()

    grad_norm, did_step = actor._teacher_optimizer_step()

    assert not torch.isfinite(grad_norm)
    assert did_step is False
    assert torch.equal(parameter.detach(), before)


def test_advantage_shaping_cap_and_mask_never_select_padding():
    actor = _make_actor("separate")
    advantages = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    student_log_probs = torch.tensor([[-2.0, -2.0, -2.0, -99.0, -99.0]])
    teacher_log_probs = torch.tensor([[-1.0, -3.0, -2.0, 99.0, 99.0]])
    response_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    distill_mask = response_mask.clone()

    shaped, evidence, shaping_mask, metrics = actor._shape_student_advantages(
        advantages=advantages,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        distill_mask=distill_mask,
        advantage_shaping_config={
            "enable": True,
            "score_source": "teacher_minus_student_logprob",
            "scale": 1.0,
            "normalize": None,
            "clip_z": None,
            "use_distill_mask": True,
            "allow_token_sign_flip": False,
            "max_delta_fraction": 1.0,
            "max_response_tokens": 2,
        },
    )

    assert torch.equal(shaping_mask, torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32))
    assert torch.equal(evidence, teacher_log_probs - student_log_probs)
    assert torch.equal(shaped[:, 2:], advantages[:, 2:])
    assert metrics["actor/advantage_shaping_pad_delta_max"] == 0.0
    assert metrics["actor/advantage_shaping_token_count"] == 2.0


def test_response_only_steering_mask_excludes_prompt_left_pad_and_response_pad():
    attention = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    response_attention = torch.tensor([[1, 1, 0], [1, 1, 1]])

    mask = OPSDDataParallelPPOActor._build_response_only_steering_mask(
        attention,
        response_attention,
    )

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, False, False, True, True, False],
                [False, False, False, True, True, True],
            ]
        ),
    )
    assert not torch.any(mask & ~attention.bool())


def test_steering_response_accumulation_is_fp32_and_excludes_prompt_and_padding():
    hidden = torch.tensor(
        [[[100.0], [200.0], [3.0], [5.0], [900.0]]], dtype=torch.bfloat16
    )
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    response = torch.tensor([[1, 1, 0]], dtype=torch.float32)

    sums, counts = OPSDDataParallelPPOActor._sum_response_activations_fp32(
        hidden, attention_mask=attention, response_mask=response
    )

    assert sums.dtype == torch.float32
    assert counts.dtype == torch.float32
    torch.testing.assert_close(sums, torch.tensor([[8.0]]), atol=0.0, rtol=0.0)
    torch.testing.assert_close(counts, torch.tensor([[2.0]]), atol=0.0, rtol=0.0)


def test_steering_packed_accumulation_excludes_prompt_and_handles_unequal_lengths():
    # Packed row 0 is [prompt=100, response=3], row 1 is
    # [prompt=200, response=7, response=9].
    packed = torch.tensor([[[100.0], [3.0], [200.0], [7.0], [9.0]]], dtype=torch.bfloat16)
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])
    response = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)

    sums, counts = OPSDDataParallelPPOActor._sum_response_activations_fp32(
        packed, attention_mask=attention, response_mask=response
    )

    torch.testing.assert_close(sums, torch.tensor([[3.0], [16.0]]), atol=0.0, rtol=0.0)
    torch.testing.assert_close(counts, torch.tensor([[1.0], [2.0]]), atol=0.0, rtol=0.0)


def test_steering_8192_token_accumulation_matches_fp32_reference():
    response_length = 8192
    hidden = torch.linspace(-2.0, 2.0, response_length, dtype=torch.float32).to(torch.bfloat16)
    hidden = hidden.reshape(1, response_length, 1)
    mask = torch.ones(1, response_length)
    sums, counts = OPSDDataParallelPPOActor._sum_response_activations_fp32(
        hidden, attention_mask=mask, response_mask=mask
    )
    expected = hidden.float().sum(dim=1)
    torch.testing.assert_close(sums, expected, atol=0.0, rtol=0.0)
    assert counts.item() == response_length


def test_empty_local_steering_sources_keep_rectangular_metric_schema():
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)
    module = torch.nn.Module()
    source_shape = (1, 2, 4)
    micro_batch = {
        "steering_source_input_ids": torch.zeros(source_shape, dtype=torch.long),
        "steering_source_attention_mask": torch.ones(source_shape, dtype=torch.long),
        "steering_source_position_ids": torch.zeros(source_shape, dtype=torch.long),
        "steering_source_response_mask": torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]]]
        ),
        "steering_source_candidate_mask": torch.zeros(1, 2, dtype=torch.bool),
        "steering_source_signs": torch.zeros(1, 2, dtype=torch.long),
    }
    opsd_config = {
        "steering": {
            "source_mode": "caa",
            "activation_aggregation": "per_rollout",
            "detach_vectors": True,
            "layer_fractions": "0.31-0.37",
            "expected_total_layers": 28,
            "expected_layer_indices": [9, 10],
        }
    }

    with (
        patch(
            "recipe.opsd.dp_actor.resolve_fractional_layer_modules",
            return_value={9: torch.nn.Identity(), 10: torch.nn.Identity()},
        ),
        patch.object(DataParallelPPOActor, "_forward_micro_batch", return_value={}),
    ):
        vectors, metrics, audit = actor._extract_sdpo_steering_vectors(
            micro_batch,
            opsd_config=opsd_config,
            module=module,
            use_remove_padding=True,
        )

    assert vectors is None
    assert metrics["actor/opsd_steering_active_rate"] == 0.0
    assert metrics["actor/opsd_steering_vector_norm_mean"] == 0.0
    assert metrics["actor/opsd_steering_vector_norm_max"] == 0.0
    assert "vector_norms" not in audit
    assert audit["accumulation_dtype"] == "float32"


def test_steering_apply_hook_supports_padded_and_packed_layouts():
    vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    apply_mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    hook = OPSDDataParallelPPOActor._make_steering_apply_hook(
        vectors, 0.5, apply_mask, apply_mask
    )

    padded = hook(None, None, torch.zeros(2, 3, 2))
    expected_padded = torch.tensor(
        [
            [[0.5, 1.0], [0.0, 0.0], [0.5, 1.0]],
            [[0.0, 0.0], [1.5, 2.0], [0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(padded, expected_padded)

    packed = hook(None, None, torch.zeros(1, 3, 2))
    torch.testing.assert_close(
        packed,
        torch.tensor([[[0.5, 1.0], [0.5, 1.0], [1.5, 2.0]]]),
    )


def test_global_steering_apply_hook_broadcasts_one_vector_to_every_rollout():
    vector = torch.tensor([[1.0, 2.0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    response_mask = torch.tensor([[0, 1, 1], [0, 1, 0]])
    hook = OPSDDataParallelPPOActor._make_steering_apply_hook(
        vector, 0.5, response_mask, attention_mask
    )

    padded = hook(None, None, torch.zeros(2, 3, 2))
    torch.testing.assert_close(
        padded,
        torch.tensor(
            [
                [[0.0, 0.0], [0.5, 1.0], [0.5, 1.0]],
                [[0.0, 0.0], [0.5, 1.0], [0.0, 0.0]],
            ]
        ),
    )

    packed = hook(None, None, torch.zeros(1, 5, 2))
    torch.testing.assert_close(
        packed,
        torch.tensor([[[0.0, 0.0], [0.5, 1.0], [0.5, 1.0], [0.0, 0.0], [0.5, 1.0]]]),
    )


def test_global_batch_caa_uses_per_rollout_means_across_microbatches():
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)

    class TinySourceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Identity()

    module = TinySourceModule()

    def make_micro(input_ids, signs):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch_size, sequence_length = input_ids.shape
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
                "position_ids": torch.arange(sequence_length).repeat(batch_size, 1),
                "responses": input_ids[:, -2:],
                "response_mask": torch.ones(batch_size, 2),
                "steering_source_outcome_sign": torch.tensor(signs, dtype=torch.float32),
            }
        )

    micro_batches = [
        make_micro([[0, 3, 5], [0, 1, 3]], [1.0, -1.0]),
        make_micro([[0, 7, 9], [0, 11, 13]], [1.0, 1.0]),
    ]

    def fake_forward(*args, **kwargs):
        model_inputs = args[0] if args else kwargs.get("micro_batch")
        if model_inputs is None:
            model_inputs = kwargs.get("data")
        input_ids = model_inputs["input_ids"].to(torch.float32)
        hidden = torch.stack((input_ids, 2.0 * input_ids), dim=-1)
        kwargs["actor_module"].layer(hidden)
        return {}

    opsd_config = {
        "steering": {
            "source_mode": "caa",
            "caa_scope": "global_batch",
            "activation_aggregation": "per_rollout",
            "detach_vectors": True,
            "normalize": "unit_norm",
            "scale": 0.5,
            "layer_fractions": "0",
        }
    }
    with (
        patch("recipe.opsd.dp_actor.get_device_id", return_value="cpu"),
        patch(
            "recipe.opsd.dp_actor.resolve_fractional_layer_modules",
            return_value={0: module.layer},
        ),
        patch.object(DataParallelPPOActor, "_forward_micro_batch", side_effect=fake_forward),
    ):
        vectors, metrics, audit, crossfit = actor._extract_global_batch_sdpo_steering_vectors(
            micro_batches,
            opsd_config=opsd_config,
            module=module,
            use_remove_padding=False,
        )

    expected = torch.tensor([[1.0, 2.0]]) / torch.sqrt(torch.tensor(5.0))
    torch.testing.assert_close(vectors[0], expected)
    torch.testing.assert_close(audit["global_class_weights"][0].squeeze(-1), torch.tensor([3.0, 1.0]))
    assert metrics["actor/opsd_steering_global_positive_rollouts"] == 3.0
    assert metrics["actor/opsd_steering_global_negative_rollouts"] == 1.0
    assert metrics["actor/opsd_steering_global_vector_cross_rank_max_abs_error"] == 0.0
    assert crossfit is None


def test_global_batch_crossfit_vectors_use_only_opposite_prompt_fold():
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)

    class TinySourceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Identity()

    module = TinySourceModule()

    def make_micro(input_ids, signs, folds):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch_size, sequence_length = input_ids.shape
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
                "position_ids": torch.arange(sequence_length).repeat(batch_size, 1),
                "responses": input_ids[:, -2:],
                "response_mask": torch.ones(batch_size, 2),
                "steering_source_outcome_sign": torch.tensor(signs, dtype=torch.float32),
                "opsd_gap_crossfit_fold": torch.tensor(folds, dtype=torch.int64),
            }
        )

    micro_batches = [
        make_micro([[0, 3, 5], [0, 1, 3]], [1.0, -1.0], [0, 0]),
        make_micro([[0, 7, 9], [0, 11, 13]], [1.0, -1.0], [1, 1]),
    ]

    def fake_forward(*args, **kwargs):
        model_inputs = args[0] if args else kwargs.get("micro_batch")
        input_ids = model_inputs["input_ids"].to(torch.float32)
        hidden = torch.stack((input_ids, 2.0 * input_ids), dim=-1)
        kwargs["actor_module"].layer(hidden)
        return {}

    opsd_config = {
        "steering": {
            "source_mode": "caa",
            "caa_scope": "global_batch",
            "activation_aggregation": "per_rollout",
            "detach_vectors": True,
            "normalize": "unit_norm",
            "scale": 0.5,
            "layer_fractions": "0",
            "gap_diagnostics": {"enabled": True, "crossfit_enabled": True},
        }
    }
    with (
        patch("recipe.opsd.dp_actor.get_device_id", return_value="cpu"),
        patch(
            "recipe.opsd.dp_actor.resolve_fractional_layer_modules",
            return_value={0: module.layer},
        ),
        patch.object(DataParallelPPOActor, "_forward_micro_batch", side_effect=fake_forward),
    ):
        _, _, audit, crossfit = actor._extract_global_batch_sdpo_steering_vectors(
            micro_batches,
            opsd_config=opsd_config,
            module=module,
            use_remove_padding=False,
        )

    unit = torch.tensor([1.0, 2.0]) / torch.sqrt(torch.tensor(5.0))
    torch.testing.assert_close(crossfit["vectors"][0][0], -unit)
    torch.testing.assert_close(crossfit["vectors"][0][1], unit)
    torch.testing.assert_close(
        crossfit["global_fold_rollout_counts"], torch.ones(2, 2)
    )
    assert crossfit["available"].tolist() == [True, True]
    assert audit["crossfit_source_fold_for_target"].tolist() == [1, 0]


@pytest.mark.parametrize("crossfit_enabled", [False, True])
def test_global_batch_policy_gradient_uses_full_batch_and_opposite_fold_diagnostics(
    crossfit_enabled,
):
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)

    class TinyProbeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Identity()

    module = TinyProbeModule()

    def make_micro(advantages, signs, fold):
        batch_size = len(advantages)
        response_width = 2
        sequence_width = 4
        return DataProto.from_dict(
            tensors={
                "input_ids": torch.zeros(batch_size, sequence_width, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, sequence_width, dtype=torch.long),
                "position_ids": torch.arange(sequence_width).repeat(batch_size, 1),
                "responses": torch.zeros(batch_size, response_width, dtype=torch.long),
                "response_attention_mask": torch.ones(batch_size, response_width),
                "response_mask": torch.ones(batch_size, response_width),
                "advantages": torch.tensor(advantages, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(-1, response_width)
                .clone(),
                "steering_source_outcome_sign": torch.tensor(signs, dtype=torch.float32),
                "opsd_gap_crossfit_fold": torch.full(
                    (batch_size,), fold, dtype=torch.int64
                ),
            }
        )

    micro_batches = [
        make_micro([2.0, -1.0], [1.0, -1.0], 0),
        make_micro([-1.0, 0.5], [-1.0, 1.0], 1),
    ]

    def fake_forward(model_inputs, **kwargs):
        hidden = torch.zeros(
            model_inputs["input_ids"].size(0),
            model_inputs["input_ids"].size(1),
            2,
            dtype=torch.float32,
        )
        hidden = kwargs["actor_module"].layer(hidden)
        response_width = model_inputs["responses"].size(-1)
        response_hidden = hidden[:, -response_width:, :]
        return {
            "log_probs": response_hidden[..., 0] + 2.0 * response_hidden[..., 1]
        }

    opsd_config = {
        "steering": {
            "source_mode": "policy_gradient",
            "caa_scope": "global_batch",
            "gradient_objective": "grpo_advantage",
            "gradient_aggregation": "per_rollout",
            "detach_vectors": True,
            "normalize": "unit_norm",
            "scale": 0.5,
            "layer_fractions": "0",
            "gap_diagnostics": {
                "enabled": crossfit_enabled,
                "crossfit_enabled": True,
            },
        }
    }
    with (
        patch("recipe.opsd.dp_actor.get_device_id", return_value="cpu"),
        patch(
            "recipe.opsd.dp_actor.resolve_fractional_layer_modules",
            return_value={0: module.layer},
        ),
        patch.object(
            DataParallelPPOActor, "_forward_micro_batch", side_effect=fake_forward
        ),
    ):
        vectors, metrics, audit, crossfit = (
            actor._extract_global_batch_policy_gradient_steering_vectors(
                micro_batches,
                opsd_config=opsd_config,
                module=module,
                use_remove_padding=False,
            )
        )

    unit = torch.tensor([1.0, 2.0]) / torch.sqrt(torch.tensor(5.0))
    torch.testing.assert_close(vectors[0][0], unit)
    if crossfit_enabled:
        torch.testing.assert_close(crossfit["vectors"][0][0], -unit)
        torch.testing.assert_close(crossfit["vectors"][0][1], unit)
        assert crossfit["available"].tolist() == [True, True]
    else:
        assert crossfit is None
    assert audit["source_mode"] == "policy_gradient"
    assert audit["nonzero_parameter_grad_count"] == 0
    assert audit["global_rollout_count"].item() == 4.0
    assert audit["global_nonzero_advantage_count"].item() == 4.0
    assert metrics["actor/opsd_policy_gradient_directional_derivative"] > 0.0


def test_packed_steering_keeps_prompt_tokens_but_applies_zero_delta_to_them():
    vectors = torch.tensor([[2.0], [4.0]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])
    response_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])
    hook = OPSDDataParallelPPOActor._make_steering_apply_hook(
        vectors,
        1.0,
        response_mask,
        attention_mask,
    )

    # Packed order is row0 positions 1,2,3 followed by row1 positions 0,1,2.
    packed = hook(None, None, torch.zeros(1, 6, 1))

    torch.testing.assert_close(
        packed,
        torch.tensor([[[0.0], [2.0], [2.0], [0.0], [4.0], [4.0]]]),
    )


def test_unit_l2_steering_normalization_preserves_zero_vectors():
    vectors = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)

    normalized = OPSDDataParallelPPOActor._normalize_steering_vectors(
        vectors, "unit_norm"
    )

    torch.testing.assert_close(normalized[0].norm(), torch.tensor(1.0))
    assert torch.equal(normalized[1], torch.zeros(2))
    assert normalized.dtype == torch.float32


def test_steering_apply_hook_builder_rechecks_depth_and_selected_layers():
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])
    vectors = {9: torch.zeros(1, 2), 10: torch.zeros(1, 2)}
    apply_mask = torch.ones(1, 1)

    handles = actor._build_steering_apply_hooks(
        module,
        steering_vectors=vectors,
        steering_scale=1.0,
        steering_apply_mask=apply_mask,
        steering_attention_mask=apply_mask,
        expected_total_layers=28,
        expected_layer_indices=[9, 10],
    )
    for handle in handles:
        handle.remove()

    with pytest.raises(ValueError, match="apply-hook layer indexes"):
        actor._build_steering_apply_hooks(
            module,
            steering_vectors=vectors,
            steering_scale=1.0,
            steering_apply_mask=apply_mask,
            steering_attention_mask=apply_mask,
            expected_total_layers=28,
            expected_layer_indices=[8, 9],
        )
