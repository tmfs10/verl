import torch

from recipe.opsd.opsd_loss import (
    balance_branch_losses,
    build_sparse_topk_state,
    compute_full_generalized_divergence,
    compute_sampled_reverse_kl_surrogate_token_loss,
    compute_sampled_reverse_kl_token_estimate,
    compute_sparse_topk_tail_generalized_divergence,
    compute_sparse_topk_tail_jsd,
    compute_teacher_is_weights,
    resolve_branch_scales_from_grad_norms,
)


def _dense_generalized_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, beta: float) -> torch.Tensor:
    student_probs = torch.softmax(student_logits, dim=-1)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log(student_probs)
    teacher_log_probs = torch.log(teacher_probs)

    if beta == 0.0:
        return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if beta == 1.0:
        return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    mixture = ((1.0 - beta) * student_probs) + (beta * teacher_probs)
    mixture_log_probs = torch.log(mixture)
    student_term = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)
    teacher_term = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
    return ((1.0 - beta) * student_term) + (beta * teacher_term)


def test_sparse_topk_tail_jsd_is_zero_for_identical_logits():
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, -1.0],
                [0.5, -0.2, 2.0],
            ]
        ],
        requires_grad=True,
    )

    jsd = compute_sparse_topk_tail_jsd(logits, logits, topk=2)
    assert torch.allclose(jsd, torch.zeros_like(jsd), atol=1e-6)


def test_sparse_topk_tail_jsd_detaches_teacher_branch():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -1.0]]], requires_grad=True)

    loss = compute_sparse_topk_tail_jsd(student_logits, teacher_logits, topk=2).sum()
    loss.backward()

    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_sparse_topk_tail_jsd_accepts_compact_state():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -1.0]]], requires_grad=True)

    student_state = build_sparse_topk_state(student_logits, topk=2)
    teacher_state = build_sparse_topk_state(teacher_logits, topk=2)

    loss = compute_sparse_topk_tail_jsd(student_state, teacher_state).sum()
    loss.backward()

    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_sparse_topk_tail_jsd_stays_bounded_with_overlapping_support():
    student_logits = torch.tensor([[[6.0, 5.0, 0.0, -1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[6.0, 4.5, 0.0, -1.0]]], requires_grad=True)

    jsd = compute_sparse_topk_tail_jsd(student_logits, teacher_logits, topk=2)

    assert torch.all(jsd >= 0.0)
    assert torch.all(jsd <= torch.log(torch.tensor(2.0)) + 1e-6)


def test_sparse_topk_generalized_divergence_matches_dense_full_support():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0, 0.25]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -0.5, 0.25]]], requires_grad=True)

    for beta in (0.0, 0.25, 0.5, 1.0):
        sparse = compute_sparse_topk_tail_generalized_divergence(
            student_logits,
            teacher_logits,
            topk=student_logits.size(-1),
            beta=beta,
        )
        dense = _dense_generalized_divergence(student_logits, teacher_logits, beta=beta)
        assert torch.allclose(sparse, dense, atol=1e-6)


def test_sparse_topk_generalized_divergence_token_clip_caps_per_token_value():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0, 0.25]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -0.5, 0.25]]], requires_grad=True)

    unclipped = compute_sparse_topk_tail_generalized_divergence(
        student_logits,
        teacher_logits,
        topk=student_logits.size(-1),
        beta=0.0,
    )
    clipped = compute_sparse_topk_tail_generalized_divergence(
        student_logits,
        teacher_logits,
        topk=student_logits.size(-1),
        beta=0.0,
        token_clip=0.05,
    )

    assert torch.all(clipped <= 0.05 + 1e-8)
    assert torch.all(clipped <= unclipped)


def test_sparse_topk_generalized_divergence_can_leave_tail_unclipped():
    student_state = {
        "topk_ids": torch.tensor([[[0]]], dtype=torch.long),
        "topk_log_probs": torch.log(torch.tensor([[[0.1]]], dtype=torch.float32)),
        "topk_probs": torch.tensor([[[0.1]]], dtype=torch.float32),
        "tail_prob": torch.tensor([[[0.9]]], dtype=torch.float32),
        "tail_log_prob": torch.log(torch.tensor([[[0.9]]], dtype=torch.float32)),
    }
    teacher_state = {
        "topk_ids": torch.tensor([[[0]]], dtype=torch.long),
        "topk_log_probs": torch.log(torch.tensor([[[0.01]]], dtype=torch.float32)),
        "topk_probs": torch.tensor([[[0.01]]], dtype=torch.float32),
        "tail_prob": torch.tensor([[[0.99]]], dtype=torch.float32),
        "tail_log_prob": torch.log(torch.tensor([[[0.99]]], dtype=torch.float32)),
    }

    clipped_with_tail = compute_sparse_topk_tail_generalized_divergence(
        student_state,
        teacher_state,
        beta=0.0,
        token_clip=0.05,
        clip_tail=True,
    )
    clipped_without_tail = compute_sparse_topk_tail_generalized_divergence(
        student_state,
        teacher_state,
        beta=0.0,
        token_clip=0.05,
        clip_tail=False,
    )

    assert torch.all(clipped_with_tail <= 0.05 + 1e-8)
    assert torch.all(clipped_without_tail > 0.05)


def test_full_generalized_divergence_matches_dense_reference():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0, 0.25]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -0.5, 0.25]]], requires_grad=True)

    for beta in (0.0, 0.25, 0.5, 1.0):
        dense = _dense_generalized_divergence(student_logits, teacher_logits, beta=beta)
        full = compute_full_generalized_divergence(student_logits, teacher_logits, beta=beta)
        assert torch.allclose(full, dense, atol=1e-6)


def test_full_generalized_divergence_accepts_log_prob_inputs_and_detaches_teacher():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -1.0]]], requires_grad=True)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    loss = compute_full_generalized_divergence(
        student_log_probs,
        teacher_log_probs,
        beta=0.0,
        inputs_are_log_probs=True,
    ).sum()
    loss.backward()

    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_full_generalized_divergence_token_clip_caps_per_token_value():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0, 0.25]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -0.5, 0.25]]], requires_grad=True)

    unclipped = compute_full_generalized_divergence(student_logits, teacher_logits, beta=0.0)
    clipped = compute_full_generalized_divergence(student_logits, teacher_logits, beta=0.0, token_clip=0.05)

    assert torch.all(clipped <= 0.05 + 1e-8)
    assert torch.all(clipped <= unclipped)


def test_sparse_topk_tail_jsd_matches_generalized_divergence_at_beta_half():
    student_logits = torch.tensor([[[3.0, 1.0, -1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[1.0, 3.0, -1.0]]], requires_grad=True)

    jsd = compute_sparse_topk_tail_jsd(student_logits, teacher_logits, topk=2)
    generalized = compute_sparse_topk_tail_generalized_divergence(
        student_logits,
        teacher_logits,
        topk=2,
        beta=0.5,
    )

    assert torch.allclose(jsd, generalized, atol=1e-6)


def test_sampled_reverse_kl_surrogate_detaches_teacher_branch():
    student_log_probs = torch.tensor([[-0.3, -0.7]], requires_grad=True)
    teacher_log_probs = torch.tensor([[-0.1, -0.9]], requires_grad=True)

    token_estimate = compute_sampled_reverse_kl_token_estimate(student_log_probs, teacher_log_probs)
    loss = compute_sampled_reverse_kl_surrogate_token_loss(student_log_probs, teacher_log_probs).sum()
    loss.backward()

    assert torch.allclose(token_estimate, torch.tensor([[-0.2, 0.2]]), atol=1e-6)
    assert student_log_probs.grad is not None
    assert teacher_log_probs.grad is None


def test_compute_teacher_is_weights_sequence_mode():
    teacher_old = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
    student_behavior = torch.log(torch.tensor([[0.25, 0.25]], dtype=torch.float32))
    response_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    weights, metrics = compute_teacher_is_weights(
        teacher_old_log_probs=teacher_old,
        student_behavior_log_probs=student_behavior,
        response_mask=response_mask,
        mode="sequence",
        clip=10.0,
    )

    assert torch.allclose(weights, torch.full_like(weights, 4.0))
    assert metrics["opsd/is_weight_max"] == 4.0


def test_compute_teacher_is_weights_sequence_mode_clamps_long_log_ratio_sum():
    teacher_old = torch.zeros((1, 4096), dtype=torch.float32)
    student_behavior = torch.full((1, 4096), 5.0, dtype=torch.float32)
    response_mask = torch.ones((1, 4096), dtype=torch.float32)

    weights, metrics = compute_teacher_is_weights(
        teacher_old_log_probs=teacher_old,
        student_behavior_log_probs=student_behavior,
        response_mask=response_mask,
        mode="sequence",
        clip=1e12,
    )

    expected = torch.exp(torch.tensor(-20.0, dtype=torch.float32))
    assert torch.allclose(weights, torch.full_like(weights, expected), atol=0.0, rtol=1e-6)
    assert metrics["opsd/is_weight_mean"] > 0.0


class _TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)


def test_balance_branch_losses_equalizes_gradient_scales():
    actor = _TinyActor()
    input_tensor = torch.tensor([[1.0, -1.0]])

    jsd_loss = actor.lm_head(input_tensor).pow(2).sum()
    rlvr_loss = 10.0 * actor.lm_head(input_tensor).abs().sum()

    total_loss, metrics = balance_branch_losses(
        actor,
        jsd_loss,
        rlvr_loss,
        mix_weight=0.5,
        balance_mode="grad_norm",
        balance_param_subset="lm_head",
    )

    assert total_loss.requires_grad
    assert metrics["opsd/jsd_grad_norm"] > 0
    assert metrics["opsd/rlvr_grad_norm"] > 0


def test_resolve_branch_scales_from_grad_norms_uses_geometric_mean_target():
    jsd_grad_norm = torch.tensor(4.0)
    rlvr_grad_norm = torch.tensor(1.0)

    jsd_scale, rlvr_scale, metrics = resolve_branch_scales_from_grad_norms(
        mix_weight=0.5,
        jsd_grad_norm=jsd_grad_norm,
        rlvr_grad_norm=rlvr_grad_norm,
    )

    assert torch.allclose(jsd_scale, torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(rlvr_scale, torch.tensor(2.0), atol=1e-6)
    assert metrics["opsd/jsd_scale"] == jsd_scale.item()
    assert metrics["opsd/rlvr_scale"] == rlvr_scale.item()
