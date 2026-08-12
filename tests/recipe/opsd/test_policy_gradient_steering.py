import pytest
import torch

from recipe.opsd.policy_gradient_steering import (
    sequence_balanced_policy_gradient_objective,
)


def test_policy_gradient_objective_has_exact_ascent_gradient():
    probe = torch.zeros(2, requires_grad=True)
    coefficients = torch.tensor(
        [
            [[1.0, 2.0], [1.0, 2.0]],
            [[1.0, 2.0], [1.0, 2.0]],
        ]
    )
    log_probs = (coefficients * probe).sum(dim=-1)
    advantages = torch.tensor([[2.0, 2.0], [-1.0, -1.0]])
    response_mask = torch.ones_like(advantages)

    objective, details = sequence_balanced_policy_gradient_objective(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
    )
    (gradient,) = torch.autograd.grad(objective, probe)

    torch.testing.assert_close(gradient, torch.tensor([1.0, 2.0]))
    assert details["rollout_count"] == 2
    assert details["nonzero_advantage_count"] == 2
    assert details["response_token_count"] == 4


def test_policy_gradient_objective_is_per_rollout_not_token_weighted():
    # The first rollout has one token; the second has three.  Their sequence
    # means, and therefore their weights in the objective, are still equal.
    log_probs = torch.tensor(
        [[-2.0, float("nan"), float("inf")], [-1.0, -1.0, -1.0]],
        requires_grad=True,
    )
    advantages = torch.tensor(
        [[1.0, float("nan"), float("inf")], [-1.0, -1.0, -1.0]]
    )
    response_mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float32)

    objective, details = sequence_balanced_policy_gradient_objective(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
    )

    torch.testing.assert_close(objective, torch.tensor(-1.0))
    torch.testing.assert_close(
        details["sequence_logprob_means"], torch.tensor([-2.0, -1.0])
    )
    assert details["response_token_count"] == 4


def test_policy_gradient_objective_rejects_token_varying_grpo_advantage():
    with pytest.raises(ValueError, match="one GRPO advantage per rollout"):
        sequence_balanced_policy_gradient_objective(
            log_probs=torch.tensor([[-1.0, -1.0]]),
            advantages=torch.tensor([[1.0, 0.5]]),
            response_mask=torch.ones(1, 2),
        )


def test_policy_gradient_objective_rejects_nonfinite_actual_response_token():
    with pytest.raises(ValueError, match="non-finite"):
        sequence_balanced_policy_gradient_objective(
            log_probs=torch.tensor([[float("nan")]]),
            advantages=torch.tensor([[1.0]]),
            response_mask=torch.ones(1, 1),
        )
