from unittest.mock import patch

import torch

from recipe.opsd.dp_actor import OPSDDataParallelPPOActor
from verl.workers.actor.dp_actor import DataParallelPPOActor


class _TinyModule(torch.nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[weight]], dtype=torch.float32))


def _make_actor(teacher_model: str) -> OPSDDataParallelPPOActor:
    actor = OPSDDataParallelPPOActor.__new__(OPSDDataParallelPPOActor)
    actor.config = {
        "opsd_teacher_model": teacher_model,
        "opsd_teacher_ema_rate": 0.25,
    }
    actor.actor_module = _TinyModule(weight=1.0)
    actor.teacher_actor_module = _TinyModule(weight=-3.0)
    actor._teacher_initialized = False
    actor.scaler = None
    return actor


def _weight(module: torch.nn.Module) -> torch.Tensor:
    return next(module.parameters()).detach().clone()


def test_fixed_teacher_snapshots_once_and_stays_frozen():
    actor = _make_actor("fixed")

    teacher_module = actor._get_distill_teacher_module({"teacher_model": "fixed", "mode": "opsd"})

    assert teacher_module is actor.teacher_actor_module
    assert actor._teacher_initialized is True
    assert torch.allclose(_weight(actor.teacher_actor_module), _weight(actor.actor_module))

    with torch.no_grad():
        next(actor.actor_module.parameters()).add_(5.0)

    teacher_module = actor._get_distill_teacher_module({"teacher_model": "fixed", "mode": "opsd"})

    assert teacher_module is actor.teacher_actor_module
    assert not torch.allclose(_weight(actor.teacher_actor_module), _weight(actor.actor_module))
    assert torch.allclose(_weight(actor.teacher_actor_module), torch.tensor([[1.0]]))


def test_ema_teacher_updates_after_optimizer_step():
    actor = _make_actor("ema")
    actor._get_distill_teacher_module({"teacher_model": "ema", "mode": "opsd"})

    with torch.no_grad():
        next(actor.actor_module.parameters()).fill_(5.0)

    with patch.object(DataParallelPPOActor, "_optimizer_step", return_value=torch.tensor(1.0)):
        grad_norm = actor._optimizer_step()

    assert torch.equal(grad_norm, torch.tensor(1.0))
    assert torch.allclose(_weight(actor.teacher_actor_module), torch.tensor([[2.0]]))
