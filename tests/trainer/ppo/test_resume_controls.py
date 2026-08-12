from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _ActorWorker:
    def __init__(self):
        self.loaded = []

    def load_checkpoint(self, path, del_local_after_load=False):
        self.loaded.append((path, del_local_after_load))


class _StatefulLoader:
    def __init__(self):
        self.loaded = []

    def load_state_dict(self, state):
        self.loaded.append(state)


def _trainer(tmp_path: Path, *, expected_step, load_dataloader_state):
    checkpoint = tmp_path / "global_step_30"
    (checkpoint / "actor").mkdir(parents=True)
    torch.save({"cursor": 17}, checkpoint / "data.pt")

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "resume_path",
                "resume_from_path": str(checkpoint),
                "default_hdfs_dir": None,
                "default_local_dir": str(tmp_path),
                "del_local_ckpt_after_load": False,
                "expected_resume_step": expected_step,
                "load_dataloader_state_on_resume": load_dataloader_state,
            }
        }
    )
    trainer.actor_rollout_wg = _ActorWorker()
    trainer.train_dataloader = _StatefulLoader()
    trainer.use_critic = False
    return trainer, checkpoint


@pytest.mark.parametrize(
    ("load_dataloader_state", "expected_loader_states"),
    [(True, [{"cursor": 17}]), (False, [])],
)
def test_resume_can_restore_or_intentionally_reset_dataloader(
    tmp_path, load_dataloader_state, expected_loader_states
):
    trainer, checkpoint = _trainer(
        tmp_path, expected_step=30, load_dataloader_state=load_dataloader_state
    )

    trainer._load_checkpoint()

    assert trainer.global_steps == 30
    assert trainer.actor_rollout_wg.loaded == [(str(checkpoint / "actor"), False)]
    assert trainer.train_dataloader.loaded == expected_loader_states


def test_resume_guard_fails_before_loading_model_state(tmp_path):
    trainer, _ = _trainer(tmp_path, expected_step=29, load_dataloader_state=True)

    with pytest.raises(RuntimeError, match="expected global step 29, discovered 30"):
        trainer._load_checkpoint()

    assert trainer.actor_rollout_wg.loaded == []
    assert trainer.train_dataloader.loaded == []


def test_requested_dataloader_restore_fails_closed_before_loading_model_state(tmp_path):
    trainer, checkpoint = _trainer(
        tmp_path, expected_step=30, load_dataloader_state=True
    )
    (checkpoint / "data.pt").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="load_dataloader_state_on_resume=true.*no dataloader state",
    ):
        trainer._load_checkpoint()

    assert trainer.actor_rollout_wg.loaded == []
    assert trainer.train_dataloader.loaded == []


def test_resume_guard_accepts_fresh_start(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "auto",
                "default_hdfs_dir": None,
                "default_local_dir": str(tmp_path),
                "expected_resume_step": 0,
            }
        }
    )

    assert trainer._load_checkpoint() == 0


@pytest.mark.parametrize(
    ("global_steps", "total_training_steps", "val_only", "expected"),
    [
        (458, 459, False, False),
        (459, 459, False, True),
        (460, 459, False, True),
        (459, 459, True, False),
    ],
)
def test_completed_training_resume_guard(global_steps, total_training_steps, val_only, expected):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {"val_only": val_only}})
    trainer.global_steps = global_steps
    trainer.total_training_steps = total_training_steps

    assert trainer._completed_training_resume() is expected
