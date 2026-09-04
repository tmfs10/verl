# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from smoke_tests.random_continuation_baseline.submit_cw_dfw import build_command


def _command(*, prompts: int, rollouts: int, points: int, continuations: int) -> list[str]:
    command, _ = build_command(
        run_tag="random-continuation-test",
        local_config_dir=Path("/tmp/config"),
        verl_root=Path("/tmp/verl"),
        reward_file=Path("/tmp/reward.py"),
        prompts=prompts,
        rollouts=rollouts,
        points=points,
        continuations=continuations,
        seed=46,
        dry_run=True,
    )
    return command


@pytest.mark.parametrize(
    ("prompts", "rollouts", "points", "continuations"),
    [(4, 2, 2, 2), (256, 4, 4, 4), (256, 1, 8, 1)],
)
def test_launcher_preserves_requested_cardinalities(prompts, rollouts, points, continuations):
    command = _command(
        prompts=prompts,
        rollouts=rollouts,
        points=points,
        continuations=continuations,
    )
    rendered = " ".join(command)
    assert f"--n_prompts {prompts}" in rendered
    assert f"--n_samples {rollouts}" in rendered
    assert f"actor_rollout_ref.rollout.n={rollouts}" in rendered
    assert f"algorithm.random_continuation_baseline.points_per_rollout={points}" in rendered
    assert f"algorithm.random_continuation_baseline.continuations_per_mark={continuations}" in rendered
    assert "actor_rollout_ref.rollout.temperature=1.0" in rendered
    assert "trainer.balance_batch=false" in rendered
    assert "--nodes 2" in rendered
    assert "--partition interactive" in rendered
    assert "--no_requeue" in rendered


@pytest.mark.parametrize("field", ["prompts", "rollouts", "points", "continuations"])
def test_launcher_rejects_nonpositive_cardinality(field):
    values = {"prompts": 4, "rollouts": 2, "points": 2, "continuations": 2}
    values[field] = 0
    with pytest.raises(ValueError, match=field):
        _command(**values)
