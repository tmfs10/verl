# Copyright 2026
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

from omegaconf import OmegaConf

from verl.trainer.main_ppo import _build_validation_reward_config


def test_validation_disables_uniform_outcome_reward_when_training_enables_it():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_manager": {"source": "register", "name": "batch"},
                "reward_kwargs": {
                    "use_response_logprob_reward_for_uniform_outcome_groups": True,
                    "uniform_outcome_group_success_threshold": 0.5,
                },
            }
        }
    )

    val_config, val_reward_kwargs = _build_validation_reward_config(config)

    assert val_config is config
    assert val_reward_kwargs["use_response_logprob_reward_for_uniform_outcome_groups"] is False
    assert val_reward_kwargs["uniform_outcome_group_success_threshold"] == 0.5


def test_validation_disables_uniform_outcome_reward_even_with_explicit_val_kwargs():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_manager": {"source": "register", "name": "batch"},
                "reward_kwargs": {
                    "use_response_logprob_reward_for_uniform_outcome_groups": True,
                },
                "val_reward_kwargs": {
                    "use_response_logprob_reward_for_uniform_outcome_groups": True,
                },
            }
        }
    )

    _, val_reward_kwargs = _build_validation_reward_config(config)

    assert val_reward_kwargs["use_response_logprob_reward_for_uniform_outcome_groups"] is False
