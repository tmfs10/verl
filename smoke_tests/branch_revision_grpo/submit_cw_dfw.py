#!/usr/bin/env python3
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
"""Dry-run-first CW-DFW launcher for branch-revision GRPO smoke."""

from pathlib import PurePosixPath

from smoke_tests.branch_revision_grpo.submit_oci_iad import SmokeClusterProfile, main

CW_DFW_PROFILE = SmokeClusterProfile(
    cluster_name="cw-dfw",
    config_filename="cw-dfw.yaml",
    ssh_alias="dfw",
    remote_output_root=PurePosixPath("/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output"),
    verl_container=("/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh"),
    replace_source_container=False,
)


if __name__ == "__main__":
    main(CW_DFW_PROFILE)
