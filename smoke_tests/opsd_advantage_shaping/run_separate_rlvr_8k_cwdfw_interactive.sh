#!/usr/bin/env bash
set -euo pipefail

# Live gate for the production separate-teacher RLVR/token-IS path at the full
# 4K prompt + 8K padded response dimensions. It uses a smaller prompt batch to
# keep the full arithmetic ledger tractable, while retaining 8 rollouts per
# prompt, two nodes, the production learning rates, and CompMath validation.
# W&B remains disabled by the parent smoke runner.

VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
export ADVANTAGE_SHAPING=1
export ADVANTAGE_SHAPING_SCALE=1.0
export ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS=null
export TEACHER_MODEL_MODE=separate
export TEACHER_RLVR_BACKWARD_SCALE=1.0
export TEACHER_SFT_WEIGHT=0.0
export OFFPOLICY_IS_MODE=token
export RLVR_WARMUP_STEPS=0
export TOTAL_STEPS=3
export SAVE_FREQ=3
export MAX_ACTOR_CKPT_TO_KEEP=2
export NODES=2
export TIME_LIMIT=04:00:00
export N_PROMPTS=16
export N_SAMPLES=8
export N_VAL_SAMPLES=4
export MAX_PROMPT_LEN=4096
export MAX_RESPONSE_LEN=8192
export MAX_TOKENS_PER_GPU=12288
export ACTOR_LR=2e-6
export TEACHER_LR=2e-6
export AUDIT_PROFILE=separate_rlvr
export AUDIT_GLOBAL_STEPS='[1,2,3]'
export AUDIT_EXPECT_WARMUP_STEPS=''
export AUDIT_EXPECT_JOINT_STEPS='1 2 3'
export DRY_RUN="${DRY_RUN:-1}"
export WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"

exec "$VERL_ROOT/smoke_tests/submit_opsd_separate_teacher_interactive.sh"
