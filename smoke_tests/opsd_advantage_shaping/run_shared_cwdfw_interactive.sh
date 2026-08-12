#!/usr/bin/env bash
set -euo pipefail

# Live gate for the shared-parameter OPSD path. The ground-truth-conditioned
# actor forward is score-only; only shaped student GRPO may update the actor.
# W&B remains disabled by the parent smoke runner.

VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
export ADVANTAGE_SHAPING=1
export ADVANTAGE_SHAPING_SCALE=1.0
export ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS=null
export TEACHER_MODEL_MODE=actor
export TEACHER_RLVR_BACKWARD_SCALE=0.0
export TEACHER_SFT_WEIGHT=0.0
export OFFPOLICY_IS_MODE=token
export RLVR_WARMUP_STEPS=0
export TOTAL_STEPS=3
export N_PROMPTS=8
export N_SAMPLES=8
export N_VAL_SAMPLES=2
export MAX_PROMPT_LEN=1024
export MAX_RESPONSE_LEN=1536
export MAX_TOKENS_PER_GPU=4096
export ACTOR_LR=2e-6
export AUDIT_PROFILE=shared
export AUDIT_GLOBAL_STEPS='[1,2,3]'
export AUDIT_EXPECT_WARMUP_STEPS=''
export AUDIT_EXPECT_JOINT_STEPS='1 2 3'
export DRY_RUN="${DRY_RUN:-1}"
export WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"

exec "$VERL_ROOT/smoke_tests/submit_opsd_separate_teacher_interactive.sh"
