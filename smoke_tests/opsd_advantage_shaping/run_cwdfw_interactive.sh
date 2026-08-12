#!/usr/bin/env bash
set -euo pipefail

# W&B-disabled, three-step separate-teacher smoke. Step 1 is teacher-SFT-only;
# steps 2-3 jointly exercise response-only student advantage shaping and the
# isolated teacher SFT optimizer. The parent runner performs data/reward
# preflight, waits for completion, downloads the full audit ledger, and invokes
# the independent standard-library verifier in this directory.

VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
export ADVANTAGE_SHAPING=1
export ADVANTAGE_SHAPING_SCALE="${ADVANTAGE_SHAPING_SCALE:-1.0}"
export ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS="${ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS:-null}"
export RLVR_WARMUP_STEPS="${RLVR_WARMUP_STEPS:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-3}"
export N_SAMPLES="${N_SAMPLES:-8}"
export TEACHER_SFT_WEIGHT="${TEACHER_SFT_WEIGHT:-1.0}"
export TEACHER_MODEL_MODE="${TEACHER_MODEL_MODE:-separate}"
export TEACHER_RLVR_BACKWARD_SCALE="${TEACHER_RLVR_BACKWARD_SCALE:-0.0}"
export OFFPOLICY_IS_MODE="${OFFPOLICY_IS_MODE:-sequence}"
export AUDIT_PROFILE="${AUDIT_PROFILE:-separate_sft_warmup}"
export AUDIT_GLOBAL_STEPS="${AUDIT_GLOBAL_STEPS:-[1,2,3]}"
export AUDIT_EXPECT_WARMUP_STEPS="${AUDIT_EXPECT_WARMUP_STEPS:-1}"
export AUDIT_EXPECT_JOINT_STEPS="${AUDIT_EXPECT_JOINT_STEPS:-2 3}"
export DRY_RUN="${DRY_RUN:-1}"
export WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"

exec "$VERL_ROOT/smoke_tests/submit_opsd_separate_teacher_interactive.sh"
