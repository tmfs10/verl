<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: min(96vw, 1800px) !important;
}
table {
  width: 100% !important;
  max-width: none !important;
}
</style>

## 2026-08-12 pre-publication verification

Before committing the accumulated OPSD and grouped-reward work, stage the
exact current tree into a fresh CW-DFW Lustre directory and run only the
focused container tests. These commands disable W&B and do not perform
generation, validation, or training:

```bash
cd /home/siddjain/workspace/verl/verl_main
RUN_TAG=20260812_precommit_final_v2 \
  bash smoke_tests/svopsd_openthoughts/run_cwdfw_container_tests.sh
RUN_TAG=20260812_precommit_final_v1 \
  bash smoke_tests/longest_success_penalty_reward/run_cwdfw_container_tests.sh
RUN_TAG=20260812_precommit_final_v1 \
  bash smoke_tests/shortest_success_reward/run_cwdfw_container_tests.sh
```

The selected configuration was `cw-dfw.yaml`, account
`nemotron_reason_code`, interactive partition, one exclusive eight-GPU node,
the pinned `verl_vllm012_flashattn_20260321.sqsh` container, and `Requeue=0`.
Jobs `15607436`, `15607477`, and `15607500` completed `0:0` with 174, 21,
and 30 passing tests respectively. An earlier request did not create a Slurm
job because `srun` timed out while communicating with the scheduler; the fresh
retry above is the authoritative run.

# Workflows

## 2026-05-26: SV-OPSD Worktree Setup

1. Create a new git worktree at `/home/siddjain/workspace/verl/verl_svopsd` from the current superproject `HEAD`.
2. Initialize the `recipe` submodule in the new worktree.
3. Implement the SDPO steering-vector variant inside the new worktree.
4. Run focused unit tests for the new helper logic before any smoke test.

## 2026-05-26: SV-OPSD Focused Unit Tests

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' pytest \
  tests/recipe/opsd/test_teacher_utils.py \
  tests/recipe/opsd/test_steering.py \
  tests/recipe/opsd/test_opsd_config.py \
  tests/recipe/opsd/test_opsd_trainer.py \
  tests/recipe/opsd/test_dp_actor.py
```

In the current local shell this direct pytest command did not run because the base Python lacks `torch`, while the available `deepseek` env has the needed runtime dependencies but not `pytest`.

Focused verification was run instead with bytecode writes disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/siddjain/workspace/verl/verl_svopsd \
  conda run -n deepseek python -c '...helper/config assertions...'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/siddjain/workspace/verl/verl_svopsd \
  conda run -n deepseek python -c '...trainer steering-field assertions with unrelated imports stubbed...'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/siddjain/workspace/verl/verl_svopsd \
  conda run -n deepseek python -c '...dp_actor steering extraction/application assertions with unrelated imports stubbed...'

PYTHONDONTWRITEBYTECODE=1 conda run -n deepseek python -c '...ast.parse changed Python files...'
```

All focused verification commands passed after the config frozen-field fix.

## 2026-05-26: SV-OPSD SciKnowEval Chemistry Smoke Test

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/siddjain/workspace/verl/verl_svopsd \
  conda run -n deepseek python smoke_tests/svopsd_chemistry_smoke.py \
  --source-train data/sciknoweval_l3_verl/chemistry_train.jsonl \
  --source-val data/sciknoweval_l3_verl/chemistry_test.jsonl \
  --output-dir /home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry \
  --tokenizer-path /home/siddjain/data/megatron_aws_dfw_smoke/Qwen3-1.7B-tokenizer \
  --train-samples 4 \
  --val-samples 2 \
  --rollouts-per-prompt 4 \
  --max-prompt-length 2048 \
  --layer-fractions 0,0.5,1
```

This local smoke does not enable W&B. It uses the real SciKnowEval Chemistry JSONL rows, writes the tiny smoke subset and result summary under `/home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry`, and validates the SV-OPSD steering path with synthetic same-uid correct/incorrect rollouts on a tiny CPU transformer module. A production-like full trainer run would use the same data paths with `trainer.logger=['console']` for smoke testing, but the current local shell has a CUDA driver/library mismatch.

Result: passed locally after fixing the smoke driver's profiler decorator stub. The run selected 4 Chemistry train rows and 2 Chemistry validation rows, built 16 synthetic same-uid rollouts, resolved `--layer-fractions 0,0.5,1` to layers `[0, 2, 3]`, and wrote `/home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry/svopsd_chemistry_smoke_summary.json`.

## 2026-05-26: SV-OPSD SciKnowEval Chemistry DFW Interactive Trainer Smoke

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default partition `batch`, and VERL container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`. For this smoke, override the partition to `interactive`, request 1 exclusive node with 8 GPUs, and disable W&B by omitting `--enable_wandb`.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
./smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

The submitter runs `recipe.opsd.main_opsd` through `skills_verl_submit.py` with:

- `--cluster cw-dfw`
- `--partition interactive`
- `--nodes 1 --gpus 8`
- `--actor_model /hf_models/Qwen3-8B`
- `--prompt_data /data/rl/sciknoweval_l3/chemistry_train.jsonl`
- `--eval_data /data/rl/sciknoweval_l3/chemistry_test.jsonl`
- `--n_prompts 8 --n_samples 8 --n_val_samples 4`
- `--max_prompt_len 2048 --max_len 4096 --max_tokens_per_gpu 8192`
- `--T 1.0 --val_T 1.0`
- `--no_sandbox`
- `++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`
- SV-OPSD args:
  - `++algorithm.opsd.teacher_source=sdpo_success_rollout`
  - `++algorithm.opsd.sdpo_conditioning_mode=steering`
  - `++algorithm.opsd.steering.layer_fractions=\"0,0.5,1\"`
  - `++algorithm.opsd.steering.correct_rollout_aggregation=all`

The submitter writes its local submit log under `/home/siddjain/data/smoke_tests/svopsd_chemistry/<expname>/submit.log`, monitors the SLURM job, and treats the smoke as passed only if the job completes and the remote training logs contain a nonzero `actor/opsd_steering_active_rate`. The escaped quotes around `layer_fractions` are intentional so Hydra receives the comma-separated selector as a string rather than a sweep.

Reference runs:

- Experiment: `svopsd_chemistry_dfw_interactive_20260526_221733`
- SLURM job: `12179959`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_chemistry_dfw_interactive_20260526_221733`
- Remote logs: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_chemistry_dfw_interactive_20260526_221733/training-logs`
- Result: completed with `actor/opsd_steering_layer_count:3.0` and `OPSD DEBUG` lines in the trainer log, but `actor/opsd_steering_active_rate:0.0` because all 512-token Chemistry rollouts clipped before valid answers. This run verifies launch/config plumbing but not active steering-vector conditioning.
- Experiment: `svopsd_chemistry_dfw_interactive_20260526_222643`
- SLURM job: `12180243`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_chemistry_dfw_interactive_20260526_222643`
- Remote logs: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_chemistry_dfw_interactive_20260526_222643/training-logs`
- Result: failed during actor update with an NCCL all-gather timeout. The dumped training rollout had `10/64` correct rows, so this run exposed the FSDP collective-safety issue in steering extraction when some ranks have no local steering candidates while others do. The code now runs a masked dummy steering-source forward on no-candidate ranks before returning no vectors.
- Experiment: `svopsd_chemistry_dfw_interactive_20260526_224708`
- SLURM job: `12181180`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_chemistry_dfw_interactive_20260526_224708`
- Remote logs: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_chemistry_dfw_interactive_20260526_224708/training-logs`
- Result: passed. The job completed and emitted active steering metrics at `step:1`: `actor/opsd_steering_layer_count:3.0`, `actor/opsd_steering_candidate_mean:0.90625`, `actor/opsd_steering_active_rate:0.59375`, `actor/opsd_distill_active_rate:0.59375`, and `actor/opsd_loss:0.27970479847863317`.

## 2026-05-27: SciKnowEval Chemistry SDPO vs SV-OPSD DFW Batch Runs

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default non-interactive partition `batch`, timeout `04:00:00`, and VERL container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`. These are non-test runs, so W&B is enabled by default.

As of the CAA steering update, the reusable submitter defaults to `sdpo svopsd_caa`; the previous `svopsd_first` and `svopsd_all` variants were superseded because CAA always uses all same-prompt positives and negatives to form `mean_positive_residual - mean_negative_residual`.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The submitter launches three independent 4-node, 8-GPU-per-node jobs:

- `sdpo`: prompt-append SDPO baseline
- `svopsd_first`: SV-OPSD variant A with `algorithm.opsd.steering.correct_rollout_aggregation=first`
- `svopsd_all`: SV-OPSD variant B with `algorithm.opsd.steering.correct_rollout_aggregation=all`

Common run settings:

- `--cluster cw-dfw`
- `--partition batch`
- `--nodes 4 --gpus 8`
- `--actor_model /hf_models/Qwen3-8B`
- `--prompt_data /data/rl/sciknoweval_l3/chemistry_train.jsonl`
- `--eval_data /data/rl/sciknoweval_l3/chemistry_test.jsonl`
- `--n_prompts 32 --n_samples 8 --n_val_samples 16 --val_batch_size 32`
- `--max_prompt_len 2048 --max_len 10240 --max_tokens_per_gpu 10240`
- `--num_epochs 5 --save_freq 20 --test_freq 20`
- `--actor_lr 1e-5 --T 1.0 --val_T 0.6 --val_top_p 0.95`
- `--enable_wandb --wandb_group chemistry_sdpo_svopsd_<run_tag>`
- `++data.apply_chat_template_kwargs.enable_thinking=False`
- `++algorithm.opsd.teacher_source=sdpo_success_rollout`
- `++algorithm.opsd.teacher_model=ema`
- `++algorithm.opsd.teacher_ema_rate=0.05`
- `++algorithm.opsd.topk=100`
- `++algorithm.opsd.distill_loss=topk_jsd`
- `++algorithm.opsd.distill_beta=0.5`
- `actor_rollout_ref.actor.optim.lr_warmup_steps=10`
- `actor_rollout_ref.actor.optim.weight_decay=0.01`
- `actor_rollout_ref.actor.optim.clip_grad=1.0`

SV-OPSD variants add:

- `++algorithm.opsd.sdpo_conditioning_mode=steering`
- `++algorithm.opsd.steering.layer_fractions=\"0.31-0.37\"`
- `++algorithm.opsd.steering.scale=1.0`

The submitter writes local submit logs and `submitted_jobs.tsv` under `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/<run_tag>`. Remote outputs map to `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/<expname>`.

Submitted run tag: `20260527_004123`.

Submitted jobs:

- `sdpo`: SLURM job `12184906`, experiment `chemistry_sdpo_qwen3_8b_4n_20260527_004123`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_20260527_004123`
- `svopsd_first`: SLURM job `12184916`, experiment `chemistry_svopsd_first_qwen3_8b_l31_37_4n_20260527_004123`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_first_qwen3_8b_l31_37_4n_20260527_004123`
- `svopsd_all`: SLURM job `12184978`, experiment `chemistry_svopsd_all_qwen3_8b_l31_37_4n_20260527_004123`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_all_qwen3_8b_l31_37_4n_20260527_004123`

Initial scheduler state after submission: all three jobs were `PENDING` on partition `batch`, 4 nodes each. Generated sbatch files include `#SBATCH --exclusive`, `#SBATCH --gpus-per-node=8`, `#SBATCH --nodes=4`, and the main `srun` uses `--nodes=4 --gpus-per-node=8`.

Later live status after the cancellation request: DFW jobs `12184906`, `12184916`, and `12184978` were no longer active and all had already completed.

## 2026-05-27: CAA SV-OPSD Chemistry DFW Interactive Smoke

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default partition `batch`, and timeout `04:00:00`. For this test workflow, override the partition to `interactive`, request one exclusive 8-GPU node, and leave W&B disabled.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
EXP_NAME=svopsd_caa_chemistry_dfw_interactive_<timestamp> \
LAYER_FRACTIONS=0.31-0.37 \
./smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

The submitter uses Qwen3-8B and the SciKnowEval Chemistry train/test JSONL files:

- `/data/rl/sciknoweval_l3/chemistry_train.jsonl`
- `/data/rl/sciknoweval_l3/chemistry_test.jsonl`

The CAA implementation keeps the teacher prompt unchanged, gathers all same-prompt verifier-correct rollouts as positives and all verifier-incorrect rollouts as negatives, extracts selected-layer residual-stream vectors as `mean(positives) - mean(negatives)` over response tokens, and applies one vector per selected layer during the teacher forward. The downstream token scoring remains the normal OPSD/SDPO distillation loss.

The submitter writes its local submit log under `/home/siddjain/data/smoke_tests/svopsd_chemistry/<expname>/submit.log`, monitors SLURM, and treats the smoke as passed only if the job completes and the remote training logs contain a nonzero `actor/opsd_steering_active_rate`.

Reference run:

- Experiment: `svopsd_caa_chemistry_dfw_interactive_20260527_090347`
- SLURM job: `12199260`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_caa_chemistry_dfw_interactive_20260527_090347`
- Remote logs: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_caa_chemistry_dfw_interactive_20260527_090347/training-logs`
- Result: passed. The job completed and emitted active CAA steering metrics at `step:1`: `actor/opsd_steering_layer_count:2.0`, `actor/opsd_steering_candidate_mean:3.75`, `actor/opsd_steering_positive_mean:0.90625`, `actor/opsd_steering_negative_mean:2.84375`, `actor/opsd_steering_active_rate:0.46875`, `actor/opsd_distill_active_rate:0.46875`, and `actor/opsd_loss:0.0032935619092313573`.

## 2026-05-27: SciKnowEval Chemistry SDPO vs SV-OPSD EOS Batch Runs

Use the EOS codegen config copied from `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/eos.yaml` to `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd_eos/config/eos.yaml`. The copied config resolves to SSH host `eos`, account `nemotron_reason_code`, default partition `batch`, timeout `04:00:00`, and output root `/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output`. EOS Slurm reports GPU H100 nodes but does not expose GPU GRES/TRES (`Gres=(null)`), so generated jobs must keep `disable_gpus_per_node: True`; requesting `--gpus-per-node=8` fails with `Invalid generic resource (gres) specification`. The run still requests exclusive full nodes, sets `trainer.n_gpus_per_node=8`, and waits for Ray to detect 32 GPUs before training starts. The copied config updates the `/home` plus `/root/.netrc` mount sources from the DFW portfolio path to EOS-local `/lustre/fsw/llmservice_nemo_reasoning/siddjain`.

Transfer the Chemistry files from DFW to EOS:

```bash
ssh eos 'mkdir -p /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/sciknoweval_l3'

ssh dfw 'cd /lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/sciknoweval_l3 && tar -cf - chemistry_train.jsonl chemistry_test.jsonl' \
  | ssh eos 'cd /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/sciknoweval_l3 && tar -xpf -'
```

Verify with `sha256sum` on both clusters before launching. The EOS-visible container data paths are:

- `/data/rl/sciknoweval_l3/chemistry_train.jsonl`
- `/data/rl/sciknoweval_l3/chemistry_test.jsonl`

Run the same three variants from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
CLUSTER=eos \
CONFIG_DIR=/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd_eos/config \
OUTPUT_BASE_DIR=/output/rl/chemistry_sdpo_svopsd \
REMOTE_OUTPUT_ROOT=/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output \
RUN_TAG=eos_<timestamp> \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The run settings match the DFW submissions: Qwen3-8B, 4 exclusive H100 nodes with the trainer configured for 8 GPUs per node, SDPO baseline plus SV-OPSD `first` and `all`, steering layer range `0.31-0.37`, W&B enabled, and paper-aligned SDPO hyperparameters.

Submitted run tag: `eos_20260527_005824`.

Submitted jobs:

- `sdpo`: SLURM job `5332410`, experiment `chemistry_sdpo_qwen3_8b_4n_eos_20260527_005824`, remote output `/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_eos_20260527_005824`
- `svopsd_first`: SLURM job `5332412`, experiment `chemistry_svopsd_first_qwen3_8b_l31_37_4n_eos_20260527_005824`, remote output `/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_first_qwen3_8b_l31_37_4n_eos_20260527_005824`
- `svopsd_all`: SLURM job `5332413`, experiment `chemistry_svopsd_all_qwen3_8b_l31_37_4n_eos_20260527_005824`, remote output `/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_all_qwen3_8b_l31_37_4n_eos_20260527_005824`

Initial scheduler state after submission: all three jobs were `PENDING` on partition `batch`, 4 nodes each, reason `Priority`. Generated sbatch files include `#SBATCH --exclusive`, `#SBATCH --nodes=4`, and `#SBATCH --partition=batch`; EOS-generated scripts intentionally omit `--gpus-per-node` because EOS Slurm rejects GPU GRES requests.

## 2026-05-27: SciKnowEval Chemistry SDPO vs SV-OPSD AWS-IAD Pool0 Runs

Use the AWS-IAD codegen config copied from `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/aws-iad.yaml` to `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd_aws_iad/config/aws-iad.yaml`. The copied config resolves to SSH host `aws-iad-cs-002-login-01.nvidia.com`, account `nemotron_reason_code`, default partition `pool0`, timeout `04:00:00`, and output root `/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output`. The copy adds the required VERL container entry using the existing AWS-IAD container `/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-verl-latest.sqsh`.

AWS-IAD exposes `pool0` GPU nodes with `gpu:8` GRES, so the generated sbatch/srun scripts should request `--gpus-per-node=8` plus `--exclusive`. Use the mounted user model path because `/hf_models/Qwen3-8B` is absent on AWS-IAD:

- model source on DFW: `/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-8B`
- model destination on AWS-IAD: `/lustre/fsw/portfolios/nemotron/users/siddjain/my_models/Qwen3-8B`
- container-visible model path: `/my_models/Qwen3-8B`

First try Data Mover through the configured `aws-iad-1` alias:

```bash
python3 /home/siddjain/workspace/scripts/src/cluster_datamover_transfer.py \
  'dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/sciknoweval_l3/chemistry_train.jsonl' \
  'aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/llm/data/rl/sciknoweval_l3/' \
  --transfer-id chemistry-train-dfw-to-aws-iad-<timestamp>

python3 /home/siddjain/workspace/scripts/src/cluster_datamover_transfer.py \
  'dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/sciknoweval_l3/chemistry_test.jsonl' \
  'aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/llm/data/rl/sciknoweval_l3/' \
  --transfer-id chemistry-test-dfw-to-aws-iad-<timestamp>

python3 /home/siddjain/workspace/scripts/src/cluster_datamover_transfer.py \
  'dfw:/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-8B' \
  'aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/my_models/' \
  --transfer-id qwen3-8b-dfw-to-aws-iad-<timestamp>
```

Verify train/test checksums and line counts against DFW, and verify the model directory exists before launching.

If Data Mover stalls because its DFW CPU Slurm runner is pending, use direct DFW-to-AWS-IAD `rsync` over SSH instead. The `aws-iad-1` alias resolves locally, while DFW resolves the explicit host `aws-iad-cs-002-login-01.nvidia.com`.

```bash
ssh aws-iad-1 'mkdir -p /lustre/fsw/portfolios/nemotron/users/siddjain/llm/data/rl/sciknoweval_l3 /lustre/fsw/portfolios/nemotron/users/siddjain/my_models'

ssh dfw 'rsync -az --info=progress2 \
  /lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/sciknoweval_l3/chemistry_train.jsonl \
  /lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/sciknoweval_l3/chemistry_test.jsonl \
  aws-iad-cs-002-login-01.nvidia.com:/lustre/fsw/portfolios/nemotron/users/siddjain/llm/data/rl/sciknoweval_l3/'

ssh dfw 'rsync -a --partial --info=progress2 \
  /lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-8B/ \
  aws-iad-cs-002-login-01.nvidia.com:/lustre/fsw/portfolios/nemotron/users/siddjain/my_models/Qwen3-8B/'
```

Run the same three variants from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
CLUSTER=aws-iad \
CONFIG_DIR=/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd_aws_iad/config \
PARTITION=pool0 \
OUTPUT_BASE_DIR=/output/rl/chemistry_sdpo_svopsd \
REMOTE_OUTPUT_ROOT=/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output \
ACTOR_MODEL=/my_models/Qwen3-8B \
RUN_TAG=aws_iad_<timestamp> \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The run settings match the DFW/EOS submissions: Qwen3-8B, 4 nodes, 8 GPUs per node, SDPO baseline plus SV-OPSD `first` and `all`, steering layer range `0.31-0.37`, W&B enabled, and paper-aligned SDPO hyperparameters.

The initial AWS-IAD submission under run tag `aws_iad_20260527_020811` allocated 4 nodes / 32 GPUs per job but failed before training because `skills_verl_submit.py` injected `pip install --upgrade one-logger-utils`, which upgraded `click` to `8.4.1`; Ray `2.46.0` in the container then failed on CLI import with `ValueError: <object object ...> is not a valid Sentinel`. A direct container diagnostic without that injected install reported Ray `2.46.0`, Click `8.1.7`, and W&B `0.19.11`, so the AWS-IAD config now points `containers.verl` at `/lustre/fsw/portfolios/nemotron/users/siddjain/containers/nemo-skills-verl-latest-onelogger.sqsh`, a symlink to the same VERL container. The `onelogger` name uses the submitter's existing guard to suppress the package install while keeping the same image.

Relaunched run tag: `aws_iad_noinstall_20260527_021548`.

Submitted jobs:

- `sdpo`: SLURM job `4154001`, experiment `chemistry_sdpo_qwen3_8b_4n_aws_iad_noinstall_20260527_021548`, remote output `/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_aws_iad_noinstall_20260527_021548`
- `svopsd_first`: SLURM job `4154005`, experiment `chemistry_svopsd_first_qwen3_8b_l31_37_4n_aws_iad_noinstall_20260527_021548`, remote output `/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_first_qwen3_8b_l31_37_4n_aws_iad_noinstall_20260527_021548`
- `svopsd_all`: SLURM job `4154021`, experiment `chemistry_svopsd_all_qwen3_8b_l31_37_4n_aws_iad_noinstall_20260527_021548`, remote output `/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_all_qwen3_8b_l31_37_4n_aws_iad_noinstall_20260527_021548`

Initial scheduler state after relaunch: all three jobs were `RUNNING` on `pool0`, 4 nodes each, with `ReqTRES`/`AllocTRES` showing `gres/gpu=32`. Generated sbatch files include `#SBATCH --exclusive`, `#SBATCH --gpus-per-node=8`, `#SBATCH --nodes=4`, `#SBATCH --partition=pool0`, and the main `srun` includes `--gpus-per-node=8 --nodes=4`. Main logs for all three jobs reached `Ray GPU resources ready: 32.0/32` and `Starting training`.

## 2026-05-27: Copy DFW VERL FlashAttention Container to AWS-IAD

Copy the DFW VERL container used by the faster Chemistry SDPO run to `aws-iad-1:~/lustre/containers/`:

```bash
python3 /home/siddjain/workspace/scripts/src/cluster_datamover_transfer.py \
  'dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh' \
  'aws-iad-1:~/lustre/containers/' \
  --transfer-id verl-vllm012-flashattn-20260321-dfw-to-aws-iad
```

Verify the copied file on both clusters:

```bash
ssh dfw 'stat -c "%n %s" /lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh'
ssh aws-iad-1 'stat -c "%n %s" ~/lustre/containers/verl_vllm012_flashattn_20260321.sqsh'
```

Result: completed. Data Mover upload job `c28504df-8b83-4040-b804-ada10b7da894` and download job `ea5f23ef-f68c-4cf4-b5b4-920df266d470` both finished with `files=1/1`, `bytes_transferred=21465190400`, and `errors=0`.

Final AWS-IAD path:

```text
/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_code/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh
```

Verified checksum on both DFW and AWS-IAD:

```text
06f5b789dfe8dc76fa64ae4eaa4771d1fa4b6459ad6078364501e4bee2040fe9
```

## 2026-05-27: SciKnowEval Chemistry CAA SV-OPSD DFW and AWS-IAD Reruns

Run only the current SV-OPSD CAA variant with the same Chemistry settings as the previous 4-node SDPO/SV-OPSD submissions. This uses Qwen3-8B, 32 prompts per PPO step, 8 rollouts per prompt, max response length 10240, 5 epochs, validation every 20 steps, W&B enabled, and steering layer selector `0.31-0.37`.

DFW uses the codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, partition `batch`, 4 exclusive nodes, and 8 GPUs per node:

```bash
RUN_TAG=dfw_caa_<timestamp> \
VARIANTS=svopsd_caa \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

AWS-IAD uses the canonical codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/aws-iad.yaml`, partition `pool0`, 4 exclusive nodes, and 8 GPUs per node. The canonical config points `containers.verl` at `/lustre/fsw/portfolios/nemotron/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`, which resolves to the copied DFW FlashAttention SQSH.

```bash
CLUSTER=aws-iad \
CONFIG_DIR=/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
PARTITION=pool0 \
OUTPUT_BASE_DIR=/output/rl/chemistry_sdpo_svopsd \
REMOTE_OUTPUT_ROOT=/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output \
ACTOR_MODEL=/my_models/Qwen3-8B \
RUN_TAG=aws_iad_caa_<timestamp> \
VARIANTS=svopsd_caa \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

Submitted run tag: `20260527_111659`.

Submitted jobs:

- DFW `svopsd_caa`: SLURM job `12203486`, experiment `chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_20260527_111659`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_20260527_111659`
- AWS-IAD `svopsd_caa`: SLURM job `4160881`, experiment `chemistry_svopsd_caa_qwen3_8b_l31_37_4n_aws_iad_caa_20260527_111659`, remote output `/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_aws_iad_caa_20260527_111659`

Initial scheduler state after submission:

- DFW job `12203486`: `PENDING` on `batch`, 4 nodes, reason `Priority`.
- AWS-IAD job `4160881`: `PENDING` on `pool0`, 4 nodes, reason `Resources`.

Generated sbatch/srun scripts for both jobs include `#SBATCH --exclusive`, `#SBATCH --nodes=4`, `#SBATCH --gpus-per-node=8`, and the main `srun` includes `--nodes=4 --gpus-per-node=8`.

## 2026-05-27: SciKnowEval Chemistry CAA SV-OPSD DFW LR Sweep

Use the same 4-node DFW batch CAA SV-OPSD Chemistry workflow as the `dfw_caa_20260527_111659` run, but override the actor learning rate with `ACTOR_LR`. The submitter default remains `1e-5`; this sweep starts one job at `1e-4` and one at `5e-5`.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
RUN_TAG=dfw_caa_lr1e4_<timestamp> \
VARIANTS=svopsd_caa \
ACTOR_LR=1e-4 \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh

RUN_TAG=dfw_caa_lr5e5_<timestamp> \
VARIANTS=svopsd_caa \
ACTOR_LR=5e-5 \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

Both jobs use:

- DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`
- non-interactive partition `batch`
- 4 exclusive nodes, 8 GPUs per node
- Qwen3-8B at `/hf_models/Qwen3-8B`
- SciKnowEval Chemistry train/test JSONL under `/data/rl/sciknoweval_l3`
- generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- CAA steering with `++algorithm.opsd.sdpo_conditioning_mode=steering` and `++algorithm.opsd.steering.layer_fractions=\"0.31-0.37\"`
- LR warmup `actor_rollout_ref.actor.optim.lr_warmup_steps=10`, weight decay `0.01`, constant LR schedule, and W&B enabled

Submitted runs:

- LR `1e-4`: SLURM job `12212000`, experiment `chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr1e4_20260527_150451`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr1e4_20260527_150451`
- LR `5e-5`: SLURM job `12212015`, experiment `chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr5e5_20260527_150517`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr5e5_20260527_150517`

Initial scheduler state after submission: both jobs were `PENDING` on `batch`, 4 nodes each, with `ReqTRES` showing `gres/gpu=32`.

## 2026-05-27: SciKnowEval Chemistry Vanilla SDPO DFW LR 1e-6

Use the reusable Chemistry SDPO/SV-OPSD DFW submitter for a vanilla prompt-append SDPO baseline, overriding only the actor learning rate to `1e-6`. This keeps the same Chemistry production settings as the earlier DFW SDPO/SV-OPSD runs.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
RUN_TAG=dfw_sdpo_lr1e6_<timestamp> \
VARIANTS=sdpo \
ACTOR_LR=1e-6 \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The run uses:

- DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`
- non-interactive partition `batch`
- 4 exclusive nodes, 8 GPUs per node
- Qwen3-8B at `/hf_models/Qwen3-8B`
- SciKnowEval Chemistry train/test JSONL under `/data/rl/sciknoweval_l3`
- generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- vanilla SDPO prompt-append conditioning via `++algorithm.opsd.sdpo_conditioning_mode=prompt_append`
- LR warmup `actor_rollout_ref.actor.optim.lr_warmup_steps=10`, weight decay `0.01`, constant LR schedule, and W&B enabled

Submitted run:

- LR `1e-6`: SLURM job `12212148`, experiment `chemistry_sdpo_qwen3_8b_4n_dfw_sdpo_lr1e6_20260527_151110`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_dfw_sdpo_lr1e6_20260527_151110`

Initial scheduler state after submission: job `12212148` was `PENDING` on `batch`, 4 nodes, with `ReqTRES` showing `gres/gpu=32`.

The vanilla LR `1e-6` job was later cancelled before it started:

```bash
ssh dfw 'scancel 12212148'
```

Accounting showed `CANCELLED` with `Elapsed=00:00:00` and no start time.

## 2026-05-27: SciKnowEval Chemistry Vanilla SDPO DFW LR 1e-5 with 16K Response Length

Use the reusable Chemistry SDPO/SV-OPSD DFW submitter for a vanilla prompt-append SDPO baseline with actor LR `1e-5` and a 16K response cap. The submitter treats `MAX_LEN` as total prompt-plus-response length, so use `MAX_LEN=18432` to obtain `data.max_response_length=16384` with `MAX_PROMPT_LEN=2048`.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
RUN_TAG=dfw_sdpo_lr1e5_resp16k_<timestamp> \
VARIANTS=sdpo \
ACTOR_LR=1e-5 \
MAX_PROMPT_LEN=2048 \
MAX_LEN=18432 \
MAX_TOKENS_PER_GPU=18432 \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The run uses:

- DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`
- non-interactive partition `batch`
- 4 exclusive nodes, 8 GPUs per node
- Qwen3-8B at `/hf_models/Qwen3-8B`
- SciKnowEval Chemistry train/test JSONL under `/data/rl/sciknoweval_l3`
- generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- vanilla SDPO prompt-append conditioning via `++algorithm.opsd.sdpo_conditioning_mode=prompt_append`
- `data.max_prompt_length=2048`, `data.max_response_length=16384`
- auto SDPO teacher prompt budget `algorithm.opsd.max_prompt_length=18432`
- auto actor/rollout/ref logprob token budget at least `34816`
- LR warmup `actor_rollout_ref.actor.optim.lr_warmup_steps=10`, weight decay `0.01`, constant LR schedule, and W&B enabled

Submitted run:

- LR `1e-5`, 16K response length: SLURM job `12213113`, experiment `chemistry_sdpo_qwen3_8b_4n_dfw_sdpo_lr1e5_resp16k_20260527_153241`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_dfw_sdpo_lr1e5_resp16k_20260527_153241`

Initial scheduler state after submission: job `12213113` was `PENDING` on `batch`, 4 nodes, with `TRES_PER_NODE=gres/gpu:8`.

## 2026-05-27: SciKnowEval Chemistry CAA SV-OPSD DFW LR 5e-5 with Layer Range 0.2-0.6

Use the same 4-node DFW batch CAA SV-OPSD Chemistry workflow as the earlier CAA runs, but override the actor learning rate to `5e-5` and expand the steering layer fraction selector to `0.2-0.6`.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
RUN_TAG=dfw_caa_lr5e5_l20_60_<timestamp> \
VARIANTS=svopsd_caa \
ACTOR_LR=5e-5 \
LAYER_FRACTIONS=0.2-0.6 \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The run uses:

- DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`
- non-interactive partition `batch`
- 4 exclusive nodes, 8 GPUs per node
- Qwen3-8B at `/hf_models/Qwen3-8B`
- SciKnowEval Chemistry train/test JSONL under `/data/rl/sciknoweval_l3`
- generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- CAA steering via `++algorithm.opsd.sdpo_conditioning_mode=steering`
- layer selector `++algorithm.opsd.steering.layer_fractions=\"0.2-0.6\"`
- LR warmup `actor_rollout_ref.actor.optim.lr_warmup_steps=10`, weight decay `0.01`, constant LR schedule, and W&B enabled

## 2026-05-28: DFW/EOS Status Snapshot

Live scheduler checks:

- DFW `squeue -u siddjain`: no active jobs.
- EOS `squeue -u siddjain`: no active jobs.
- Direct AWS-IAD check failed with `ssh: Could not resolve hostname aws-iad: Temporary failure in name resolution`; no fallback route was used.

DFW job accounting:

| Job | Run | State | Last / Latest Useful Metric |
|---:|---|---|---|
| `12212000` | Chemistry CAA SV-OPSD LR `1e-4`, layers `0.31-0.37` | `COMPLETED`, `03:58:12` | Latest validation step `80`: sample `acc_mean=0.000000`; final train response clip ratio `0.99609375`, steering active `0.0`. |
| `12213113` | Chemistry vanilla SDPO LR `1e-5`, 16K response | `FAILED`, `01:59:55` | Latest validation step `40`: sample `acc_mean=0.607276`; failed at train step `46` with `NotImplementedError: sequence_length=21284 is larger than max_length=18432`. |
| `12213198` | Chemistry CAA SV-OPSD LR `5e-5`, layers `0.2-0.6` | `COMPLETED`, `03:57:18` | Latest validation step `80`: sample `acc_mean=0.000000`; final train response clip ratio `1.0`, steering active `0.0`. |
| `12214042` | Chemistry CAA SV-OPSD LR `1e-5`, layers `0.2-0.6` | `COMPLETED`, `03:57:00` | Latest validation step `80`: sample `acc_mean=0.000000`; final train response clip ratio `1.0`, steering active `0.0`. |
| `12215167` | DeepMath GRPO LR `5e-6` | `COMPLETED`, `03:55:48` | Latest validation step `20`: AIME24 pass@1 `0.487500`, AIME25 `0.345833`, HMMT `0.280612`. |
| `12215171` | DeepMath vanilla SDPO LR `5e-6` | `TIMEOUT`, `04:00:27` | Latest validation step `20`: AIME24/AIME25/HMMT pass@1 all `0.000000`; final train response clip ratio `1.0`. |
| `12215183` | DeepMath SV-SDPO CAA LR `5e-6` | `TIMEOUT`, `04:00:28` | Latest validation step `20`: AIME24 pass@1 `0.466667`, AIME25 `0.300000`, HMMT `0.248724`; steering active `0.1171875`. |

EOS job accounting:

- Smoke job `5337181` remains `FAILED` after `00:09:06`; root cause was vLLM memory startup at `gpu_memory_utilization=0.7`.

Submitted run:

- LR `1e-5`, layer selector `0.2-0.6`: SLURM job `12214042`, experiment `chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr1e5_l20_60_20260527_161839`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr1e5_l20_60_20260527_161839`

Initial scheduler state after submission: job `12214042` was `PENDING` on `batch`, 4 nodes, with `TRES_PER_NODE=gres/gpu:8`.

## 2026-05-27: DeepMath/CompMath GRPO, Vanilla SDPO, and SV-SDPO CAA DFW Runs

Launch three 4-node DFW batch runs on DeepMath train with CompMath validation, keeping the established DeepMath 30B setup and setting actor LR to `5e-6` with an 8K training response cap.

Run from any directory:

```bash
RUN_TAG=dfw_deepmath_sdpo_svopsd_<timestamp> \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Common settings:

- DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`
- partition `batch`
- 4 exclusive nodes, 8 GPUs per node
- model `/hf_models/Qwen3-30B-A3B`
- train `/data/rl/mathgen/deepmath_verl.jsonl`
- validation `/data/rl/mathgen/comp_math_verl.jsonl`
- rollout tensor parallel size `4`
- `n_prompts=32`
- `n_samples=8`
- `n_val_samples=8`
- `actor_lr=5e-6`
- generation temperature `0.85`
- validation temperature `0.6`, validation top-p `1.0`
- `max_prompt_len=2k`
- `max_len=10k`, which maps to `data.max_response_length=8192`
- `max_tokens_per_gpu=10240`
- `num_epochs=5`
- `num_ppo_iter=2`
- `save_freq=20`
- `test_freq=20`
- W&B enabled
- no-placeholder safety for DeepMath/CompMath:
  - `data.filter_overlong_prompts=False`
  - `++data.dynamic_masked_solution=False`
  - `++data.min_masked_fraction=null`
  - `++data.max_masked_fraction=null`
  - `++data.mask_seed=null`

Variants:

- `grpo`: `script_module=verl.trainer.main_ppo`
- `sdpo`: `script_module=recipe.opsd.main_opsd`, `++algorithm.opsd.sdpo_conditioning_mode=prompt_append`
- `svsdpo_caa`: `script_module=recipe.opsd.main_opsd`, `++algorithm.opsd.sdpo_conditioning_mode=steering`, `++algorithm.opsd.steering.layer_fractions=\"0.31-0.37\"`

The SDPO and SV-SDPO CAA variants also use the same OPSD/SDPO settings as the Chemistry SDPO/SV-SDPO runs: EMA teacher, `topk_jsd`, `topk=100`, `distill_beta=0.5`, `sdpo_distill_only_failed=True`, `sdpo_exclude_self_success=True`, token-level off-policy IS clipping, constant actor LR schedule, weight decay `0.01`, grad clip `1.0`, and rollout weight-update bucket `4096 MB`.

Submitted run tag: `dfw_deepmath_sdpo_svopsd_20260527_170000`.

Submitted jobs:

- `grpo`: SLURM job `12215167`, experiment `grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`
- `sdpo`: SLURM job `12215171`, experiment `sdpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/sdpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`
- `svsdpo_caa`: SLURM job `12215183`, experiment `svsdpo_caa_qwen3_30b_a3b_deepmath_compmath_resp8k_l31_37_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/svsdpo_caa_qwen3_30b_a3b_deepmath_compmath_resp8k_l31_37_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`

Initial scheduler state after submission: all three jobs were `PENDING` on `batch`, 4 nodes each, with reason `Priority`.

## 2026-05-27: DFW VERL Container Copy to EOS

Copy the DFW VERL container to the EOS path requested as `eos:~/lustre/containers/`.

Requested source:

- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`

Requested destination:

- `eos:~/lustre/containers/`
- This resolves on EOS to `/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/`.

Primary Data Mover attempt:

```bash
python3 /home/siddjain/workspace/scripts/src/cluster_datamover_transfer.py \
  --transfer-id dfw-eos-verl-vllm012-flashattn-20260321-20260527 \
  'dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh' \
  'eos:~/lustre/containers/'
```

Data Mover upload to staging completed as job `e9d386a7-9767-443a-b42c-32f069b8f953`, launched through DFW Slurm job `12215316`, with `files=1/1`, `bytes_transferred=21465190400`, and `errors=0`. The EOS download leg was created as Data Mover job `92560c15-18d4-4d6c-b345-505fbcfbb44a`, but its EOS Slurm runner `5337021` was pending on `Priority` with estimated start `2026-05-28T07:41:00`, so the pending download leg was cancelled and the transfer was completed with direct DFW-initiated rsync.

DFW-to-EOS direct rsync:

```bash
ssh dfw 'set -euo pipefail; rsync -avP --info=progress2 \
  -e "ssh -o IdentitiesOnly=yes -i ~/.ssh/id_rsa" \
  /lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh \
  siddjain@login-eos01.eos.clusters.nvidia.com:/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/'
```

Result:

- Destination file: `eos:/home/siddjain/lustre/containers/verl_vllm012_flashattn_20260321.sqsh`
- Resolved destination file: `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`
- Final size: `21465190400` bytes
- Final mtime: `2026-03-30 15:51:26 -0700`

## 2026-05-27: DeepMath/CompMath Data Sync and EOS Interactive Smoke

Replace EOS DeepMath/CompMath data with the DFW copies and run an EOS interactive smoke using DeepMath as train data and CompMath as validation.

Data sync source paths:

- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/deepmath_verl.jsonl`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl`

Data sync destination paths:

- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/deepmath_verl.jsonl`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl`

The sync must delete the existing EOS `deepmath_verl.jsonl` before copying. Since the files are small enough and EOS Data Mover download runners can sit pending on `Priority`, use direct DFW-to-EOS `rsync` with the explicit DFW-side SSH identity:

```bash
ssh dfw 'set -euo pipefail
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_rsa siddjain@login-eos01.eos.clusters.nvidia.com \
  "mkdir -p /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen && rm -f /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/deepmath_verl.jsonl /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl"
rsync -avP -e "ssh -o IdentitiesOnly=yes -i ~/.ssh/id_rsa" \
  /lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/deepmath_verl.jsonl \
  /lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl \
  siddjain@login-eos01.eos.clusters.nvidia.com:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/'
```

Verify with `sha256sum` on DFW and EOS before launching.

Smoke submitter:

```bash
smoke_tests/submit_deepmath_compmath_eos_interactive.sh
```

Smoke settings:

- cluster `eos`
- partition `interactive`
- 1 exclusive node
- trainer configured for 8 GPUs per node
- W&B disabled via `trainer.logger=['console']`
- model `/hf_models/Qwen3-30B-A3B`
- train `/data/rl/mathgen/deepmath_verl.jsonl`
- validation `/data/rl/mathgen/comp_math_verl.jsonl`
- `n_prompts=8`, `n_samples=8`, `n_val_samples=4`
- `max_prompt_len=2k`
- `max_len=10k`, mapping to 8K response length
- generation temperature `0.85`
- validation temperature `0.6`, validation top-p `1.0`
- no-placeholder safety:
  - `data.filter_overlong_prompts=False`
  - `++data.dynamic_masked_solution=False`
  - `++data.min_masked_fraction=null`
  - `++data.max_masked_fraction=null`
  - `++data.mask_seed=null`

To avoid the previous EOS failure mode, the smoke uses an EOS config under `~/data` that keeps EOS-local mounts and points `containers.verl` at an `onelogger`-named symlink to the same VERL image. The submitter injects `pip install --upgrade one-logger-utils` unless `containers.verl` contains `onelogger`; using the stock container path directly would reintroduce the Ray CLI import failure from the earlier EOS jobs.

Executed data sync:

- DFW and EOS `deepmath_verl.jsonl` checksum: `33ad3ccd84c61eefa6dbac9c948030deeb2bb9578cced2376ca51a1cdc701bed`
- DFW and EOS `comp_math_verl.jsonl` checksum: `570589a20298c6ab031aea6acf1108fbcf01352351a5cff6fe05590213815f64`
- Line counts: DeepMath `102795`, CompMath `256`

Smoke submit details:

- Submitter: `/home/siddjain/workspace/verl/verl_main/smoke_tests/submit_deepmath_compmath_eos_interactive.sh`
- Generated EOS config: `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd_eos/config_onelogger/eos.yaml`
- EOS VERL image symlink: `/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/verl_vllm012_flashattn_20260321_onelogger.sqsh`
- Dry-run guard: passed without `one-logger-utils` install injection.
- First real submission attempt failed before job creation because the generated config used EOS batch's `04:00:00` timeout on the `interactive` partition. EOS reports interactive `MaxTime=02:00:00`, so the submitter now generates an explicit `interactive: 02:00:00` timeout.
- Submitted smoke run tag: `eos_deepmath_compmath_smoke_20260527_180554`
- Variant: `svsdpo_caa`
- SLURM job: `5337181`
- Initial scheduler state: `RUNNING` on `eos0070`
- Remote output: `/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/smoke_tests/deepmath_compmath_eos/deepmath_compmath_svsdpo_caa_eos_interactive_eos_deepmath_compmath_smoke_20260527_180554`
- Local submit log: `/home/siddjain/data/smoke_tests/deepmath_compmath_eos/deepmath_compmath_svsdpo_caa_eos_interactive_eos_deepmath_compmath_smoke_20260527_180554/submit.log`

Status update:

- Final scheduler state: `FAILED` after `00:09:06`.
- The run avoided the previous EOS `one-logger-utils`/Ray CLI failure: the job log starts with `No preamble command to execute`, Ray started, datasets loaded, and the trainer computed `Total training steps: 1`.
- Root cause: vLLM server startup rejected `actor_rollout_ref.rollout.gpu_memory_utilization=0.7` because the allocated EOS H100 node had only about `50.38-53.16 GiB` free on several `79.11 GiB` GPUs, below the requested `55.38 GiB`.
- Next EOS smoke retry should lower `actor_rollout_ref.rollout.gpu_memory_utilization`, for example to `0.55` or `0.5`, while keeping the same `onelogger` image path and EOS-local mounts.

## 2026-05-27: SciKnowEval Chemistry CAA SV-OPSD DFW LR 1e-5 with Layer Range 0.2-0.6

Use the same 4-node DFW batch CAA SV-OPSD Chemistry workflow as job `12213198`, but set the actor learning rate to `1e-5` while keeping the expanded steering layer fraction selector `0.2-0.6`.

Run from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
RUN_TAG=dfw_caa_lr1e5_l20_60_<timestamp> \
VARIANTS=svopsd_caa \
ACTOR_LR=1e-5 \
LAYER_FRACTIONS=0.2-0.6 \
./scripts/submit_chemistry_sdpo_svopsd_dfw.sh
```

The run uses:

- DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`
- non-interactive partition `batch`
- 4 exclusive nodes, 8 GPUs per node
- Qwen3-8B at `/hf_models/Qwen3-8B`
- SciKnowEval Chemistry train/test JSONL under `/data/rl/sciknoweval_l3`
- generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- CAA steering via `++algorithm.opsd.sdpo_conditioning_mode=steering`
- layer selector `++algorithm.opsd.steering.layer_fractions=\"0.2-0.6\"`
- LR warmup `actor_rollout_ref.actor.optim.lr_warmup_steps=10`, weight decay `0.01`, constant LR schedule, and W&B enabled

## 2026-05-28: OPSD Centered Advantage-Shaping Tests

Run focused tests from `/home/siddjain/workspace/verl/verl_svopsd` after changing the advantage-shaping implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/siddjain/workspace/verl/verl_svopsd \
  pytest tests/recipe/opsd/test_opsd_loss.py tests/recipe/opsd/test_opsd_config.py tests/recipe/opsd/test_dp_actor.py
```

The tests cover centered correctness shaping, response-total advantage conservation, distill submasks, sign-flip prevention, config validation, SDPO steering with `opsd_rlvr` under advantage shaping, and existing steering-vector extraction behavior.

## 2026-05-28: DFW Interactive SDPO/SV-SDPO CAA Advantage-Shaping Smoke

Use DFW interactive smoke runs with W&B disabled, temperature `1.0`, exclusive 1-node / 8-GPU allocation, and the existing SciKnowEval Chemistry Qwen3-8B pipeline. The new shaping mode uses `recipe.opsd.main_opsd` with:

- `++algorithm.opsd.mode=opsd_rlvr`
- `++algorithm.opsd.mix_weight=1.0`
- `++algorithm.opsd.advantage_shaping.enable=True`
- `++algorithm.opsd.advantage_shaping.scale=1.0`
- `++algorithm.opsd.advantage_shaping.normalize=std`
- `++algorithm.opsd.advantage_shaping.clip_z=3.0`
- `++algorithm.opsd.advantage_shaping.use_distill_mask=True`
- `++algorithm.opsd.advantage_shaping.allow_token_sign_flip=True`
- `++algorithm.opsd.teacher_source=sdpo_success_rollout`
- `++algorithm.opsd.sdpo_distill_only_failed=True`
- `++algorithm.opsd.sdpo_exclude_self_success=True`

Run both variants from `/home/siddjain/workspace/verl/verl_svopsd` after the submitter is updated:

```bash
VARIANT=sdpo_advshape ./smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
VARIANT=svsdpo_caa_advshape LAYER_FRACTIONS=0.31-0.37 ./smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

The vanilla SDPO smoke uses `++algorithm.opsd.sdpo_conditioning_mode=prompt_append`. The SV-SDPO CAA smoke uses `++algorithm.opsd.sdpo_conditioning_mode=steering` and `++algorithm.opsd.steering.layer_fractions=\"0.31-0.37\"`. Passing criteria are job completion plus nonzero `actor/advantage_shaping_active_rate`; the CAA variant should also report nonzero `actor/opsd_steering_active_rate`.

Verification notes:

- Local syntax check passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  recipe/opsd/dp_actor.py \
  recipe/opsd/opsd_loss.py \
  verl/trainer/config/algorithm.py
```

- A lightweight local shaping-helper assertion passed under the `deepseek` conda env with bytecode disabled.
- Full local pytest is currently blocked by local dependency issues: the base environment lacks the expected pytest/runtime stack, and the temporary dependency path fails collection on missing `pyvers`. The cluster trainer smoke is therefore the authoritative end-to-end check for this change.

First vanilla attempt:

- Variant: `sdpo_advshape`
- Experiment: `sdpo_advshape_chemistry_dfw_interactive_20260528_110351`
- SLURM job: `12250196`
- Result: failed at optimizer step with an Adam foreach dtype/device mismatch. The fix was to make the shaping-only distill scorer forward no-grad and detach support state when `mix_weight=1.0`.

Passing vanilla SDPO advantage-shaping smoke:

- Command:

```bash
VARIANT=sdpo_advshape LAYER_FRACTIONS=0.31-0.37 POLL_SECONDS=30 \
  /home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

- Experiment: `sdpo_advshape_chemistry_dfw_interactive_20260528_111451`
- SLURM job: `12250448`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_111451`
- Result: completed and passed.
- Step 1 metrics included `actor/advantage_shaping_active_rate:0.390625`, `actor/advantage_shaping_token_rate:0.390625`, `actor/opsd_distill_weight:0.0`, `actor/student_rlvr_loss:0.00013899803161621094`, and `actor/grad_norm:0.38262978196144104`.

Passing SV-SDPO CAA advantage-shaping smoke:

- Command:

```bash
VARIANT=svsdpo_caa_advshape LAYER_FRACTIONS=0.31-0.37 POLL_SECONDS=30 \
  /home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

- Experiment: `svsdpo_caa_advshape_chemistry_dfw_interactive_20260528_112235`
- SLURM job: `12250818`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_chemistry_dfw_interactive_20260528_112235`
- Result: completed and passed.
- Step 1 metrics included `actor/opsd_steering_layer_count:2.0`, `actor/opsd_steering_candidate_mean:2.125`, `actor/opsd_steering_positive_mean:0.546875`, `actor/opsd_steering_negative_mean:1.578125`, `actor/opsd_steering_active_rate:0.265625`, `actor/advantage_shaping_active_rate:0.265625`, `actor/advantage_shaping_token_rate:0.265625`, `actor/opsd_distill_weight:0.0`, `actor/student_rlvr_loss:0.00010930327698588371`, and `actor/grad_norm:0.40108227729797363`.

## 2026-05-28: DeepMath/CompMath SDPO-RLVR and SV-SDPO-RLVR CAA DFW Runs

Use the reusable DFW DeepMath launcher:

```bash
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

The launcher uses:

- DFW batch partition
- 4 exclusive nodes
- 8 GPUs per node
- Qwen3-30B-A3B at `/hf_models/Qwen3-30B-A3B`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- LR `5e-6`
- response length 8K via `MAX_LEN=10k`
- rollout TP `4`
- W&B enabled
- no sandbox

Start the two new RLVR runs with temperature `1.0` for both train and validation generation, per the current project instruction:

```bash
RUN_TAG=dfw_deepmath_rlvr_$(date +%Y%m%d_%H%M%S) \
VARIANTS='sdpo_rlvr svsdpo_rlvr_caa' \
ACTOR_LR=5e-6 \
LAYER_FRACTIONS=0.2-0.6 \
TRAIN_T=1.0 \
VAL_T=1.0 \
VAL_TOP_P=1.0 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

The `sdpo_rlvr` variant uses:

- `++algorithm.opsd.mode=opsd_rlvr`
- `++algorithm.opsd.mix_weight=1.0`
- `++algorithm.opsd.sdpo_conditioning_mode=prompt_append`
- `++algorithm.opsd.advantage_shaping.enable=True`
- `++algorithm.opsd.advantage_shaping.scale=1.0`
- `++algorithm.opsd.advantage_shaping.normalize=std`
- `++algorithm.opsd.advantage_shaping.clip_z=3.0`

The `svsdpo_rlvr_caa` variant uses the same RLVR advantage-shaping path plus:

- `++algorithm.opsd.sdpo_conditioning_mode=steering`
- `++algorithm.opsd.steering.layer_fractions=\"0.2-0.6\"`
- `++algorithm.opsd.steering.scale=1.0`
- `++algorithm.opsd.steering.normalize=null`
- `++algorithm.opsd.steering.apply_positions=all_nonpad`

Resume the previous GRPO run from its existing output directory and checkpoint state. Preserve the original run tag, previous train/validation temperatures, and previous W&B id:

```bash
RUN_TAG=dfw_deepmath_sdpo_svopsd_20260527_170000 \
VARIANTS=grpo \
ACTOR_LR=5e-6 \
TRAIN_T=0.85 \
VAL_T=0.6 \
VAL_TOP_P=1.0 \
WANDB_ID=grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000_27-16-54-23 \
LOCAL_SUBMIT_LOG_ROOT=/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000_resume_$(date +%Y%m%d_%H%M%S) \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

The previous GRPO run completed through `training/global_step:25` and has a checkpoint at:

```text
dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000/checkpoints/global_step_25
```

Submitted RLVR run tag: `dfw_deepmath_rlvr_20260528_131831`.

Submitted RLVR jobs:

- `sdpo_rlvr`: SLURM job `12255932`, experiment `sdpo_rlvr_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_rlvr_20260528_131831`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/sdpo_rlvr_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_rlvr_20260528_131831`
- `svsdpo_rlvr_caa`: SLURM job `12256035`, experiment `svsdpo_rlvr_caa_qwen3_30b_a3b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_rlvr_20260528_131831`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/svsdpo_rlvr_caa_qwen3_30b_a3b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_rlvr_20260528_131831`

Submitted GRPO resume:

- Local submit root: `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000_resume_20260528_131936`
- SLURM job: `12256113`
- Experiment: `grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`

Initial live DFW scheduler check after submission:

- `12255932`: `PENDING`, reason `Priority`
- `12256035`: `PENDING`, reason `None`
- `12256113`: `PENDING`, reason `None`

Cancellation update:

- `12255932`: `CANCELLED by 140634`, elapsed `00:00:00`
- `12256035`: `CANCELLED by 140634`, elapsed `00:00:00`
- `12256113`: `CANCELLED by 140634`, elapsed `00:00:00`

Reason: these submissions inherited the DeepMath launcher's existing `ACTOR_MODEL=/hf_models/Qwen3-30B-A3B` default. Do not reuse this submission as-is when the intended DeepMath model is Qwen3-8B; pass `ACTOR_MODEL=/hf_models/Qwen3-8B` explicitly before launching replacements.

## 2026-05-28: Corrected Qwen3-8B DeepMath/CompMath SDPO-RLVR and SV-SDPO-RLVR CAA DFW Runs

The corrected restart uses Qwen3-8B and the effective Qwen3-8B DeepMath settings from the existing DFW runs under `/output/rl/mathgen`:

- model `/hf_models/Qwen3-8B`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- DFW batch partition
- 4 exclusive nodes
- 8 GPUs per node
- train batch/prompts `32`
- rollouts per prompt `8`
- validation samples `16`
- response length `8192`
- prompt length `2048`
- max batched tokens / actor token budget `10240`
- `num_ppo_iter=1`, yielding actor PPO minibatch size `32`
- train temperature `1.0`
- validation temperature `0.6`
- validation top-p `0.95`
- rollout TP final override `4`
- save/test freq `20`
- total epochs `5`
- W&B enabled

Requested changes relative to the existing Qwen3-8B DeepMath runs:

- LR `5e-6`
- `sdpo_rlvr`: `opsd.mode=opsd_rlvr`, prompt-append conditioning, advantage shaping enabled
- `svsdpo_rlvr_caa`: CAA steering conditioning with `layer_fractions=0.2-0.6`, advantage shaping enabled

Launch the two unambiguous RLVR jobs with:

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_rlvr_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS='sdpo_rlvr svsdpo_rlvr_caa' \
ACTOR_LR=5e-6 \
LAYER_FRACTIONS=0.2-0.6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=1 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Do not submit a Qwen3-8B GRPO resume unless a compatible Qwen3-8B GRPO checkpoint path is identified. The earlier cancelled GRPO resume targeted the Qwen3-30B-A3B run and cannot be made into an 8B resume by changing `ACTOR_MODEL`.

Submitted corrected Qwen3-8B RLVR run tag: `dfw_deepmath_qwen3_8b_rlvr_20260528_140606`.

Submitted jobs:

- `sdpo_rlvr`: SLURM job `12257178`, experiment `sdpo_rlvr_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlvr_20260528_140606`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/sdpo_rlvr_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlvr_20260528_140606`
- `svsdpo_rlvr_caa`: SLURM job `12257181`, experiment `svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_rlvr_20260528_140606`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_rlvr_20260528_140606`

Initial live DFW scheduler check:

- `12257178`: `PENDING`, reason `None`
- `12257181`: `PENDING`, reason `None`

## 2026-05-29: Matched Qwen3-8B DeepMath/CompMath Pure GRPO DFW Run

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default partition `batch`, and VERL container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`. This is a non-test run, so W&B is enabled.

This run is a pure GRPO baseline matched to the current Qwen3-8B `SDPO_RLVR` run where relevant. It uses no `algorithm.opsd.*`, no SDPO teacher, no prompt-append conditioning, no steering, and no OPSD advantage shaping.

Key launch parameters:

- `--script_module verl.trainer.main_ppo`
- `--cluster cw-dfw --partition batch --nodes 4 --gpus 8`
- `--actor_model /hf_models/Qwen3-8B`
- `--prompt_data /data/rl/mathgen/deepmath_verl.jsonl`
- `--eval_data /data/rl/mathgen/comp_math_verl.jsonl`
- `--n_prompts 32 --n_samples 8 --n_val_samples 16 --val_batch_size 32`
- `--max_prompt_len 2k --max_len 10k --max_tokens_per_gpu 10240`
- `--actor_lr 5e-6 --T 1.0 --val_T 0.6 --val_top_p 0.95`
- `--save_freq 20 --test_freq 20 --num_epochs 5 --num_ppo_iter 1`
- `--ae grpo --kl_coef 0.0 --clip_ae 0.2,0.28 --reward_manager batch`
- `--enable_wandb --no_sandbox`

Extra overrides:

- `data.filter_overlong_prompts=False`
- `++data.dynamic_masked_solution=False`
- `++data.min_masked_fraction=null`
- `++data.max_masked_fraction=null`
- `++data.mask_seed=null`
- `++actor_rollout_ref.rollout.tensor_model_parallel_size=4`
- `algorithm.norm_adv_by_std_in_grpo=False`
- `actor_rollout_ref.actor.optim.lr_warmup_steps=10`
- `actor_rollout_ref.actor.optim.weight_decay=0.01`
- `actor_rollout_ref.actor.optim.clip_grad=1.0`
- `actor_rollout_ref.actor.optim.lr_scheduler_type=constant`
- `++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`

Submitted run:

- Run tag: `dfw_deepmath_qwen3_8b_grpo_20260529_011018`
- Local submit log: `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_20260529_011018/grpo/submit.log`
- SLURM job: `12275498`
- Experiment: `grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_20260529_011018`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_20260529_011018`
- Initial scheduler state: `PENDING` on DFW `batch`, 4 nodes, reason `None`

## 2026-05-29: Qwen3-8B DeepMath/CompMath Non-RLVR SV-SDPO CAA DFW Run

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default partition `batch`, and VERL container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`. This is a non-test run, so W&B is enabled.

This run is the non-RLVR CAA counterpart for the current Qwen3-8B DeepMath/CompMath SDPO-RLVR experiments. It uses `recipe.opsd.main_opsd` with `algorithm.opsd.mode=opsd`, steering conditioning, and no RLVR advantage shaping.

Launch with:

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_svsdpo_caa_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS=svsdpo_caa \
ACTOR_LR=5e-6 \
LAYER_FRACTIONS=0.2-0.6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=1 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Key settings inherited from the launcher:

- `--cluster cw-dfw --partition batch --nodes 4 --gpus 8`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- train prompts `32`, rollouts per prompt `8`, validation samples `16`
- prompt length `2048`, response length `8192`, max token budget `10240`
- rollout TP `4`, save/test freq `20`, total epochs `5`
- OPSD teacher source `sdpo_success_rollout`, EMA teacher, top-k JSD with `topk=100`, `distill_beta=0.5`
- CAA steering selector `0.2-0.6`

Submitted run:

- Run tag: `dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834`
- Local submit log: `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834/svsdpo_caa/submit.log`
- SLURM job: `12277118`
- Experiment: `svsdpo_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834`
- Remote output: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834`
- Initial scheduler state: `PENDING` on DFW `batch`, 4 nodes, reason `None`

## 2026-05-29: Qwen3-8B DeepMath/CompMath GRPO Restart and Positive-Only SV-SDPO DFW Runs

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default partition `batch`, and VERL container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`. These are non-test runs, so W&B is enabled.

The submitter `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh` now supports:

- `grpo`: pure GRPO through `verl.trainer.main_ppo`
- `svsdpo`: non-RLVR positive-only steering through `recipe.opsd.main_opsd`, `algorithm.opsd.mode=opsd`, `sdpo_conditioning_mode=steering`, and `algorithm.opsd.steering.source_mode=positive`

The `svsdpo` variant averages all same-prompt correct rollout response-token residual activations per selected layer and does not subtract incorrect-rollout activations. This is the non-CAA steering baseline. The CAA baseline remains `svsdpo_caa`, which uses positive minus negative residual means.

Launch with:

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS='grpo svsdpo' \
ACTOR_LR=5e-6 \
LAYER_FRACTIONS=0.2-0.6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=1 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Key settings:

- `--cluster cw-dfw --partition batch --nodes 4 --gpus 8`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- train prompts `32`, rollouts per prompt `8`, validation samples `16`
- prompt length `2048`, response length `8192`, max token budget `10240`
- rollout TP `4`, save/test freq `20`, total epochs `5`, PPO iter `1`
- actor LR `5e-6`, generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`

## 2026-05-29: DFW Interactive SDPO_RLVR Advantage-Shaping Prefix-Cap Smoke

Purpose: verify `algorithm.opsd.advantage_shaping.max_response_tokens` for both prompt-appended SDPO_RLVR scoring and SV-SDPO_RLVR CAA steering scoring. The smoke keeps W&B disabled through `trainer.logger=['console']` and uses DFW `interactive` with 1 node / 8 GPUs.

Vanilla SDPO_RLVR advantage-shaping smoke:

```bash
VARIANT=sdpo_advshape \
EXP_NAME=sdpo_advshape_prefixcap_chemistry_dfw_interactive_$(date +%Y%m%d_%H%M%S) \
ADV_SHAPING_MAX_RESPONSE_TOKENS=16 \
WAIT_FOR_COMPLETION=1 \
POLL_SECONDS=60 \
/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

SV-SDPO_RLVR CAA advantage-shaping smoke:

```bash
VARIANT=svsdpo_caa_advshape \
EXP_NAME=svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_$(date +%Y%m%d_%H%M%S) \
ADV_SHAPING_MAX_RESPONSE_TOKENS=16 \
LAYER_FRACTIONS=0.2-0.6 \
WAIT_FOR_COMPLETION=1 \
POLL_SECONDS=60 \
/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

Default production behavior after the code change is `ADV_SHAPING_MAX_RESPONSE_TOKENS=1024`, unless overridden or set to `null` through Hydra to shape all active response tokens.

Submitted smoke results:

- SDPO_RLVR prefix-cap smoke: experiment `sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214`, SLURM job `12307521`, state `COMPLETED`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214`, local submit log `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214/submit.log`.
- SDPO_RLVR key metrics: `actor/advantage_shaping_active_rate=0.21875`, `actor/advantage_shaping_token_rate=0.001708984375`, `actor/advantage_shaping_total_error_max=4.76837158203125e-07`, `actor/advantage_shaping_response_total_error_max=6.103515625e-05`.
- SV-SDPO_RLVR CAA prefix-cap smoke: experiment `svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953`, SLURM job `12307762`, state `COMPLETED`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953`, local submit log `/home/siddjain/data/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953/submit.log`.
- SV-SDPO_RLVR CAA key metrics: `actor/opsd_steering_layer_count=14.0`, `actor/opsd_steering_active_rate=0.125`, `actor/advantage_shaping_active_rate=0.125`, `actor/advantage_shaping_token_rate=0.0009765625`, `actor/advantage_shaping_total_error_max=9.5367431640625e-07`, `actor/advantage_shaping_response_total_error_max=0.0`.

## 2026-05-29: Qwen3-8B DeepMath/CompMath SDPO_RLVR Prefix-1024 Production Runs

Submit matched SDPO_RLVR and SV-SDPO_RLVR CAA production jobs on DFW with advantage shaping limited to the first 1024 response tokens. The DFW cluster config is `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to account `nemotron_reason_code`, partition `batch`, and container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`.

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS="sdpo_rlvr svsdpo_rlvr_caa" \
ACTOR_LR=5e-6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=2 \
LAYER_FRACTIONS=0.2-0.6 \
ADV_SHAPING_MAX_RESPONSE_TOKENS=1024 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Key settings:

- `--cluster cw-dfw --partition batch --nodes 4 --gpus 8`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- train prompts `32`, rollouts per prompt `8`, validation samples `16`
- prompt length `2048`, response length `8192`, max token budget `10240`
- rollout TP `4`, save/test freq `20`, total epochs `5`, PPO iter `2`
- actor LR `5e-6`, generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- SDPO_RLVR advantage shaping uses `scale=1.0`, `normalize=std`, `clip_z=3.0`, `use_distill_mask=True`, `allow_token_sign_flip=True`, and `max_response_tokens=1024`
- SV-SDPO_RLVR CAA uses steering layers `0.2-0.6`

Submitted run tag: `dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`.

Submitted jobs:

- `sdpo_rlvr`: SLURM job `12313004`, experiment `sdpo_rlvr_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/sdpo_rlvr_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053/sdpo_rlvr/submit.log`
- `svsdpo_rlvr_caa`: SLURM job `12313042`, experiment `svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053/svsdpo_rlvr_caa/submit.log`

Initial scheduler state after submission:

- `12313004`: `PENDING`, reason `Priority`
- `12313042`: `PENDING`, reason `Priority`

Note: this submission used the `NUM_PPO_ITER=2` value from the params listed immediately before launch, and the submitter emitted `actor_rollout_ref.actor.ppo_mini_batch_size=16`. The earlier corrected Qwen3-8B SDPO_RLVR run log emitted `ppo_mini_batch_size=32`, corresponding to `NUM_PPO_ITER=1`.

## 2026-05-29: RLSD Correct-Rollout Privileged Source DFW Interactive Smoke

Run from `/home/siddjain/workspace/verl/verl_main`:

```bash
VARIANT=rlsd_rollout \
EXP_NAME=rlsd_rollout_chemistry_dfw_interactive_$(date +%Y%m%d_%H%M%S) \
WAIT_FOR_COMPLETION=1 \
POLL_SECONDS=60 \
/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

Smoke settings:

- cluster `cw-dfw`, partition `interactive`, account `nemotron_reason_code`
- 1 node, 8 GPUs per node
- model `/hf_models/Qwen3-8B`
- train data `/data/rl/sciknoweval_l3/chemistry_train.jsonl`
- validation data `/data/rl/sciknoweval_l3/chemistry_test.jsonl`
- train generation temperature `1.0`
- validation temperature `1.0`, top-p `0.95`
- trainer logger `['console']`, no W&B
- RLSD source `correct_rollout`, fixed teacher, `opsd_rlvr`, prompt-append teacher branch only
- RLSD defaults: `lambda_init=0.5`, `lambda_final=0.0`, `lambda_decay_steps=50`, `eps_w=0.2`, `teacher_sync_interval=10`

Submitted smoke result:

- experiment `rlsd_rollout_chemistry_dfw_interactive_20260529_132521`
- SLURM job `12302510`
- state `COMPLETED`
- remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/rlsd_rollout_chemistry_dfw_interactive_20260529_132521`
- local submit log `/home/siddjain/data/smoke_tests/svopsd_chemistry/rlsd_rollout_chemistry_dfw_interactive_20260529_132521/submit.log`
- verification metric present: `actor/rlsd_active_rate:0.3125`
- key smoke metrics: `actor/rlsd_lambda:0.5`, `actor/rlsd_token_rate:0.3125`, `actor/rlsd_no_privileged_fallback_rate:0.6875`, `actor/student_rlvr_loss:0.007555665913969278`, `response_length/clip_ratio:0.875`, `reward/acc/mean:0.0625`

## 2026-05-29: Qwen3-8B DeepMath/CompMath RLSD DFW Runs

Use the DFW codegen config `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/cw-dfw.yaml`, which resolves to SSH host `dfw`, account `nemotron_reason_code`, default partition `batch`, and VERL container `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`. These are non-test runs, so W&B is enabled.

Shared SDPO-matched settings:

- model `/hf_models/Qwen3-8B`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- DFW batch partition, 4 exclusive nodes, 8 GPUs per node
- actor LR `5e-6`
- train generation temperature `1.0`
- validation temperature `0.6`, validation top-p `0.95`, validation samples `16`
- prompt length `2048`, response length `8192`, max token budget `10240`
- rollout TP `4`
- `num_ppo_iter=1`
- save/test freq `20`, total epochs `5`
- `opsd.mode=opsd_rlvr`, `opsd.mix_weight=1.0`, fixed RLSD teacher

Submit `rlsd_rollout` with the SDPO-style rollout group shape:

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_rlsd_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS=rlsd_rollout \
ACTOR_LR=5e-6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=1 \
N_PROMPTS=32 \
N_SAMPLES=8 \
MAX_PROMPT_LEN=2k \
MAX_LEN=10k \
MAX_TOKENS_PER_GPU=10240 \
NODES=4 \
GPUS=8 \
ROLLOUT_TP=4 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Submit `rlsd_gt` with the requested single-generation shape:

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_rlsd_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS=rlsd_gt \
ACTOR_LR=5e-6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=1 \
N_PROMPTS=256 \
N_SAMPLES=1 \
MAX_PROMPT_LEN=2k \
MAX_LEN=10k \
MAX_TOKENS_PER_GPU=10240 \
NODES=4 \
GPUS=8 \
ROLLOUT_TP=4 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Notes:

- `rlsd_rollout` uses `algorithm.opsd.rlsd.privileged_source=correct_rollout` and `algorithm.opsd.teacher_source=sdpo_success_rollout`.
- `rlsd_gt` uses `algorithm.opsd.rlsd.privileged_source=ground_truth_answer` and `algorithm.opsd.teacher_source=ground_truth`.
- `rlsd_gt` intentionally uses `N_SAMPLES=1`; this preserves the requested shape even though GRPO normally relies on multiple samples per prompt for within-prompt comparison.

Submitted run tag: `dfw_deepmath_qwen3_8b_rlsd_20260529_134546`.

Submitted jobs:

- `rlsd_rollout`: SLURM job `12303531`, experiment `rlsd_rollout_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlsd_20260529_134546`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/rlsd_rollout_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlsd_20260529_134546`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlsd_20260529_134546/rlsd_rollout/submit.log`
- `rlsd_gt`: SLURM job `12303568`, experiment `rlsd_gt_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlsd_20260529_134546`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/rlsd_gt_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlsd_20260529_134546`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlsd_20260529_134546/rlsd_gt/submit.log`

Initial scheduler state:

- `12303531`: `RUNNING` on DFW `batch`, nodes `pool0-[00321,00332-00333,00338]`
- `12303568`: `PENDING` on DFW `batch`, reason `None`

Submitted run:

- `grpo`: SLURM job `12300861`, experiment `grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603/grpo/submit.log`

Initial scheduler state:

- `12300861`: `PENDING` on DFW `batch`, reason `Resources`, submitted `2026-05-29T12:46:23`

## 2026-05-29: RLSD Implementation Checks and DFW Interactive Smoke

Focused local checks from `/home/siddjain/workspace/verl/verl_svopsd`:

```bash
python -m py_compile \
  verl/trainer/config/algorithm.py \
  recipe/opsd/opsd_loss.py \
  recipe/opsd/opsd_trainer.py \
  recipe/opsd/dp_actor.py

python -m pytest \
  tests/recipe/opsd/test_opsd_config.py \
  tests/recipe/opsd/test_opsd_loss.py \
  tests/recipe/opsd/test_dp_actor.py
```

Local check result:

- `python -m py_compile ...` passed.
- `python -m pytest ...` could not collect in the local base environment because `torch` and `tensordict` are not installed there; verification continued through the DFW interactive smoke.

DFW interactive RLSD smoke, W&B disabled through console-only trainer logging:

```bash
VARIANT=rlsd_gt \
EXP_NAME=rlsd_gt_chemistry_dfw_interactive_$(date +%Y%m%d_%H%M%S) \
WAIT_FOR_COMPLETION=1 \
POLL_SECONDS=60 \
/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

Correct-rollout privileged-source smoke:

```bash
VARIANT=rlsd_rollout \
EXP_NAME=rlsd_rollout_chemistry_dfw_interactive_$(date +%Y%m%d_%H%M%S) \
WAIT_FOR_COMPLETION=1 \
POLL_SECONDS=60 \
/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh
```

Smoke settings:

- cluster `cw-dfw`, partition `interactive`, account `nemotron_reason_code`
- 1 node, 8 GPUs per node
- model `/hf_models/Qwen3-8B`
- train data `/data/rl/sciknoweval_l3/chemistry_train.jsonl`
- validation data `/data/rl/sciknoweval_l3/chemistry_test.jsonl`
- train generation temperature `1.0`
- validation temperature `1.0`, top-p `0.95`
- trainer logger `['console']`, no W&B

Submitted smoke result:

- experiment `rlsd_gt_chemistry_dfw_interactive_20260529_131445`
- SLURM job `12302026`
- state `COMPLETED`
- remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/rlsd_gt_chemistry_dfw_interactive_20260529_131445`
- local submit log `/home/siddjain/data/smoke_tests/svopsd_chemistry/rlsd_gt_chemistry_dfw_interactive_20260529_131445/submit.log`
- verification metric present: `actor/rlsd_active_rate:1.0`
- key smoke metrics: `actor/rlsd_lambda:0.5`, `actor/rlsd_token_rate:1.0`, `actor/rlsd_no_privileged_fallback_rate:0.0`, `actor/student_rlvr_loss:0.0040476275607943535`
- positive-only SV-SDPO selector `0.2-0.6`

Submitted run tag: `dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704`.

Submitted runs:

- `grpo`: SLURM job `12299551`, experiment `grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704/grpo/submit.log`
- `svsdpo`: SLURM job `12299555`, experiment `svsdpo_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704/svsdpo/submit.log`

Initial scheduler state:

- `12299551`: `RUNNING` on DFW `batch`, nodes `pool0-[00326-00329]`
- `12299555`: `PENDING` on DFW `batch`, reason `None`

Follow-up scheduler check shortly after submission:

- `12299551`: `RUNNING`, elapsed `00:01:24`, nodes `pool0-[00326-00329]`
- `12299555`: `RUNNING`, elapsed `00:01:03`, nodes `pool0-[01261,01269,01350,01458]`

Follow-up after user requested stopping positive-only SV-SDPO:

- `12299555`: cancelled via `scancel`; SLURM accounting state `CANCELLED by 140634`, elapsed `00:11:52`
- `12299551`: failed independently before the cancel request completed; root error was `AssertionError: Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32) is too large to fit in the bucket. Please increase rollout.update_weights_bucket_megabytes(2048 MB).`

## 2026-05-29: Qwen3-8B DeepMath/CompMath GRPO Bucket-Fix Resubmission

Fix `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh` so the rollout checkpoint-engine weight-update bucket override is part of `common_extra_args`, not only OPSD args:

```bash
++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
```

Resubmit only the pure GRPO replacement on DFW:

```bash
RUN_TAG=dfw_deepmath_qwen3_8b_grpo_bucket4096_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS=grpo \
ACTOR_LR=5e-6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=1 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Key settings:

- `--cluster cw-dfw --partition batch --nodes 4 --gpus 8`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- train prompts `32`, rollouts per prompt `8`, validation samples `16`
- prompt length `2048`, response length `8192`, max token budget `10240`
- rollout TP `4`, save/test freq `20`, total epochs `5`, PPO iter `1`
- actor LR `5e-6`, generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`

## 2026-05-29: Qwen3-8B DeepMath/CompMath SV-SDPO_RLVR CAA Prefix-1024 Resubmission

The first prefix-1024 CAA submission failed before training because the generated experiment/W&B name exceeded W&B's 128-character name limit. Resubmit only the failed CAA variant with the same substantive settings and a shorter run tag.

Run from the local machine:

```bash
RUN_TAG=p1024caa_$(date +%m%d_%H%M) \
OUTPUT_BASE_DIR=/output/rl/mathgen \
ACTOR_MODEL=/hf_models/Qwen3-8B \
VARIANTS=svsdpo_rlvr_caa \
ACTOR_LR=5e-6 \
TRAIN_T=1.0 \
VAL_T=0.6 \
VAL_TOP_P=0.95 \
N_VAL_SAMPLES=16 \
NUM_PPO_ITER=2 \
LAYER_FRACTIONS=0.2-0.6 \
ADV_SHAPING_MAX_RESPONSE_TOKENS=1024 \
/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh
```

Key settings:

- `--cluster cw-dfw --partition batch --nodes 4 --gpus 8`
- train data `/data/rl/mathgen/deepmath_verl.jsonl`
- validation data `/data/rl/mathgen/comp_math_verl.jsonl`
- model `/hf_models/Qwen3-8B`
- train prompts `32`, rollouts per prompt `8`, validation samples `16`
- prompt length `2048`, response length `8192`, max token budget `10240`
- rollout TP `4`, save/test freq `20`, total epochs `5`, PPO iter `2`
- actor LR `5e-6`, generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`
- SV-SDPO_RLVR CAA layers `0.2-0.6`
- advantage shaping limited to first `1024` response tokens via `++algorithm.opsd.advantage_shaping.max_response_tokens=1024`

Submitted run tag: `p1024caa_0529_1736`.

Submitted job:

- `svsdpo_rlvr_caa`: SLURM job `12314248`, experiment `svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_p1024caa_0529_1736`, remote output `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_p1024caa_0529_1736`, local submit log `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/p1024caa_0529_1736/svsdpo_rlvr_caa/submit.log`

Initial scheduler state:

- `12314248`: `PENDING` on DFW `batch`, reason `None`
