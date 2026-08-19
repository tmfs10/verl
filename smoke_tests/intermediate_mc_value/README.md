<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: 96vw !important;
}
table { width: 100% !important; }
</style>

# Intermediate Monte Carlo value-model test plan

This plan verifies the synchronous implementation only. It deliberately does
not exercise VeRL's fully asynchronous policy trainer, payload queues, or
asynchronous reward paths. Internal composite-agent request concurrency is
allowed only behind a blocking `generate_sequences` call.

## Coverage layers

| Layer | Command or artifact | Required assertions |
|---|---|---|
| Pure numerical CPU | `pytest tests/trainer/ppo/test_intermediate_mc_value_on_cpu.py` | Independent head/selector config; random/EMA/variance selectors; float64 critic-value EMA with the configured baseline and baseline-anchored gap; mark aggregation and trained `V(s0)`; native GAE alignment; critique normalization; scalar MSE/BCE; Beta mean/variance/concentration with an FP32 prediction clamp independent of the target transform; reward-normalized value clipping; finite gradients; exact conditioned and unconditioned critic boundaries. |
| Native orchestration CPU | `pytest tests/trainer/ppo/test_intermediate_mc_orchestration_on_cpu.py` | Native warmup step semantics; no feature checkpoint state; synchronous reward-loop selection and rollout-ID preservation; rejection of grouped rewards, registered `conditional_logprob`, HDFS critic paths, and impossible optimizer cardinalities; `V(s0)` plus dense/EOS labels duplicated across critique streams; averaged-value EMA after critic scoring; `num_critiques=0` one-stream/solution-only behavior; full-sequence actor packing; conventional/packed log-prob equivalence; recorded behavior denominators; continuations excluded; no optimizer dummies; internal child-field stripping; fail-closed runtime gating; blocking rollout lifecycle cleanup. |
| Composite generation failure injection | Same orchestration suite | Every critique/continuation task starts on one sticky key and is drained; any critique failure aborts only after drain; individual continuation generation failures are omitted; temperature/logprob/max-token request fields are forced. Reward and invalid-value failures are iteration-fatal and never retried. |
| Native worker compatibility | `pytest tests/workers/test_intermediate_mc_critic_worker.py` | Feature-off compute/update calls retain VeRL's original critic signatures and never request the feature DP group; enabled config applies actor-style overrides and then forces scalar/Beta head width; disabled config preserves native behavior. |
| Two-rank dynamic batching | `pytest tests/workers/critic/test_intermediate_mc_dynamic_batching.py` | A two-rank Gloo group receives unequal sequence workloads yet executes the same dynamic critic microbatch count through the feature critic API and restores row order; active value positions outside or on padding fail instead of clamping. |
| Hydra/config generation | `bash scripts/generate_trainer_config.sh` plus composition of `intermediate_mc_ppo_trainer` | Base feature-disabled config remains unchanged; preset enables the feature, native `dppo_tv`, 30 critic RPC warmup, and reward-normalized clip epsilon 0.2; caller native-loss override survives; generated reference YAML is deterministic. |
| GPU critic components | Focused CI jobs for `DataParallelIntermediateMCCritic` under legacy FSDP and FSDP2, with/without remove-padding | Scalar and two-logit heads agree with a single-process reference; target masks give zero gradient outside `V(s0)`/solution positions; loss aggregation matches actor configuration; checkpoints round-trip head width and optimizer state. |
| Two-step end-to-end GPU | `run_smoke.sh` | Step 1 performs one blocking critic RPC, requests configured critiques but no continuations, and freezes actor optimization. Step 2 performs immediate random children or critic-scored deferred EMA/variance continuations, bounded continuation rewards, critic update, actor update, native checkpointing, and native rollout resynchronization. Both self-critique and no-self-critique modes are verified. |
| Native resume GPU | Optional `RUN_RESUME=1` | Resume native step 2 and execute step 3 without any feature-owned counter or contract file and without repeating warmup. Native actor, critic, dataloader, and global-step checkpoint state remain authoritative. |

## GPU smoke matrix

By default the runner executes twelve cells: `scalar_random`, `scalar_ema`, and
`beta_variance`, each under legacy `fsdp` and `fsdp2`, once with two critiques
and once with `num_critiques=0`. It uses temperature `1.0` for solutions,
critiques, continuations, and validation; uses console logging only; and sets
`WANDB_MODE=disabled`.

```bash
cd /home/siddjain/workspace/verl/verl_intermediate_mc_value_model
export MODEL_PATH=/absolute/path/to/a/small/text-only/instruction-model
export TRAIN_FILE=/absolute/path/to/a/two-or-more-row/verl-train.parquet
export VAL_FILE=/absolute/path/to/a/verl-validation.parquet
export GPU_COUNT=2
bash smoke_tests/intermediate_mc_value/run_smoke.sh
```

Compose and print the exact Hydra jobs without allocating workers or requiring a GPU by adding `DRY_RUN=1`.

Outputs, checkpoints, logs, audit JSONL, and resumability markers default to `/home/siddjain/data/intermediate_mc_value_model/verl/smoke`; no runtime artifacts are written into the repository. Override `SMOKE_ROOT` with another data directory if needed. To run one cell, for example:

```bash
CELLS=beta_variance BACKENDS=fsdp2 CRITIQUE_COUNTS="2 0" RUN_RESUME=1 \
  bash smoke_tests/intermediate_mc_value/run_smoke.sh
```

## Acceptance criteria

The CPU suites, configuration generation, static checks, and all selected GPU
matrix cells must pass. A production launch should additionally use the
intended production tokenizer/model pair and synchronous reward function,
preserve temperature `1.0`, inspect EMA/variance selection and continuation
failure metrics, confirm actor and critic row divisibility for the production
DP sizes, and perform one native checkpoint resume before scaling out.
