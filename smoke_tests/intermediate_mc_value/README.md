<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: 96vw !important;
}
table { width: 100% !important; }
</style>

# Intermediate Monte Carlo value-model test plan

This plan verifies the synchronous implementation only. It deliberately does not exercise VeRL's fully asynchronous policy trainer, payload queues, or asynchronous reward paths.

## Coverage layers

| Layer | Command or artifact | Required assertions |
|---|---|---|
| Pure numerical CPU | `pytest tests/trainer/ppo/test_intermediate_mc_value_on_cpu.py` | Exact context boundaries; nonterminal mark bounds; uniform gap-constrained random selection; variance tie-breaking and fallback; continuation aggregation; terminal-at-final-valid-token behavior; GAE; critique normalization; scalar MSE/BCE clipping; Beta mean, variance, concentration, endpoint transform, clipping, and finite gradients. |
| Synchronous orchestration CPU | `pytest tests/trainer/ppo/test_intermediate_mc_orchestration_on_cpu.py` | Warmup labels only the terminal token; dense mark labels never touch the delimiter; critique averaging feeds solution GAE; actor batches contain solutions and critiques only; behavior log probabilities are the actor denominator; runtime gating; dummy padding has zero loss mass; checkpoint contracts restore the exact critic-update count. |
| Hydra/config generation | `scripts/generate_trainer_config.sh` | Structured defaults compose, generated reference YAML is current, the default actor objective is `dppo_tv`, and unsupported recipes/backends/reward modes fail before worker allocation. |
| GPU component tests | Add focused CI jobs for `DataParallelIntermediateMCCritic` under legacy FSDP and FSDP2, with and without remove-padding, comparing gathered logits/losses against a single-process reference on the same tiny model and batch. | Scalar and two-logit heads agree across sharding modes; explicit critic positions gather delimiter plus every solution state; masked labels have exactly zero gradient; FSDP checkpoint round-trips preserve head width and optimizer state. |
| Two-step end-to-end GPU | `run_smoke.sh` | Update 1 performs critic warmup with no continuations and no actor update. Update 2 selects marks, samples continuations, trains the critic, then trains the actor. The audit verifier proves continuations never enter actor batches, continuation rewards are bounded, dense labels survive, and the checkpoint count is exact. |
| Resume GPU | Enabled by `RUN_RESUME=1` in `run_smoke.sh` | Resume from update 2, validate the feature/tokenizer contract before model restore, run update 3 without repeating warmup, append to the same audit, and save `critic_update_count=3`. |
| Failure injection | Run orchestration tests with mocked batch failure followed by row-level success/failure. | Partial continuation failures average successful samples; an all-failed mark is omitted; incomplete critique groups are not trainable; generation cleanup always sleeps inference replicas; invalid or non-finite rewards/critic outputs fail closed. |

## GPU smoke matrix

By default the runner executes all four combinations of `scalar_random`/`beta_variance` and legacy `fsdp`/`fsdp2`. It uses temperature `1.0` for solutions, critiques, continuations, and validation; uses console logging only; and sets `WANDB_MODE=disabled`.

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
RECIPES=beta_variance BACKENDS=fsdp2 RUN_RESUME=1 \
  bash smoke_tests/intermediate_mc_value/run_smoke.sh
```

## Acceptance criteria

The CPU suites, configuration generation, static checks, and all selected GPU matrix cells must pass. A production launch should additionally use the intended production tokenizer/model pair and reward function, preserve temperature `1.0`, inspect the audit for label density and continuation failure rates, and perform one checkpoint resume before scaling out.
