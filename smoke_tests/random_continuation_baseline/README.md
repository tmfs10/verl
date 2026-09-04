<style>
body,
main,
article,
.markdown-body,
.rendered_html,
.jp-RenderedHTMLCommon,
.jp-MarkdownOutput {
  max-width: none !important;
  width: min(98vw, 1800px) !important;
}
table {
  width: 100% !important;
}
</style>

# Random-continuation baseline

This synchronous evaluation generates multiple independent Qwen3-4B
originals for each OPSD Math 30K prompt, selects uniformly random valid
token-prefix positions strictly after 10% of each original, and samples
multiple independent natural continuations from every exact prefix. It runs
for one iteration at temperature 1.0 without critic, critique, actor
optimization, validation, checkpointing, or W&B.

All marks leave at least 128 response tokens and use the production unmatched
display-math, LaTeX-environment, and code-fence checks. The response, prompt,
and model-context limits are 8,192, 2,048, and 32,768 tokens.

## Two-node CW-DFW smoke

The smoke uses four prompts, two originals per prompt, two random marks per
original, and two continuations per mark: eight originals, 16 marks, and 32
continuations.

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=random-continuation-n2m2c2-smoke-YYYYMMDDTHHMMSSZ
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/random_continuation_baseline/$RUN_TAG

python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --prompts 4 --rollouts 2 --points 2 --continuations 2
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --prompts 4 --rollouts 2 --points 2 --continuations 2
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
/home/siddjain/anaconda3/envs/deepseek/bin/python \
  -m smoke_tests.random_continuation_baseline.submit_cw_dfw verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

## Full one-iteration evaluation

The full evaluation uses 256 prompts, four originals per prompt, four random
marks per original, and four continuations per mark: 1,024 originals, 4,096
marks, and 16,384 continuations.

```bash
RUN_TAG=random-continuation-n4m4c4-prod-YYYYMMDDTHHMMSSZ
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/random_continuation_baseline/$RUN_TAG

python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --prompts 256 --rollouts 4 --points 4 --continuations 4
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --prompts 256 --rollouts 4 --points 4 --continuations 4
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
/home/siddjain/anaconda3/envs/deepseek/bin/python \
  -m smoke_tests.random_continuation_baseline.submit_cw_dfw verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

The authoritative result is `collected/audit/summary.json`. Verification
requires exact prompt/original/mark/continuation cardinalities, zero generation
failures or selection shortfalls, structural eligibility, token/log-probability
alignment, binary rewards, correct leave-one-out IID statistics, and no actor
optimizer evidence.
