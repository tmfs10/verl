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

This evaluation generates one Qwen3-4B original for each of 256 OPSD Math
30K prompts, selects eight uniformly random valid token-prefix positions
strictly after 10% of each original, and samples one natural continuation from
each prefix. It runs for exactly one iteration, at temperature 1.0, without
critic, critique, actor optimization, validation, checkpointing, or W&B.

Run on two exclusive CW-DFW interactive nodes:

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=random-continuation-cw-dfw-2n-YYYYMMDDTHHMMSSZ
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/random_continuation_baseline/$RUN_TAG

python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.random_continuation_baseline.submit_cw_dfw verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

The authoritative result is `collected/audit/summary.json`. The verifier
checks the requested token budgets, all prefix locations, structural
eligibility, continuation accounting, numerator/denominator statistics, and
the absence of optimizer evidence.
