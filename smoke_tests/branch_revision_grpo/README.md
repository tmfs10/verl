<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: 96vw !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
table { width: 100% !important; }
</style>

# Branch-revision GRPO smoke

This is a production-faithful, fully synchronous, one-step acceptance smoke for Qwen3-1.7B on the OPSD Math 30K training data. It uses one exclusive eight-GPU OCI-IAD node, four original rollouts per each of eight prompts, four IID branch critiques for every incorrect original, one revised continuation per strictly valid edit, a real actor optimizer step, temperature `1.0` for every generation, and no W&B logging. Original solutions have a 2,048-token cap, and a valid edit must leave at least 128 tokens of continuation capacity.

The smoke allows up to 2,560 critique tokens inside a 6,144-token context. The critique is encoded as a genuine, context-aware follow-up user turn using the rollout worker's actor tokenizer and exact original conversation. The worker records the exact prompt token IDs used for sampling, and the trainer reuses those IDs verbatim; Qwen's hidden-thinking prefill is disabled for that turn so the complete generated critique remains visible and trainable. The strict parser requires all three nonempty numbered findings before the edit tags. The larger hard cap is a fail-safe rather than permission to omit the structured edit.

The live GPU cell uses the default `dppo_tv` loss. CPU tests separately exercise both `dppo_tv` and native PPO clipping (`vanilla`) with identical global normalization. The verifier fails unless it observes mixed rewards in at least one original-solution GRPO group, mixed rewards in at least one self-critique GRPO group, a successful revised continuation, exact recorded critique prompt IDs, the configured continuation budget, binary environment outcomes, exact critique rewards of `continuation_outcome - original_prompt_pass@1`, correct actor regrouping/masking counts, and a finite positive optimizer gradient. This prevents an all-zero batch from being certified as an optimizer smoke.

## OCI-IAD workflow

Use a timestamped directory outside the code workspace. Submission is intentionally impossible until the exact clean, pushed revision and exact rendered command have passed a dry run.

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=branch-revision-smoke-YYYYMMDD-HHMMSS
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/branch_revision_grpo_smoke/$RUN_TAG

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

The launcher derives an execution-only cluster YAML from the authoritative OCI-IAD configuration, targets the previously authorized `iad-2` SSH route, uses the validated shared VeRL image, explicitly disables requeue, disables W&B, and stores all runtime artifacts under `/home/siddjain/data` locally and `/output/smoke_tests/branch_revision_grpo` remotely.
