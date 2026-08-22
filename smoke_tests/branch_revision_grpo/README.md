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

This is a production-faithful, fully synchronous, one-step acceptance smoke for Qwen3-1.7B or Qwen3-4B on the OPSD Math 30K training data. Its default cell uses Qwen3-1.7B on one exclusive eight-GPU OCI-IAD node, four original rollouts per each of eight prompts, four IID recovery critiques for every incorrect original, four IID compression critiques for every correct original, one revised continuation per structurally valid edit, a real actor optimizer step, temperature `1.0` for every generation, and no W&B logging. Original solutions have a 2,048-token cap, and a valid edit must leave at least 128 tokens of continuation capacity. `--model-path`, `--n-prompts`, `--n-samples`, `--num-critiques`, and `--seed` may select or scale a timestamped acceptance retry; the launcher applies the critique count to both objectives and keeps the model, dataset batch, rollout multiplicity, and verifier expectations synchronized.

The smoke allows up to 2,560 critique tokens inside a 6,144-token context. Each critique is encoded as a genuine, context-aware follow-up user turn using the rollout worker's actor tokenizer and exact original conversation. The worker records the exact prompt token IDs used for sampling, and the trainer reuses those IDs verbatim; Qwen's hidden-thinking prefill is disabled for that turn so the complete generated critique remains visible and trainable. Analysis is free-form. The strict parser requires nonempty analysis, one exact unique branch, one concise replacement, stable token boundaries, no final-answer marker in the replacement, ordered tags, and no trailing text. It also rejects a branch boundary inside `$$...$$`, `\[...\]`, a LaTeX environment, or a fenced code block before any continuation is launched.

Each revised-continuation request asks vLLM for `prompt_logprobs=1` and reuses the chosen-token prompt probabilities for the replacement under the exact pre-branch context. VeRL slices the result on the rollout replica, transports float32 replacement token scores, and fails closed on any alignment error; no separate trainer-model forward pass is required. `learnability_logprob_statistic` selects either `mean` (the average replacement-token log probability) or `min` (the least likely replacement token). For each distinct replacement length, the trainer enumerates every contiguous window of exactly that length from every eligible original rollout and gives every window equal mass. Float64 prefix sums compute exhaustive means, while an exact reusable range-minimum index computes exhaustive minima; each length-specific distribution is built and sorted once. Edits below the configured 20th percentile are not reward-evaluated or solution-trained, and accepted critique credit ramps to full strength at the 50th percentile. Correct-parent critique credit additionally requires a correct continuation and scales with relative token compression, reaching full credit at 25% compression.

For recovery reporting, `branch_revision/flip/success_per_valid_continuation` retains the historical accepted-edit denominator. `branch_revision/flip/success_per_continuation` uses every structurally valid recovery edit as its denominator, including generations rejected by the learnability gate, so it exposes the corresponding zero-outcome attempts.

The live GPU cell uses the default `dppo_tv` loss. CPU tests separately exercise both `dppo_tv` and native PPO clipping (`vanilla`) with identical global normalization. Every invocation writes a new schema-v3 audit-attempt directory, and `status.json` identifies the exact completed attempt to verify; an incomplete prior attempt can therefore coexist with a resumed or retried step without blocking it or being mistaken for current evidence. Schema v3 records compact exhaustive window counts and hashes instead of serializing every window, while the verifier retains schema-v2 support for historical evidence. Attempt metadata retains and hashes the exact runtime configuration. The verifier compares it with the saved launch configuration after applying only VeRL's deterministic pre-step normalization of resource-pool sizes and optimizer training-step counts; any other difference fails. The integrity verifier independently reconstructs the vLLM scoring slice and exhaustive length-matched reference, then reconstructs every post-balancing actor row—including masks and behavior log probabilities—from source events and tensor hashes. Strict acceptance additionally requires mixed rewards in recovery and compression critique groups, successful recovery, successful positive compression, and a finite positive optimizer gradient. Thus a complete zero-signal run can prove plumbing integrity without being mislabeled as algorithmic acceptance.

## OCI-IAD workflow

Use a timestamped directory outside the code workspace. Submission is intentionally impossible until the exact clean, pushed revision and exact rendered command have passed a dry run.

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=branch-revision-smoke-YYYYMMDD-HHMMSS
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/branch_revision_grpo_smoke/$RUN_TAG

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --model-path /hf_models/Qwen3-1.7B \
  --n-prompts 8 --n-samples 4 --num-critiques 4 --seed 43 \
  --learnability-logprob-statistic min

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --model-path /hf_models/Qwen3-1.7B \
  --n-prompts 8 --n-samples 4 --num-critiques 4 --seed 43 \
  --learnability-logprob-statistic min

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad verify-integrity \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

Use the same `--learnability-logprob-statistic` value on dry-run and submit; `mean` remains the default. Add `--loss-mode vanilla` to both commands for the native clipped-PPO parity cell; omitting it selects `dppo_tv`. `verify-integrity` proves the completed audit and trained-row plumbing even if sampling produced no accepted or successful revision. Only `verify` is the strict algorithmic acceptance gate.

The launcher derives an execution-only cluster YAML from the authoritative OCI-IAD configuration, targets the previously authorized `iad-2` SSH route, uses the validated shared VeRL image, explicitly disables requeue, disables W&B, and stores all runtime artifacts under `/home/siddjain/data` locally and `/output/smoke_tests/branch_revision_grpo` remotely.

## Acceptance matrix

Local tests cover both prompt constants, free-form strict parsing, overlapping and token-boundary ambiguity, final-answer markers, display-math/LaTeX/code-fence boundaries, exhaustive equal-per-window references, mean/min ranking and float32 boundaries, percentile thresholds, recovery and compression equations, rejected-continuation filtering, exact vLLM prompt-logprob seed slicing/alignment, critique-failure draining, tokenizer-aware context headroom, retry-safe audit attempts, schema-v2/v3 verification, post-balance actor hashes, prompt/parent grouping, suffix-only continuation masks, both denominator metrics, and both `dppo_tv` and native clipped PPO (`vanilla`). Configuration composition must retain temperature `1.0`, `top_p=1`, `top_k=-1`, repetition penalty `1`, processed rollout log-probabilities, critic disablement, and synchronous reward evaluation.

The first live acceptance cell should use `dppo_tv` on FSDP and must pass `verify_smoke.py`, including a successful recovery and a correct shorter positive continuation. After that anchor, run the same cell with `loss_mode=vanilla`, then FSDP2, then a two-step checkpoint followed by one resumed step. Resume acceptance requires restored actor/optimizer/dataloader/global-step state, no duplicate step evidence, and the same learnability/reference invariants. Multi-node scaling is a separate final parity cell; none of these variants may weaken the semantic acceptance gates merely to obtain a nonzero gradient.
