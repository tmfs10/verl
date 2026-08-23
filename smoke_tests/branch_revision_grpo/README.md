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

This is a production-faithful, fully synchronous, multi-step acceptance smoke for Qwen3-1.7B or Qwen3-4B on the OPSD Math 30K training data. Its small default cell uses Qwen3-1.7B on one exclusive eight-GPU OCI-IAD node, four original rollouts per each of eight prompts, four IID recovery critiques for every incorrect original, four IID compression critiques for every correct original, one revised continuation per learnability-accepted edit, a real actor optimizer step, temperature `1.0` for every generation, and no W&B logging. Prompt-logprob OOM acceptance uses the documented five-step Qwen3-4B cell below; a one-step run is only a plumbing check. Original solutions have a 2,048-token cap, and a valid edit must leave at least 128 tokens of continuation capacity. `--model-path`, `--n-prompts`, `--n-samples`, `--num-critiques`, and `--seed` may select or scale a timestamped acceptance retry; the launcher applies the critique count to both objectives and keeps the model, dataset batch, rollout multiplicity, and verifier expectations synchronized.

The default smoke allows up to 2,560 critique tokens inside an 8,192-token context; the launcher exposes prompt, answer, critique, context, per-GPU token-budget, and one/two-node dimensions for larger cells. Each critique is encoded as a genuine, context-aware follow-up user turn using the rollout worker's actor tokenizer and exact original conversation. The worker records the exact prompt token IDs used for sampling, and the trainer reuses those IDs verbatim; Qwen's hidden-thinking prefill is disabled for that turn so the complete generated critique remains visible and trainable. Free-form analysis is encouraged but optional and an otherwise valid response may begin directly with `<prefix>`. A critique ends with a locator `<prefix>P</prefix>` and a joint `<prefix + new continuation>P+C</prefix + new continuation>` field. The joint field must begin character-for-character with `P`; only the nonempty appended `C` is the new continuation. The parser prefers one exact unique location for `P`. When that text is absent, it lowercases and deletes non-alphanumeric characters, requires one uniquely best maximum-prefix location without a length threshold, and recovers only a stable token boundary inside the adjacent formatting-only gap. The revision replaces the original trajectory from the matched location onward with generated `P+C`, allowing the model to produce a natural transition jointly. Both boundaries around `P` must be stable token boundaries and outside `$$...$$`, `\[...\]`, LaTeX environments, and fenced code blocks; a complete delimited block may be included. The parser also requires ordered tags, no trailing text, and no final-answer marker in `C`.

After critique parsing, a separate blocking score phase asks vLLM for `prompt_logprobs=0` and `max_tokens=1`. It reuses the chosen-token prompt probabilities for only `C`, conditioned on the exact original prompt plus the matched pre-prefix trajectory and generated `P`. VeRL slices the result on the rollout replica, transports float32 `C` token scores, and fails closed on any alignment error; no separate trainer-model forward pass is required. The trainer then applies the global learnability gate and launches long suffix generation only for accepted edits, without prompt log probabilities. `learnability_logprob_statistic` selects either `mean` (the average `C`-token log probability) or `min` (the least likely `C` token). For each distinct `C` length, the trainer enumerates every contiguous window of exactly that length from every eligible original rollout and gives every window equal mass. Float64 prefix sums compute exhaustive means, while an exact reusable range-minimum index computes exhaustive minima; each length-specific distribution is built and sorted once. The default `stddev` threshold mode accepts an edit when its score is no more than 15 population standard deviations below its length-matched reference mean; accepted edits receive learnability weight one and rejected edits receive zero. Explicit `percentile` mode retains the historical 20th-percentile gate and linear credit ramp to full strength at the 50th percentile. Correct-parent critique credit additionally requires a correct continuation and scales with relative token compression, reaching full credit at 25% compression.

Prompt-logprob scoring is opt-in and bounded independently on every vLLM server. The production smoke sets an 8,192-token weighted admission budget: concurrent score prompts may proceed only while their complete prompt-token sum fits the budget, while a single larger prompt runs alone. The permit is released immediately after vLLM exposes prompt log probabilities, including failure cleanup. Normal original, critique, and suffix generation retain `max_num_seqs=32` and the normal 32K batched-token setting. The repaired cell also reserves allocator headroom with vLLM memory utilization `0.6` and propagates `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to every Ray worker. Per-request admission evidence and per-server high-water marks are retained in schema-v5 audit events without changing the schema number.

For recovery reporting, `branch_revision/flip/success_per_valid_continuation` retains the historical accepted-edit denominator. `branch_revision/flip/success_per_continuation` uses every structurally valid recovery edit as its denominator, including generations rejected by the learnability gate, so it exposes the corresponding zero-outcome attempts.

The live GPU cell uses the default `dppo_tv` loss. CPU tests separately exercise both `dppo_tv` and native PPO clipping (`vanilla`) with identical global normalization. Every invocation writes a new schema-v5 audit-attempt directory, and `status.json` identifies the exact completed attempt to verify; an incomplete prior attempt can therefore coexist with a resumed or retried step without blocking it or being mistaken for current evidence. Schema v5 adds the threshold mode, exhaustive population mean and standard deviation, acceptance floor, and standardized edit distance to schema v4's generated-prefix and joint-boundary evidence; the verifier retains schema-v2, schema-v3, and schema-v4 support for historical evidence. Attempt metadata retains and hashes the exact runtime configuration. The verifier compares it with the saved launch configuration after applying only VeRL's deterministic pre-step normalization of resource-pool sizes and optimizer training-step counts; any other difference fails. For fresh and resumed multi-step runs it derives the exact expected contiguous step range from `starting_global_step`, requires every step in that range to finish, validates prompt-logprob admission evidence on every completed step, and deeply reconstructs the step with the largest admitted prompt-token volume. The integrity verifier independently reconstructs the joint boundary, vLLM `C` scoring slice, admission budget, exhaustive length-matched reference, threshold decision, and every post-balancing actor row—including masks and behavior log probabilities—from source events and tensor hashes. Strict acceptance additionally requires mixed rewards in recovery and compression critique groups, successful recovery, successful positive compression, and a finite positive optimizer gradient. Thus a complete zero-signal run can prove plumbing integrity without being mislabeled as algorithmic acceptance.

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

Use the same learnability arguments on dry-run and submit. `mean`, `stddev`, and `15.0` are the defaults; use `--learnability-threshold-mode percentile` to restore the historical percentile gate, or `--max-seed-window-stddevs` to change the sigma cutoff. Add `--loss-mode vanilla` to both commands for the native clipped-PPO parity cell; omitting it selects `dppo_tv`. `verify-integrity` proves the completed audit and trained-row plumbing even if sampling produced no accepted or successful revision. Only `verify` is the strict algorithmic acceptance gate.

The two-node 32K-context cell keeps every original and revised answer within an 8,192-token response budget. The continuation worker enforces this as `8192 - len(revised_prefix_ids)`, so the generated prefix, edit seed, and remaining suffix together cannot exceed 8,192 response tokens.

```bash
python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --model-path /hf_models/Qwen3-4B \
  --n-prompts 64 --n-samples 8 --num-critiques 2 --seed 43 \
  --nodes 2 --max-prompt-length 2048 --max-response-length 8192 \
  --critique-max-response-length 8192 --max-model-len 32768 \
  --max-tokens-per-gpu 32768 \
  --prompt-logprob-max-inflight-tokens 8192 \
  --gpu-memory-utilization 0.6 \
  --training-steps 5 \
  --partition interactive \
  --loss-mode dppo_tv --learnability-logprob-statistic mean \
  --learnability-threshold-mode stddev --max-seed-window-stddevs 15

# Repeat the identical arguments with action `submit`, then use the standard
# status, collect, verify-integrity, and verify actions above.
```

Use `--training-steps 5` for prompt-logprob memory acceptance. A one-step cell
does not exercise the higher structured-edit compliance observed after policy
updates and is not sufficient evidence against prompt-logprob OOMs.
The `interactive` partition is limited to at most two nodes. An unchanged
four-node reproduction may omit `--partition`; it then uses the configured
normal OCI-IAD batch partitions.

The launcher derives an execution-only cluster YAML from the authoritative OCI-IAD configuration, targets the previously authorized `iad-2` SSH route, uses the validated shared VeRL image, explicitly disables requeue, disables W&B, and stores all runtime artifacts under `/home/siddjain/data` locally and `/output/smoke_tests/branch_revision_grpo` remotely.

## Acceptance matrix

Local tests cover both prompt constants, free-form strict parsing, exact generated-prefix/joint agreement, overlapping and token-boundary ambiguity, final-answer markers, both sides of display-math/LaTeX/code-fence boundaries, exhaustive equal-per-window references, mean/min ranking and float32 boundaries, population-standard-deviation and percentile thresholds, zero-variance references, recovery and compression equations, rejected-continuation filtering, exact vLLM score-only prompt-logprob slicing/alignment, accepted-only suffix generation, weighted-admission concurrency/oversize/cancellation behavior, critique-failure draining, tokenizer-aware context headroom, retry-safe audit attempts, multi-step high-pressure selection, schema-v2/v3/v4/v5 verification, post-balance actor hashes, prompt/parent grouping, suffix-only continuation masks, both denominator metrics, and both `dppo_tv` and native clipped PPO (`vanilla`). Configuration composition must retain temperature `1.0`, `top_p=1`, `top_k=-1`, repetition penalty `1`, processed rollout log-probabilities, critic disablement, and synchronous reward evaluation.

The first live acceptance cell should use `dppo_tv` on FSDP and must pass `verify_smoke.py`, including a successful recovery and a correct shorter positive continuation. After that anchor, run the same cell with `loss_mode=vanilla`, then FSDP2, then a two-step checkpoint followed by one resumed step. Resume acceptance requires restored actor/optimizer/dataloader/global-step state, no duplicate step evidence, and the same learnability/reference invariants. Multi-node scaling is a separate final parity cell; none of these variants may weaken the semantic acceptance gates merely to obtain a nonzero gradient.
