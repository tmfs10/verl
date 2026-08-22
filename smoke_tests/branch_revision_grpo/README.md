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

The smoke allows up to 2,560 critique tokens inside a 6,144-token context. Each critique is encoded as a genuine, context-aware follow-up user turn using the rollout worker's actor tokenizer and exact original conversation. The worker records the exact prompt token IDs used for sampling, and the trainer reuses those IDs verbatim; Qwen's hidden-thinking prefill is disabled for that turn so the complete generated critique remains visible and trainable. Analysis is free-form. The strict parser requires nonempty analysis, one exact unique branch, one concise replacement, stable token boundaries, no final-answer marker in the replacement, ordered tags, and no trailing text.

Each revised-continuation request asks vLLM for `prompt_logprobs=1` and reuses the chosen-token prompt probabilities for the replacement under the exact pre-branch context. VeRL slices the result on the rollout replica, transports only the replacement token IDs and scores, and fails closed on any alignment error; no separate trainer-model forward pass is required. The trainer compares the seed's mean log-probability with deterministic, length-matched windows from every original rollout in the iteration, with equal reference mass per eligible rollout. Edits below the configured 20th percentile are not reward-evaluated or solution-trained, and accepted critique credit ramps to full strength at the 50th percentile. Correct-parent critique credit additionally requires a correct continuation and scales with relative token compression, reaching full credit at 25% compression.

The live GPU cell uses the default `dppo_tv` loss. CPU tests separately exercise both `dppo_tv` and native PPO clipping (`vanilla`) with identical global normalization. The verifier fails unless it observes both objectives, exact learnability evidence, mixed rewards in recovery and compression critique groups, successful recovery, successful positive compression, exact recorded critique prompt IDs, the configured continuation budget, binary solution outcomes, objective-correct shaped critique rewards, correct actor regrouping/masking counts, and a finite positive optimizer gradient. This prevents an all-zero or negative-only batch from being certified as acceptance evidence.

## OCI-IAD workflow

Use a timestamped directory outside the code workspace. Submission is intentionally impossible until the exact clean, pushed revision and exact rendered command have passed a dry run.

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=branch-revision-smoke-YYYYMMDD-HHMMSS
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/branch_revision_grpo_smoke/$RUN_TAG

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --model-path /hf_models/Qwen3-1.7B \
  --n-prompts 8 --n-samples 4 --num-critiques 4 --seed 43

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --model-path /hf_models/Qwen3-1.7B \
  --n-prompts 8 --n-samples 4 --num-critiques 4 --seed 43

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

Add `--loss-mode vanilla` to both dry-run and submit commands for the native clipped-PPO parity cell; omitting it selects `dppo_tv`.

The launcher derives an execution-only cluster YAML from the authoritative OCI-IAD configuration, targets the previously authorized `iad-2` SSH route, uses the validated shared VeRL image, explicitly disables requeue, disables W&B, and stores all runtime artifacts under `/home/siddjain/data` locally and `/output/smoke_tests/branch_revision_grpo` remotely.

## Acceptance matrix

Local tests cover both prompt constants, free-form strict parsing, overlapping and token-boundary ambiguity, final-answer markers, deterministic stratified windows, equal per-rollout reference weighting, percentile thresholds, recovery and compression equations, rejected-continuation filtering, exact vLLM prompt-logprob seed slicing/alignment, prompt/parent grouping, suffix-only continuation masks, and both `dppo_tv` and native clipped PPO (`vanilla`). Configuration composition must retain temperature `1.0`, `top_p=1`, `top_k=-1`, repetition penalty `1`, processed rollout log-probabilities, critic disablement, and synchronous reward evaluation.

The first live acceptance cell should use `dppo_tv` on FSDP and must pass `verify_smoke.py`, including a successful recovery and a correct shorter positive continuation. After that anchor, run the same cell with `loss_mode=vanilla`, then FSDP2, then a two-step checkpoint followed by one resumed step. Resume acceptance requires restored actor/optimizer/dataloader/global-step state, no duplicate step evidence, and the same learnability/reference invariants. Multi-node scaling is a separate final parity cell; none of these variants may weaken the semantic acceptance gates merely to obtain a nonzero gradient.
