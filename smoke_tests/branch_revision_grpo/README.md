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

This is a production-faithful, fully synchronous, multi-step acceptance smoke for Qwen3-1.7B or Qwen3-4B on the OPSD Math 30K training data. Every launch must explicitly choose `--revision-mode branch_only` or `--revision-mode seeded_revision`; there is no implicit behavior-changing default. `branch_only` asks the critique policy to diagnose and locate a branch point, discards the original suffix, and lets the actor generate a natural continuation from the exact retained prefix. `seeded_revision` retains the earlier joint prefix-plus-candidate-continuation workflow and its learnability gate. All generation uses temperature `1.0`, and smoke runs disable W&B by default.

The small seeded-revision cell uses Qwen3-1.7B on one exclusive eight-GPU OCI-IAD node, four original rollouts per each of eight prompts, four IID recovery critiques for every incorrect original whose prompt also has a successful original, four IID compression critiques for every correct original, one revised continuation per learnability-accepted edit, and a real actor optimizer step. Each recovery critique receives an independently selected successful original from its prompt. Prompt-logprob OOM acceptance applies only to `seeded_revision` and uses the documented five-step Qwen3-4B cell below; a one-step run is only a plumbing check. Original solutions have a 2,048-token cap, and a valid branch must leave at least 128 tokens of continuation capacity. `--model-path`, `--n-prompts`, `--n-samples`, `--num-critiques`, and `--seed` may select or scale a timestamped acceptance retry; the launcher applies the critique count to both objectives and keeps the model, dataset batch, rollout multiplicity, and verifier expectations synchronized.

The default smoke allows up to 2,560 critique tokens inside an 8,192-token context; the launcher exposes prompt, answer, critique, context, per-GPU token-budget, and one/two-node dimensions for larger cells. Each critique is encoded as a genuine, context-aware follow-up user turn using the rollout worker's actor tokenizer and exact original conversation. The worker records the exact prompt token IDs used for sampling, and the trainer reuses those IDs verbatim. Chat-template options pass through unchanged, so Qwen uses its native thinking mode unless the run explicitly supplies `enable_thinking=False`; all generated thinking and visible critique tokens remain part of the trainable response. Free-form analysis is encouraged but optional and an otherwise valid response may begin directly with `<prefix>`.

In `branch_only`, the response ends with exactly one locator `<prefix>P</prefix>` and contains no proposed direction or continuation. The uniquely matched end of `P` is the branch boundary. Everything through that exact original token boundary is retained, the original suffix is discarded, and the actor samples the complete new suffix naturally. In `seeded_revision`, the response ends with `<prefix>P</prefix>` and `<prefix + new continuation>P+C</prefix + new continuation>`; only nonempty `C` is the proposed seed. Both parsers prefer one exact unique location for `P`. When that text is absent, they lowercase and delete non-alphanumeric characters, require one uniquely best maximum-prefix location without a length threshold, and recover only a stable token boundary inside the adjacent formatting-only gap. Every accepted boundary must be a stable token boundary outside `$$...$$`, `\[...\]`, LaTeX environments, and fenced code blocks; a complete delimited block may be included.

Only `seeded_revision` runs the blocking vLLM score phase. It asks for `prompt_logprobs=0` and `max_tokens=1`, reuses the chosen-token prompt probabilities for only `C`, and applies the global learnability gate before launching long suffix generation. `branch_only` has no generated seed to score, so it launches no prompt-logprob request, builds no reference-window distribution, and applies no learnability rejection. In both modes, the actor trains only on naturally generated suffix tokens; retained original-prefix tokens are context and remain masked from the continuation-row loss.

Prompt-logprob scoring is uncapped by default: the production path does not use the earlier 8,192-token weighted admission mechanism. It still requests only chosen-token probabilities with `prompt_logprobs=0`, scores edits in a separate blocking phase with `max_tokens=1`, applies the global learnability gate, and generates long suffixes only for accepted edits without prompt log probabilities. Normal original, critique, and suffix generation retain `max_num_seqs=32` and the normal 32K batched-token setting. The repaired cell reserves headroom with vLLM memory utilization `0.6`. It deliberately does not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, because vLLM sleep mode uses a CUDA memory pool that rejects expandable segments during engine startup. Schema-v5 audits require admission evidence to be absent when the configured cap is `null`. The optional `--prompt-logprob-max-inflight-tokens` argument remains available for a controlled capped comparison, but it is not part of the default fix.

`critique_advantage_mode=counterfactual_uplift` adds one paired no-diagnosis control for every recovery critique. The same critique-policy snapshot and exact critique prompt are reused, with the literal assistant prefill `<think>\n\n</think>\n\n<prefix>` forced before sampling. In `branch_only`, diagnostic and control each select a branch, then receive independent natural actor continuations from their exact retained prefixes. In `seeded_revision`, both paths retain the structural parser and learnability gate before actor suffix generation. The diagnostic critique reward is `max(0, diagnostic_success - control_success)`; control tokens never enter either policy batch. Compression rows in a mixed run retain their existing correctness-times-compression reward and do not receive controls.

For recovery reporting, `branch_revision/flip/success_per_valid_continuation` retains the historical accepted-edit denominator. `branch_revision/flip/success_per_continuation` uses every structurally valid recovery edit as its denominator, including generations rejected by the learnability gate, so it exposes the corresponding zero-outcome attempts.

`branch_revision/self_critique_reward/mean` reports the iteration mean of the raw self-critique reward `continuation_success - original_prompt_pass@1` over every generated critique. Invalid, learnability-rejected, and unsuccessful proposals have continuation success zero. This diagnostic is intentionally separate from the configured critique optimization reward, including compression and learnability weighting.

Recovery critiques can optionally receive an independently sampled successful
original rollout from the same prompt by setting
`recovery_reference_mode=successful_original`. Selection is uniform with
replacement over the successful original rollouts and is repeated for every
critique using a deterministic hash of the configured selection seed, global
step, prompt, incorrect parent, and critique index. An incorrect original is
skipped when its prompt has no successful original; positive compression is
unchanged. The successful rollout appears only inside the critique-policy
prompt as privileged diagnostic evidence. It never becomes a main-actor row,
and the generated locator must still identify a prefix of the incorrect
rollout. The prompt explicitly requires the proposed next move to remain
causally justified when both the successful reference and the incorrect
rollout suffix are hidden.

For external `pass_at_1` critique advantages,
`critique_prompt_weighting=equal_prompt` is the default. Every critique first
receives weight `1 / critiques_for_prompt`, followed by global mean-one
normalization, so each eligible prompt contributes equal total weight even
when it has a different number of incorrect parents or valid edits. The
historical `headroom` option remains selectable and additionally multiplies
each prompt's mass by `(1 - pass@1) ** exponent`.

`algorithm.branch_revision_grpo.critique_grpo_grouping` controls only critique-row normalization. The backward-compatible `per_original` mode makes the IID critiques for each original rollout one GRPO group. `batch` puts every critique row from the complete synchronous training iteration into one logical `critique:batch` group. Advantages are computed on that complete driver-side batch before data-parallel balancing, so `batch` never means one group per rank, optimizer minibatch, or dynamic microbatch. Original and revised solutions retain their prompt-level `solution:{prompt_group_id}` groups. Because prompt pass@1 varies across prompts, its subtraction cancels under `per_original` centering but remains an active difficulty-dependent signal under `batch` centering.

An optional independent critique policy is enabled with `algorithm.branch_revision_grpo.separate_critique_model=true`. It is initialized from the exact same `actor_rollout_ref.model.path` as the main actor but has its own native FSDP model, optimizer, scheduler, rollout replicas, and checkpoint under `global_step_N/critique_actor_rollout`. The two policies use disjoint Ray GPU pools. The critique policy samples and trains only critique rows; the main actor continues to sample originals, score edit learnability, and sample/train revised suffixes. During the first `critique_warmup_steps` optimizer steps, a separate critique policy trains while the main actor is completely frozen. With a shared policy, the same warmup trains only critique rows and excludes original and revised-continuation rows. Warmup is derived from the restored global step, so resumption does not repeat completed warmup updates.

The smoke launcher can prove a real process-boundary restore with `--resume-from-path /output/.../global_step_N --expected-resume-step N`. A resumed smoke fails before mutation if the exact expected step, actor checkpoint, independent `critique_actor_rollout` checkpoint, or dataloader state is missing. Its source and final manifests separately require native model, optimizer, and extra-state shards for both policies; FSDP extra state contains the scheduler and RNG state. The resumed run writes a new output root and audit attempt, trains only steps after `N`, and saves both policies again at the new terminal step. Fresh smokes retain `resume_mode=disable` and require an expected step of zero.

The live GPU cell uses the default `dppo_tv` loss. CPU tests separately exercise both `dppo_tv` and native PPO clipping (`vanilla`) with identical global normalization. Every invocation writes a new schema-v8 audit-attempt directory, and `status.json` identifies the exact completed attempt to verify. Schema v8 records the explicit revision mode and, for `branch_only`, the original decoded source, locator match kind and offsets, matched source span, exact retained text/tokens, natural continuation tokens/log probabilities, masks, rewards, and absence of seeded-revision scoring evidence. The verifier retains schema-v2 through schema-v7 support for historical evidence. Attempt metadata retains and hashes the exact runtime configuration. Strict acceptance requires a nonuniform logical critique reward group, a successful recovery, a finite positive optimizer gradient, and—when positive compression is enabled—a successful positive compression. Thus a complete zero-signal run can prove plumbing integrity without being mislabeled as algorithmic acceptance.

## OCI-IAD workflow

Use a timestamped directory outside the code workspace. Submission is intentionally impossible until the exact clean, pushed revision and exact rendered command have passed a dry run.

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=branch-revision-smoke-YYYYMMDD-HHMMSS
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/branch_revision_grpo_smoke/$RUN_TAG

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --revision-mode seeded_revision \
  --model-path /hf_models/Qwen3-1.7B \
  --n-prompts 8 --n-samples 4 --num-critiques 4 --seed 43 \
  --learnability-logprob-statistic min

python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad submit \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --revision-mode seeded_revision \
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
  --revision-mode seeded_revision \
  --model-path /hf_models/Qwen3-4B \
  --n-prompts 64 --n-samples 8 --num-critiques 2 --seed 43 \
  --nodes 2 --max-prompt-length 2048 --max-response-length 8192 \
  --critique-max-response-length 8192 --max-model-len 32768 \
  --max-tokens-per-gpu 32768 \
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

The two-step independent-policy lifecycle smoke allocates one actor node and one critique-policy node. Step 1 must contain only a `critique_actor` policy batch; step 2 must contain distinct `actor` and `critique_actor` batches. The verifier reconstructs both batches independently and checks the warmup update flags.

```bash
python3 -m smoke_tests.branch_revision_grpo.submit_oci_iad dry-run \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR" \
  --revision-mode seeded_revision \
  --model-path /hf_models/Qwen3-4B \
  --n-prompts 8 --n-samples 4 --num-critiques 2 --seed 43 \
  --nodes 2 --partition interactive --training-steps 2 \
  --separate-critique-model --critique-warmup-steps 1 \
  --critique-grpo-grouping batch --disable-positive-compression \
  --critique-model-nnodes 1 --critique-model-n-gpus-per-node 8

# Repeat the identical arguments with action `submit`, then collect and verify.
```

The launcher derives an execution-only cluster YAML from the authoritative OCI-IAD configuration, targets the previously authorized `iad-2` SSH route, uses the validated shared VeRL image, explicitly disables requeue, disables W&B, and stores all runtime artifacts under `/home/siddjain/data` locally and `/output/smoke_tests/branch_revision_grpo` remotely.

## CW-DFW workflow

The CW launcher shares the OCI algorithm arguments and verifier but derives its
execution snapshot from authoritative `cw-dfw.yaml`. It preserves CW's native
`verl_vllm012_flashattn_20260321.sqsh` container, uses the documented `dfw`
route, requests two complete eight-GPU `interactive` nodes, disables W&B and
validation, and forces the submitted scheduler record to `Requeue=0`.

The branch-only acceptance cell mirrors the intended mixed recovery/compression
production shape at smaller scale. It uses one actor node and one independent
critique-policy node, prompt-level solution groups, one iteration-wide critique
group with equal prompt weight, and paired counterfactual uplift for recovery.
No candidate continuation or prompt-logprob scoring is permitted in this mode.

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=branch-only-cw-dfw-2n-YYYYMMDDTHHMMSSZ
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/branch_only/smoke/$RUN_TAG

COMMON_ARGS=(
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
  --revision-mode branch_only
  --model-path /hf_models/Qwen/Qwen3-4B
  --n-prompts 32 --n-samples 8 --num-critiques 2
  --ppo-mini-batch-size 32 --seed 43
  --nodes 2 --partition interactive --training-steps 2
  --max-prompt-length 2048 --max-response-length 8192
  --critique-max-response-length 8192 --max-model-len 32768
  --max-tokens-per-gpu 32768 --gpu-memory-utilization 0.6
  --separate-critique-model --critique-warmup-steps 1
  --critique-model-nnodes 1 --critique-model-n-gpus-per-node 8
  --critique-grpo-grouping batch
  --critique-advantage-mode counterfactual_uplift
  --critique-prompt-weighting equal_prompt
  --positive-compression-target 0.75
  --recovery-reference-mode none --loss-mode dppo_tv
)

python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw dry-run "${COMMON_ARGS[@]}"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw submit "${COMMON_ARGS[@]}"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw verify-integrity \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

The external-pass@1 recovery acceptance cell uses one actor node and one
independent critique-policy node. It keeps every production algorithm setting
except the deliberately shortened one-step warmup and two-step lifetime:

```bash
cd /home/siddjain/workspace/verl/verl_branch_revision_grpo
RUN_TAG=branch-revision-cw-dfw-external-pass-at1-YYYYMMDDTHHMMSSZ
RUN_DIR=/home/siddjain/data/intermediate_mc_value_model/verl/cw_dfw_external_pass_at1/smoke/$RUN_TAG

COMMON_ARGS=(
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
  --revision-mode seeded_revision
  --model-path /hf_models/Qwen/Qwen3-4B
  --n-prompts 32 --n-samples 8 --num-critiques 2 --seed 43
  --nodes 2 --partition interactive --training-steps 2
  --max-prompt-length 2048 --max-response-length 8192
  --critique-max-response-length 8192 --max-model-len 32768
  --max-tokens-per-gpu 32768 --gpu-memory-utilization 0.6
  --separate-critique-model --critique-warmup-steps 1
  --critique-model-nnodes 1 --critique-model-n-gpus-per-node 8
  --critique-grpo-grouping batch --critique-advantage-mode pass_at_1
  --critique-prompt-weighting equal_prompt
  --recovery-reference-mode successful_original
  --recovery-reference-selection-seed 0
  --disable-positive-compression --loss-mode dppo_tv
  --learnability-logprob-statistic mean
  --learnability-threshold-mode stddev --max-seed-window-stddevs 15
)

python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw dry-run "${COMMON_ARGS[@]}"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw submit "${COMMON_ARGS[@]}"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw status \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw collect \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw verify-integrity \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
python3 -m smoke_tests.branch_revision_grpo.submit_cw_dfw verify \
  --run-tag "$RUN_TAG" --local-run-dir "$RUN_DIR"
```

Strict verification additionally reconstructs external pass@1 raw advantages,
invalid and learnability-rejection penalties, RMS scaling, equal-prompt
weights, successful-reference assignments, clipping, and the final actor
tensor hashes. Do not weaken the
successful-recovery requirement for a stochastic run; use a new timestamped
seed-44 retry if seed 43 passes integrity but lacks algorithm signal.

## Acceptance matrix

Local tests cover all three prompt constants, free-form strict parsing, exact generated-prefix/joint agreement, overlapping and token-boundary ambiguity, final-answer markers, both sides of display-math/LaTeX/code-fence boundaries, exhaustive equal-per-window references, mean/min ranking and float32 boundaries, population-standard-deviation and percentile thresholds, zero-variance references, recovery and compression equations, rejected-continuation filtering, exact vLLM score-only prompt-logprob slicing/alignment, accepted-only suffix generation, weighted-admission concurrency/oversize/cancellation behavior, critique-failure draining, tokenizer-aware context headroom including the successful reference, deterministic per-critique same-prompt reference sampling, all-failure prompt skipping, retry-safe audit attempts, multi-step high-pressure selection, schema-v2 through schema-v6 verification, equal-prompt and headroom weighting, post-balance actor hashes, prompt/parent grouping, suffix-only continuation masks, both denominator metrics, and both `dppo_tv` and native clipped PPO (`vanilla`). Configuration composition must retain temperature `1.0`, `top_p=1`, `top_k=-1`, repetition penalty `1`, processed rollout log-probabilities, critic disablement, and synchronous reward evaluation.

The first live acceptance cell should use `dppo_tv` on FSDP and must pass `verify_smoke.py`, including a successful recovery and a correct shorter positive continuation. After that anchor, run the same cell with `loss_mode=vanilla`, then FSDP2, then a two-step checkpoint followed by one resumed step. Resume acceptance requires restored actor/optimizer/dataloader/global-step state, no duplicate step evidence, and the same learnability/reference invariants. Multi-node scaling is a separate final parity cell; none of these variants may weaken the semantic acceptance gates merely to obtain a nonzero gradient.
