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

# Changes

## 2026-08-12

- Audited the complete dirty state of the parent `verl` checkout and nested
  `verl-recipe` repository before publication; no credentials or private keys
  were present in the added files.
- Split the accumulated work into scoped commits for Article-RAG launchers,
  OPSD teacher/steering objectives, forward-only logprob tooling, grouped
  success rewards and resume guards, smoke/audit workflows, and tracking docs.
- Preserved the two already-local parent commits rather than rewriting branch
  history. The parent submodule pointer now targets the committed recipe tree.
- Reverified the exact pre-commit tree in the pinned CW-DFW VERL container:
  174 OPSD tests, 21 grouped-reward tests, and 30 grouped-reward/resume tests
  passed. Local Python and shell syntax checks also passed.

## 2026-05-26

- Started implementation tracking for the SV-OPSD worktree and steering-vector SDPO variant.
- Created `/home/siddjain/workspace/verl/verl_svopsd` as a dedicated worktree on branch `svopsd`.
- Added SDPO steering-vector conditioning support in the `verl_svopsd` worktree:
  - nested OPSD steering config and validation
  - SDPO all-correct candidate selection
  - fractional layer selection parsing
  - steering-source tensor construction without prompt mutation
  - per-layer vector extraction and application hooks in the OPSD actor
  - focused unit tests for helpers and actor hook mechanics
- Fixed nested OPSD steering config construction to respect the repo's frozen `BaseConfig` behavior.
- Added a SciKnowEval Chemistry SV-OPSD smoke-test driver that builds tiny Chemistry subsets under `~/data` and exercises SDPO steering-vector candidate selection, extraction, and application on a tiny CPU model.
- Ran the SciKnowEval Chemistry SV-OPSD smoke test successfully after fixing the smoke driver's local profiler decorator stub.
- Added a DFW interactive trainer smoke submitter for SV-OPSD on SciKnowEval Chemistry using `recipe.opsd.main_opsd`, Qwen3-8B, 1 interactive DFW node, 8 GPUs, W&B disabled, and temperature 1.0.
- Fixed the DFW SV-OPSD smoke submitter to pass the comma-separated steering layer selector to Hydra as an escaped quoted string.
- Fixed the DFW SV-OPSD smoke submitter to use `++algorithm.opsd...` overrides now that the mounted code includes `algorithm.opsd`.
- Increased the DFW SV-OPSD smoke rollout weight-sync bucket to `4096 MB` for Qwen3-8B.
- Fixed the SciKnowEval reward function to handle array-valued `extra_infos` from the trainer by checking `None` explicitly.
- Ran a DFW interactive SV-OPSD Chemistry trainer smoke that completed as SLURM job `12179959`, but tightened the pass criterion afterward because the 512-token response budget yielded no correct rollouts and `actor/opsd_steering_active_rate:0.0`.
- Increased the DFW SV-OPSD Chemistry smoke response budget to `--max_len 4096` and `--max_tokens_per_gpu 8192`, and changed the pass criterion to require nonzero `actor/opsd_steering_active_rate`.
- Ran DFW interactive SV-OPSD Chemistry job `12180243`; it generated correct Chemistry rollouts (`10/64` in the pre-actor rollout dump) but failed during actor update with an NCCL all-gather timeout.
- Fixed SV-OPSD steering extraction so ranks with no local steering candidates still participate in a dummy steering-source forward before returning no vectors, avoiding FSDP collective desynchronization when other ranks do have candidates.
- Added a focused regression test covering the no-local-candidate dummy forward behavior in `tests/recipe/opsd/test_dp_actor.py`.
- Reran the DFW interactive SV-OPSD Chemistry trainer smoke as SLURM job `12181180`; it completed and passed the stricter steering-active criterion with `actor/opsd_steering_active_rate:0.59375`, `actor/opsd_steering_candidate_mean:0.90625`, and `actor/opsd_distill_active_rate:0.59375`.

## 2026-05-27

- Added a reusable DFW batch submitter for the SciKnowEval Chemistry SDPO baseline and two SV-OPSD variants on Qwen3-8B, using 4 nodes, 8 GPUs per node, W&B enabled, paper-aligned SDPO hyperparameters, and steering layer range `0.31-0.37`.
- Submitted the three DFW batch jobs under run tag `20260527_004123`: SDPO job `12184906`, SV-OPSD first-correct job `12184916`, and SV-OPSD all-correct job `12184978`.
- Transferred the SciKnowEval Chemistry train/test JSONL files from DFW to EOS and verified matching line counts and SHA-256 checksums.
- Added an EOS-specific copied cluster config under `~/data` with EOS-local `/home` and `/root/.netrc` mounts. Kept `disable_gpus_per_node: True` because EOS Slurm does not expose GPU GRES/TRES and rejects `--gpus-per-node=8`; EOS runs use exclusive full H100 nodes with `trainer.n_gpus_per_node=8` and Ray's 32-GPU readiness check.
- Submitted the three EOS batch jobs under run tag `eos_20260527_005824`: SDPO job `5332410`, SV-OPSD first-correct job `5332412`, and SV-OPSD all-correct job `5332413`.
- Added an AWS-IAD-specific copied cluster config under `~/data` that keeps the stock `pool0` settings and adds the required `containers.verl` entry pointing at `/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-verl-latest.sqsh`.
- Tried Data Mover for DFW-to-AWS-IAD Chemistry data transfer, but the DFW CPU Slurm runner stayed pending on priority; cancelled Data Mover job `dd2781e6-9d52-45ee-9de4-c7fa4a6c49a6` / Slurm job `12186572` and switched the AWS-IAD transfer workflow to direct DFW-to-AWS-IAD `rsync`.
- Synced the SciKnowEval Chemistry train/test JSONL files and Qwen3-8B HF model directory from DFW to AWS-IAD. Verified the Chemistry checksums/line counts, model file count, model key metadata checksums, and an rsync dry-run comparison.
- Submitted the first AWS-IAD Chemistry SDPO/SV-OPSD run under run tag `aws_iad_20260527_020811`; all three jobs allocated 4 nodes / 32 GPUs but failed before training because the injected `one-logger-utils` install upgraded `click` and broke Ray CLI import.
- Updated the AWS-IAD copied config to point at an `onelogger`-named symlink to the same VERL container, suppressing the submitter's package-install wrapper while preserving the intended container and W&B-enabled run configuration.
- Relaunched the AWS-IAD Chemistry SDPO/SV-OPSD jobs under run tag `aws_iad_noinstall_20260527_021548`: SDPO job `4154001`, SV-OPSD first-correct job `4154005`, and SV-OPSD all-correct job `4154021`. Verified all three were running on `pool0` with 4 nodes / 32 GPUs and had reached Ray GPU readiness plus training startup.
- Checked the previously launched DFW/EOS/AWS-IAD Chemistry jobs after the cancellation request. No matching jobs were active: DFW jobs `12184906`, `12184916`, and `12184978` had completed; EOS jobs `5332410`, `5332412`, and `5332413` had failed early; AWS-IAD jobs `4154001` and `4154021` had timed out and `4154005` had completed.
- Reworked SV-OPSD steering extraction to use CAA-style contrastive residual vectors. For each prompt group, the code now gathers all verifier-correct rollouts as positives and all verifier-incorrect rollouts as negatives, extracts selected-layer response-token residual activations, and computes `mean_positive - mean_negative` per selected layer without averaging across layers.
- Updated the SV-OPSD batch fields, metrics, smoke driver, and focused tests to carry positive/negative source signs instead of single-correct or all-correct-only aggregation.
- Patched the CAA steering-field path so the active distillation mask is constructed on the same device as the response mask.
- Ran the DFW interactive CAA SV-OPSD Chemistry smoke as SLURM job `12199260` with `LAYER_FRACTIONS=0.31-0.37`; it completed successfully with nonzero CAA steering activity (`actor/opsd_steering_active_rate:0.46875`).
- Updated the reusable Chemistry SDPO/SV-OPSD DFW submitter to accept `ACTOR_LR` as an environment override, while preserving the default `1e-5` LR.
- Submitted two DFW batch CAA SV-OPSD Chemistry LR-sweep jobs: LR `1e-4` as SLURM job `12212000`, and LR `5e-5` as SLURM job `12212015`.
- Submitted a DFW batch vanilla SDPO Chemistry run with LR `1e-6` as SLURM job `12212148`.
- Cancelled the DFW batch vanilla SDPO Chemistry LR `1e-6` job `12212148` before it started.
- Recorded the DFW batch vanilla SDPO Chemistry LR `1e-5` / 16K-response workflow, using `MAX_LEN=18432` so the submitter produces `data.max_response_length=16384`.
- Submitted the DFW batch vanilla SDPO Chemistry LR `1e-5` / 16K-response run as SLURM job `12213113`.
- Submitted an additional DFW batch CAA SV-OPSD Chemistry run with LR `5e-5` and steering layer selector `0.2-0.6` as SLURM job `12213198`.
- Submitted an additional DFW batch CAA SV-OPSD Chemistry run with LR `1e-5` and steering layer selector `0.2-0.6` as SLURM job `12214042`.
- Added a reusable launcher under `~/data` for DFW DeepMath/CompMath GRPO, vanilla SDPO, and SV-SDPO CAA 30B runs with LR `5e-6` and 8K response length.
- Submitted the DFW DeepMath/CompMath 30B runs under run tag `dfw_deepmath_sdpo_svopsd_20260527_170000`: GRPO job `12215167`, vanilla SDPO job `12215171`, and SV-SDPO CAA job `12215183`. All three were pending on 4 DFW batch nodes with reason `Priority` immediately after submission.
- Copied `verl_vllm012_flashattn_20260321.sqsh` from DFW to EOS `~/lustre/containers/`. Data Mover upload to staging completed, but the EOS Data Mover download runner was pending until the next morning, so the pending download leg was cancelled and a DFW-initiated rsync completed the 21,465,190,400-byte copy.
- Added an EOS interactive DeepMath/CompMath smoke submitter under `smoke_tests/` that disables W&B, uses the copied 8K-response DeepMath settings, and points the generated EOS config at an `onelogger`-named symlink to the VERL image to avoid the previous one-logger package upgrade failure.
- Fixed the EOS smoke submitter to add an explicit `interactive` timeout entry to the generated EOS config; the stock EOS YAML has `batch,interactive`, but the submitter's partition resolver requires a separate `interactive` key.
- Adjusted the generated EOS smoke config's explicit `interactive` timeout to `02:00:00`, matching EOS Slurm's `interactive` partition max time.
- Deleted the stale EOS DeepMath file, copied DFW `deepmath_verl.jsonl` and `comp_math_verl.jsonl` to EOS, and verified exact checksum/line-count matches for both files.
- Started the EOS interactive DeepMath/CompMath SV-SDPO CAA smoke as SLURM job `5337181`; the generated sbatch uses the `onelogger` VERL image symlink and `#SBATCH --time=0-02:00:00`.
- Checked EOS smoke job `5337181`; it failed after Ray and dataset initialization because vLLM required `55.38 GiB` at `gpu_memory_utilization=0.7` but only about `50.38-53.16 GiB` was free on the allocated GPUs. This confirms the earlier one-logger/Ray CLI issue was avoided.
- Checked DFW Chemistry CAA SV-OPSD LR `5e-5` job `12212015`; it failed during Ray cluster startup with a node-registration timeout before trainer startup. The later LR `5e-5` layer-range `0.2-0.6` job `12213198` remains the active replacement run.

## 2026-05-28

- Checked live DFW and EOS queues; both had no active jobs for `siddjain`.
- Checked DFW accounting for the active Chemistry and DeepMath runs. Chemistry SV-OPSD CAA jobs `12212000`, `12213198`, and `12214042` completed but their latest validation sample accuracy was `0.0`; the vanilla SDPO 16K job `12213113` failed because the appended conditioning sequence reached `21284` tokens, above `max_length=18432`.
- Checked DeepMath/CompMath DFW accounting and validation outputs. GRPO `12215167` completed with latest pass@1 AIME24/AIME25/HMMT of `0.487500`/`0.345833`/`0.280612`; vanilla SDPO `12215171` timed out with zero latest validation pass@1; SV-SDPO CAA `12215183` timed out with latest pass@1 `0.466667`/`0.300000`/`0.248724`.

## 2026-05-28

- Added an OPSD advantage-shaping config path in `/home/siddjain/workspace/verl/verl_svopsd`:
  - `algorithm.opsd.advantage_shaping.enable`
  - centered token-correctness shaping with configurable scale, normalization, clipping, distill-mask usage, and sign-flip allowance
  - validation requiring `opsd.mode=opsd_rlvr` and `opsd.mix_weight=1.0` when shaping is enabled
- Added centered-correctness advantage redistribution using `-distill_token_loss` as token correctness, preserving each response's total GRPO advantage while giving higher-correctness tokens higher shaped advantages.
- Integrated the shaping path into OPSD actor policy updates so the shaped advantage trains the student/original-prompt policy branch rather than directly optimizing the absolute OPSD distillation loss.
- Allowed SV-SDPO CAA steering with `opsd_rlvr` only when advantage shaping is enabled, so steering vectors can score tokens while the student prompt remains unchanged.
- Extended the DFW interactive Chemistry smoke submitter to run `sdpo_advshape` and `svsdpo_caa_advshape` variants and to check for nonzero advantage-shaping activity.
- Fixed the shaping-only scoring forward to run without gradient tracking and with detached support state when `mix_weight=1.0`, after the first DFW vanilla SDPO shaping smoke exposed an Adam foreach dtype/device optimizer failure.
- Reran the vanilla SDPO advantage-shaping Chemistry smoke on DFW interactive as SLURM job `12250448`; it completed with nonzero shaping activity and no optimizer dtype/device failure.
- Ran the SV-SDPO CAA advantage-shaping Chemistry smoke on DFW interactive as SLURM job `12250818`; it completed with nonzero steering and advantage-shaping activity.
- Extended the reusable DeepMath/CompMath DFW launcher under `~/data` to support `sdpo_rlvr` and `svsdpo_rlvr_caa` variants, explicit train/validation sampling temperatures, validation top-p, and a W&B id override for faithful GRPO resumes.
- Submitted the DFW DeepMath/CompMath SDPO-RLVR job `12255932` and SV-SDPO-RLVR CAA layer `0.2-0.6` job `12256035`, both with LR `5e-6`, 4 batch nodes, 8 GPUs per node, and train/validation temperature `1.0`.
- Submitted GRPO resume job `12256113` for the previous DeepMath/CompMath GRPO run, preserving the original experiment name, output path, W&B id, train temperature `0.85`, and validation temperature `0.6`.
- Cancelled DFW jobs `12255932`, `12256035`, and `12256113` before they started because they used the DeepMath launcher's existing `Qwen3-30B-A3B` default instead of the intended Qwen3-8B model.
- Updated the reusable DeepMath/CompMath DFW launcher to derive experiment-name model tags from `ACTOR_MODEL`, so Qwen3-8B submissions no longer produce `qwen3_30b_a3b` experiment names when `ACTOR_MODEL=/hf_models/Qwen3-8B` is passed.
- Submitted corrected DFW Qwen3-8B DeepMath/CompMath SDPO-RLVR job `12257178` and SV-SDPO-RLVR CAA layer `0.2-0.6` job `12257181` with LR `5e-6`, train temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`, validation samples `16`, and PPO minibatch size `32`.
- Did not submit a Qwen3-8B GRPO resume because no compatible Qwen3-8B GRPO checkpoint path was found; the earlier GRPO resume target was a Qwen3-30B-A3B run and cannot be resumed as Qwen3-8B.
- Tried direct AWS-IAD status via `ssh aws-iad`, but hostname resolution failed; no fallback route was used.

## 2026-05-29

- Submitted a matched pure GRPO Qwen3-8B DeepMath/CompMath DFW batch run as SLURM job `12275498`, using the same relevant parameters as the active Qwen3-8B `SDPO_RLVR` run: LR `5e-6`, train temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`, validation samples `16`, 8K response length, 4 nodes, 8 GPUs per node, rollout TP `4`, PPO minibatch size `32`, LR warmup `10`, weight decay `0.01`, constant LR scheduler, and `algorithm.norm_adv_by_std_in_grpo=False`.
- Fixed the DeepMath/CompMath DFW launcher so non-RLVR `svsdpo_caa` experiment names use the actual `LAYER_FRACTIONS` label instead of always hardcoding `l31_37`.
- Submitted a Qwen3-8B DeepMath/CompMath non-RLVR SV-SDPO CAA DFW batch run as SLURM job `12277118`, using `opsd` mode, CAA steering layers `0.2-0.6`, LR `5e-6`, train temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`, validation samples `16`, and no RLVR advantage shaping.
- Added `algorithm.opsd.steering.source_mode`, defaulting to `caa`, with `positive` mode for the non-CAA SV-SDPO baseline.
- Updated SDPO steering source construction so `source_mode=positive` gathers all eligible same-prompt correct rollouts, marks them with positive signs only, and leaves the prompt unchanged.
- Updated steering-vector extraction so CAA remains `mean_positive_residual - mean_negative_residual`, while positive-only mode applies `mean_positive_residual`.
- Added focused regression coverage for positive-only steering config, source tensor construction, and vector extraction.
- Updated the DeepMath DFW submitter with an `svsdpo` variant and explicit final Hydra overrides for actor LR, train temperature, validation temperature/top-p, validation sample count, and rollout TP.
- Submitted corrected pure GRPO restart job `12299551` and non-RLVR positive-only SV-SDPO job `12299555` on DFW with Qwen3-8B, DeepMath train, CompMath validation, LR `5e-6`, generation temperature `1.0`, validation temperature `0.6`, validation top-p `0.95`, validation samples `16`, and response length `8K`.
- Cancelled positive-only SV-SDPO job `12299555` after the user requested stopping it.
- Recorded that GRPO restart job `12299551` failed independently with a rollout weight-update bucket assertion requiring a bucket larger than `2048 MB` for `model.embed_tokens.weight`.
- Fixed the DeepMath DFW submitter so `++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096` is part of the common args used by GRPO as well as OPSD variants.
- Resubmitted the Qwen3-8B DeepMath/CompMath pure GRPO run with the bucket fix as DFW SLURM job `12300861`; initial scheduler state was `PENDING` with reason `Resources`.
- Added RLSD configuration and advantage shaping in `/home/siddjain/workspace/verl/verl_svopsd`: configurable privileged source (`ground_truth_answer` or `correct_rollout`), multiplicative teacher/student sampled-logprob evidence weights, lambda decay, clipping, and active/fallback metrics.
- Added periodic fixed-teacher synchronization for RLSD so `teacher_model=fixed` can be used in `opsd_rlvr` mode when `algorithm.opsd.rlsd.enable=True`.
- Extended the DFW Chemistry interactive smoke launcher with `rlsd_gt` and `rlsd_rollout` variants, and extended the DeepMath DFW submitter with matching production variants.
- Ran the DFW interactive `rlsd_gt` Chemistry smoke as SLURM job `12302026`; it completed with active RLSD metrics (`actor/rlsd_active_rate=1.0`, `actor/rlsd_token_rate=1.0`, `actor/rlsd_no_privileged_fallback_rate=0.0`).
- Ran the DFW interactive `rlsd_rollout` Chemistry smoke as SLURM job `12302510`; it completed with active correct-rollout RLSD metrics (`actor/rlsd_active_rate=0.3125`, `actor/rlsd_token_rate=0.3125`, `actor/rlsd_no_privileged_fallback_rate=0.6875`).
- Submitted DFW Qwen3-8B DeepMath/CompMath RLSD jobs under run tag `dfw_deepmath_qwen3_8b_rlsd_20260529_134546`: `rlsd_rollout` job `12303531` with `N_PROMPTS=32`, `N_SAMPLES=8`; and `rlsd_gt` job `12303568` with `N_PROMPTS=256`, `N_SAMPLES=1`.
- Cancelled active non-vanilla-GRPO DFW jobs `12303531` (`rlsd_rollout`) and `12303568` (`rlsd_gt`), leaving vanilla GRPO job `12300861` running.
- Added `algorithm.opsd.advantage_shaping.max_response_tokens`, default `1024`, so SDPO_RLVR/SV-SDPO_RLVR CAA advantage shaping can be limited to the first K response tokens without capping distillation-token scoring.
- Updated centered-correctness advantage shaping so unchanged tokens outside the shaping mask keep their original GRPO advantages and the primary conservation metric checks the total advantage over the shaped token region; added a separate response-total conservation metric.
- Extended the DFW interactive Chemistry smoke submitter to pass `ADV_SHAPING_MAX_RESPONSE_TOKENS` through to SDPO_RLVR and SV-SDPO_RLVR CAA advantage-shaping variants.
- Ran the DFW interactive SDPO_RLVR prefix-cap smoke as SLURM job `12307521` with `ADV_SHAPING_MAX_RESPONSE_TOKENS=16`; it completed with nonzero shaping activity and prefix conservation (`actor/advantage_shaping_token_rate=0.001708984375`, `actor/advantage_shaping_total_error_max=4.76837158203125e-07`).
- Ran the DFW interactive SV-SDPO_RLVR CAA prefix-cap smoke as SLURM job `12307762` with layers `0.2-0.6` and `ADV_SHAPING_MAX_RESPONSE_TOKENS=16`; it completed with nonzero steering and shaping activity (`actor/opsd_steering_active_rate=0.125`, `actor/advantage_shaping_token_rate=0.0009765625`, `actor/advantage_shaping_total_error_max=9.5367431640625e-07`).
- Updated the DeepMath/CompMath DFW submitter under `~/data` to pass `ADV_SHAPING_MAX_RESPONSE_TOKENS` explicitly into `algorithm.opsd.advantage_shaping.max_response_tokens` for `sdpo_rlvr` and `svsdpo_rlvr_caa` production runs, defaulting to `1024`.
- Submitted DFW Qwen3-8B DeepMath/CompMath prefix-1024 SDPO_RLVR jobs under run tag `dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`: SDPO_RLVR job `12313004` and SV-SDPO_RLVR CAA layers `0.2-0.6` job `12313042`, both with explicit `algorithm.opsd.advantage_shaping.max_response_tokens=1024`.
