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

## 2026-08-12 pre-publication verification artifacts

- `/home/siddjain/data/smoke_tests/svopsd_openthoughts/container_tests_20260812_precommit_final_v2/pytest.log` — exact-tree OPSD test log for job `15607436` (`174 passed`).
- `/home/siddjain/data/smoke_tests/longest_success_penalty_reward/container_tests_20260812_precommit_final_v1/pytest.log` — grouped-reward test log for job `15607477` (`21 passed`).
- `/home/siddjain/data/smoke_tests/shortest_success_reward/container_tests_20260812_precommit_final_v1/pytest.log` — grouped-reward and resume-control test log for job `15607500` (`30 passed`).

# Paths

## Created

- `/home/siddjain/workspace/verl/verl_main/TRACKING/CHANGES.md`
- `/home/siddjain/workspace/verl/verl_main/TRACKING/WORKFLOWS.md`
- `/home/siddjain/workspace/verl/verl_main/TRACKING/PATHS.md`

## Planned

- `/home/siddjain/workspace/verl/verl_svopsd`

## Created During SV-OPSD

- `/home/siddjain/workspace/verl/verl_svopsd`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/steering.py`
- `/home/siddjain/workspace/verl/verl_svopsd/tests/recipe/opsd/test_steering.py`
- `/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/svopsd_chemistry_smoke.py`
- `/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh`
- `/home/siddjain/workspace/verl/verl_svopsd/scripts/submit_chemistry_sdpo_svopsd_dfw.sh`
- `/home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry`
- `/home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry/chemistry_train_tiny.jsonl`
- `/home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry/chemistry_test_tiny.jsonl`
- `/home/siddjain/data/verl_svopsd_smoke/sciknoweval_chemistry/svopsd_chemistry_smoke_summary.json`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/20260527_004123`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/20260527_004123/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/20260527_004123/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/20260527_004123/svopsd_first/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704/grpo/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_svsdpo_positive_20260529_121704/svsdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/20260527_004123/svopsd_all/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd_eos/config/eos.yaml`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/eos_20260527_005427/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/eos_20260527_005824`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/eos_20260527_005824/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/eos_20260527_005824/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/eos_20260527_005824/svopsd_first/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/eos_20260527_005824/svopsd_all/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd_aws_iad/config/aws-iad.yaml`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/sciknoweval_l3/chemistry_train.jsonl`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/sciknoweval_l3/chemistry_test.jsonl`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_eos_20260527_005824`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_first_qwen3_8b_l31_37_4n_eos_20260527_005824`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_all_qwen3_8b_l31_37_4n_eos_20260527_005824`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/llm/data/rl/sciknoweval_l3/chemistry_train.jsonl`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/llm/data/rl/sciknoweval_l3/chemistry_test.jsonl`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/my_models/Qwen3-8B`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/containers/nemo-skills-verl-latest-onelogger.sqsh`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_20260527_020811`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_noinstall_20260527_021548`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_noinstall_20260527_021548/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_noinstall_20260527_021548/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_noinstall_20260527_021548/svopsd_first/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_noinstall_20260527_021548/svopsd_all/submit.log`
- `/home/siddjain/data/python_test_deps/verl_svopsd_caa`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/svopsd_caa_chemistry_dfw_interactive_20260527_090347`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/svopsd_caa_chemistry_dfw_interactive_20260527_090347/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svopsd_caa_chemistry_dfw_interactive_20260527_090347`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_aws_iad_noinstall_20260527_021548`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_first_qwen3_8b_l31_37_4n_aws_iad_noinstall_20260527_021548`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_all_qwen3_8b_l31_37_4n_aws_iad_noinstall_20260527_021548`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_code/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_20260527_111659`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_20260527_111659/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_20260527_111659/svopsd_caa/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_caa_20260527_111659`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_caa_20260527_111659/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/aws_iad_caa_20260527_111659/svopsd_caa/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr1e4_20260527_150451`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr1e4_20260527_150451/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr1e4_20260527_150451/svopsd_caa/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr5e5_20260527_150517`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr5e5_20260527_150517/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr5e5_20260527_150517/svopsd_caa/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_sdpo_lr1e6_20260527_151110`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_sdpo_lr1e6_20260527_151110/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_sdpo_lr1e6_20260527_151110/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_sdpo_lr1e5_resp16k_20260527_153241`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_sdpo_lr1e5_resp16k_20260527_153241/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_sdpo_lr1e5_resp16k_20260527_153241/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr5e5_l20_60_20260527_153958`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr5e5_l20_60_20260527_153958/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr5e5_l20_60_20260527_153958/svopsd_caa/submit.log`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr1e5_l20_60_20260527_161839`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr1e5_l20_60_20260527_161839/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/chemistry_sdpo_svopsd/dfw_caa_lr1e5_l20_60_20260527_161839/svopsd_caa/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/scripts/submit_deepmath_grpo_sdpo_svopsd_dfw.sh`
- `/home/siddjain/workspace/verl/verl_main/smoke_tests/submit_deepmath_compmath_eos_interactive.sh`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd_eos/config_onelogger/eos.yaml`
- `/home/siddjain/data/smoke_tests/deepmath_compmath_eos/deepmath_compmath_svsdpo_caa_eos_interactive_eos_deepmath_compmath_smoke_20260527_180554`
- `/home/siddjain/data/smoke_tests/deepmath_compmath_eos/deepmath_compmath_svsdpo_caa_eos_interactive_eos_deepmath_compmath_smoke_20260527_180554/submit.log`
- `/home/siddjain/data/smoke_tests/deepmath_compmath_eos/deepmath_compmath_svsdpo_caa_eos_interactive_eos_deepmath_compmath_smoke_20260527_180554/submitted_jobs.tsv`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/deepmath_verl.jsonl`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/verl_vllm012_flashattn_20260321_onelogger.sqsh`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output/smoke_tests/deepmath_compmath_eos/deepmath_compmath_svsdpo_caa_eos_interactive_eos_deepmath_compmath_smoke_20260527_180554`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000/grpo/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000/sdpo/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000/svsdpo_caa/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/sdpo_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/svsdpo_caa_qwen3_30b_a3b_deepmath_compmath_resp8k_l31_37_lr5e6_dfw_deepmath_sdpo_svopsd_20260527_170000`
- `eos:/home/siddjain/lustre/containers/verl_vllm012_flashattn_20260321.sqsh`
- `eos:/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_20260527_111659`
- `aws-iad-1:/lustre/fsw/portfolios/nemotron/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_aws_iad_caa_20260527_111659`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr1e4_20260527_150451`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr5e5_20260527_150517`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_dfw_sdpo_lr1e6_20260527_151110`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_sdpo_qwen3_8b_4n_dfw_sdpo_lr1e5_resp16k_20260527_153241`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr5e5_l20_60_20260527_153958`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/chemistry_sdpo_svopsd/chemistry_svopsd_caa_qwen3_8b_l31_37_4n_dfw_caa_lr1e5_l20_60_20260527_161839`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_rlvr_20260528_131831`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_rlvr_20260528_131831/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_rlvr_20260528_131831/sdpo_rlvr/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_rlvr_20260528_131831/svsdpo_rlvr_caa/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000_resume_20260528_131936`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000_resume_20260528_131936/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_sdpo_svopsd_20260527_170000_resume_20260528_131936/grpo/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/sdpo_rlvr_qwen3_30b_a3b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_rlvr_20260528_131831`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/deepmath_sdpo_svopsd/svsdpo_rlvr_caa_qwen3_30b_a3b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_rlvr_20260528_131831`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlvr_20260528_140606`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlvr_20260528_140606/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlvr_20260528_140606/sdpo_rlvr/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlvr_20260528_140606/svsdpo_rlvr_caa/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/sdpo_rlvr_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlvr_20260528_140606`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_rlvr_20260528_140606`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_110351`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_110351/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_110351`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_111451`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_111451/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/sdpo_advshape_chemistry_dfw_interactive_20260528_111451`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_chemistry_dfw_interactive_20260528_112235`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_chemistry_dfw_interactive_20260528_112235/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_chemistry_dfw_interactive_20260528_112235`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_20260529_011018`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_20260529_011018/grpo/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_20260529_011018/submitted_jobs.tsv`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_20260529_011018`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834/svsdpo_caa/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834/submitted_jobs.tsv`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_svsdpo_caa_20260529_021834`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603/grpo/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/grpo_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_grpo_bucket4096_20260529_124603`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/rlsd_gt_chemistry_dfw_interactive_20260529_131445`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/rlsd_gt_chemistry_dfw_interactive_20260529_131445/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/rlsd_gt_chemistry_dfw_interactive_20260529_131445`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/rlsd_gt_chemistry_dfw_interactive_20260529_131445/training-logs/main_rlsd_gt_chemistry_dfw_interactive_20260529_131445-ppo-0_12302026_srun.log`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/rlsd_rollout_chemistry_dfw_interactive_20260529_132521`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/rlsd_rollout_chemistry_dfw_interactive_20260529_132521/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/rlsd_rollout_chemistry_dfw_interactive_20260529_132521`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/rlsd_rollout_chemistry_dfw_interactive_20260529_132521/training-logs/main_rlsd_rollout_chemistry_dfw_interactive_20260529_132521-ppo-0_12302510_srun.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlsd_20260529_134546`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlsd_20260529_134546/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlsd_20260529_134546/rlsd_rollout/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_rlsd_20260529_134546/rlsd_gt/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/rlsd_rollout_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlsd_20260529_134546`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/rlsd_gt_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_rlsd_20260529_134546`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214/training-logs/main_sdpo_advshape_prefixcap_chemistry_dfw_interactive_20260529_151214-ppo-0_12307521_srun.log`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953`
- `/home/siddjain/data/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/smoke_tests/svopsd_chemistry/svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953/training-logs/main_svsdpo_caa_advshape_prefixcap_chemistry_dfw_interactive_20260529_151953-ppo-0_12307762_srun.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053/sdpo_rlvr/submit.log`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053/svsdpo_rlvr_caa/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/sdpo_rlvr_qwen3_8b_deepmath_compmath_resp8k_lr5e6_dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_dfw_deepmath_qwen3_8b_sdpo_rlvr_prefix1024_20260529_171053`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/p1024caa_0529_1736`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/p1024caa_0529_1736/submitted_jobs.tsv`
- `/home/siddjain/data/verl_runs/deepmath_sdpo_svopsd/p1024caa_0529_1736/svsdpo_rlvr_caa/submit.log`
- `dfw:/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/mathgen/svsdpo_rlvr_caa_qwen3_8b_deepmath_compmath_resp8k_l2_6_lr5e6_p1024caa_0529_1736`

## Modified During SV-OPSD

- `/home/siddjain/workspace/verl/verl_svopsd/verl/trainer/config/algorithm.py`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/teacher_utils.py`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/opsd_trainer.py`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/dp_actor.py`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/opsd_loss.py`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/main_opsd.py`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/config/opsd_trainer.yaml`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/opsd/config/opsdv2_trainer.yaml`
- `/home/siddjain/workspace/verl/verl_svopsd/smoke_tests/submit_svopsd_chemistry_dfw_interactive.sh`
- `/home/siddjain/workspace/verl/verl_svopsd/recipe/sciknoweval/reward_fn_sciknoweval.py`
- `/home/siddjain/workspace/verl/verl_svopsd/tests/recipe/opsd/test_teacher_utils.py`
- `/home/siddjain/workspace/verl/verl_svopsd/tests/recipe/opsd/test_opsd_config.py`
- `/home/siddjain/workspace/verl/verl_svopsd/tests/recipe/opsd/test_opsd_loss.py`
- `/home/siddjain/workspace/verl/verl_svopsd/tests/recipe/opsd/test_opsd_trainer.py`
- `/home/siddjain/workspace/verl/verl_svopsd/tests/recipe/opsd/test_dp_actor.py`
