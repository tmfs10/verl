<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: 96vw !important;
}
table { width: 100% !important; }
</style>

# OCI-IAD intermediate-MC topology benchmark

This workflow finds separate throughput-optimal synchronous topologies for
Qwen3-1.7B and Qwen3-4B on the OPSD math 30K dataset. It measures both
`scalar_random` and `beta_variance`, each with M4 self-critiques and with the
matched M0 no-self-critique baseline.

M0 does **not** disable the feature. It keeps intermediate MC enabled, uses one
unconditioned critic stream, selects one mark, requests one continuation, and
trains the solution actor and critic. It only removes critique generation and
critique actor rows. Vanilla feature-disabled VeRL is outside this benchmark.

The training loop remains synchronous. VeRL's existing internal vLLM request
machinery may execute independent child requests concurrently behind the
blocking iteration barrier; this benchmark does not add an asynchronous
trainer, payload queue, critic, or optimizer path.

## Fixed workload contract

| Property | Value |
|---|---:|
| Prompts per iteration | 64 |
| Rollouts per prompt | 8 |
| Solutions per iteration | 512 |
| Maximum prompt tokens | 4096 |
| Maximum solution tokens | 8192 |
| Maximum critique tokens | 8192 |
| Critiques | 0 or 4 |
| Continuations per mark | 1 |
| Marks per solution | 1 |
| Generation temperature | 1.0 |
| Initial coarse run | 1 stabilization + 2 measured iterations |
| Finalist repeat | 3 seeds, each 1 stabilization + 4 measured iterations |

The production training file is pinned to 29,427 rows and SHA-256
`f79a42fe155218db2f1927ee903afd101929724f2d0516352bdbb91cdb139178`.
The measured entrypoint verifies that checksum and both model `config.json`
files before allocating training work.

## Search stages

1. `two-node-core` runs all twelve topology families for scalar/random. Beta/
   variance begins with seven families: the four principal FSDP2 sharding/SP
   anchors, both rollout-TP2 anchors, and one legacy-FSDP anchor. This first
   wave contains 76 candidates across both models and M0/M4.
2. `batching-expansion` applies fixed batching and two higher-throughput
   dynamic/vLLM profiles only to leading core families. Expand Beta to the
   omitted families if a selected point lies at a boundary, the top points are
   within 3%, continuation generation exceeds 20% of the iteration, or Beta's
   bottleneck differs from scalar's.
3. `finalists` repeats each workload winner over three seeds. Do not combine
   M0 and M4 when choosing winners.
4. `four-node` promotes at most the best two families per workload. Each source
   topology is tested once with its two-node FSDP shard sizes preserved and
   once with those groups doubled. Select four nodes only when median raw
   iterations/hour improves by at least 5% without a token-volume rejection.
5. P05 disables gradient checkpointing and reshard-after-forward. It is
   deliberately launch-gated; use it only after a prior measured candidate
   reports enough memory headroom, and pass `--allow-memory-gated` explicitly.

The primary metric is `3600 / median(measured timing_s/step)`. Secondary
metrics include p95 step time, generation tokens/hour, GPU-hours/iteration,
phase fractions, and peak reserved memory. Any run with an OOM, engine error,
wrong hardware, incomplete optimizer updates, changed workload, missing mark
work, or non-finite metric is rejected. A candidate whose generated output
volume differs by more than 10% from its matched workload/seed group is marked
for rerun rather than credited as faster.

## Reproducible workflow

Runtime artifacts must stay outside the repository:

```bash
cd /home/siddjain/workspace/verl/verl_intermediate_mc_value_model

RUN_TAG=imc-topology-$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=/home/siddjain/data/intermediate_mc_value_model/verl/topology/$RUN_TAG
mkdir -p "$RUN_ROOT"

python3 -m smoke_tests.intermediate_mc_value.topology.matrix two-node-core \
  --seed 1234 \
  --output "$RUN_ROOT/two_node_core.jsonl"

python3 -m smoke_tests.intermediate_mc_value.topology.submit_oci_iad render \
  --manifest "$RUN_ROOT/two_node_core.jsonl" \
  --run-tag "$RUN_TAG" \
  --local-run-dir "$RUN_ROOT/core"
```

`render` is completely local. Inspect `rendered_commands.jsonl`, then run the
launcher's real remote dry-run. The successful dry-run marker is bound to the
manifest hash, clean pushed VeRL revision, exact per-candidate command hash,
and memory-gate setting, so changed commands cannot be submitted using stale
evidence.

```bash
python3 -m smoke_tests.intermediate_mc_value.topology.submit_oci_iad dry-run \
  --manifest "$RUN_ROOT/two_node_core.jsonl" \
  --run-tag "$RUN_TAG" \
  --local-run-dir "$RUN_ROOT/core"

python3 -m smoke_tests.intermediate_mc_value.topology.submit_oci_iad submit \
  --manifest "$RUN_ROOT/two_node_core.jsonl" \
  --run-tag "$RUN_TAG" \
  --local-run-dir "$RUN_ROOT/core" \
  --max-candidates 8
```

Use `--candidate-id ID` repeatedly or `--max-candidates N` to submit bounded
waves. A bare whole-manifest submission is rejected unless `--submit-all` is
given explicitly. A successful whole-manifest dry-run may authorize later
bounded subsets because each candidate has its own command hash. The submitter
appends `jobs.tsv` after each successful submission and skips already-recorded
candidates on resume. Two-node jobs use OCI-IAD's batch-plus-interactive
eligibility; four-node jobs remain batch-only. Every job requests exclusive
full nodes, all eight GPUs per node, four hours, and `Requeue=0`. W&B,
validation, checkpoints, detailed audit JSONL, and rollout dumps are disabled
in the measurement window.

The installed NeMo Run wrapper may invoke `sbatch --requeue` even when its
generated script contains `#SBATCH --no-requeue`. Immediately after receiving
the job ID, the submitter therefore sets `Requeue=0`, verifies it from live
`scontrol` output, and records the scheduler contract. If either operation
fails, it cancels the job before adding it to the local submission ledger.

Check state and collect completed benchmark artifacts through the pinned
`iad-2` route:

```bash
python3 -m smoke_tests.intermediate_mc_value.topology.submit_oci_iad status \
  --manifest "$RUN_ROOT/two_node_core.jsonl" \
  --run-tag "$RUN_TAG" \
  --local-run-dir "$RUN_ROOT/core"

python3 -m smoke_tests.intermediate_mc_value.topology.submit_oci_iad collect \
  --manifest "$RUN_ROOT/two_node_core.jsonl" \
  --run-tag "$RUN_TAG" \
  --local-run-dir "$RUN_ROOT/core"

python3 -m smoke_tests.intermediate_mc_value.topology.analyze \
  --manifest "$RUN_ROOT/two_node_core.jsonl" \
  --collected-root "$RUN_ROOT/core/collected" \
  --output-dir "$RUN_ROOT/core/report"
```

The analyzer writes `summary.json` and a wide-page `REPORT.md`. Use the leading
families from that report to generate the batching, finalist, and four-node
manifests:

```bash
python3 -m smoke_tests.intermediate_mc_value.topology.matrix batching-expansion \
  --workloads qwen3-1p7b-scalar_random-m0,qwen3-1p7b-scalar_random-m4 \
  --topologies T02,T05,T09 \
  --seed 1234 \
  --output "$RUN_ROOT/batching.jsonl"

python3 -m smoke_tests.intermediate_mc_value.topology.matrix finalists \
  --workload qwen3-1p7b-scalar_random-m0 \
  --topology T05 \
  --profile P03 \
  --seeds 1234,2345,3456 \
  --output "$RUN_ROOT/finalist.jsonl"

python3 -m smoke_tests.intermediate_mc_value.topology.matrix four-node \
  --workload qwen3-1p7b-scalar_random-m0 \
  --topologies T05,T09 \
  --profile P03 \
  --seed 1234 \
  --output "$RUN_ROOT/four_node.jsonl"

python3 -m smoke_tests.intermediate_mc_value.topology.matrix memory-expansion \
  --workload qwen3-1p7b-scalar_random-m0 \
  --topology T05 \
  --seed 1234 \
  --output "$RUN_ROOT/memory_gated.jsonl"

python3 -m smoke_tests.intermediate_mc_value.topology.matrix four-node-finalists \
  --workload qwen3-1p7b-scalar_random-m0 \
  --source-topology T05 \
  --variant expanded \
  --profile P03 \
  --seeds 1234,2345,3456 \
  --output "$RUN_ROOT/four_node_finalist.jsonl"
```

The submitter resolves `iad-2` to login node 02 and writes a data-owned
execution copy of `oci-iad.yaml` under each local run directory. That copy
changes `ssh_tunnel.host` and selects the shared OCI VeRL 0.7.0 image
(`vllm==0.8.5.post1`, `flash-attn==2.7.4.post1`, `torch==2.6.0`), whose varlen
kernel was exercised successfully on an A100-SXM4-80GB before topology runs;
the source/generated hashes and both explicit replacements are recorded in
`cluster_config/provenance.json`. The authoritative source YAML is never
modified. Do not substitute another OCI login alias or cluster route silently.
