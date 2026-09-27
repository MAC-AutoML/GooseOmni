# GooseOmni

GooseOmni is a trajectory-derived Theory-of-Mind benchmark built from a six-POV aligned Goose Goose Duck replay. It evaluates whether an omni model can separate oracle truth from each player perspective, reconstruct pre-reveal false beliefs, and predict strategic communication effects.

## Benchmark Question

```text
Given a real multi-agent social-game trajectory, can a model distinguish:
1. what globally happened;
2. what a specific player saw or heard at a cutoff;
3. what that player did not see, but another player did;
4. whether a heard claim conflicts with oracle events;
5. whether the target player can know that conflict;
6. what false or incomplete belief existed before reveal;
7. whether the model can reconstruct that prior belief after truth is revealed;
8. how one player should expect another player to interpret a strategic utterance.
```

## Project Layout

```text
benchmark/gooseomni_v1/           Frozen historical benchmark
benchmark/gooseomni_v2/           Current model-verified benchmark
src/gooseomni/data_pipeline/      Declarative data build and provenance stages
src/gooseomni/evaluation/         Unified tracks, runner, scoring, and reports
src/gooseomni/benchmark/          ToM schemas, builders, validators, and scorers
src/gooseomni/annotation/         Aligned-video annotation algorithms reused by the data pipeline
tools/                            Build, annotation, evaluation, packaging, and validation CLIs
configs/slurm/                    HPC job templates and runtime examples
src/gooseomni/models/              Local model server and client integrations
docs/                             Benchmark design and annotation documentation
tests/                            Unit and release validation tests
```

The repository root is intentionally small. Generated annotations, local runs, raw videos, and model responses stay outside Git.

## Benchmark Data

Current benchmark version (v1 remains frozen):

```text
benchmark/gooseomni_v2/
```

Important files:

| file | content |
|---|---|
| `public/leaderboard_core/trials.jsonl` | Structured leaderboard core, 889 public trials. |
| `public/leaderboard_core/interactive_prompts.jsonl` | A/B/C/D public prompts for interactive diagnostics. |
| `public/leaderboard_core/probe_groups_public.jsonl` | Public probe-group metadata with internal review fields removed. |
| `public/raw_video_smoke/raw_video_smoke.jsonl` | 48 raw-video smoke trials. |
| `public/agentic_midgame_prediction/trials.jsonl` | 160 supplementary POV-to-oracle and future-behavior prediction trials. |
| `private/agentic_midgame_prediction/hidden_gold.jsonl` | Scorer-only hidden POV and future-outcome gold for the agentic track. |
| `private/leaderboard_core/gold.jsonl` | Scoring metadata. |
| `private/leaderboard_core/hidden_gold.jsonl` | Scorer-only answers. Never put this file into model prompts. |
| `tables/README.md` | Human-readable metric and stratified tables. |
| `manifest.json` | File list, SHA-256 hashes, line counts, and public-file leak scan. |

Core counts:

```text
leaderboard_core trials          889
leaderboard_core hidden gold     889
raw-video smoke trials            48
agentic midgame trials          160
public probe groups              276
public_file_leak_hits             []
```

## Diagnostic Design

GooseOmni follows a Decrypto-style diagnostic-generation pipeline:

```text
6-POV aligned replay trajectory
-> oracle trajectory ledger
-> visibility / belief / memory projection
-> grouped A/B/C/D ToM probes
-> leaderboard trials
-> controlled scoring
```

The storage unit is an aligned video segment or phase clip. The benchmark unit is a trajectory node:

```text
cutoff_abs_sec + target_player + query_variable + probe_type
```

Probe groups are generated around information gaps, private observations, verifiable claims, delayed public reveals, and strategic communication events.

## Benchmark Tracks

| track | purpose |
|---|---|
| `leaderboard_core` | Decrypto-style A/B/C/D Theory-of-Mind diagnostics. |
| `raw_video_smoke` | Small omni video-input smoke track. |
| `agentic_midgame_prediction` | Supplementary 5-human + 1-omni middle-layer track for hidden-state estimation, offscreen action prediction, future behavior prediction, and deception-state inference. |

## Probe Types

| probe type | purpose |
|---|---|
| `A_pre_reveal_belief` | Ask what the target player can believe before truth reveal, using only target-available evidence. |
| `B_post_reveal_reconstruct_previous_belief` | Reveal oracle truth, then ask the model to reconstruct the target player earlier belief without hindsight leakage. |
| `C_other_agent_false_belief` | Ask what another player would believe before reveal, given that player limited information. |
| `D_perspective_taking_prediction` | Ask how a speaker should expect a listener to interpret a strategic claim or accusation. |

## Metrics

| metric | meaning |
|---|---|
| `RC_weak` / `RC_strong` | Representational-change consistency after reveal. |
| `FB_weak` / `FB_strong` | False-belief recognition and reconstruction. |
| `PT_weak` / `PT_strong` | Perspective-taking and strategic-communication prediction. |
| `claim_verification_global` | Whether a claim is globally supported, contradicted, or unresolved. |
| `claim_verification_local` | Whether the target player can know the truth status of the claim. |
| `perspective_leakage_rate` | Whether a response cites information unavailable to the simulated player. |
| `forbidden_evidence_usage_rate` | Whether hidden scorer-only evidence is used in the answer. |
| `death_skill_overclaim_rate` | Whether visible death/body/blood evidence is overclaimed as killer, role, alignment, skill, or mechanism. |
| `evidence_support_rate` | Whether cited evidence is compatible with the allowed input condition. |
| `json_parse_success` / `schema_validation_success` | Output reliability and schema compliance. |

Aggregate score:

```text
SoG-ToM-Core =
  0.20 * RC_strong
+ 0.20 * FB_strong
+ 0.20 * PT_strong
+ 0.15 * claim_verification_local
+ 0.10 * evidence_support_rate
+ 0.10 * (1 - perspective_leakage_rate)
+ 0.05 * (1 - death_skill_overclaim_rate)
```

## Agentic Midgame Metrics

The supplementary agentic track reports `hidden_world_state_accuracy`, `offscreen_action_accuracy`, `next_behavior_top1_accuracy`, `next_behavior_topk_accuracy`, `deception_detection_global`, `deception_awareness_local`, `oracle_overclaim_rate`, `pov_evidence_support_rate`, and `death_skill_overclaim_rate`. Its aggregate is `SoG-Agentic-Midgame`; it is reported separately from `SoG-ToM-Core`.

## Evaluation Tools

The supported public interface is the unified CLI:

```bash
gooseomni models list
gooseomni models check
gooseomni models serve qwen3_omni --host 127.0.0.1 --port 5090
gooseomni annotate run -- --backend mock --limit 3
gooseomni data validate-input --config configs/datasets/gooseomni_v2.yaml
gooseomni data build --config configs/datasets/gooseomni_v2.yaml --resume
gooseomni data status --run runs/gooseomni_v2
gooseomni data validate --run runs/gooseomni_v2
gooseomni eval plan --benchmark gooseomni_v2 --models all
gooseomni eval run --benchmark gooseomni_v2 --model qwen3_omni --track raw_video_smoke --modalities text v a av --limit 1 --resume
gooseomni eval score --benchmark gooseomni_v2 --run runs/eval/qwen3_smoke
gooseomni eval report --run runs/eval/qwen3_smoke
```

Slurm jobs call the same CLI. Files under `tools/` are internal implementation
entrypoints and are not a separate public API.

```text
configs/slurm/qwen3_omni_gooseomni_eval.slurm
```

The two former Qwen evaluation scripts remain only as deprecated thin forwards
to `gooseomni eval`; new automation should call the unified CLI directly.
The Slurm runner starts the same `/health` + `/v1/infer` service and does not
upload a placeholder video for text-only inference.

## Local-Only Artifacts

The following paths are intentionally ignored and should not be pushed:

```text
runs/
annotations/
data/source_videos/
results/*
tmp_pass*.py
work/
.codex_tmp_sync/
benchmark/* except benchmark/gooseomni_v1/ and benchmark/gooseomni_v2/
```

Model responses and raw run scores are not part of the clean benchmark. Local evaluation summaries may exist under ignored workspaces such as `runs/gooseomni_eval_qwen3omni_pass001/LOCAL_RESULTS.md`.

## Validation

Recommended pre-push checks:

```bash
find src tests tools -type f -name "*.py" ! -name "._*" -print0 | xargs -0 .venv/bin/python -m py_compile
bash -n configs/slurm/*.slurm configs/slurm/*.sh
.venv/bin/ruff check src/gooseomni/data_pipeline src/gooseomni/evaluation
.venv/bin/python -m pytest -q
uv lock --check --offline
```

## Citation

This repository is being organized as the GooseOmni benchmark foundation. Add the paper citation here once the benchmark paper metadata is finalized.
