# GooseOmni v1 Benchmark

This directory is the official tracked benchmark package for GooseOmni v1. It is derived from the pass248 death-skill scope audit and organized as a standard benchmark release rather than an experimental run folder.

## Layout

```text
public/       Model-facing inputs and public metadata. No scorer-only answers.
private/      Gold and hidden gold used only by local scorers.
tables/       Human-readable metric summaries and CSV tables.
manifest.json File hashes, line counts, and public leak-scan results.
```

## Public Inputs

- `public/leaderboard_core/trials.jsonl`: structured leaderboard core, 889 trials.
- `public/leaderboard_core/interactive_prompts.jsonl`: A/B/C/D public prompts.
- `public/leaderboard_core/probe_groups_public.jsonl`: public probe group metadata, 276 groups.
- `public/static_trials/trials.jsonl`: static benchmark trials.
- `public/interactive_diagnostics/prompts.jsonl`: interactive diagnostic prompts.
- `public/raw_video_smoke/raw_video_smoke.jsonl`: 48 raw-video smoke trials.
- `public/agentic_midgame_prediction/trials.jsonl`: 160 supplementary agentic midgame trials.
- `public/agentic_midgame_prediction/scoring_rules.json`: public scoring rule descriptions for the agentic track.

## Supplementary Agentic Track

`agentic_midgame_prediction` evaluates the 5-human + 1-omni middle layer: from one ego POV, infer hidden world state, predict another player's offscreen current action, predict future behavior within a bounded window, and distinguish global claim truth from ego-local knowability. This track uses scorer-private hidden POV and future-outcome gold and is reported separately from the Decrypto-style `leaderboard_core`.

## Scorer-Private Files

`private/` contains scorer-only gold and hidden gold. These files are intentionally tracked for local scoring and benchmark maintenance, but they must never be passed to an evaluated model.

## Validation Snapshot

```text
public_file_leak_hits                              []
public/leaderboard_core/trials.jsonl               889
private/leaderboard_core/hidden_gold.jsonl         889
public/raw_video_smoke/raw_video_smoke.jsonl        48
public/agentic_midgame_prediction/trials.jsonl  160
private/agentic_midgame_prediction/hidden_gold.jsonl 160
public/leaderboard_core/probe_groups_public.jsonl  276
```

## Local Results

Model responses, run logs, and local model pass scores are ignored by Git. They are not part of this benchmark release.
