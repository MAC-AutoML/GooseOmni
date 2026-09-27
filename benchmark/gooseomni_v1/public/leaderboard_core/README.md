# GooseOmni v1 Leaderboard Core

This directory is the paper/leaderboard-grade core split derived from the full GooseOmni v1 benchmark.

Inclusion policy:

- Include all A_pre_reveal_belief, B_post_reveal_reconstruct_previous_belief, and C_other_agent_false_belief trials.
- Include D_perspective_taking_prediction only when the diagnostic hidden gold has behavior_checked_public_scope=true.
- Exclude ungrounded D trials from leaderboard scoring until independent public behavior or response evidence is human verified.

Current counts:

{
  "behavior_checked_D_groups": 60,
  "benchmark_variant": "gooseomni_v1_leaderboard_core",
  "excluded_D_ungrounded": 84,
  "gold_source": "human_verified",
  "interactive_prompt_counts": {
    "A_pre_reveal_belief": 276,
    "B_post_reveal_reconstruct_previous_belief": 276,
    "C_other_agent_false_belief": 276,
    "D_perspective_taking_prediction": 60
  },
  "policy": "A/B/C are included; D_perspective_taking_prediction is included only when behavior_checked_public_scope is present in diagnostic hidden gold.",
  "probe_groups": 276,
  "public_leak_policy": "leaderboard_core/trials.jsonl must not expose hidden gold, forbidden event ids, behavior evidence internals, or excluded D response evidence",
  "source_pass": "runs/gooseomni_decrypto_human_verified_combined_pass240_leaderboard_core_split",
  "static_trial_counts": {
    "A_pre_reveal_belief": 276,
    "B_post_reveal_reconstruct_previous_belief": 276,
    "C_other_agent_false_belief": 276,
    "D_perspective_taking_prediction": 60
  },
  "total_core_prompts": 888,
  "total_core_static_trials": 888
}

Files:

- trials.jsonl: public model-facing core trials, no hidden scorer answers.
- gold.jsonl: scorer-visible gold metadata.
- private_gold.jsonl: scorer-only hidden gold. Do not release to model participants.
- interactive_prompts.jsonl: core interactive diagnostic prompts.
- probe_groups.jsonl: probe groups with core inclusion policy annotations.
- excluded_D_ungrounded.jsonl: D trials kept in the extended benchmark but excluded from leaderboard-core scoring.

## Pass241 adjudication

All 84 D_perspective_taking_prediction trials excluded from leaderboard-core were individually adjudicated in pass241. The per-trial records live in `excluded_D_ungrounded.jsonl` and the internal diagnostic audit is `annotations_evaluated_model/diagnostics/review/d_ungrounded_adjudication_pass241.jsonl`.