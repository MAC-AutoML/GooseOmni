# GooseOmni-v1 Benchmark Card

GooseOmni-v1 is a trajectory-derived Theory-of-Mind diagnostic benchmark built from a strictly aligned 6-POV Goose Goose Duck replay. It uses oracle trajectory ledgers, visibility projections, claim-truth links, and Decrypto-style A/B/C/D probe groups to evaluate representational change, false belief, claim verification, perspective taking, strategy communication, and perspective leakage.

Segments are storage units only. The benchmark unit is a trajectory node: `cutoff_abs_sec + target_player + query_variable + information_gap`. Each node is selected from an oracle trajectory ledger and projected into player-local knowledge states before probe generation.

Probe groups follow a Decrypto-style diagnostic mechanism. A asks for the target player's pre-reveal belief using only target-available evidence. B reveals oracle truth but asks the model to reconstruct the target's earlier belief without truth contamination. C asks for another player's false or incomplete belief. D asks a speaker to predict how a listener will interpret a strategic claim from speaker-safe context.

Interactive diagnostics evaluate A/B/C/D consistency and representational change across multiple prompts. Static trials provide frozen public prompts and separate hidden gold for leaderboard-style scoring. `trials.jsonl` must not expose hidden gold or forbidden event IDs; `private_gold.jsonl` is scorer-only.

Perspective leakage means that an answer simulating a limited player perspective cites hidden oracle facts, forbidden event IDs, or evidence only available to another POV. `gold_source` is `evaluated_model_weak` by default, `evaluated_model_checked` only after high-quality video review and reviewer merge gating, and `human_verified` only after explicit human review.

- probe_groups: 276
- prompts: 972
- human_review_queue: 0
- gold_source: evaluated_model_weak unless later promoted by explicit checking or human review