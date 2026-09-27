# Agentic Midgame Prediction

This supplementary GooseOmni track evaluates the online middle-layer skills needed by a hypothetical omni player in a 5-human + 1-omni game: hidden-state estimation, offscreen player action prediction, future behavior prediction, and deception-state inference from one limited POV.

It does not replace `leaderboard_core`. The core A/B/C/D benchmark measures Decrypto-style Theory-of-Mind diagnostics. This track measures POV-to-oracle state estimation and POV-to-future behavior prediction, with scorer-private hidden gold from the aligned 6-POV replay.

- total_trials: 160
- `E_hidden_world_state_estimation`: 40
- `F_other_player_current_action_prediction`: 40
- `G_next_behavior_prediction`: 40
- `H_deception_state_inference`: 40

Public files contain only ego-available context and prompts. Scorer-only hidden POV facts, future outcomes, and forbidden event IDs live under `private/agentic_midgame_prediction/` and must never be sent to evaluated models.
