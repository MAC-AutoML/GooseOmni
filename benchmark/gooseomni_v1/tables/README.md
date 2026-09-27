# GooseOmni pass248 benchmark 表格摘要

## 主指标

| track | metric | value |
|---|---|---:|
| structured | trials_scored | 889 |
| structured | SoG_ToM_Core | 0.342256 |
| structured | RC_weak | 0.992537 |
| structured | RC_strong | 0.000000 |
| structured | FB_weak | 0.522388 |
| structured | FB_strong | 0.253623 |
| structured | PT_weak | 0.245902 |
| structured | PT_strong | 0.245902 |
| structured | claim_verification_global | 0.000000 |
| structured | claim_verification_local | 0.000000 |
| structured | evidence_support_rate | 0.956130 |
| structured | perspective_leakage_rate | 0.029246 |
| structured | forbidden_evidence_usage_rate | 0.029246 |
| structured | death_skill_overclaim_rate | 0.006749 |
| structured | json_parse_success | 0.998875 |
| structured | schema_validation_success | 0.469066 |
| structured | error_count | 0 |
| raw_video_smoke | trials_scored | 48 |
| raw_video_smoke | SoG_ToM_Core | 0.229167 |
| raw_video_smoke | RC_weak | 1.000000 |
| raw_video_smoke | RC_strong | 0.000000 |
| raw_video_smoke | FB_weak | 0.000000 |
| raw_video_smoke | FB_strong | 0.000000 |
| raw_video_smoke | PT_weak | 0.000000 |
| raw_video_smoke | PT_strong | 0.000000 |
| raw_video_smoke | claim_verification_global | 0.000000 |
| raw_video_smoke | claim_verification_local | 0.000000 |
| raw_video_smoke | evidence_support_rate | 0.854167 |
| raw_video_smoke | perspective_leakage_rate | 0.062500 |
| raw_video_smoke | forbidden_evidence_usage_rate | 0.062500 |
| raw_video_smoke | death_skill_overclaim_rate | 0.000000 |
| raw_video_smoke | json_parse_success | 1.000000 |
| raw_video_smoke | schema_validation_success | 0.750000 |
| raw_video_smoke | error_count | 0 |

## by_probe_type

| track | group | n | SoG | RC_s | FB_s | PT_s | evidence | leakage | forbidden | overclaim | parse | schema |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structured | A_pre_reveal_belief | 276 | 0.245652 |  |  |  | 0.956522 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.485507 |
| structured | B_post_reveal_reconstruct_previous_belief | 276 | 0.248007 | 0.000000 |  |  | 0.992754 | 0.003623 | 0.003623 | 0.018116 | 0.996377 | 0.485507 |
| structured | C_other_agent_false_belief | 276 | 0.282609 |  | 0.253623 |  | 0.909420 | 0.090580 | 0.090580 | 0.000000 | 1.000000 | 0.485507 |
| structured | D_perspective_taking_prediction | 61 | 0.298361 |  |  | 0.245902 | 1.000000 | 0.000000 | 0.000000 | 0.016393 | 1.000000 | 0.245902 |
| raw_video_smoke | A_pre_reveal_belief | 12 | 0.216667 |  |  |  | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | B_post_reveal_reconstruct_previous_belief | 12 | 0.250000 | 0.000000 |  |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | C_other_agent_false_belief | 12 | 0.200000 |  | 0.000000 |  | 0.750000 | 0.250000 | 0.250000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | D_perspective_taking_prediction | 12 | 0.250000 |  |  | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |

## by_input_condition

| track | group | n | SoG | RC_s | FB_s | PT_s | evidence | leakage | forbidden | overclaim | parse | schema |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structured | global_to_perspective | 142 | 0.313380 |  | 0.492958 |  | 0.823944 | 0.176056 | 0.176056 | 0.000000 | 1.000000 | 0.943662 |
| structured | oracle_truth_revealed | 276 | 0.248007 | 0.000000 |  |  | 0.992754 | 0.003623 | 0.003623 | 0.018116 | 0.996377 | 0.485507 |
| structured | public_history_only | 134 | 0.250000 |  | 0.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | speaker_perspective | 61 | 0.298361 |  |  | 0.245902 | 1.000000 | 0.000000 | 0.000000 | 0.016393 | 1.000000 | 0.245902 |
| structured | target_available_events | 276 | 0.245652 |  |  |  | 0.956522 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.485507 |
| raw_video_smoke | global_to_perspective | 12 | 0.200000 |  | 0.000000 |  | 0.750000 | 0.250000 | 0.250000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | oracle_truth_revealed | 12 | 0.250000 | 0.000000 |  |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | speaker_perspective | 12 | 0.250000 |  |  | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| raw_video_smoke | target_available_events | 12 | 0.216667 |  |  |  | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |

## by_template

| track | group | n | SoG | RC_s | FB_s | PT_s | evidence | leakage | forbidden | overclaim | parse | schema |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structured | claim_backed_hidden_event | 6 | 0.250000 | 0.000000 | 0.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| structured | critical_hidden_event | 42 | 0.269048 | 0.000000 | 0.214286 |  | 0.880952 | 0.119048 | 0.119048 | 0.000000 | 1.000000 | 1.000000 |
| structured | critical_hidden_event_scope_limited | 27 | 0.390741 | 0.000000 | 0.777778 |  | 0.925926 | 0.074074 | 0.074074 | 0.000000 | 1.000000 | 1.000000 |
| structured | critical_hidden_event_scope_limited_proximity | 12 | 0.450000 | 0.000000 | 1.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| structured | hidden_event_awareness | 3 | 0.250000 | 0.000000 | 0.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| structured | manual_behavior_checked_alternative_evidence_after_accusation | 4 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | manual_behavior_checked_alternative_information_request | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_counterevidence_uncertainty_response | 4 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | manual_behavior_checked_defense_after_vote_acceptance | 4 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | manual_behavior_checked_followup_hypothesis_question_after_location_claim | 4 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | manual_behavior_checked_followup_hypothesis_question_after_sighting | 4 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | manual_behavior_checked_public_accusation_verbal_uptake | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_public_accusation_vote_follow | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_public_no_info_location_frame_uptake | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_public_vote_suggestion_counterevidence_followup | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_self_defense_after_no_info_frame | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_vote_call | 16 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_vote_deferral | 4 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_vote_frame | 8 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 |
| structured | manual_behavior_checked_vote_strategy_followup | 4 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | meeting_claim_audio_confirmed | 338 | 0.249112 |  | 0.000000 | 0.000000 | 0.997041 | 0.000000 | 0.000000 | 0.011834 | 0.997041 | 0.000000 |
| structured | meeting_claim_reviewer_text_gate_behavior_checked_d | 8 | 0.250000 |  | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | private_witness | 66 | 0.439394 | 0.000000 | 1.000000 |  | 0.893939 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| structured | public_ui_reveal | 33 | 0.259091 | 0.000000 | 0.181818 |  | 0.878788 | 0.121212 | 0.121212 | 0.060606 | 1.000000 | 1.000000 |
| structured | route_hidden_event | 93 | 0.234946 | 0.000000 | 0.000000 |  | 0.924731 | 0.075269 | 0.075269 | 0.000000 | 1.000000 | 1.000000 |
| structured | route_location_hidden_event_scope_limited | 60 | 0.450000 | 0.000000 | 1.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| structured | route_proximity_hidden_event_scope_limited | 36 | 0.450000 | 0.000000 | 1.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| structured | scope_limited_public_suspect_belief | 41 | 0.450000 |  | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.048780 |
| structured | scope_limited_vote_selection_next_action | 24 | 0.250000 |  | 0.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structured | vote_influence | 24 | 0.162500 | 0.000000 | 0.000000 |  | 0.458333 | 0.333333 | 0.333333 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | claim_backed_hidden_event | 6 | 0.216667 | 0.000000 | 0.000000 |  | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | critical_hidden_event | 18 | 0.250000 | 0.000000 | 0.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | hidden_event_awareness | 3 | 0.250000 | 0.000000 | 0.000000 |  | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | meeting_claim_audio_confirmed | 12 | 0.250000 |  |  | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| raw_video_smoke | route_hidden_event | 3 | 0.183333 | 0.000000 | 0.000000 |  | 0.666667 | 0.333333 | 0.333333 | 0.000000 | 1.000000 | 1.000000 |
| raw_video_smoke | vote_influence | 6 | 0.150000 | 0.000000 | 0.000000 |  | 0.333333 | 0.333333 | 0.333333 | 0.000000 | 1.000000 | 1.000000 |