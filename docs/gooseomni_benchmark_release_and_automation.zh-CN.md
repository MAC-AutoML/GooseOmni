# Omni Goose Benchmark 发布与自动化说明

## 当前可发布对象

当前已经完成的是 `gooseomni` 的 GooseOmni / Theory-of-Mind 弱标注 benchmark。

权威完成证据：

```text
runs/gooseomni_oracle_pass1/completion_validation.json
runs/gooseomni_oracle_pass1/completion_validation.md
```

当前验证状态：

```text
complete: true
segments: 82
candidate_trials: 456
benchmark_trials: 2280
issue_counts: {}
```

## 推荐发布包层级

建议发布分为三个层级，不要把所有本地运行产物混在一起发布。

### A. Minimal Benchmark Release

适合先发给合作者或做 paper appendix artifact。

包含：

```text
README.md
release_manifest.json
SHA256SUMS
completion_validation.json
completion_validation.md
data/gooseomni/manifest.json
data/gooseomni/segments.json
data/gooseomni/segments.jsonl
data/gooseomni/skipped.json
annotations/candidate_trials/g001_candidate_trials.jsonl
benchmark/weak/trials.jsonl
benchmark/weak/gold_weak.jsonl
benchmark/weak/metadata.json
benchmark/human_review_queue.jsonl
benchmark/reports/dataset_card.md
benchmark/reports/tom_benchmark_card.md
benchmark/reports/annotation_quality.md
```

默认不包含：

```text
videos/
raw videos
slurm logs
job manifests
errors/
model raw responses
ssh keys
AGENTS.md
docs/local_*.md
```

### B. Video Benchmark Release

如果合作者需要直接跑 `single_pov_video` 条件，则需要额外包含：

```text
data/gooseomni/videos/g001/{segment_id}/{player_id}.mp4
```

发布前必须确认：

```text
1. 视频中没有非对局内容。
2. 视频中没有不应公开的个人隐私信息。
3. 语音内容是否允许公开。
4. 平台和游戏内容是否适合公开分发。
```

### C. Full Annotation Release

如果要让合作者复查标注链路，可以额外发布结构化中间标注：

```text
annotations/pov_events/
annotations/utterances/
annotations/phase_events/
annotations/global_events/
annotations/information_states/
annotations/memory_states/
annotations/belief_states/
annotations/candidate_trials/
```

当前 `data build` 产生的 run-local 初标使用扁平布局：

```text
runs/<run_name>/annotations/pov_events/{clip_id}.json
```

每个文件包含 `clip` 和 `events` 两个字段。完整性校验器会自动识别这种布局，
并同时校验 clip、事件 schema、玩家和时间窗口；旧的层级布局仍保持兼容。

不建议公开发布：

```text
annotations/errors/
raw model responses
configs/slurm/logs/
runs/*/job_manifests/
```

这些内容可能包含调试信息、本地路径、模型失败输出或不适合公开的中间文本。

## 当前发布打包命令

新增脚本：

```text
tools/package/package_gooseomni_benchmark_release.py
```

生成 minimal release：

```bash
.venv/bin/python tools/package/package_gooseomni_benchmark_release.py \
  --dataset-root data/gooseomni \
  --run-root runs/gooseomni_oracle_pass1 \
  --output-dir runs/gooseomni_oracle_pass1/release_minimal_v1 \
  --version v1 \
  --overwrite
```

生成包含视频的 release：

```bash
.venv/bin/python tools/package/package_gooseomni_benchmark_release.py \
  --dataset-root data/gooseomni \
  --run-root runs/gooseomni_oracle_pass1 \
  --output-dir runs/gooseomni_oracle_pass1/release_video_v1 \
  --version v1 \
  --include-videos \
  --overwrite
```

生成包含结构化中间标注的 release：

```bash
.venv/bin/python tools/package/package_gooseomni_benchmark_release.py \
  --dataset-root data/gooseomni \
  --run-root runs/gooseomni_oracle_pass1 \
  --output-dir runs/gooseomni_oracle_pass1/release_full_annotation_v1 \
  --version v1 \
  --include-annotations \
  --overwrite
```

脚本默认要求：

```text
completion_validation.json 里的 complete 必须为 true。
```

如未完成会拒绝打包，除非显式加：

```bash
--allow-incomplete
```

## 发布前必须检查

发布前建议固定执行：

```bash
.venv/bin/python tools/validate/validate_gooseomni_completion.py \
  --dataset-root data/gooseomni \
  --annotation-root runs/gooseomni_oracle_pass1/annotations \
  --benchmark-root runs/gooseomni_oracle_pass1/benchmark \
  --output runs/gooseomni_oracle_pass1/completion_validation.json

.venv/bin/python tools/report/report_benchmark_quality.py \
  --dataset-root data/gooseomni \
  --benchmark-root runs/gooseomni_oracle_pass1/benchmark
```

验收标准：

```text
completion_validation complete: true
issue_counts: {}
candidate rows 覆盖全部 segment
benchmark rows = candidate rows * input_conditions 数量
human_review_queue 已生成
SHA256SUMS 已生成
```

当前 raw release 的最小复现校验命令：

```bash
uv run python tools/validate/validate_gooseomni_completion.py \
  --dataset-root data/gooseomni \
  --annotation-root runs/gooseomni_v2_raw_release/annotations \
  --benchmark-root benchmark/gooseomni_v2_raw_release \
  --output runs/gooseomni_v2_raw_release/completion_validation.json \
  --fail-on-incomplete
```

## 新一组多视角视频的自动化流程

后续如果提供新一组多视角视频，pipeline 可以自动化，但输入必须先满足几个硬条件。

### 新数据输入要求

建议为每一局准备：

```text
game_id
player_id 列表
每个 player 的原始视频路径
同步 offset 或可由模型/人工确认的同步事件
对局有效时间范围
需要排除的非对局片段范围
```

当前脚本默认玩家是：

```text
Gemini
baile
beigang
mojiang
saoyi
xiaolu
```

如果下一批玩家不同，需要把玩家列表参数化，至少涉及：

```text
src/gooseomni/benchmark/schema.py
tools/annotation/submit_gooseomni_oracle_jobs.py
各阶段 prompt 和 validation
```

如果仍是 6 个固定玩家，则可以直接复用当前 pipeline。

### 自动化阶段

推荐把新数据 pipeline 拆成 6 个阶段：

```text
1. raw videos -> aligned strict 6-POV dataset
2. aligned videos -> Qwen3-Omni upstream annotations
3. upstream annotations -> global / information / memory / belief states
4. belief states -> ToM candidate trials
5. candidate trials -> benchmark export
6. validation + package release
```

### 当前已有自动化能力

已经有的脚本：

```text
tools/package/export_gooseomni_aligned_videos.py
tools/annotation/submit_gooseomni_oracle_jobs.py
tools/annotation/recover_gooseomni_error_outputs.py
tools/annotation/report_gooseomni_progress.py
tools/package/export_tom_benchmark.py
tools/report/report_benchmark_quality.py
tools/validate/validate_gooseomni_completion.py
tools/package/package_gooseomni_benchmark_release.py
```

Slurm 自动化脚本：

```text
configs/slurm/gooseomni_promote_export_monitor.slurm
configs/slurm/qwen3_omni_oracle_2gpu_stage.slurm
configs/slurm/qwen3_omni_oracle_4x2_local.slurm
```

### 建议的新数据复跑命令骨架

1. 生成 aligned dataset：

```bash
.venv/bin/python tools/package/export_gooseomni_aligned_videos.py \
  --manifest-path data/processed/<new_clip_manifest>.jsonl \
  --sync-offsets data/processed/<new_sync_offsets>.json \
  --round-boundaries-dir annotations/<new_round_boundaries> \
  --raw-dir data/raw \
  --output-dir data/<new_dataset_name> \
  --game-id <new_game_id> \
  --overwrite
```

2. 提交 Qwen3-Omni 标注任务：

```bash
sbatch \
  --export=ALL,DATASET_ROOT=data/<new_dataset_name>,ANNOTATION_ROOT=runs/<new_run_name>/annotations,MAX_TOTAL_JOBS=60,MAX_PENDING_JOBS=55 \
  configs/slurm/gooseomni_promote_export_monitor.slurm
```

3. 中途查看进度：

```bash
.venv/bin/python tools/annotation/report_gooseomni_progress.py \
  --dataset-root data/<new_dataset_name> \
  --annotation-root runs/<new_run_name>/annotations \
  --benchmark-root runs/<new_run_name>/benchmark \
  --output runs/<new_run_name>/progress_report.json
```

4. 最终导出和验证：

```bash
.venv/bin/python tools/package/export_tom_benchmark.py \
  --dataset-root data/<new_dataset_name> \
  --annotation-root runs/<new_run_name>/annotations \
  --benchmark-root runs/<new_run_name>/benchmark

.venv/bin/python tools/report/report_benchmark_quality.py \
  --dataset-root data/<new_dataset_name> \
  --benchmark-root runs/<new_run_name>/benchmark

.venv/bin/python tools/validate/validate_gooseomni_completion.py \
  --dataset-root data/<new_dataset_name> \
  --annotation-root runs/<new_run_name>/annotations \
  --benchmark-root runs/<new_run_name>/benchmark \
  --output runs/<new_run_name>/completion_validation.json
```

5. 打包 release：

```bash
.venv/bin/python tools/package/package_gooseomni_benchmark_release.py \
  --dataset-root data/<new_dataset_name> \
  --run-root runs/<new_run_name> \
  --output-dir runs/<new_run_name>/release_minimal_v1 \
  --version v1 \
  --overwrite
```

## 还需要加强的自动化点

当前 pipeline 已经可以自动跑完标注和 benchmark 导出，但下一批新视频如果要更稳，建议补强：

```text
1. 玩家列表参数化，不再写死 6 个 player_id。
2. 新数据 ingest config，用一个 YAML/JSON 描述 raw videos、game_id、players、sync offsets、输出目录。
3. 自动视觉 QA：抽帧检查 6 POV 是否同一局、是否非对局、是否黑屏或错位。
4. 自动 release smoke test：抽样读取 benchmark trial，确认 input_video_files 都存在。
5. 隐私扫描：检测桌面、浏览器、直播间、非游戏画面，并要求人工确认。
6. human verification 子集抽样：从 high/medium leakage 和 low certainty 样本中固定抽查。
```

## 发布表述建议

当前数据建议称为：

```text
GooseOmni-ToM v1
```

标注性质建议写清楚：

```text
Qwen3-Omni weakly supervised benchmark with structured video understanding,
memory/belief state construction, candidate ToM trial generation, and a
human-review queue for verification.
```

不要声称：

```text
fully human verified
```

除非后续真的完成了人工复核。
