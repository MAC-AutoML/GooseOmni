# Qwen3-Omni 多视角初标流水线

> 本文前半部分记录旧初标工具。严格 ToM Pilot 不使用
> `data/processed/sync_offsets.json`、人工发布门禁或固定窗口作为语义边界；其唯一时间
> 来源是 run 内的 `alignment.json` 和审核后的 episode phases。严格流程以
> `docs/gooseomni_v2_pilot_pipeline.zh-CN.md` 为准。

本文档记录当前机器上《鹅鸭杀》多视角录像初标的固定用法。所有命令默认在项目根目录执行：

```bash
cd /public/home/xty/workdir/omni_goose
```

## 目录约定

- 原始视频：`data/raw/{game_id}/{player_id}.mp4`
- 对齐配置：`data/processed/sync_offsets.json`
- 同步审查片段：`data/processed/sync_review/{game_id}/{player_id}_sync_review.mp4`
- 切片输出：`data/processed/clips/{game_id}/{player_id}/{clip_id}.mp4`
- 切片清单：`data/processed/clip_manifest.jsonl`
- Qwen 初标：`annotations/pov_events/{clip_id}.json`
- 错误输出：`annotations/errors/{clip_id}.json`

不要提交视频、API key、私人语音数据、`data/processed/*` 或 `annotations/*`。

## 1. 标记第一局开始时间

不同玩家录屏的开始时间和总时长不一致，必须先为每个 POV 标记“第一局游戏开始”在原始录屏中的秒数。

推荐先用 Qwen3-Omni 生成同步初稿。该步骤会截取每个原始视频开头一段审查片段，让模型判断第一局正式开始的 `raw_start_sec`：

```bash
sbatch \
  --gpus-per-node=2 \
  --export=ALL,RUN_SYNC=1,SKIP_SPLIT=1,LIMIT=0,GAME_ID=g001,RESUME=0,OVERWRITE_SYNC_REVIEW=1 \
  configs/slurm/qwen3_annotation_pipeline.slurm
```

只在已有本地服务上调试同步：

```bash
uv run python tools/build/infer_sync_offsets.py \
  --backend local \
  --server-url http://127.0.0.1:5090 \
  --game-id g001 \
  --no-resume
```

也可以先生成人工模板：

```bash
uv run python tools/build/mark_game_starts.py --raw-dir data/raw --probe-duration
```

同步结果必须人工抽查 `data/processed/sync_offsets.json` 和 `data/processed/sync_review/`。每条记录里的 `raw_start_sec` 是第一局开始时刻。例如：

```json
{
  "game_id": "g001",
  "player_id": "P1",
  "raw_start_sec": 123.5,
  "evidence": "第一局加载结束，玩家进入地图",
  "confidence": 0.9
}
```

后续切片的 `start_sec/end_sec` 都是对齐到第一局开始后的全局秒数。

共同知识和标注边界见：

```text
docs/goose_goose_duck_common_knowledge.zh-CN.md
```

## 2. 生成 Qwen3-Omni 审查窗口

这里生成的是 Qwen3-Omni 输入审查窗口，不是最终回合切分结果。窗口长度、overlap 和 `clip_id` 只用于让模型分段读取长视频，不能直接写成 `round_start` / `round_end`。

最终回合边界必须由 Qwen3-Omni 基于视频内容推理得到，并保留可审查证据、置信度和来源 POV。禁止用固定时长、固定 gap、均匀切分或纯聚类结果直接决定一局游戏内的回合边界。

```bash
uv run python tools/build/split_raw_videos.py --overwrite
```

审查窗口参数：

- 窗口长度是模型输入工程参数，不代表回合长度
- overlap 用于避免语义边界落在窗口边缘
- clip_id：`{game_id}_{player_id}_{start_sec}_{end_sec}`

推荐流程：

1. 先生成带 overlap 的审查窗口。
2. 让 Qwen3-Omni 在窗口内识别游戏语义边界，输出 evidence 和 confidence。
3. 融合多 POV 结果，确认 `game -> round/phase -> event -> aligned clip` 层级。
4. 用同步 offset 把对齐时间换回各 POV 原始视频时间后再裁剪。

只检查命令、不写文件：

```bash
uv run python tools/build/split_raw_videos.py --dry-run
```

## 3. Slurm 提交 Qwen3-Omni 初标

模型路径固定为：

```text
/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct
```

提交一个小样本冒烟任务：

```bash
sbatch \
  --nodelist=gpu8 \
  --gpus-per-node=2 \
  --export=ALL,LIMIT=3,GAME_ID=g001,RESUME=1 \
  configs/slurm/qwen3_annotation_pipeline.slurm
```

跑完整任务：

```bash
sbatch \
  --nodelist=gpu8 \
  --gpus-per-node=2 \
  --export=ALL,RESUME=1 \
  configs/slurm/qwen3_annotation_pipeline.slurm
```

常用变量：

- `RAW_DIR`：默认 `data/raw`
- `SYNC_OFFSETS`：默认 `data/processed/sync_offsets.json`
- `RUN_SYNC=1`：启动模型后先推断同步起点，再切片或标注
- `SYNC_REVIEW_WINDOW_SEC`：同步审查使用原始视频开头多少秒，默认 `600`
- `OVERWRITE_SYNC_REVIEW=1`：重新生成同步审查片段
- `MANIFEST_PATH`：默认 `data/processed/clip_manifest.jsonl`
- `LIMIT`：限制标注 clip 数，用于冒烟测试
- `GAME_ID` / `PLAYER_ID`：只跑指定游戏或玩家
- `RESUME=1`：跳过已有 `annotations/pov_events/*.json`
- `SKIP_SPLIT=1`：已有 manifest 时跳过切片
- `OVERWRITE_CLIPS=1`：重新覆盖切片
- `QWEN3_OMNI_MAX_TOKENS`：默认 `2048`，用于输出 JSON 标注
- `QWEN3_OMNI_VIDEO_FPS`：默认 `0.5`，控制视频采样帧率，降低显存占用
- `QWEN3_OMNI_VIDEO_MAX_FRAMES`：默认 `32`，限制单 clip 输入帧数
- `QWEN3_OMNI_VIDEO_MAX_PIXELS`：默认 `200704`，限制单帧像素预算

Slurm 作业会在同一个任务内启动本地 Qwen3-Omni HTTP 服务，等待 `/health` 就绪后再运行标注脚本。

## 4. 本地调试

如果已经手工启动了 Qwen3-Omni 服务：

```bash
uv run python tools/annotation/run_qwen_annotation.py \
  --backend local \
  --server-url http://127.0.0.1:5090 \
  --game-id g001 \
  --limit 3 \
  --resume
```

不调用模型的 mock 测试：

```bash
uv run python tools/annotation/run_qwen_annotation.py --backend mock --limit 3
```

## 5. 后处理

后处理统一由数据流水线编排，输出位于该 run 的 `artifacts/fusion/`，
不再使用 `data/processed/` 根目录下的全局事件、信息状态和候选题旧产物。
在前置阶段已完成的同一个 run 上执行：

```bash
uv run gooseomni data build \
  --config configs/datasets/gooseomni_v2_raw.yaml \
  --from-stage event_fusion --to-stage oracle_and_belief --resume
```

`tools/build/` 中的单阶段调试脚本必须显式指定输入和输出路径；
`annotate postprocess` 批量旧入口已移除。严格 Pilot 请使用其独立配置和阶段。

`merge_global_events` 和候选 trial 结果只是候选/初稿。最终回合组织必须经过 Qwen3-Omni 语义边界融合和人工抽查；没有清晰证据的边界应标为待复核，不能自动落成最终标注。

## 6. 测试

```bash
uv sync --dev
uv run pytest tests/test_annotation_pipeline.py -q
```

如果 `uv run` 因远程 `flash-attn` wheel 网络问题卡住，可使用已有虚拟环境：

```bash
.venv/bin/python -m pytest tests/test_annotation_pipeline.py -q
```
