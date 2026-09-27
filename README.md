# GooseOmni

GooseOmni 是一个面向多模态模型的《鹅鸭杀》多智能体 Theory-of-Mind 基准与数据流水线。项目把六个 POV 的对齐录像转换为可追溯的事件、信息状态、信念状态和评测 trial，用来测试模型是否能区分全局事实、玩家可见证据、错误信念和策略性交流。

## 先跑起来

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 3.10 环境，所有依赖和虚拟环境都放在项目内：

```bash
uv sync
uv run gooseomni --help
uv run pytest -q
uv run ruff check src tools tests
```

需要视频、音频或图像处理依赖时，安装 `media` extra：

```bash
uv sync --extra media --dev
```

不需要手动设置 `PYTHONPATH`。推荐使用 `uv run gooseomni ...` 或 `uv run python ...`；`src/` 布局、项目内 `.venv` 和 `uv.lock` 保证命令使用同一套环境。

## 目录结构

```text
configs/
  datasets/             数据流水线配置（raw、pilot、smoke）
  slurm/                Qwen3-Omni、对齐和评测作业模板
data/                   本地原始数据与处理中间结果，默认不入 Git
annotations/            标注和审核产物，默认不入 Git
benchmark/
  gooseomni_v1/         冻结的历史基准
  gooseomni_v2/         当前可发布的基准文件和 manifest
src/gooseomni/
  annotation/           POV 标注、后处理和提交队列
  benchmark/            schema、解析、oracle ledger、probe 和 scorer
  data_pipeline/        声明式数据制作流水线
  evaluation/           评测计划、运行、计分和报告
  models/               模型客户端、服务端和 pipeline
  cli.py                gooseomni 命令行入口
tools/
  build/                数据构建脚本
  annotation/           标注与审核脚本
  eval/                 评测辅助脚本
  package/               发布物打包脚本
  validate/             数据和发布物校验脚本
docs/                   数据协议、Pilot 流程和游戏标注规范
tests/                  单元测试、解析测试和发布校验测试
```

代码按“浅接口、深实现”组织：CLI 负责参数和阶段编排，核心逻辑位于 `src/gooseomni`；`tools/` 是内部脚本入口，不构成稳定公共 API。原始视频、API key、私人语音、`runs/`、`annotations/` 和 `data/` 中间产物不要提交。

## 三条常用路径

### 1. 数据制作流水线

权威配置是 `configs/datasets/gooseomni_v2_raw.yaml` 或严格 Pilot 配置 `configs/datasets/gooseomni_v2_pilot.yaml`。标准阶段为：

```text
ingest → sync/align → segment/episode → perception
→ event/trajectory fusion → information/belief state
→ trial build → review → validate → package
```

普通 v2 数据流水线：

```bash
uv run gooseomni data validate-input \
  --config configs/datasets/gooseomni_v2_raw.yaml

uv run gooseomni data build \
  --config configs/datasets/gooseomni_v2_raw.yaml

uv run gooseomni data status \
  --run runs/gooseomni_v2_raw_release

uv run gooseomni data validate \
  --run runs/gooseomni_v2_raw_release
```

严格 Pilot 使用独立 run 和 phase-aware 对齐，不复用旧的固定窗口同步结果：

```bash
uv run gooseomni data validate-input \
  --config configs/datasets/gooseomni_v2_pilot.yaml

uv run gooseomni data build \
  --config configs/datasets/gooseomni_v2_pilot.yaml \
  --from-stage align --resume
```

多视角真实 Qwen3-Omni 感知必须通过 Slurm 执行，入口见 `configs/slurm/gooseomni_v2_pilot_qwen.slurm`。同步、phase 边界和 review 规则详见：

- `docs/gooseomni_data_pipeline.zh-CN.md`
- `docs/gooseomni_v2_pilot_pipeline.zh-CN.md`
- `docs/qwen3_omni_annotation_pipeline.zh-CN.md`
- `docs/goose_goose_duck_common_knowledge.zh-CN.md`

### 2. 基准评测

当前发布基准位于 `benchmark/gooseomni_v2/`，其 `manifest.json` 记录文件清单、SHA-256、行数和公开泄漏检查结果。主要 track：

| track | 用途 |
| --- | --- |
| `leaderboard_core` | A/B/C/D Theory-of-Mind 诊断 |
| `raw_video_smoke` | 小规模视频输入冒烟 |
| `agentic_midgame_prediction` | 中局隐藏状态、行为预测和欺骗意识 |

统一评测入口：

```bash
uv run gooseomni eval plan \
  --benchmark gooseomni_v2 \
  --models all \
  --tracks leaderboard_core raw_video_smoke

uv run gooseomni eval run \
  --benchmark gooseomni_v2 \
  --model gpt4o \
  --track raw_video_smoke \
  --modalities text \
  --limit 1 \
  --run runs/eval/gpt4o_smoke

uv run gooseomni eval score \
  --benchmark gooseomni_v2 \
  --run runs/eval/gpt4o_smoke

uv run gooseomni eval report \
  --run runs/eval/gpt4o_smoke
```

`private/` 下的 gold 只供 scorer 使用，不能进入模型 prompt。模态不兼容时应记录为 `skipped`，不应计为模型失败。

### 3. 模型和标注调试

```bash
uv run gooseomni models list
uv run gooseomni models check
uv run gooseomni models serve qwen3_omni --host 127.0.0.1 --port 5090

uv run gooseomni annotate sync --help
uv run gooseomni annotate split --help
uv run gooseomni annotate run --help
uv run gooseomni benchmark validate --help
```

真实模型调用需要相应的本地服务、Slurm 资源或 API 环境变量。测试和开发优先使用 mock backend；不要把 token 写进配置或命令历史。

## 解析与数据契约

初标与融合记录使用 `annotation.schemas`；segment 标注与评测记录使用
`benchmark.schema`。两者由流水线转换，不能交叉校验。后处理产物归属于
具体 run 的 `artifacts/fusion/`，旧的全局处理 JSON 和 `annotate postprocess`
入口已移除；单阶段工具必须显式指定输入和输出路径。

JSON/JSONL 解析集中在 `src/gooseomni/benchmark/io.py` 和 `pipeline_runtime.py`：

- `safe_json_loads` 支持普通 JSON 和 fenced Markdown JSON。
- `parse_json_array`、`parse_json_object` 负责顶层类型校验。
- `parse_partial_json_array_objects` 可从截断数组恢复完整对象，并自动降低 certainty、标记 `needs_human_review`。
- Pydantic schema 位于 `benchmark/schema_events.py`、`schema_benchmark.py` 和 `schema_oracle.py`。
- 所有重要数据阶段都保留 segment、player、时间窗口、来源 POV 和 review 状态，禁止用没有证据的推理替代观测事实。

## 开发与验证

新增代码应保持模块单一职责，单文件不超过 500 行，并配套最小 pytest。提交前运行：

```bash
uv run ruff check src tools tests
uv run pytest -q
uv lock --check --offline

files=$(rg --files src tools tests -g '*.py' -g '!**/._*')
uv run python -m py_compile $files
git diff --check
```

开发分支统一使用 `xty-YYYYMMDD` 命名；完成验证后先推送该分支，再合并到 `main`。未经明确授权不要执行 `git commit`、`git push` 或 `git reset --hard`。

## 相关文档

- `docs/gooseomni_data_pipeline.zh-CN.md`：数据制作闭环
- `docs/gooseomni_v2_pilot_pipeline.zh-CN.md`：严格同步和 Pilot
- `docs/gooseomni_decrypto_diagnostics.zh-CN.md`：Decrypto 风格诊断
- `docs/gooseomni_benchmark_release_and_automation.zh-CN.md`：发布与自动化
- `docs/model_capabilities.md`：模型能力和评测约束
