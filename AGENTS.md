# AGENTS.md

## 项目工作规则

- 始终使用简体中文回复。
- 先阅读 `docs/` 下与任务相关的文档，再改代码；Qwen3-Omni 初标流水线优先读 `docs/qwen3_omni_annotation_pipeline.zh-CN.md`。
- 《鹅鸭杀》标注需要同时读取 `docs/goose_goose_duck_common_knowledge.zh-CN.md`，特别注意任务 UI/点击操作不等同于地图移动。
- Python 环境由 `uv` 管理，常用命令：
  - `uv add 包名`
  - `uv add --dev 包名`
  - `uv sync`
  - `uv run python xxx.py`
  - `uv run pytest`
- 本项目已纳入 Git 管理。修改前后都要查看 `git status --short`，不要提交视频、API key、私人语音数据、中间产物。
- 未经用户明确确认，不执行 `git commit`、`git push`、`git reset --hard`。
- 后续开发分支统一命名为 `xty-YYYYMMDD`，其中日期使用创建分支当天的日期；完成验证后先推送该分支，再合并到 `main` 并推送 `main`。

## Qwen3-Omni 与 Slurm

- 当前机器的 Qwen3-Omni 模型路径：
  `/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct`
- 多视角初标必须通过 Slurm 提交 GPU 作业，使用：
  `configs/slurm/qwen3_annotation_pipeline.slurm`
- 冒烟测试示例：
  `sbatch --nodelist=gpu8 --gpus-per-node=2 --export=ALL,LIMIT=3,GAME_ID=g001,RESUME=1 configs/slurm/qwen3_annotation_pipeline.slurm`
- 完整流程参考：
  `docs/qwen3_omni_annotation_pipeline.zh-CN.md`

## 数据与产物约束

- 原始视频放在 `data/raw/{game_id}/{player_id}.mp4`。
- 第一局对齐起点记录在 `data/processed/sync_offsets.json`。
- 多视角录像必须先以第一局正式开始对齐，再切片和标注；同步审查产物放在 `data/processed/sync_review/`。
- 中间结果只写入 `data/processed/` 或 `annotations/`。
- 不训练模型，不下载模型，不把视频、API key、私人语音或中间产物提交进 Git。

## 代码质量约束

- 单文件不超过 500 行，超过必须拆分。
- 优先模块化，避免把切片、模型调用、schema、后处理耦合在一个文件。
- 遵循 SOLID、KISS、DRY、YAGNI。
- 新增代码需要配套最小 pytest 测试；真实 Qwen 调用用 mock 测试覆盖。
