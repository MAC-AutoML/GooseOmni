# GooseOmni Slurm Jobs

本目录用于提交 GPU 节点任务，日志统一写入 `configs/slurm/logs/`。

## 冒烟测试

```bash
sbatch --nodelist=gpu8 --gpus-per-node=2 configs/slurm/gooseomni_autoserver.slurm
```

也可以使用 `.sh` 入口：

```bash
sbatch --nodelist=gpu8 --gpus-per-node=2 configs/slurm/gooseomni_autoserver.sh
```

默认会从 `configs/gooseomni.yaml` 的 `benchmark.level1.model` 读取模型。当前默认参数等价于在同一个 Slurm 作业里启动 `qwen3_omni` 服务、等待 ready、执行评测、最后清理服务进程：

```bash
MODEL=qwen3_omni LEVEL=1
```

模型服务在作业开始时只启动一次，权重会一直驻留在 GPU 上；评测过程中每个样例复用同一个本地 HTTP 服务，整轮测试结束后脚本再停止服务。

## 指定模型和任务

```bash
sbatch \
  --job-name=qwen3_l2 \
  --nodelist=gpu8 \
  --gpus-per-node=2 \
  --export=ALL,MODEL=qwen3_omni,LEVEL=2,RESUME=1 \
  configs/slurm/gooseomni_autoserver.slurm
```

## 冒烟测试

指定 `MAX_SAMPLES` 可以只跑少量样本：

```bash
sbatch \
  --job-name=qwen3_smoke \
  --nodelist=gpu8 \
  --gpus-per-node=2 \
  --export=ALL,MODEL=qwen3_omni,LEVEL=1,MAX_SAMPLES=1,RESUME=1 \
  configs/slurm/gooseomni_autoserver.slurm
```

## 常用变量

- `MODEL`：模型键名，来自 `src/gooseomni/models/model_server/clients.py`
- 不传 `MODEL` 时，脚本按 `LEVEL` 读取 `configs/gooseomni.yaml` 里的 `benchmark.level{LEVEL}.model`
- `LEVEL`：`1` 或 `2`
- `MAX_SAMPLES`：评测样本数；不设置表示跑完整任务
- `RESUME`：`1`/`true` 表示自动续跑
- `REUSE_SERVER`：默认 `0`，表示必须由当前作业启动服务；调试时可设为 `1` 复用已有服务
- `SERVER_TIMEOUT`：服务启动等待秒数，默认 `1800`
- `TEST_TIMEOUT`：评测超时秒数，默认 `86400`

模型权重路径由 `configs/gooseomni.yaml` 管理，当前默认指向 `/publicssd/xty/models`。

## Qwen3-Omni 多视角初标

初标流水线使用独立作业脚本：

```bash
sbatch \
  --nodelist=gpu8 \
  --gpus-per-node=2 \
  --export=ALL,LIMIT=3,GAME_ID=g001,RESUME=1 \
  configs/slurm/qwen3_annotation_pipeline.slurm
```

完整说明见 `docs/qwen3_omni_annotation_pipeline.zh-CN.md`。该作业会在同一个 Slurm 任务内启动
`/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct` 对应的本地 HTTP 服务，再调用
`tools/annotation/run_qwen_annotation.py --backend local`。
