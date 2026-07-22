# Agentix nano-vLLM 复现

这个仓库提供 Agentix 应用层和 nano-vLLM 调度器的测试入口。当前 workload 以 **program** 为到达和完成单位：不同 program 可以并发运行，但同一个 program 在前一次 LLM call 完成后才会提交下一次 call，不会在测试开始时一次性提交全部 call。

## 运行环境

代码会把 `nano-vllm` 作为同级目录加载，并以 `agentix_app` 作为 Python 包名。推荐目录结构如下：

```text
<workspace>/
├── nano-vllm/
└── agentix_app/       # 本仓库
```

从 `<workspace>` 目录运行下文命令。模型需要已经下载到本地；运行时建议显式传入 `--model-path`，不要依赖代码中的开发机默认路径。

## 测试方式

### 快速冒烟测试

仓库自带 16 个 ShareGPT program 和 16 个 BFCL program 的小型 fixture。下面的命令运行 ShareGPT fixture：

```bash
python -m agentix_app.dataset_runner \
  --dataset sharegpt \
  --input agentix_app/fixtures/sharegpt_fixture_16.json \
  --limit 16 \
  --model-path /path/to/Qwen3-0.6B \
  --scheduler-policy mlfq_plas \
  --arrival-rate 2 \
  --arrival-seed 0 \
  --shuffle-programs \
  --out results/sharegpt-smoke.jsonl
```

BFCL fixture 使用相同入口。`--replay-steps` 控制每个 BFCL program 包含多少次顺序 LLM call：

```bash
python -m agentix_app.dataset_runner \
  --dataset bfcl \
  --input agentix_app/fixtures/bfcl_fixture_16.jsonl \
  --limit 16 \
  --replay-steps 3 \
  --model-path /path/to/Qwen3-0.6B \
  --scheduler-policy mlfq_plas \
  --arrival-rate 2 \
  --arrival-seed 0 \
  --shuffle-programs \
  --out results/bfcl-smoke.jsonl
```

fixture 适合检查脚本、模型和调度器能否正常协作，不代表论文规模的性能结果。

### 一次运行三种调度策略

一次 `dataset_runner` 进程只测试一种策略。可以使用下面的循环依次测试 `fcfs`、`plas` 和 `mlfq_plas`：

```bash
MODEL=/path/to/Qwen3-0.6B
DATASET=agentix_app/fixtures/sharegpt_fixture_16.json

for policy in fcfs plas mlfq_plas; do
  python -m agentix_app.dataset_runner \
    --dataset sharegpt \
    --input "$DATASET" \
    --limit 16 \
    --model-path "$MODEL" \
    --scheduler-policy "$policy" \
    --arrival-rate 2 \
    --arrival-seed 0 \
    --shuffle-programs \
    --out "results/sharegpt-${policy}.jsonl"
done
```

正式测试时，把 fixture 换成完整数据集并增大 `--limit`。为了让策略之间可比，以下参数必须保持一致：

- 数据集文件和 `--limit`
- `--arrival-rate`、`--arrival-seed` 和是否启用 `--shuffle-programs`
- 模型、`--max-tokens`、batch 配置和其他推理参数
- 运行机器和 GPU 配置

建议使用多个 `--arrival-seed` 重复测试，并扫描多档 `--arrival-rate`。到达率过低时几乎没有资源竞争，不容易体现调度策略的差异。

### Program-level replay 语义

测试过程如下：

1. 加载数据集，每条样本对应一个 program。
2. 可选地随机打乱 program，然后根据 `--arrival-rate` 生成泊松到达时间。
3. program 到达时只提交它的第一个 LLM call。
4. 一个 call 完成后，才提交同一 program 的下一个 call。
5. 所有 call 完成后，该 program 才计为完成。

因此，同一个 program 任意时刻最多有一个活跃 call，而不同 program 的 call 可以同时进入调度器。这与 Agent 场景中“大程序驱动多步小调用”的执行方式一致。

当 `--arrival-rate` 小于或等于 `0` 时，所有 program 都在测试开始时到达，但每个 program 内的 call 仍然顺序提交。

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--dataset` | 必填 | `sharegpt` 或 `bfcl` |
| `--input` | 必填 | 本地 JSON/JSONL 数据集路径 |
| `--limit` | `1` | 最多加载的 program 数量；正式测试需要显式增大 |
| `--scheduler-policy` | `mlfq_plas` | `fcfs`、`plas` 或 `mlfq_plas` |
| `--model-path` | 开发机路径 | 本地模型目录，建议总是显式设置 |
| `--arrival-rate` | `0` | 泊松 program 到达率，单位为 program/s；非正数表示同时到达 |
| `--arrival-seed` | `0` | program 打乱和到达时间采样所用随机种子 |
| `--shuffle-programs` | 关闭 | 在分配到达时间前随机打乱 program |
| `--replay-steps` | `3` | 每个 BFCL program 的顺序 call 数量 |
| `--max-tokens` | `8` | 每次 call 的最大输出 token 数 |
| `--temperature` | `0.7` | 生成采样温度 |
| `--ignore-eos` | 开启 | 是否忽略 EOS 并生成至 `--max-tokens`；可用 `--no-ignore-eos` 关闭 |
| `--max-model-len` | `4096` | 模型最大上下文长度 |
| `--max-num-seqs` | `512` | nano-vLLM 最大并发序列数 |
| `--max-num-batched-tokens` | `16384` | 每个 batch 的最大 token 数 |
| `--out` | 必填 | program 明细 JSONL 输出路径 |
| `--summary-out` | `<out>.summary.json` | 汇总 JSON 路径 |

完整参数可以通过以下命令查看：

```bash
python -m agentix_app.dataset_runner --help
```

## 测试结果

每次运行生成两个文件：

```text
results/sharegpt-mlfq_plas.jsonl
results/sharegpt-mlfq_plas.jsonl.summary.json
```

### Program 明细 JSONL

`--out` 指定的文件使用 JSONL 格式，每行对应一个 program。

| 字段 | 说明 |
| --- | --- |
| `program_id` | program 唯一标识 |
| `dataset` | 数据集名称 |
| `kind` | program 类型，例如 `chatbot` 或 `react` |
| `scheduler_policy` | 本次使用的调度策略 |
| `call_submission_mode` | 当前固定为 `program_sequential` |
| `arrival_offset_sec` | 该 program 相对测试开始时间的计划到达偏移 |
| `arrival_lag_sec` | 实际接纳相对计划到达时间的延迟 |
| `num_calls` | program 包含的 call 总数 |
| `num_submitted_calls` | 已提交 call 数 |
| `num_finished_calls` | 已完成 call 数 |
| `output_tokens` | program 所有 call 的输出 token 总数 |
| `started_at` | program 实际开始时的单调时钟值 |
| `ended_at` | program 完成时的单调时钟值 |
| `last_call_ended_at` | 最后一次 call 完成时的单调时钟值 |
| `program_latency_sec` | 从 program 实际开始到完成的延迟 |
| `service_time` | 调度器记录的 program 服务时间 |
| `wait_time` | 调度器记录的 program 等待时间 |
| `active_call_ids` | 测试结束时仍活跃的 call；正常完成时应为空 |
| `status` | `ok` 表示正常完成，`not_finished` 表示测试结束时未完成 |

`started_at`、`ended_at` 和 `last_call_ended_at` 来自进程内单调时钟，只用于计算时间差，不是 Unix 时间戳。

### 汇总 JSON

summary 文件用于不同调度策略之间的主要比较。

| 字段 | 说明 |
| --- | --- |
| `num_programs` | program 总数 |
| `num_ok` | 正常完成的 program 数 |
| `total_calls` | call 总数 |
| `total_output_tokens` | 输出 token 总数 |
| `elapsed_sec` | 整次测试耗时 |
| `avg_program_latency_sec` | program 平均延迟 |
| `p50_program_latency_sec` | program 延迟 P50 |
| `p95_program_latency_sec` | program 延迟 P95 |
| `p99_program_latency_sec` | program 延迟 P99 |
| `throughput_program_per_sec` | 每秒完成的 program 数 |
| `throughput_call_per_sec` | 每秒完成的 call 数 |
| `throughput_output_tok_per_sec` | 每秒生成的输出 token 数 |
| `dataset` | 数据集名称 |
| `scheduler_policy` | 调度策略 |
| `call_submission_mode` | call 提交模式，当前为 `program_sequential` |
| `arrival_rate_program_per_sec` | 配置的 program 到达率 |
| `arrival_seed` | 到达随机种子 |
| `shuffle_programs` | 是否打乱 program |
| `model_path` | 模型路径 |
| `max_tokens` | 单次 call 最大输出 token 数 |
| `max_num_seqs` | 最大并发序列数 |
| `max_num_batched_tokens` | batch 最大 token 数 |
| `ignore_eos` | 是否忽略 EOS，持续生成至 `max_tokens` |

比较调度器时，优先关注 `p95_program_latency_sec`、`p99_program_latency_sec` 和 `throughput_program_per_sec`。尾延迟越低、program 吞吐越高越好，同时要先确认各结果的 `num_ok`、`total_calls` 和 workload 参数一致。

## 查看结果

只使用 Python 查看格式化后的 summary：

```bash
python -m json.tool results/sharegpt-mlfq_plas.jsonl.summary.json
```

查看第一条 program 明细：

```bash
head -n 1 results/sharegpt-mlfq_plas.jsonl | python -m json.tool
```

安装了 `jq` 时，可以提取三种策略的核心指标：

```bash
jq '{
  scheduler_policy,
  num_ok,
  avg_program_latency_sec,
  p95_program_latency_sec,
  p99_program_latency_sec,
  throughput_program_per_sec
}' results/sharegpt-*.summary.json
```

查看延迟最高的 10 个 program：

```bash
jq -s '
  sort_by(.program_latency_sec)
  | reverse
  | .[:10]
  | map({program_id, program_latency_sec, wait_time, service_time, num_calls})
' results/sharegpt-mlfq_plas.jsonl
```

计算 program 的平均等待时间和平均服务时间：

```bash
jq -s '{
  avg_wait_time: (map(.wait_time) | add / length),
  avg_service_time: (map(.service_time) | add / length)
}' results/sharegpt-mlfq_plas.jsonl
```
