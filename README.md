# Agentix nano-vLLM 复现

这个仓库提供 Agentix 应用层和 nano-vLLM 调度器的测试入口。当前 workload 以 **program** 为到达和完成单位：不同 program 可以并发运行，但同一个 program 在前一次 LLM call 完成后才会提交下一次 call，不会在测试开始时一次性提交全部 call。

## 运行环境

从**本仓库根目录**运行命令。代码会把 `nano-vllm` 作为本仓库的同级目录加载，推荐目录结构如下：

```text
<workspace>/
├── nano-vllm/
└── agentix_vllm_reproduce/   # 本仓库，目录名不需要改成 agentix_app
```

进入仓库根目录：

```bash
cd <workspace>/agentix_vllm_reproduce
```

模型需要已经下载到本地；正式运行时建议显式传入 `--model-path` 或脚本里的 `MODEL`，不要依赖代码中的开发机默认路径。

## 测试方式

### 真实 ShareGPT 数据

ShareGPT workload 应该重放完整多轮 conversation，而不是只使用第一轮。本仓库的数据脚本默认使用 [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)，并固定到 revision `192ab2185289094fc556ec8ce5ce1e8e587154ca` 下的 `ShareGPT_V3_unfiltered_cleaned_split.json`：

| 项目 | 值 |
| --- | --- |
| 原始文件大小 | `672837942` bytes，约 642 MiB |
| 许可证 | Apache-2.0 |
| program | 一条完整 conversation |
| LLM call | 一个 user/assistant turn |

仓库已经直接包含可运行的真实样本：

```text
benchmark_data/sharegpt-real-5000-seed0.jsonl.gz
benchmark_data/sharegpt-real-5000-seed0.stats.json
```

这份数据从完整数据集的 92,825 个有效 program 中以 seed `0` 做 reservoir sampling，包含 5,000 个 program 和 17,696 次 LLM call，压缩后约 9.1 MiB。runner 可以直接读取 `.jsonl.gz`，无需解压或额外下载。

下面的命令可以从固定 revision 重新生成完全相同的仓库内样本：

```bash
python3 scripts/prepare_sharegpt.py \
  --num-programs 5000 \
  --seed 0 \
  --output benchmark_data/sharegpt-real-5000-seed0.jsonl.gz
```

完整的 673 MB 原始文件缓存到已忽略的 `data/raw/`。stats 文件记录全量扫描数、有效 program 数、call 分布、数据 revision、许可证和压缩样本 SHA-256。

### 先检查数据集

在没有 nano-vLLM 或模型的机器上，可以先只检查数据集解析和 workload 规模：

```bash
python3 -m agentix_app.dataset_runner \
  --dataset sharegpt \
  --input benchmark_data/sharegpt-real-5000-seed0.jsonl.gz \
  --limit 5000 \
  --arrival-rate 2 \
  --arrival-seed 0 \
  --shuffle-programs \
  --validate-only
```

也可以用一键脚本做同样检查，此时不需要设置 `MODEL`：

```bash
VALIDATE_ONLY=1 \
DATASET=fixtures/sharegpt_fixture_16.json \
LIMIT=16 \
./scripts/run_benchmark.sh
```

这个检查不会启动推理，只会输出 program 数、call 数、每个 program 的 call 分布和 prompt 字符数分布。真正运行推理时会使用模型 tokenizer 执行 `--max-model-len` 过滤，summary 里会记录 `num_skipped_overlong_programs`。

### 快速冒烟测试

仓库自带 16 个 ShareGPT program 和 16 个 BFCL program 的小型 fixture。下面的命令在正式推理机器上运行 ShareGPT fixture：

```bash
DATASET=fixtures/sharegpt_fixture_16.json \
MODEL=/path/to/Qwen3-0.6B \
LIMIT=16 \
ARRIVAL_RATE=2 \
ARRIVAL_SEEDS="0" \
SCHEDULER_POLICIES="mlfq_plas" \
./scripts/run_benchmark.sh
```

BFCL fixture 使用相同入口。`REPLAY_STEPS` 控制每个 BFCL program 包含多少次顺序 LLM call：

```bash
DATASET_TYPE=bfcl \
DATASET=fixtures/bfcl_fixture_16.jsonl \
MODEL=/path/to/Qwen3-0.6B \
LIMIT=16 \
REPLAY_STEPS=3 \
ARRIVAL_RATE=2 \
ARRIVAL_SEEDS="0" \
SCHEDULER_POLICIES="mlfq_plas" \
./scripts/run_benchmark.sh
```

fixture 适合检查脚本、模型和调度器能否正常协作，不代表论文规模的性能结果。

### 一键跑三种策略

已有完整 ShareGPT 数据集时，可以直接指定路径，并启用真实回复长度 replay：

```bash
DATASET=/data/ShareGPT_V3_unfiltered_cleaned_split.json \
MODEL=/models/Qwen3-0.6B \
LIMIT=1000 \
ARRIVAL_RATE=2 \
ARRIVAL_SEEDS="0 1 2" \
OUTPUT_LENGTH_MODE=reference \
MAX_TOKENS=512 \
./scripts/run_benchmark.sh
```

脚本默认依次测试 `fcfs`、`plas` 和 `mlfq_plas`，结果保存在 `results/` 下。为了让策略之间可比，以下参数必须保持一致：

- 数据集文件和 `LIMIT`
- `ARRIVAL_RATE`、`ARRIVAL_SEEDS`、是否启用 `SHUFFLE_PROGRAMS`
- 模型、`MAX_TOKENS`、batch 配置和其他推理参数
- 运行机器和 GPU 配置

仓库内已经有 5000 条真实数据，一键测试无需下载数据：

```bash
./scripts/run_sharegpt_benchmark.sh /models/Qwen3-0.6B
```

这一条命令会先检查数据，再使用以下正式默认矩阵运行测试：

- 真实 ShareGPT program：`5000`
- program 到达率：`1 2 4` program/s
- seed：`0 1 2`
- 调度策略：`fcfs plas mlfq_plas`
- 输出长度：数据集真实 assistant 回复的 token 数，单次 call 最多 `512`

结果写入 `results/sharegpt-real/`。这会启动 27 次推理测试，可能运行数小时；中断后可以通过环境变量缩小矩阵重新运行。只有显式指定其他 `PREPARE_PROGRAMS` 或 `DATASET` 时，脚本才会准备另一份数据。

例如只做一次较短的推理链路检查：

```bash
LIMIT=16 \
ARRIVAL_RATES="2" \
ARRIVAL_SEEDS="0" \
./scripts/run_sharegpt_benchmark.sh /models/Qwen3-0.6B
```

fixture 只有 16 个 program，只适合冒烟，不够支撑正式性能结论。正式测试建议至少 `LIMIT=1000`；如果要看 P99，优先使用 5000 个以上 program，并使用多个 seed 重复测试。还需要扫描多档 `ARRIVAL_RATE`，因为到达率过低时几乎没有资源竞争，不容易体现调度策略差异。

增大 `LIMIT` 本身不会制造调度收益。它的作用是让到达过程持续更久、减少均值和 P95/P99 的随机波动，并覆盖更多长短不同的 program。调度器只有在多个 program 同时等待 GPU、且 program 的 call 数和 token 数存在差异时，才有优化空间；系统处于低负载时，三种策略接近是正常结果。

判断负载是否足够时，先看 FCFS summary：如果 `throughput_program_per_sec` 基本跟随配置的到达率，且 program 延迟随到达率升高仍没有明显上升，说明还未进入排队区间。此时应提高到达率，例如：

```bash
ARRIVAL_RATES="2 4 8" \
./scripts/run_sharegpt_benchmark.sh /models/Qwen3-0.6B
```

因此，5,000 条数据会让结论更可信，但不能保证调度器一定优于 FCFS。应在出现排队竞争的到达率下，比较三种策略的平均 program 延迟、P95/P99 和吞吐；调度优化通常首先体现在 program 延迟分布，而不一定体现在总吞吐。

### Program-level replay 语义

测试过程如下：

1. 加载数据集，每条样本对应一个 program。
2. 可选地随机打乱 program，然后根据 `--arrival-rate` 生成泊松到达时间。
3. program 到达时只提交它的第一个 LLM call。
4. 一个 call 完成后，才提交同一 program 的下一个 call。
5. 所有 call 完成后，该 program 才计为完成。

因此，同一个 program 任意时刻最多有一个活跃 call，而不同 program 的 call 可以同时进入调度器。这与 Agent 场景中“大程序驱动多步小调用”的执行方式一致。

当 `--arrival-rate` 小于或等于 `0` 时，所有 program 都在测试开始时到达，但每个 program 内的 call 仍然顺序提交。

正式 ShareGPT 测试使用 `--output-length-mode reference --ignore-eos`：每次 call 按数据集中对应 assistant 回复的 tokenizer 长度执行 decode，并受 `--max-tokens` 上限约束。后续 prompt 使用数据集中的真实 conversation history，从而稳定重放相同的 prefill/decode workload。

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--dataset` | 必填 | `sharegpt` 或 `bfcl` |
| `--input` | 必填 | 本地 JSON/JSONL 数据集路径 |
| `--limit` | `1` | 打乱和上下文过滤后选取的 program 数量；`0` 表示全部 |
| `--scheduler-policy` | `mlfq_plas` | `fcfs`、`plas` 或 `mlfq_plas` |
| `--model-path` | 开发机路径 | 本地模型目录，建议总是显式设置 |
| `--arrival-rate` | `0` | 泊松 program 到达率，单位为 program/s；非正数表示同时到达 |
| `--arrival-seed` | `0` | program 到达时间采样所用随机种子 |
| `--sample-seed` | `--arrival-seed` | `--shuffle-programs` 的随机种子；用于在 `--limit` 前随机抽样 |
| `--shuffle-programs` | 关闭 | 在上下文过滤和 `--limit` 前随机打乱 program |
| `--replay-steps` | `3` | 每个 BFCL program 的顺序 call 数量 |
| `--max-tokens` | `8` | 每次 call 的最大输出 token 数 |
| `--output-length-mode` | `fixed` | `fixed` 使用固定 `--max-tokens`；`reference` 使用参考答案长度并受 `--max-tokens` 封顶 |
| `--temperature` | `0.7` | 生成采样温度 |
| `--ignore-eos` | 开启 | 是否忽略 EOS 并生成至 `--max-tokens`；可用 `--no-ignore-eos` 关闭 |
| `--max-model-len` | `4096` | 模型最大上下文长度 |
| `--max-num-seqs` | `512` | nano-vLLM 最大并发序列数 |
| `--max-num-batched-tokens` | `16384` | 每个 batch 的最大 token 数 |
| `--skip-overlong-programs` | 开启 | 跳过任一 call 超过 `--max-model-len` 的 program |
| `--validate-only` | 关闭 | 只解析和汇总 workload，不启动 nano-vLLM |
| `--out` | 必填 | program 明细 JSONL 输出路径；`--validate-only` 时不需要 |
| `--summary-out` | `<out>.summary.json` | 汇总 JSON 路径 |

完整参数可以通过以下命令查看：

```bash
python3 -m agentix_app.dataset_runner --help
```

## 测试结果

每次运行生成两个文件：

```text
results/sharegpt-limit1000-rate2-seed0-mlfq_plas.jsonl
results/sharegpt-limit1000-rate2-seed0-mlfq_plas.jsonl.summary.json
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
| `sample_seed` | program 抽样和打乱随机种子 |
| `shuffle_programs` | 是否打乱 program |
| `model_path` | 模型路径 |
| `max_tokens` | 单次 call 最大输出 token 数 |
| `output_length_mode` | 输出长度模式 |
| `max_num_seqs` | 最大并发序列数 |
| `max_num_batched_tokens` | batch 最大 token 数 |
| `ignore_eos` | 是否忽略 EOS，持续生成至 `max_tokens` |
| `skip_overlong_programs` | 是否启用上下文长度过滤 |
| `num_loaded_programs` | 数据集中解析出的 runnable program 数 |
| `num_skipped_overlong_programs` | 因上下文过长被跳过的 program 数 |
| `num_selected_programs` | 实际进入 replay 的 program 数 |

比较调度器时，优先关注 `p95_program_latency_sec`、`p99_program_latency_sec` 和 `throughput_program_per_sec`。尾延迟越低、program 吞吐越高越好，同时要先确认各结果的 `num_ok`、`total_calls`、`num_selected_programs` 和 workload 参数一致。

## 查看结果

只使用 Python 查看格式化后的 summary：

```bash
python3 -m json.tool results/sharegpt-limit1000-rate2-seed0-mlfq_plas.jsonl.summary.json
```

查看第一条 program 明细：

```bash
head -n 1 results/sharegpt-limit1000-rate2-seed0-mlfq_plas.jsonl | python3 -m json.tool
```

安装了 `jq` 时，可以提取三种策略的核心指标：

```bash
jq '{
  scheduler_policy,
  num_ok,
  num_selected_programs,
  num_skipped_overlong_programs,
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
' results/sharegpt-limit1000-rate2-seed0-mlfq_plas.jsonl
```

计算 program 的平均等待时间和平均服务时间：

```bash
jq -s '{
  avg_wait_time: (map(.wait_time) | add / length),
  avg_service_time: (map(.service_time) | add / length)
}' results/sharegpt-limit1000-rate2-seed0-mlfq_plas.jsonl
```

## BFCL 状态

论文的 BFCL workload 来自 Apache-2.0 的 [Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)。当前 `fixtures/bfcl_fixture_16.jsonl` 和 `datasets/bfcl.py` 只实现带 scripted Observation 的链路测试，没有执行 BFCLv3 的真实工具环境，因此不能用于复现论文 BFCL 结果。当前正式支持的数据集是上述真实 ShareGPT workload。
