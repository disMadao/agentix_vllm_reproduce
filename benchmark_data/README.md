# Bundled ShareGPT benchmark sample

`sharegpt-real-5000-seed0.jsonl.gz` contains 5,000 complete, real ShareGPT conversations for program-level replay.

- Source: `anon8231489123/ShareGPT_Vicuna_unfiltered`
- Source file: `ShareGPT_V3_unfiltered_cleaned_split.json`
- Source revision: `192ab2185289094fc556ec8ce5ce1e8e587154ca`
- Source URL: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- License declared by the source dataset: Apache-2.0
- Sampling: reservoir sampling over all runnable conversations
- Sampling seed: `0`
- Program count: `5,000`
- LLM call count: `17,696`
- Compressed size: approximately `9.1 MiB`
- SHA-256: `291a228f926d95c13ba87c78d19c1f8ac3892ae93403a8028cd59317d1771202`

The companion `.stats.json` file records the full scan and sampled workload statistics. The gzip file is deterministic (`mtime=0`) and can be read directly by `dataset_runner.py`; do not extract it before use.

To reproduce the file from the pinned source revision:

```bash
python3 scripts/prepare_sharegpt.py \
  --num-programs 5000 \
  --seed 0 \
  --output benchmark_data/sharegpt-real-5000-seed0.jsonl.gz
```
