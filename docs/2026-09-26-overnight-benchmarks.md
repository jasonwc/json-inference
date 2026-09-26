# 2026-09-26: first full benchmark sweep

Every model definition, benchmarked back to back with `inference sweep` on
prose and code (`json_inference/bench.py` describes the tests). The raw
numbers are in `results/`, and `inference results` prints the latest run of
each model.

## Results

"Just me" is single-stream decode speed, which is what one user feels.
Prefill is prompt tokens/s at ~32k tokens. "x4" is aggregate tokens/s over
four concurrent requests. Start is seconds from launch until the API
answers, on a warm second boot.

| Model | Nodes | Just me: prose / code | Prefill 32k | x4 prose / code | Start |
|---|---|---|---|---|---|
| gpt-oss-120b-tp2 | 2 | **53.6 / 54.1** | **4,601** | **148 / 172** | 166 s |
| qwen3.8-flash-next-nvfp4 | 2 | 39.9 / 51.7 | 2,633 | 118 / 118 | 768 s |
| gpt-oss-120b | 1 | 38.7 / 38.6 | 2,842 | 94 / 121 | 516 s |
| deepseek-v4-flash-dspark | 2 | 32.7 / 46.2 | 1,904 | 80 / 86 | 570 s |
| deepseek-v4.1-flash-exl3 | 2 | 28.2 / 46.0 | 1,240 | 40 / 54 | 567 s |
| mimo-v2.6-flash | 2 | 27.0 / 32.7 | 2,217 | 59 / 76 | 440 s |
| glm-5.3-flash-exl3 | 2 | 21.6 / 32.3 | 1,532 | 50 / 50 | 199 s |
| mimo-v2.6-flash-dflash | 2 | 20.5 / 39.6 | 2,327 | 45 / 81 | 458 s |
| qwen3.8-27b-nvfp4 | 1 | 19.8 / 24.2 | 1,919 | 76 / 83 | 328 s |

All nine pass the smoke test.

How to read it:
- **Speed isn't quality.** gpt-oss-120b is the fastest by far, but it's the
  weakest on current open-model leaderboards. Artificial Analysis'
  intelligence index puts gpt-oss-120b at ~12, against 34–42 for the
  Qwen3.8, DeepSeek-V4.x and GLM-5.3 models.
- **Code runs faster than prose** on the models with speculative decoding
  (MTP, DSpark, DFlash), because code is more predictable. The code test runs
  at temperature 0, the prose test at 0.7, and thinking stays at each
  recipe's default. Recipe READMEs often quote higher numbers measured with
  thinking off.

## Suggested picks

- **Interactive coding and agents:** `deepseek-v4-flash-dspark` or
  `qwen3.8-flash-next-nvfp4`. Both decode code at ~46–52 tok/s and prefill
  ~1.9–2.6k tok/s. DSpark has a 1M-token context and the better 4-stream
  scaling among the DeepSeeks; Qwen3.8-Flash-Next is faster on prose.
- **Long unattended jobs where correctness matters more than speed:**
  `glm-5.3-flash-exl3`. It's the highest-ranked of the set on quality.
  Tool calling on GLM-5.x in vLLM has known bugs, so check it with your
  harness.
- **One Spark, leaving the other free:** `qwen3.8-27b-nvfp4`. It's slow for
  a single user (~20–24 tok/s) but scales well (76–83 tok/s across 4
  requests).
- **Maximum raw speed:** `gpt-oss-120b-tp2`.
- **Superseded:** `deepseek-v4.1-flash-exl3`. DSpark matches it on code,
  beats it on prose, prefill and concurrency, and boots in the same time.

## Changes made during the run

| Change | Where | Effect |
|---|---|---|
| QSFP MTU 1500 → 9000 (RoCE MTU 1024 → 4096) | json-lab `sparks/common.nix` | NCCL all_gather 21.4 → 23.4 GB/s |
| DeepSeek-V4.1 `pack`: Engram rows on each node's NVMe instead of NFS | `models/deepseek-v4.1-flash-exl3.toml` | prefill 455–739 → 977–1,240 tok/s; boot 1,141 → 567 s |
| vLLM compile caches on the host | `engines.py` | gpt-oss TP=2 boot ~7 min → 166 s |
| Qwen3.8-27B MTP: 5 → 3 speculative tokens | `models/qwen3.8-27b-nvfp4.toml` | x4 prose 66 → 76 tok/s; x4 code went from failing to 83 tok/s |
| MiMo `MEM_FRACTION_STATIC` 0.93 → 0.90 | `models/mimo-v2.6-flash*.toml` | at 0.93 the recipe's memory guard killed the worker during boot |
| MiMo DFlash variant | `models/mimo-v2.6-flash-dflash.toml` | code +21%, prose −24% vs EAGLE |

## Problems found and fixed in json-inference

- **Recipes' NFS servers collided.** Each two-Spark recipe runs its own NFS
  server on the head, all on port 2049, and none of their stop commands
  remove it. Kernel nfsd threads also outlive the container: its init hangs
  reaping them, and a new server never registers with rpcbind. `down` now
  removes each definition's `nfs_container` and stops kernel nfsd.
- **The DSpark recipe runs compose on the worker from the same path.** The
  launcher can now mirror its checkout (`worker_checkout`), name its env file
  (`env_file`) and image key (`image_key`), and pre-pull the image on both
  nodes before the recipe's own pull step, which refuses to run without it.
- **`down` stopped recipes in alphabetical order.** MiMo's stop waits for the
  GPU, so it sat waiting while another recipe still held it, and a switch
  took over 6 minutes. The serving recipe now stops first.
- **Sweeps:** `inference sweep` benchmarks models unattended and logs
  failures instead of stopping.

## Not changed, worth knowing

- **DeepSeek-V4.1 serves only 2 requests at a time** (`MAX_NUM_SEQS=2`), so
  4 streams queue (x4 ≈ x2). Its recipe says raising this costs memory it
  barely has, and that `SPEC_METHOD=none` gives better 4-stream throughput
  but slower single streams. It stays on the recipe defaults.
- **Qwen3.8-27B on SGLang with a DFlash drafter** (MiaAI's
  Qwen3.8-27B-SGLang-DGX-Spark recipe) reports ~51 tok/s on code, about 2×
  this vLLM + MTP setup. It's the obvious next variant to try.
- **Prose vs code conditions:** recipe READMEs measure with thinking off.
  A thinking-off benchmark variant would make the numbers directly
  comparable to theirs.
