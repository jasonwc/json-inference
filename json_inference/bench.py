"""Benchmark a running model through its OpenAI-compatible API.

Measures, all client-side over streaming chat completions:
  smoke     17 x 19 = 323 answered correctly (catches a broken model/template)
  decode    one stream: time to first token and decode tokens/s
  prefill   prompts of N tokens: prompt tokens / time to first visible token.
            Prompts are random words, so the prefix cache can't help.
  parallel  1/2/4 concurrent streams: aggregate decode tokens/s

Token counts come from the server's usage report and include reasoning
tokens, since those cost the same time to generate.
"""

import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .config import REPO_ROOT, RESULTS_DIR, Model

DECODE_PROMPT = "Write a detailed, 600-word short story about a lighthouse keeper who finds a message in a bottle."
WORDS = (
    "river stone lantern orbit copper meadow signal harbor quiet ember "
    "falcon glacier thunder violet cedar canyon beacon marble prism tide"
).split()


def _post(url: str, body: dict, timeout: float = 1800):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    usage = None
    with _post(f"{base}/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                    first = time.perf_counter()
    end = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError("stream ended without tokens or usage")
    out_tokens = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": out_tokens,
        "ttft_s": round(first - start, 3),
        "decode_tok_s": round((out_tokens - 1) / (end - first), 2) if out_tokens > 1 and end > first else None,
        "start": start,
        "end": end,
    }


def smoke(base: str, model: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the number."}],
        "max_tokens": 4096,
        "temperature": 0,
    }
    with _post(f"{base}/chat/completions", body, timeout=600) as resp:
        msg = json.load(resp)["choices"][0]["message"]
    answer = (msg.get("content") or "").strip()
    return {"pass": "323" in answer, "answer": answer[-200:]}


def random_prompt(tokens: int) -> str:
    rng = random.Random()
    # ~1.3 tokens per word for these words; the server reports the real count.
    words = " ".join(rng.choice(WORDS) for _ in range(int(tokens / 1.3)))
    return f"Here is a list of words:\n{words}\nHow many times does the word 'river' appear? Answer briefly."


def parallel(base: str, model: str, streams: int) -> dict:
    results: list[dict] = []
    errors: list[str] = []

    def one():
        try:
            results.append(stream_chat(base, model, DECODE_PROMPT, 256))
        except Exception as e:  # noqa: BLE001 - recorded in the results
            errors.append(str(e))

    threads = [threading.Thread(target=one) for _ in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        return {"streams": streams, "error": errors[0]}
    span = max(r["end"] for r in results) - min(r["start"] for r in results)
    total = sum(r["completion_tokens"] for r in results)
    return {
        "streams": streams,
        "aggregate_tok_s": round(total / span, 2),
        "mean_ttft_s": round(sum(r["ttft_s"] for r in results) / streams, 3),
    }


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("start", "end")}


def _git_rev() -> str:
    rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True)
    return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def run(base: str, model: Model) -> dict:
    name = model.served_name
    report: dict = {
        "model": model.name,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "json_inference_rev": _git_rev(),
        "endpoint": base,
        "definition": model.raw,
    }

    print("smoke ...", flush=True)
    report["smoke"] = smoke(base, name)
    print(f"  {'pass' if report['smoke']['pass'] else 'FAIL'}: {report['smoke']['answer']!r}")

    print("decode (1 stream, 512 tokens) ...", flush=True)
    stream_chat(base, name, "Say hi.", 8)  # warm-up
    d = _strip(stream_chat(base, name, DECODE_PROMPT, 512))
    report["decode"] = d
    print(f"  ttft {d['ttft_s']} s, decode {d['decode_tok_s']} tok/s")

    report["prefill"] = []
    for n in model.prefill_tokens:
        print(f"prefill (~{n} tokens) ...", flush=True)
        try:
            # 16, not 1: some chat formats (gpt-oss harmony) spend their first
            # tokens on markup that never reaches the client. TTFT is the first
            # visible token, still dominated by prefill at these lengths.
            r = stream_chat(base, name, random_prompt(n), 16)
            p = {
                "target_tokens": n,
                "prompt_tokens": r["prompt_tokens"],
                "ttft_s": r["ttft_s"],
                "prefill_tok_s": round(r["prompt_tokens"] / r["ttft_s"], 1),
            }
        except (urllib.error.URLError, RuntimeError) as e:
            p = {"target_tokens": n, "error": str(e)}
        report["prefill"].append(p)
        print(f"  {p}")

    report["parallel"] = []
    for streams in (1, 2, 4):
        print(f"parallel ({streams} streams x 256 tokens) ...", flush=True)
        p = parallel(base, name, streams)
        report["parallel"].append(p)
        print(f"  {p}")

    out_dir = RESULTS_DIR / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['timestamp'].replace(':', '')}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return report


def summary() -> None:
    """Latest result per model, side by side."""
    rows = []
    for model_dir in sorted(RESULTS_DIR.glob("*/")):
        latest = sorted(model_dir.glob("*.json"))
        if not latest:
            continue
        r = json.loads(latest[-1].read_text())
        prefill = {p["target_tokens"]: p.get("prefill_tok_s", "err") for p in r["prefill"]}
        agg = {p["streams"]: p.get("aggregate_tok_s", "err") for p in r["parallel"]}
        rows.append(
            [
                r["model"],
                r["timestamp"][:10],
                "ok" if r["smoke"]["pass"] else "FAIL",
                r["decode"]["ttft_s"],
                r["decode"]["decode_tok_s"],
                " / ".join(f"{k // 1024}k:{v}" for k, v in sorted(prefill.items())),
                agg.get(4, "-"),
            ]
        )
    header = ["model", "date", "smoke", "ttft s", "decode tok/s", "prefill tok/s", "x4 agg tok/s"]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
