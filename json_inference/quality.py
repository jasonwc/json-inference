"""Quick quality check of a running model: `inference bench MODEL --quality`.

Four small suites, scored deterministically (no judge model), temperature 0:
  math         word problems with one exact integer answer (last number in the reply)
  knowledge    multiple choice A-D (last standalone letter in the reply)
  instructions replies checked against verifiable format rules
  recall       a fact buried in filler text at ~2k / 8k / 32k tokens, asked for at the end

Items are written for this repo (json_inference/quality_items.json), so they can't
have leaked into training sets as-is; they're small on purpose: this is a smoke test for
quality regressions between models and quants, not a leaderboard. max_tokens is 2048 so
reasoning models have room to think before answering (Paper marks calls that return no
text as failed; anything under ~256 risks that).
Saved as results/<model>/quality-<timestamp>.json.
"""

import json
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT, RESULTS_DIR, Model
from .engines import git_rev

ITEMS = json.loads((Path(__file__).parent / "quality_items.json").read_text())
MAX_TOKENS = 2048
WORDS = ("river stone lantern copper meadow signal orchard harbor violet engine paper "
         "summit thread winter canyon pocket marble garden ladder beacon").split()


def ask(base: str, model: str, prompt: str) -> tuple[str, float]:
    body = {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        msg = json.load(r)["choices"][0]["message"]
    text = msg.get("content") or ""
    # Engines that inline their reasoning: keep only what follows it.
    text = re.sub(r"(?s)^.*</think>", "", text).strip()
    return text, round(time.monotonic() - t0, 2)


def last_int(text: str):
    nums = re.findall(r"-?\d[\d,]*", text)
    return int(nums[-1].replace(",", "")) if nums else None


def last_letter(text: str):
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1] if m else None


def check_instruction(rule: dict, text: str) -> bool:
    kind = rule["check"]
    lines = [l for l in text.splitlines() if l.strip()]
    if kind == "bullets":
        return len(lines) == rule["n"] and all(l.startswith("- ") for l in lines)
    if kind == "lowercase":
        return bool(text) and text == text.lower() and any(c.isalpha() for c in text)
    if kind == "json_keys":
        try:
            obj = json.loads(re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip())
        except ValueError:
            return False
        return isinstance(obj, dict) and sorted(obj) == sorted(rule["keys"])
    if kind == "word_count":
        return len(re.findall(r"[A-Za-z']+", text)) == rule["n"]
    if kind == "ends_with":
        return text.rstrip().rstrip(".!").endswith(rule["word"])
    if kind == "no_letter":
        return bool(text) and rule["letter"] not in text.lower()
    raise ValueError(kind)


def haystack(tokens: int, needle: str, seed: int) -> str:
    rnd = random.Random(seed)
    words = [rnd.choice(WORDS) for _ in range(int(tokens * 0.75))]
    sentences = [" ".join(words[i:i + 12]).capitalize() + "." for i in range(0, len(words), 12)]
    sentences.insert(len(sentences) // 2, needle)
    return " ".join(sentences)


def run(base: str, model: Model) -> dict:
    name = model.served_name
    report = {"model": model.name, "kind": "quality",
              "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "json_inference_rev": git_rev(), "endpoint": base, "max_tokens": MAX_TOKENS,
              "suites": {}}

    def suite(label, items, score):
        rows = []
        for it in items:
            try:
                text, secs = ask(base, name, it["prompt"])
                ok = score(it, text)
            except (urllib.error.URLError, RuntimeError, KeyError, ValueError) as e:
                text, secs, ok = f"error: {e}", None, False
            rows.append({"id": it["id"], "pass": ok, "seconds": secs, "reply": text[-300:]})
        passed = sum(r["pass"] for r in rows)
        report["suites"][label] = {"passed": passed, "total": len(rows), "items": rows}
        print(f"  {label}: {passed}/{len(rows)}", flush=True)

    print("quality (temperature 0) ...", flush=True)
    suite("math", ITEMS["math"], lambda it, t: last_int(t) == it["answer"])
    suite("knowledge", ITEMS["knowledge"], lambda it, t: last_letter(t) == it["answer"])
    suite("instructions", ITEMS["instructions"], lambda it, t: check_instruction(it, t))
    recall = []
    for i, n in enumerate((2000, 8000, 32000)):
        code = 1000 + random.Random(n).randrange(9000)
        needle = f"The access code for the Heron project is {code}."
        prompt = (haystack(n, needle, n) + "\n\nWhat is the access code for the Heron project? "
                  "Answer with just the number.")
        recall.append({"id": f"recall-{n // 1000}k", "prompt": prompt, "answer": code})
    suite("recall", recall, lambda it, t: last_int(t) == it["answer"])

    total = sum(s["passed"] for s in report["suites"].values())
    count = sum(s["total"] for s in report["suites"].values())
    report["score"] = round(100 * total / count, 1)
    print(f"  overall: {total}/{count} ({report['score']}%)")
    out_dir = RESULTS_DIR / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quality-{report['timestamp'].replace(':', '')}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return report


def latest(model_dir: Path):
    files = sorted(model_dir.glob("quality-*.json"))
    return json.loads(files[-1].read_text()) if files else None
