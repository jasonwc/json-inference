"""`inference serve-ui`: a small web page for swapping the served model.

Shows what the Sparks serve right now (with vLLM's live running/waiting
requests), every definition in models/ with its latest benchmark, and offers
Swap (`inference up MODEL`) and Stop (`inference down`). Pulls, benchmarks and
sweeps stay on the command line.

One operation at a time: the job runs as a child `inference` process, its state
and log live under ~/.local/state/json-inference/ui/ so a page reload (or a
restart of this server) still shows the last result.

Security: Authelia's Remote-User from INFERENCE_UI_TRUSTED_PROXIES (the lab ingress hosts), else
HTTP basic auth from INFERENCE_UI_USER / INFERENCE_UI_PASSWORD
(required; the server refuses to start without them), and POSTs must carry the
X-Inference-UI header with a same-origin Origin (a cross-site form can't), so a
logged-in browser can't be tricked into swapping. Model names are checked against
the definitions and passed to the CLI as argv, never through a shell.

--dry-run replaces the CLI with a short simulation, for testing the page without
touching the Sparks.
"""

import base64
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import REPO_ROOT, RESULTS_DIR, list_models, load_cluster

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "json-inference/ui"
JOB_FILE = STATE_DIR / "job.json"
LOG_FILE = STATE_DIR / "job.log"
# Paper's gateway routes by exact model name, and Jason's production org (qse15, gateway
# `default`, backend json-sparks) routes only these; any other served name gets an empty
# 500 there, so Byrdpip falls back to the local endpoint uncaptured. Override with
# INFERENCE_UI_PAPER_MODELS (comma-separated).
PAPER_ROUTED = {m.strip() for m in os.environ.get("INFERENCE_UI_PAPER_MODELS", "deepseek-v4-flash-vision-exp").split(",") if m.strip()}

_lock = threading.Lock()


# --- state ----------------------------------------------------------------


def _read_job() -> dict:
    try:
        return json.loads(JOB_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_job(job: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = JOB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(job))
    tmp.replace(JOB_FILE)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _recover() -> None:
    """After a restart: a job marked running is interrupted if its process is
    gone, or watched until it exits (its exit code is lost: result unknown)."""
    job = _read_job()
    if job.get("state") != "running":
        return
    if not _alive(job.get("pid")):
        job.update(state="interrupted", ended=_now())
        _write_job(job)
        return

    def watch(pid) -> None:
        while _alive(pid):
            time.sleep(2)
        with _lock:
            j = _read_job()
            if j.get("pid") == pid and j.get("state") == "running":
                j.update(state="interrupted", ended=_now())
                _write_job(j)

    threading.Thread(target=watch, args=(job["pid"],), daemon=True).start()


def _log_tail(lines: int = 40) -> str:
    try:
        data = LOG_FILE.read_bytes()[-16384:]
    except OSError:
        return ""
    return "\n".join(data.decode(errors="replace").splitlines()[-lines:])


# --- what's serving ---------------------------------------------------------


def _head() -> str:
    return load_cluster().head.ssh


def _served() -> list[str] | None:
    try:
        with urllib.request.urlopen(f"http://{_head()}:8000/v1/models", timeout=3) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except (OSError, ValueError, KeyError):
        return None


def _activity() -> dict:
    out = {}
    try:
        with urllib.request.urlopen(f"http://{_head()}:8000/metrics", timeout=3) as r:
            text = r.read().decode(errors="replace")
    except OSError:
        return out
    for key, metric in (("running", "vllm:num_requests_running"), ("waiting", "vllm:num_requests_waiting")):
        vals = [float(m) for m in re.findall(rf"^{re.escape(metric)}\{{[^}}]*\}} ([0-9.e+-]+)$", text, re.M)]
        if vals:
            out[key] = int(sum(vals))
    return out


def _latest_results(name: str) -> dict:
    d = RESULTS_DIR / name
    benches = sorted(p for p in d.glob("*.json") if not p.name.startswith(("up-", "quality-")))
    quals = sorted(d.glob("quality-*.json"))
    ups = sorted(d.glob("up-*.json"))
    r: dict = {}
    if benches:
        try:
            b = json.loads(benches[-1].read_text())
            r = {
                "date": b.get("timestamp", "")[:10],
                "smoke": (b.get("smoke") or {}).get("pass"),
                "ttft_s": (b.get("decode") or {}).get("ttft_s"),
                "decode_tok_s": (b.get("decode") or {}).get("decode_tok_s"),
                "code_tok_s": (b.get("decode_code") or {}).get("decode_tok_s"),
            }
        except (OSError, ValueError):
            pass
    if quals:
        try:
            r["quality"] = json.loads(quals[-1].read_text()).get("score")
        except (OSError, ValueError):
            pass
    if ups:
        try:
            r["start_s"] = json.loads(ups[-1].read_text()).get("start_s")
        except (OSError, ValueError):
            pass
    return r


def _models() -> list[dict]:
    return [
        {
            "name": m.name,
            "served_name": m.served_name,
            "description": m.description,
            "engine": m.engine,
            "nodes": m.nodes,
            "paper_unrouted": m.served_name not in PAPER_ROUTED,
            "results": _latest_results(m.name),
        }
        for m in list_models()
    ]


def state() -> dict:
    job = _read_job()
    if job:
        job["log"] = _log_tail()
    return {"serving": _served(), "activity": _activity(), "job": job, "models": _models()}


# --- jobs -------------------------------------------------------------------


def _command(op: str, model: str | None, dry_run: bool) -> list[str]:
    if dry_run:
        what = f"inference {op}{' ' + model if model else ''}"
        script = (
            "import time\n"
            f"print({'[dry-run] would run: ' + what!r}, flush=True)\n"
            "for step in ('stopping whatever is serving', 'starting', 'waiting for the API'):\n"
            "    print('[dry-run]', step, flush=True); time.sleep(3)\n"
            "print('[dry-run] done', flush=True)\n"
        )
        return [sys.executable, "-c", script]
    return [sys.executable, "-m", "json_inference", op] + ([model] if model else [])


def start_job(op: str, model: str | None, dry_run: bool) -> tuple[bool, str]:
    if op not in ("up", "down"):
        return False, "unknown operation"
    if op == "up":
        names = {m.name for m in list_models()}
        if model not in names:
            return False, "unknown model"
    else:
        model = None
    with _lock:
        if _read_job().get("state") == "running":
            return False, "another operation is still running"
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "wb")
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            _command(op, model, dry_run), cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        log.close()
        job = {"op": op, "model": model, "state": "running", "pid": proc.pid, "started": _now(), "dry_run": dry_run}
        _write_job(job)

    def wait() -> None:
        rc = proc.wait()
        with _lock:
            j = _read_job()
            if j.get("pid") == proc.pid:
                j.update(state="ok" if rc == 0 else "failed", rc=rc, ended=_now())
                _write_job(j)

    threading.Thread(target=wait, daemon=True).start()
    return True, "started"


# --- HTTP -------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "inference-ui"
    user = ""
    password = ""
    dry_run = False
    trusted_proxies: frozenset = frozenset()

    def log_message(self, fmt, *args):  # one line per request, no auth headers
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _authed(self) -> bool:
        # Behind the lab ingress, Authelia has already signed the user in and passes them
        # in Remote-User. Trusted only from the ingress hosts (INFERENCE_UI_TRUSTED_PROXIES,
        # the k3s nodes: their traffic arrives SNATed to the node address) and only for the
        # configured user; anyone else, e.g. direct LAN access, still needs basic auth.
        if self.client_address[0] in self.trusted_proxies:
            remote = self.headers.get("Remote-User", "")
            if remote and hmac.compare_digest(remote, self.user):
                return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(h[6:]).decode().partition(":")
            except (ValueError, UnicodeDecodeError):
                return False
            return hmac.compare_digest(user, self.user) & hmac.compare_digest(pw, self.password)
        return False

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _gate(self) -> bool:
        if self._authed():
            return True
        self._send(401, b"login required\n", "text/plain", {"WWW-Authenticate": 'Basic realm="inference"'})
        return False

    def do_GET(self) -> None:
        if not self._gate():
            return
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(200, state())
        else:
            self._send(404, b"not found\n", "text/plain")

    def do_POST(self) -> None:
        if not self._gate():
            return
        origin = self.headers.get("Origin")
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host", "")
        if self.headers.get("X-Inference-UI") != "1" or (origin and origin.split("://", 1)[-1] != host):
            self._json(403, {"error": "cross-site request refused"})
            return
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)) or b"{}")
        except ValueError:
            body = {}
        if self.path == "/api/up":
            ok, msg = start_job("up", str(body.get("model", "")), self.dry_run)
        elif self.path == "/api/down":
            ok, msg = start_job("down", None, self.dry_run)
        else:
            self._send(404, b"not found\n", "text/plain")
            return
        self._json(200 if ok else 409, {"ok": ok, "message": msg})


def serve(host: str, port: int, dry_run: bool) -> None:
    user, password = os.environ.get("INFERENCE_UI_USER", ""), os.environ.get("INFERENCE_UI_PASSWORD", "")
    if not user or len(password) < 12:
        raise SystemExit("set INFERENCE_UI_USER and INFERENCE_UI_PASSWORD (12+ characters)")
    Handler.user, Handler.password, Handler.dry_run = user, password, dry_run
    Handler.trusted_proxies = frozenset(
        a.strip() for a in os.environ.get("INFERENCE_UI_TRUSTED_PROXIES", "").split(",") if a.strip())
    _recover()
    print(f"[inference] serve-ui on http://{host}:{port}{' (dry run)' if dry_run else ''}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model swap</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d2330;--muted:#667085;--card:#fff;--line:#e3e6ec;--accent:#2563eb;--ok:#15803d;--bad:#b42318;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--fg:#e6e9ef;--muted:#98a2b3;--card:#171b23;--line:#262c38;--accent:#5b8def;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1000px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.now{display:flex;flex-wrap:wrap;gap:18px;align-items:baseline}.big{font-size:20px;font-weight:600}.muted{color:var(--muted)}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600}td.num{text-align:right;white-space:nowrap}
button{font:inherit;padding:6px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.45;cursor:default}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:999px;border:1px solid currentColor;margin-left:6px}
.serving{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
pre{white-space:pre-wrap;word-break:break-all;background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:8px;max-height:280px;overflow:auto;font-size:12px}
@media (max-width:640px){.hide-sm{display:none}}
</style></head><body><main>
<h1>Model swap <span class="muted" style="font-size:14px">· DGX Sparks</span></h1>
<div class="card"><div class="now"><div><div class="muted">Serving</div><div class="big" id="serving">…</div></div>
<div><div class="muted">Running</div><div class="big" id="running">–</div></div>
<div><div class="muted">Waiting</div><div class="big" id="waiting">–</div></div>
<div style="margin-left:auto"><button id="stop">Stop</button></div></div></div>
<div class="card" id="jobcard" hidden><div><b id="jobtitle"></b> <span id="jobstate"></span></div><pre id="joblog"></pre></div>
<div class="card"><table><thead><tr><th>Model</th><th class="num">TTFT s</th><th class="num">prose tok/s</th>
<th class="num hide-sm">code tok/s</th><th class="num hide-sm">quality %</th><th class="num hide-sm">start s</th><th class="hide-sm">benched</th><th></th></tr></thead>
<tbody id="models"></tbody></table>
<p class="muted" style="margin:10px 0 0">Swapping stops the current model first: Byrdpip, Osty, voice and Paper have no model until the new one answers (several minutes).</p></div>
</main><script>
const WARN="Byrdpip, Osty, voice and Paper lose the model for several minutes while the Sparks switch.";
let busy=false;
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const v=x=>x===undefined||x===null?"–":x;
async function post(path,body){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Inference-UI":"1"},body:JSON.stringify(body||{})});
 const j=await r.json().catch(()=>({message:r.status}));if(!r.ok)alert(j.message||j.error||r.status);refresh();}
function swap(name,unrouted){let m="Serve "+name+"?\\n\\n"+WARN;if(unrouted)m+="\\n\\nNote: Paper doesn't route this model, so Byrdpip's turns fall back to the local endpoint (not captured).";
 if(confirm(m))post("/api/up",{model:name});}
document.getElementById("stop").onclick=()=>{if(confirm("Stop every model on the Sparks?\\n\\n"+WARN+"\\nNothing will be served until you swap one in."))post("/api/down");};
async function refresh(){let s;try{s=await (await fetch("/api/state",{cache:"no-store"})).json();}catch(e){return;}
 const serving=s.serving||[];document.getElementById("serving").textContent=s.serving===null?"nothing (API not answering)":(serving.join(", ")||"nothing");
 document.getElementById("running").textContent=v(s.activity.running);document.getElementById("waiting").textContent=v(s.activity.waiting);
 const j=s.job||{};busy=j.state==="running";const jc=document.getElementById("jobcard");
 if(j.op){jc.hidden=false;document.getElementById("jobtitle").textContent=(j.op==="up"?"Swap to "+j.model:"Stop")+(j.dry_run?" (dry run)":"");
  const st=document.getElementById("jobstate");st.textContent={running:"· running…",ok:"· done",failed:"· failed (exit "+j.rc+")",interrupted:"· interrupted"}[j.state]||"";
  st.className=j.state==="failed"||j.state==="interrupted"?"bad":j.state==="ok"?"serving":"muted";
  const pre=document.getElementById("joblog");const atEnd=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-4;pre.textContent=j.log||"";if(atEnd)pre.scrollTop=pre.scrollHeight;}
 document.getElementById("stop").disabled=busy;
 document.getElementById("models").innerHTML=s.models.map(m=>{const r=m.results||{};const on=serving.includes(m.served_name);
  return `<tr><td><b>${esc(m.name)}</b>${on?'<span class="badge serving">serving</span>':""}${m.paper_unrouted?'<span class="badge warn" title="Paper routes exact model names; this one isn\\'t routed, so Byrdpip falls back to local (uncaptured)">Paper doesn\\'t route this model</span>':""}${r.smoke===false?'<span class="badge bad">smoke failed</span>':""}
  <div class="muted">${esc(m.description)} · ${m.nodes} Spark${m.nodes>1?"s":""} · ${esc(m.engine)}</div></td>
  <td class="num">${v(r.ttft_s)}</td><td class="num">${v(r.decode_tok_s)}</td><td class="num hide-sm">${v(r.code_tok_s)}</td><td class="num hide-sm">${v(r.quality)}</td><td class="num hide-sm">${v(r.start_s)}</td>
  <td class="hide-sm">${esc(r.date||"never")}</td><td><button class="primary" ${busy||on?"disabled":""} data-m="${esc(m.name)}" data-u="${m.paper_unrouted?1:0}">Swap</button></td></tr>`}).join("");
 document.querySelectorAll("button[data-m]").forEach(b=>b.onclick=()=>swap(b.dataset.m,b.dataset.u==="1"));}
refresh();setInterval(refresh,3000);
</script></body></html>
"""
