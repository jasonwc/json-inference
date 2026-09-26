"""Start, stop and stage models on the Sparks.

Two engines:
  vllm      json-inference runs vLLM containers itself, following NVIDIA's
            dgx-spark-playbooks/playbook-vllm: one container on the head for
            nodes = 1, or a Ray cluster across both Sparks (TP=2) for nodes = 2.
  launcher  a third-party recipe with its own scripts (e.g. MiaAI-Lab's
            DeepSeek recipe), checked out at a pinned commit on the head with a
            rendered .env.

Only one model runs at a time: large models take most of both nodes' unified
memory, so `up` always stops whatever else is running first.
"""

import json
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

from . import remote
from .config import REPO_ROOT, RESULTS_DIR, Cluster, Model, Node, list_models, render

LABEL = "json-inference.model"
STATE_DIR = "~/json-inference"
HF_CACHE = "$HOME/.cache/huggingface"
COMPILE_CACHE = "$HOME/.cache/json-inference-compile"
READY_TIMEOUT_S = 45 * 60


# --- detached remote jobs --------------------------------------------------
# Long steps (downloads, launcher boots) run under nohup on the Spark, so a
# dropped SSH session or a closed laptop doesn't kill them. The CLI follows the
# log until the job exits; Ctrl-C stops following, not the job. Each job is its
# own session (setsid), so `cancel` stops it and everything it spawned by
# process group; child processes' names (e.g. hf's python wrapper) don't matter.


def run_detached(cluster: Cluster, node: Node, job: str, script: str) -> str:
    log_path = f"{STATE_DIR}/logs/{job}.log"
    pid = remote.output(
        cluster,
        node,
        f"""mkdir -p {STATE_DIR}/logs
pidfile={STATE_DIR}/logs/{job}.pid
if [ -f "$pidfile" ] && kill -0 -- "-$(cat "$pidfile")" 2>/dev/null; then
  echo "{job} is already running (pgid $(cat "$pidfile")); \\`inference cancel\\` it first" >&2
  exit 1
fi
cat > {STATE_DIR}/logs/{job}.sh <<'JSON_INFERENCE_JOB'
set -euo pipefail
{script}
JSON_INFERENCE_JOB
setsid nohup bash -l {STATE_DIR}/logs/{job}.sh > {log_path} 2>&1 < /dev/null &
echo $! > "$pidfile"
echo $!""",
    )
    remote.log(node, f"{job}: pid {pid}, log {node.name}:{log_path}")
    return pid


def follow(cluster: Cluster, node: Node, job: str, pid: str) -> None:
    log_path = f"{STATE_DIR}/logs/{job}.log"
    try:
        remote.run(cluster, node, f"tail -n +1 -f --pid={pid} {log_path}", check=False)
    except KeyboardInterrupt:
        remote.log(node, f"{job} keeps running; its log is {node.name}:{log_path}")
        raise SystemExit(130)
    # The job's exit status isn't kept past the process; its script ends by
    # writing a marker line on success.
    if remote.output(cluster, node, f"tail -n 1 {log_path}") != f"[{job}] ok":
        raise SystemExit(f"{node.name}: {job} failed; see {log_path}")


def cancel(cluster: Cluster, model: Model) -> None:
    """Stop the model's detached jobs (pull/start) on both nodes."""
    for node in (cluster.head, cluster.worker):
        out = remote.output(
            cluster,
            node,
            f"""for pidfile in {STATE_DIR}/logs/*-{model.name}.pid; do
  [ -f "$pidfile" ] || continue
  pgid=$(cat "$pidfile")
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid"; echo "stopped $(basename "$pidfile" .pid) (pgid $pgid)"
  fi
  rm -f "$pidfile"
done""",
        )
        for line in out.splitlines():
            remote.log(node, line)


def detached(cluster: Cluster, node: Node, job: str, script: str, wait: bool = True) -> None:
    pid = run_detached(cluster, node, job, script + f'\necho "[{job}] ok"')
    if wait:
        follow(cluster, node, job, pid)


# --- vllm ------------------------------------------------------------------


def _docker_base(model: Model, name: str) -> list[str]:
    return [
        "docker", "run", "-d",
        "--name", name,
        "--label", f"{LABEL}={model.name}",
        "--gpus", "all",
        "--ipc", "host",
        "--network", "host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--shm-size", "16g",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        # Compile caches (vLLM's torch.compile, Triton, FlashInfer JIT) survive
        # the container, so only a model's first `up` pays for kernel builds.
        # Entries are keyed by vLLM version and model config, so images and
        # models share the directory safely.
        "-v", f"{COMPILE_CACHE}:/root/.cache/compile",
        "-v", f"{COMPILE_CACHE}/flashinfer:/root/.cache/flashinfer",
        "-e", "VLLM_CACHE_ROOT=/root/.cache/compile/vllm",
        "-e", "TRITON_CACHE_DIR=/root/.cache/compile/triton",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/root/.cache/compile/inductor",
        # Weights are staged by `pull`; never download at serve time.
        "-e", "HF_HUB_OFFLINE=1",
    ]  # fmt: skip


def _serve_args(model: Model) -> list[str]:
    v = model.vllm
    return [
        "vllm", "serve", v["model"],
        "--host", "0.0.0.0",
        "--port", str(model.port),
        "--served-model-name", model.served_name,
        *(["--revision", v["revision"]] if "revision" in v else []),
        *v.get("args", []),
    ]  # fmt: skip


def _q(args: list[str]) -> str:
    # $HOME stays expandable on the remote side; everything else is quoted.
    return " ".join(a if a.startswith("$HOME") or "=$HOME" in a else shlex.quote(a) for a in args)


def _nccl_env(cluster: Cluster, node: Node) -> list[str]:
    # Same pins NVIDIA's two-node vLLM playbook passes to run_cluster.sh.
    netdev = cluster.cx7_netdev
    env = {
        "VLLM_HOST_IP": node.cx7_ip,
        "MASTER_ADDR": cluster.head.cx7_ip,
        "UCX_NET_DEVICES": netdev,
        "NCCL_SOCKET_IFNAME": netdev,
        # No NCCL_IB_HCA pin: QSFP port 0 is two RoCE devices (one per PCIe
        # path) and NCCL needs both for full bandwidth, as in json-lab's
        # sparks/nccl-test.sh.
        "NCCL_IB_GID_INDEX": str(cluster.cx7_gid_index),
        "OMPI_MCA_btl_tcp_if_include": netdev,
        "GLOO_SOCKET_IFNAME": netdev,
        "TP_SOCKET_IFNAME": netdev,
        "RAY_memory_monitor_refresh_ms": "0",
    }
    return [a for k, v in env.items() for a in ("-e", f"{k}={v}")]


def vllm_pull(cluster: Cluster, model: Model, wait: bool) -> None:
    v = model.vllm
    revision = f" --revision {shlex.quote(v['revision'])}" if "revision" in v else ""
    # Some repos ship the same weights in several formats (gpt-oss: original/,
    # metal/); `download_exclude` skips the ones vLLM doesn't load.
    revision += "".join(f" --exclude {shlex.quote(p)}" for p in v.get("download_exclude", []))
    for node in cluster.nodes(model.nodes):
        detached(
            cluster,
            node,
            f"pull-{model.name}",
            f"docker pull -q {shlex.quote(v['image'])}\n"
            f"hf download {shlex.quote(v['model'])}{revision}",
            wait=wait,
        )


def vllm_up(cluster: Cluster, model: Model) -> None:
    image = model.vllm["image"]
    head = cluster.head
    if model.nodes == 1:
        name = f"inference-{model.name}"
        remote.log(head, f"starting {name}")
        remote.run(
            cluster, head, _q([*_docker_base(model, name), "--entrypoint", "", image, *_serve_args(model)])
        )
        return

    # nodes = 2: a Ray cluster over the QSFP link, then TP=2 `vllm serve` from
    # the head (NVIDIA playbook-vllm, "Two nodes (direct QSFP cable)").
    ray = "python3 -c 'import ray' 2>/dev/null || pip install -q --root-user-action=ignore 'ray[default]>=2.9'; "
    for node in (head, cluster.worker):
        if node == head:
            start = f"ray start --block --head --port=6379 --node-ip-address={head.cx7_ip}"
        else:
            start = f"ray start --block --address={head.cx7_ip}:6379 --node-ip-address={node.cx7_ip}"
        name = f"inference-{model.name}-ray"
        remote.log(node, f"starting {name}")
        remote.run(
            cluster,
            node,
            _q(
                [
                    *_docker_base(model, name),
                    *_nccl_env(cluster, node),
                    # RDMA for NCCL. Without the device NCCL silently falls back
                    # to TCP over the QSFP netdev (NVIDIA's run_cluster.sh omits it).
                    "--device", "/dev/infiniband",
                    "--cap-add", "IPC_LOCK",
                    "--entrypoint", "/bin/bash",
                    image, "-c", ray + start,
                ]  # fmt: skip
            ),
        )

    remote.log(head, "waiting for both Ray nodes")
    container = f"inference-{model.name}-ray"
    deadline = time.time() + 10 * 60
    while True:
        status = remote.run(
            cluster, head, f"docker exec {container} ray status 2>/dev/null || true", capture=True, check=False
        ).stdout
        if "/2.0 GPU" in status:
            break
        if time.time() > deadline:
            raise SystemExit("Ray cluster didn't reach 2 GPUs in 10 minutes; see `inference logs`")
        time.sleep(10)

    serve = _serve_args(model) + ["--tensor-parallel-size", "2", "--distributed-executor-backend", "ray"]
    remote.log(head, "starting vllm serve (TP=2)")
    remote.run(
        cluster,
        head,
        f"docker exec -d {container} bash -c {shlex.quote(_q(serve) + ' > /tmp/vllm-serve.log 2>&1')}",
    )


def vllm_logs(cluster: Cluster, model: Model, follow_logs: bool) -> None:
    f = "-f " if follow_logs else ""
    if model.nodes == 1:
        cmd = f"docker logs {f}--tail 200 inference-{model.name}"
    else:
        cmd = f"docker exec inference-{model.name}-ray tail {f}-n 200 /tmp/vllm-serve.log"
    remote.run(cluster, cluster.head, cmd + " 2>&1", check=False)


# --- launcher ----------------------------------------------------------------


def _checkout(model: Model) -> str:
    return f"{STATE_DIR}/recipes/{model.name}"


def launcher_prepare(cluster: Cluster, model: Model) -> None:
    """Pinned checkout on the head, plus the rendered env file. Recipes that
    run their own scripts on the worker (`worker_checkout = true`) get the
    same directory mirrored there at the same path."""
    lc = model.launcher
    env_lines = "\n".join(f"{k}={render(str(v), cluster)}" for k, v in lc.get("env", {}).items())
    template = shlex.quote(lc.get("env_template", ".env.example"))
    env_file = shlex.quote(lc.get("env_file", ".env"))
    worker = f"{cluster.user}@{cluster.worker.cx7_ip}"
    mirror = (
        f'ssh -o BatchMode=yes {worker} mkdir -p "$dir"\n'
        f'rsync -a --delete "$dir/" {worker}:"$dir/"'
        if lc.get("worker_checkout")
        else ""
    )
    remote.run(
        cluster,
        cluster.head,
        f"""dir=$(eval echo {_checkout(model)})
[ -d "$dir/.git" ] || git clone -q {shlex.quote(lc['repo'])} "$dir"
git -C "$dir" fetch -q origin
git -C "$dir" -c advice.detachedHead=false checkout -q {shlex.quote(lc['rev'])}
{{ cat "$dir/"{template}; printf '\\n# --- json-inference overrides (models/{model.name}.toml) ---\\n'; cat <<'ENV'
{env_lines}
ENV
}} > "$dir/"{env_file}
# The recipe reaches the worker over the QSFP link; it SSHes with BatchMode.
ssh-keygen -F {cluster.worker.cx7_ip} >/dev/null 2>&1 \\
  || ssh-keyscan -t ed25519 {cluster.worker.cx7_ip} 2>/dev/null >> ~/.ssh/known_hosts
{mirror}""",
    )


def launcher_run(cluster: Cluster, model: Model, step: str, wait: bool = True) -> None:
    lc = model.launcher
    script = f"cd {_checkout(model)}\n{lc[step]}"
    if step == "pull":
        # Some recipes' prepare steps refuse to run without their image, so
        # try pulling it first too (best effort: a tag the recipe builds
        # itself doesn't exist yet at this point).
        image_key = lc.get("image_key", "IMAGE")
        env_file = shlex.quote(lc.get("env_file", ".env"))
        script = (
            f"cd {_checkout(model)}\n"
            f'IMAGE=$(set -a; . ./{env_file}; echo "${{{image_key}:-}}")\n'
            '[ -z "$IMAGE" ] || docker pull -q "$IMAGE" >/dev/null 2>&1 || true\n'
            f"{lc[step]}"
        )
        # Some recipes' download steps fetch only weights and pull their image
        # at launch, on each node; pull it here so `up` doesn't wait on it.
        # A tag the recipe builds itself isn't pullable, which is fine as long
        # as the pull step already built it.
        image_key = lc.get("image_key", "IMAGE")
        env_file = shlex.quote(lc.get("env_file", ".env"))
        have = 'docker pull -q "$IMAGE" >/dev/null 2>&1 || docker image inspect "$IMAGE" >/dev/null'
        script += f'\nIMAGE=$(set -a; . ./{env_file}; echo "${{{image_key}:-}}")\n[ -z "$IMAGE" ] || {have}'
        if model.nodes == 2:
            worker = f"{cluster.user}@{cluster.worker.cx7_ip}"
            script += f"\n[ -z \"$IMAGE\" ] || ssh -o BatchMode=yes {worker} {shlex.quote(have.replace('$IMAGE', '__IMG__'))}".replace(
                "__IMG__", "'\"$IMAGE\"'"
            )
    detached(cluster, cluster.head, f"{step}-{model.name}", script, wait)


def launcher_stop(cluster: Cluster, model: Model) -> None:
    remote.run(
        cluster,
        cluster.head,
        f"[ ! -d {_checkout(model)} ] || (cd {_checkout(model)} && {model.launcher['stop']})",
        check=False,
    )


# --- engine-independent ------------------------------------------------------


def git_rev() -> str:
    rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True)
    return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def endpoint(cluster: Cluster, model: Model) -> str:
    return f"http://{cluster.head.ssh}:{model.port}/v1"


def served_models(cluster: Cluster, port: int) -> list[str] | None:
    try:
        with urllib.request.urlopen(f"http://{cluster.head.ssh}:{port}/v1/models", timeout=3) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except OSError:
        return None


def pull(cluster: Cluster, model: Model, wait: bool = True) -> None:
    if model.engine == "vllm":
        vllm_pull(cluster, model, wait)
    else:
        launcher_prepare(cluster, model)
        launcher_run(cluster, model, "pull", wait)


def down(cluster: Cluster) -> None:
    for node in (cluster.head, cluster.worker):
        remote.run(
            cluster,
            node,
            f'ids=$(docker ps -aq --filter label={LABEL}); [ -z "$ids" ] || docker rm -f $ids >/dev/null',
        )
    # Stop whichever recipe is actually serving first: some stops (MiMo's)
    # wait for the GPU to be released, which never happens while another
    # recipe still holds it.
    serving = set(served_models(cluster, 8000) or [])
    launchers = [m for m in list_models() if m.engine == "launcher"]
    for model in sorted(launchers, key=lambda m: m.served_name not in serving):
        launcher_stop(cluster, model)
    # Each recipe runs its own NFS server container on the head (host network,
    # port 2049) and its stop leaves it up, so the next recipe's server can't
    # bind. The container starts the kernel's nfsd threads, which belong to its
    # PID namespace, so `docker rm -f` hangs with init stuck reaping them until
    # nfsd is shut down; do that (the write reports an I/O error but works).
    for name in {m.launcher["nfs_container"] for m in list_models() if "nfs_container" in m.launcher}:
        remote.run(
            cluster,
            cluster.head,
            f"""docker inspect {name} >/dev/null 2>&1 || exit 0
docker rm -f {name} >/dev/null 2>&1 && exit 0
mountpoint -q /proc/fs/nfsd || sudo mount -t nfsd nfsd /proc/fs/nfsd
sudo sh -c 'echo 0 > /proc/fs/nfsd/threads' 2>/dev/null || true
pid=$(docker inspect -f '{{{{.State.Pid}}}}' {name})
[ "$pid" -gt 0 ] && sudo kill -9 "$pid" 2>/dev/null
for _ in $(seq 60); do
  [ "$(docker inspect -f '{{{{.State.Running}}}}' {name})" = false ] && break
  sleep 1
done
docker rm -f {name} >/dev/null""",
            check=False,
        )
    remote.log(None, "nothing running")


def up(cluster: Cluster, model: Model) -> None:
    t0 = time.time()
    down(cluster)
    t1 = time.time()
    if model.engine == "vllm":
        vllm_up(cluster, model)
    else:
        launcher_prepare(cluster, model)
        launcher_run(cluster, model, "start")
    wait_ready(cluster, model)
    t2 = time.time()
    # How long a switch takes, so `inference results` can show it next to
    # speed. start_s runs from launch to the API answering.
    record = {
        "model": model.name,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "json_inference_rev": git_rev(),
        "stop_s": round(t1 - t0),
        "start_s": round(t2 - t1),
    }
    out = RESULTS_DIR / model.name / f"up-{record['timestamp'].replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    remote.log(None, f"stop {record['stop_s']} s, start {record['start_s']} s (saved {out.relative_to(REPO_ROOT)})")


def wait_ready(cluster: Cluster, model: Model) -> None:
    remote.log(None, f"waiting for {endpoint(cluster, model)} to serve {model.served_name}")
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if model.served_name in (served_models(cluster, model.port) or []):
            remote.log(None, f"{model.name} is up at {endpoint(cluster, model)}")
            return
        if model.engine == "vllm":
            running = remote.output(
                cluster, cluster.head, f"docker ps -q --filter label={LABEL}={model.name} | wc -l"
            )
            if running == "0":
                raise SystemExit(f"{model.name}'s container exited; see `inference logs {model.name}`")
        time.sleep(15)
    raise SystemExit(f"{model.name} not ready after {READY_TIMEOUT_S // 60} minutes")


def logs(cluster: Cluster, model: Model, follow_logs: bool) -> None:
    if model.engine == "vllm":
        vllm_logs(cluster, model, follow_logs)
    elif "logs" in model.launcher:
        remote.run(cluster, cluster.head, f"cd {_checkout(model)} && {model.launcher['logs']}", check=False)
    else:
        # Recipes without a logs command: the output of their start script.
        f = "-f " if follow_logs else ""
        remote.run(cluster, cluster.head, f"tail {f}-n 200 {STATE_DIR}/logs/start-{model.name}.log", check=False)


def status(cluster: Cluster) -> None:
    for node in (cluster.head, cluster.worker):
        out = remote.output(
            cluster,
            node,
            "docker ps --format '  {{.Names}}  {{.Status}}  {{.Image}}'; "
            "awk '/MemAvailable/ {printf \"  MemAvailable %.1f GiB\\n\", $2/1048576}' /proc/meminfo",
        )
        print(f"{node.name}:\n{out or '  (no containers)'}")
    ports = sorted({m.port for m in list_models()})
    for port in ports:
        served = served_models(cluster, port)
        if served:
            print(f"serving on :{port}: {', '.join(served)}  ({cluster.head.ssh})")
