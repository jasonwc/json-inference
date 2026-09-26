"""Run commands on the Sparks over SSH."""

import shlex
import subprocess
import sys

from .config import Cluster, Node


def ssh_target(cluster: Cluster, node: Node) -> str:
    return f"{cluster.user}@{node.ssh}"


def run(
    cluster: Cluster,
    node: Node,
    script: str,
    *,
    check: bool = True,
    capture: bool = False,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a bash script on a node. Login shell, so system-manager's PATH (hf,
    git) is available."""
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        ssh_target(cluster, node),
        "bash -lc " + shlex.quote("set -euo pipefail\n" + script),
    ]
    result = subprocess.run(cmd, text=True, capture_output=capture, input=stdin)
    if check and result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"{node.name}: command failed (exit {result.returncode})")
    return result


def output(cluster: Cluster, node: Node, script: str) -> str:
    return run(cluster, node, script, capture=True).stdout.strip()


def log(node: Node | None, msg: str) -> None:
    prefix = f"[inference {node.name}]" if node else "[inference]"
    print(f"{prefix} {msg}", flush=True)
