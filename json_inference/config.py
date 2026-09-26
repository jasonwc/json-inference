"""Cluster and model definitions (cluster.toml, models/*.toml)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class Node:
    name: str
    ssh: str
    cx7_ip: str


@dataclass(frozen=True)
class Cluster:
    user: str
    head: Node
    worker: Node
    cx7_netdev: str
    cx7_ibdev: str
    cx7_gid_index: int

    def nodes(self, count: int) -> list[Node]:
        return [self.head, self.worker][:count]

    def template_vars(self) -> dict[str, str]:
        return {
            "user": self.user,
            "head.name": self.head.name,
            "head.ssh": self.head.ssh,
            "head.cx7_ip": self.head.cx7_ip,
            "worker.name": self.worker.name,
            "worker.ssh": self.worker.ssh,
            "worker.cx7_ip": self.worker.cx7_ip,
            "cx7.netdev": self.cx7_netdev,
            "cx7.ibdev": self.cx7_ibdev,
            "cx7.gid_index": str(self.cx7_gid_index),
        }


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    description: str
    engine: str
    nodes: int
    port: int
    served_name: str
    raw: dict = field(repr=False)

    @property
    def vllm(self) -> dict:
        return self.raw.get("vllm", {})

    @property
    def launcher(self) -> dict:
        return self.raw.get("launcher", {})

    @property
    def prefill_tokens(self) -> list[int]:
        return self.raw.get("bench", {}).get("prefill_tokens", [2048, 8192])


def load_cluster() -> Cluster:
    data = tomllib.loads((REPO_ROOT / "cluster.toml").read_text())
    node = lambda d: Node(d["name"], d["ssh"], d["cx7_ip"])  # noqa: E731
    return Cluster(
        user=data["user"],
        head=node(data["head"]),
        worker=node(data["worker"]),
        cx7_netdev=data["cx7"]["netdev"],
        cx7_ibdev=data["cx7"]["ibdev"],
        cx7_gid_index=data["cx7"]["gid_index"],
    )


def load_model(name: str) -> Model:
    path = MODELS_DIR / f"{name}.toml"
    if not path.exists():
        known = ", ".join(m.name for m in list_models())
        raise SystemExit(f"unknown model '{name}' (known: {known})")
    raw = tomllib.loads(path.read_text())
    engine = raw["engine"]
    if engine not in ("vllm", "launcher"):
        raise SystemExit(f"{path}: unknown engine '{engine}'")
    nodes = raw.get("nodes", 1)
    if nodes not in (1, 2):
        raise SystemExit(f"{path}: nodes must be 1 or 2")
    return Model(
        name=name,
        path=path,
        description=raw.get("description", ""),
        engine=engine,
        nodes=nodes,
        port=raw.get("port", 8000),
        # vLLM models are served under their definition name, so clients and
        # benchmark results use the same id regardless of the HF repo.
        served_name=raw.get("served_name", name),
        raw=raw,
    )


def list_models() -> list[Model]:
    return [load_model(p.stem) for p in sorted(MODELS_DIR.glob("*.toml"))]


def render(value: str, cluster: Cluster) -> str:
    for key, sub in cluster.template_vars().items():
        value = value.replace("{" + key + "}", sub)
    return value
