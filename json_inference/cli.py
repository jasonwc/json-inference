"""inference: run and compare models on the DGX Spark cluster."""

import argparse

from . import bench, engines
from .config import list_models, load_cluster, load_model


def main() -> None:
    p = argparse.ArgumentParser(prog="inference", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="model definitions in models/")
    for name, help_ in (
        ("pull", "stage a model's image and weights on the nodes it needs"),
        ("up", "stop whatever is running, then serve MODEL"),
        ("bench", "benchmark MODEL (must be up) and save results/"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("model")
        if name == "pull":
            s.add_argument("--no-wait", action="store_true", help="start the download and return")
    cn = sub.add_parser("cancel", help="stop MODEL's background jobs (downloads, recipe boots)")
    cn.add_argument("model")
    lg = sub.add_parser("logs", help="server logs for MODEL")
    lg.add_argument("model")
    lg.add_argument("-f", "--follow", action="store_true")
    sw = sub.add_parser("sweep", help="up + bench each MODEL in turn (default: all), continuing past failures")
    sw.add_argument("models", nargs="*")
    sub.add_parser("down", help="stop every model on both Sparks")
    sub.add_parser("status", help="containers, free memory and what's serving")
    sub.add_parser("results", help="latest benchmark per model, side by side")
    args = p.parse_args()

    if args.cmd == "list":
        for m in list_models():
            print(f"{m.name:<28} {m.engine:<9} {m.nodes} node{'s' if m.nodes > 1 else ' '}  {m.description}")
        return
    if args.cmd == "results":
        bench.summary()
        return

    cluster = load_cluster()
    if args.cmd == "sweep":
        sweep(cluster, args.models or [m.name for m in list_models()])
        return
    if args.cmd == "down":
        engines.down(cluster)
    elif args.cmd == "status":
        engines.status(cluster)
    else:
        model = load_model(args.model)
        if args.cmd == "pull":
            engines.pull(cluster, model, wait=not args.no_wait)
        elif args.cmd == "up":
            engines.up(cluster, model)
        elif args.cmd == "cancel":
            engines.cancel(cluster, model)
        elif args.cmd == "logs":
            engines.logs(cluster, model, args.follow)
        elif args.cmd == "bench":
            bench.run(engines.endpoint(cluster, model), model)


def sweep(cluster, names: list[str]) -> None:
    """Benchmark several models unattended. A model that fails to start or to
    benchmark is logged and skipped, so one broken recipe doesn't stall the
    rest; the last model that came up is left serving."""
    failed: list[str] = []
    for name in names:
        print(f"\n===== {name} =====", flush=True)
        try:
            model = load_model(name)
            engines.up(cluster, model)
            bench.run(engines.endpoint(cluster, model), model)
        except (SystemExit, Exception) as e:  # noqa: BLE001 - keep sweeping
            print(f"[sweep] {name} failed: {e}", flush=True)
            failed.append(name)
    print("\n===== results =====")
    bench.summary()
    if failed:
        print(f"\n[sweep] failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
