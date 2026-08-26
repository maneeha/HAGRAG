from __future__ import annotations

import argparse
import subprocess
import sys


ALGORITHMS = ("leiden", "louvain", "agglomerative")
WEIGHTINGS = ("abstract", "equal", "specific", "adaptive")


def run(*args: str) -> None:
    command = [sys.executable, "-m", "hagrag", *args]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and evaluate the paper configurations")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=["build", "evaluate", "all"], default="all")
    parser.add_argument("--clear-graph", action="store_true")
    args = parser.parse_args()

    prefix = ["--config", args.config]
    if args.stage in {"build", "all"}:
        for index, algorithm in enumerate(ALGORITHMS):
            extra = ["--clear-graph"] if args.clear_graph and index == 0 else []
            run(*prefix, "build", "--algorithm", algorithm, *extra)
    if args.stage in {"evaluate", "all"}:
        for algorithm in ALGORITHMS:
            for weighting in WEIGHTINGS:
                run(*prefix, "evaluate", "--algorithm", algorithm, "--weighting", weighting)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
