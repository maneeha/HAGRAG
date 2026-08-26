from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .config import load_config
from .data import load_qa_dataset
from .errors import HAGRAGError
from .evaluation import mean_metrics, response_metrics, retrieval_metrics
from .io_utils import ensure_dir, write_json
from .runtime import build_system, create_embedder, load_query_engine
from .storage import Neo4jStore

LOGGER = logging.getLogger("hagrag")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hagrag", description="HAGRAG research implementation")
    parser.add_argument("--config", default="configs/paper.yaml", help="YAML configuration file")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Validate configuration, data paths and Neo4j connectivity")
    check.add_argument("--require-data", action="store_true")

    build = sub.add_parser("build", help="Build graph, hierarchy and C-HNSW checkpoints")
    build.add_argument("--algorithm", choices=["leiden", "louvain", "agglomerative"], default="leiden")
    build.add_argument("--clear-graph", action="store_true")

    query = sub.add_parser("query", help="Run one HAGRAG query")
    query.add_argument("question")
    query.add_argument("--algorithm", choices=["leiden", "louvain", "agglomerative"], default="leiden")
    query.add_argument("--weighting", choices=["abstract", "equal", "specific", "adaptive"], default=None)

    evaluate = sub.add_parser("evaluate", help="Evaluate a built HAGRAG configuration")
    evaluate.add_argument("--algorithm", choices=["leiden", "louvain", "agglomerative"], default="leiden")
    evaluate.add_argument("--weighting", choices=["abstract", "equal", "specific", "adaptive"], default=None)
    return parser


def _question_and_reference(row: dict) -> tuple[str, str]:
    question = row.get("question") or row.get("QUESTION") or row.get("query") or ""
    reference = (
        row.get("answer")
        or row.get("reference_answer")
        or row.get("ground_truth")
        or row.get("final_decision")
        or ""
    )
    return str(question).strip(), str(reference).strip()


def command_check(config, require_data: bool) -> dict:
    result = {"configuration": "ok"}
    if require_data:
        pdf_dir = Path(config.paths.pdf_dir)
        dataset = Path(config.paths.qa_dataset)
        result["pdfs"] = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
        result["qa_dataset"] = "ok" if dataset.exists() else "missing"
    try:
        store = Neo4jStore.from_env(config.neo4j)
        store.verify()
        result["neo4j"] = "ok"
        store.close()
    except Exception as exc:
        result["neo4j"] = f"unavailable: {exc}"
    result["hf_token"] = "set" if os.getenv("HF_TOKEN") else "not set"
    return result


def command_evaluate(config, algorithm: str, weighting: str | None) -> dict:
    engine = load_query_engine(config, algorithm, weighting)
    dataset = load_qa_dataset(config.paths.qa_dataset, config.evaluation.max_queries)
    embedder = create_embedder(config)
    rows = []
    retrieval_rows = []
    detailed = []
    for item in dataset:
        question, reference = _question_and_reference(item)
        if not question or not reference:
            continue
        result = engine.query(question)
        evidence = [entry["text"] for entry in result["context"]]
        metrics = response_metrics(
            question,
            result["response"],
            reference,
            evidence,
            embedder,
            config.evaluation.faithfulness_threshold,
            config.evaluation.relevance_threshold,
            config.evaluation.correctness_threshold,
        )
        rmetrics = retrieval_metrics(evidence, reference, question, embedder)
        rows.append(metrics)
        retrieval_rows.append(rmetrics)
        detailed.append({**result, "reference": reference, "metrics": metrics, "retrieval_metrics": rmetrics})

    output_dir = ensure_dir(Path(config.paths.output_dir) / algorithm / (weighting or config.retrieval.weighting))
    write_json(output_dir / "predictions.json", detailed)
    summary = {"generation": mean_metrics(rows), "retrieval": mean_metrics(retrieval_rows), "queries": len(detailed)}
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.command == "check":
            result = command_check(config, args.require_data)
        elif args.command == "build":
            result = build_system(config, args.algorithm, args.clear_graph)
        elif args.command == "query":
            engine = load_query_engine(config, args.algorithm, args.weighting)
            result = engine.query(args.question)
        elif args.command == "evaluate":
            result = command_evaluate(config, args.algorithm, args.weighting)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except HAGRAGError as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
