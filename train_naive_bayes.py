#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_pipeline import DEFAULT_DATASET_CSV, DEFAULT_RESULTS_DIR, configure_hardware, load_dataset, prepare_labeled_data, run_tfidf_cv, save_json, save_sklearn_artifact, set_global_determinism, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Naive Bayes baseline.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--save-artifact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_determinism(args.seed)
    configure_hardware()
    df = load_dataset(args.dataset_csv, limit_rows=args.limit_rows)
    texts, labels = prepare_labeled_data(df, seed=args.seed, apply_preprocess=True)
    metrics, vectorizer, model = run_tfidf_cv(texts, labels, model_kind="tfidf-nb", folds=args.folds, seed=args.seed)
    save_json(args.results_dir / "naive_bayes.json", {"summary": summarize(metrics), "folds": [m.__dict__ for m in metrics]})
    if args.save_artifact:
        save_sklearn_artifact(args.results_dir / "artifacts" / "naive_bayes", model, vectorizer, {"model_type": "naive_bayes", "apply_preprocess": True})
    print(summarize(metrics))


if __name__ == "__main__":
    main()
