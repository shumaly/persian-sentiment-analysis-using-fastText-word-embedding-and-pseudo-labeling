#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_pipeline import DEFAULT_DATASET_CSV, DEFAULT_EMBEDDING_PATH, DEFAULT_RESULTS_DIR, configure_hardware, load_dataset, prepare_labeled_data, run_neural_cv, save_json, set_global_determinism, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BiLSTM model from the paper pipeline.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=400)
    parser.add_argument("--num-words", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_determinism(args.seed)
    hardware = configure_hardware()
    df = load_dataset(args.dataset_csv, limit_rows=args.limit_rows)
    texts, labels = prepare_labeled_data(df, seed=args.seed, apply_preprocess=True)
    metrics, meta = run_neural_cv(texts, labels, args.embedding_path, "bilstm", hardware, args.folds, args.seed, args.epochs, args.batch_size, args.max_seq_len, args.num_words)
    save_json(args.results_dir / "bilstm.json", {"summary": summarize(metrics), "folds": [m.__dict__ for m in metrics], "meta": meta})
    print(summarize(metrics))


if __name__ == "__main__":
    main()
