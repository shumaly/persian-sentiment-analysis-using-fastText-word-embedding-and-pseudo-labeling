#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentiment_pipeline import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_CSV,
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_RESULTS_DIR,
    configure_hardware,
    load_dataset,
    metrics_to_rows,
    prepare_labeled_data,
    pseudo_label_dataset,
    run_neural_cv,
    run_tfidf_cv,
    save_json,
    set_global_determinism,
    summarize,
    train_and_save_final_cnn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the paper tables and save them as CSV files.")
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
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_determinism(args.seed)
    hardware = configure_hardware()
    df = load_dataset(args.dataset_csv, limit_rows=args.limit_rows)

    results_dir = args.results_dir
    artifacts_dir = args.artifacts_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing datasets...")
    processed_texts, processed_labels = prepare_labeled_data(df, seed=args.seed, apply_preprocess=True)
    raw_texts, raw_labels = prepare_labeled_data(df, seed=args.seed, apply_preprocess=False)
    pseudo_texts, pseudo_labels, pseudo_meta = pseudo_label_dataset(
        df,
        embedding_path=args.embedding_path,
        hardware_bundle=hardware,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_words=args.num_words,
    )

    print("Running BiLSTM...")
    bilstm_metrics, bilstm_meta = run_neural_cv(processed_texts, processed_labels, args.embedding_path, "bilstm", hardware, args.folds, args.seed, args.epochs, args.batch_size, args.max_seq_len, args.num_words)
    print("Running CNN after preprocessing...")
    cnn_after_metrics, cnn_after_meta = run_neural_cv(processed_texts, processed_labels, args.embedding_path, "cnn", hardware, args.folds, args.seed, args.epochs, args.batch_size, args.max_seq_len, args.num_words)
    print("Running TF-IDF + Naive Bayes...")
    nb_metrics, _, _ = run_tfidf_cv(processed_texts, processed_labels, model_kind="tfidf-nb", folds=args.folds, seed=args.seed)
    print("Running TF-IDF + Logistic Regression...")
    lr_metrics, _, _ = run_tfidf_cv(processed_texts, processed_labels, model_kind="tfidf-lr", folds=args.folds, seed=args.seed)
    print("Running CNN before preprocessing...")
    cnn_before_metrics, _ = run_neural_cv(raw_texts, raw_labels, args.embedding_path, "cnn", hardware, args.folds, args.seed, args.epochs, args.batch_size, args.max_seq_len, args.num_words)
    print("Running CNN after pseudo labeling...")
    cnn_pseudo_metrics, cnn_pseudo_meta = run_neural_cv(pseudo_texts, pseudo_labels, args.embedding_path, "cnn", hardware, args.folds, args.seed, args.epochs, args.batch_size, args.max_seq_len, args.num_words)

    table4_rows = []
    table4_rows.extend(metrics_to_rows("BiLSTM", bilstm_metrics))
    table4_rows.extend(metrics_to_rows("CNN", cnn_after_metrics))
    table4_rows.extend(metrics_to_rows("Naive Bayes", nb_metrics))
    table4_rows.extend(metrics_to_rows("Logistic Regression", lr_metrics))
    pd.DataFrame(table4_rows).to_csv(results_dir / "table4_model_comparison.csv", index=False)

    table5_rows = []
    table5_rows.extend(metrics_to_rows("Before Prep.", cnn_before_metrics))
    table5_rows.extend(metrics_to_rows("After Prep.", cnn_after_metrics))
    table5_rows.extend(metrics_to_rows("After Pseudo labeling", cnn_pseudo_metrics))
    pd.DataFrame(table5_rows).to_csv(results_dir / "table5_cnn_situations.csv", index=False)

    save_json(
        results_dir / "summary.json",
        {
            "hardware": hardware["hardware"],
            "table4": {
                "bilstm": summarize(bilstm_metrics),
                "cnn": summarize(cnn_after_metrics),
                "naive_bayes": summarize(nb_metrics),
                "logistic_regression": summarize(lr_metrics),
            },
            "table5": {
                "cnn_before_prep": summarize(cnn_before_metrics),
                "cnn_after_prep": summarize(cnn_after_metrics),
                "cnn_after_pseudo_labeling": summarize(cnn_pseudo_metrics),
            },
            "meta": {
                "bilstm": bilstm_meta,
                "cnn_after_prep": cnn_after_meta,
                "pseudo_labeling": pseudo_meta,
                "cnn_after_pseudo_labeling": cnn_pseudo_meta,
            },
        },
    )

    print("Training final best CNN artifact...")
    train_and_save_final_cnn(
        texts=pseudo_texts,
        labels=pseudo_labels,
        embedding_path=args.embedding_path,
        hardware_bundle=hardware,
        artifact_dir=artifacts_dir / "best_cnn_model",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_words=args.num_words,
    )

    print(f"Saved {results_dir / 'table4_model_comparison.csv'}")
    print(f"Saved {results_dir / 'table5_cnn_situations.csv'}")
    print(f"Saved {artifacts_dir / 'best_cnn_model'}")


if __name__ == "__main__":
    main()
