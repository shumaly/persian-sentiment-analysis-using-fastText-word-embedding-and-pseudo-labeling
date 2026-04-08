#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_pipeline import (
    DEFAULT_DATASET_CSV,
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_RESULTS_DIR,
    configure_hardware,
    load_dataset,
    prepare_labeled_data,
    pseudo_label_dataset,
    run_neural_cv,
    save_json,
    set_global_determinism,
    summarize,
    train_and_save_final_cnn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN variants from the original paper pipeline.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--variant", choices=("before-prep", "after-prep", "pseudo"), default="after-prep")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=400)
    parser.add_argument("--num-words", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--save-artifact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_determinism(args.seed)
    hardware = configure_hardware()
    df = load_dataset(args.dataset_csv, limit_rows=args.limit_rows)

    if args.variant == "before-prep":
        texts, labels = prepare_labeled_data(df, seed=args.seed, apply_preprocess=False)
    elif args.variant == "after-prep":
        texts, labels = prepare_labeled_data(df, seed=args.seed, apply_preprocess=True)
    else:
        texts, labels, _ = pseudo_label_dataset(
            df,
            embedding_path=args.embedding_path,
            hardware_bundle=hardware,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            num_words=args.num_words,
        )

    metrics, meta = run_neural_cv(
        texts=texts,
        labels=labels,
        embedding_path=args.embedding_path,
        model_kind="cnn",
        hardware_bundle=hardware,
        folds=args.folds,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_words=args.num_words,
    )
    summary = summarize(metrics)
    save_json(args.results_dir / f"cnn_{args.variant.replace('-', '_')}.json", {"summary": summary, "folds": [m.__dict__ for m in metrics], "meta": meta})

    if args.save_artifact:
        train_and_save_final_cnn(
            texts=texts,
            labels=labels,
            embedding_path=args.embedding_path,
            hardware_bundle=hardware,
            artifact_dir=args.results_dir / "artifacts" / f"cnn_{args.variant.replace('-', '_')}",
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            num_words=args.num_words,
        )

    print(summary)


if __name__ == "__main__":
    main()
