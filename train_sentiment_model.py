import argparse
from pathlib import Path

import pandas as pd

from digikala_sentiment.utils.download_dataset import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_EMBEDDING_ARCHIVE_PATH,
    DEFAULT_EMBEDDING_OUTPUT_DIR,
    download_dataset,
    ensure_embedding_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNN sentiment model with pseudo-labeling and save artifacts."
    )
    parser.add_argument(
        "--dataset",
        default=f"artifacts/data/raw/{DEFAULT_DATASET_FILENAME}",
        help="Path to the dataset CSV file. It will be downloaded from Kaggle if missing.",
    )
    parser.add_argument(
        "--embedding-archive",
        default=str(DEFAULT_EMBEDDING_ARCHIVE_PATH),
        help="Path to the zipped fastText embedding archive stored in the repository.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=str(DEFAULT_EMBEDDING_OUTPUT_DIR),
        help="Directory where the embedding archive will be extracted.",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/cnn_pseudolabel",
        help="Directory for the saved model and tokenizer artifacts.",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/reports/cnn_pseudolabel_report.txt",
        help="Path for the training report text file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout size for evaluation.")
    return parser.parse_args()


def format_report(
    dataset_path: Path,
    embeddings_path: Path,
    training_config: dict[str, object],
    initial_metrics: dict[str, object],
    final_metrics: dict[str, object],
    labeled_count: int,
    unlabeled_count: int,
    pseudo_count: int,
    split_counts: dict[str, int],
) -> str:
    lines = [
        "Digikala CNN Training Report",
        "=" * 30,
        "",
        f"Dataset: {dataset_path}",
        f"Embeddings: {embeddings_path}",
        "",
        "Training configuration",
        f"- Epochs: {training_config['epochs']}",
        f"- Batch size: {training_config['batch_size']}",
        f"- Max sequence length: {training_config['max_sequence_length']}",
        f"- Pseudo negative max probability: {training_config['pseudo_negative_max']}",
        f"- Pseudo positive min probability: {training_config['pseudo_positive_min']}",
        "",
        "Dataset sizes",
        f"- Labeled rows: {labeled_count}",
        f"- Unlabeled rows: {unlabeled_count}",
        f"- Train rows before pseudo-labeling: {split_counts['train']}",
        f"- Test rows: {split_counts['test']}",
        f"- Selected pseudo-labeled rows: {pseudo_count}",
        f"- Final training rows before oversampling: {split_counts['final_train']}",
        "",
        "Initial CNN metrics on holdout set",
        f"- AUC: {initial_metrics['auc']:.6f}",
        f"- F1 score: {initial_metrics['f1']:.6f}",
        f"- Confusion matrix: {initial_metrics['confusion_matrix']}",
        "",
        "Final CNN + pseudo-labeling metrics on holdout set",
        f"- AUC: {final_metrics['auc']:.6f}",
        f"- F1 score: {final_metrics['f1']:.6f}",
        f"- Confusion matrix: {final_metrics['confusion_matrix']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    from digikala_sentiment import pipeline

    pipeline.set_seed()

    dataset_path = download_dataset(Path(args.dataset))
    embeddings_path = ensure_embedding_file(
        archive_path=Path(args.embedding_archive),
        output_dir=Path(args.embedding_dir),
    )
    model_dir = Path(args.model_dir)
    report_path = Path(args.report_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = pipeline.load_and_prepare_dataset(dataset_path)
    split = pipeline.split_labeled_data(prepared.labeled_df, test_size=args.test_size)

    initial_tokenizer = pipeline.create_tokenizer(split.train_texts)
    initial_embedding_matrix = pipeline.build_embedding_matrix(embeddings_path, initial_tokenizer.word_index)

    train_texts_os, train_labels_os = pipeline.oversample_positive_class(split.train_texts, split.train_labels)
    train_sequences = pipeline.vectorize_texts(initial_tokenizer, train_texts_os)
    test_sequences = pipeline.vectorize_texts(initial_tokenizer, split.test_texts)

    initial_model = pipeline.create_cnn_model(len(initial_tokenizer.word_index) + 1, initial_embedding_matrix)
    initial_model.fit(
        train_sequences,
        train_labels_os,
        validation_data=(test_sequences, split.test_labels),
        epochs=pipeline.EPOCHS,
        batch_size=pipeline.BATCH_SIZE,
        verbose=1,
        shuffle=True,
    )
    initial_probabilities = initial_model.predict(test_sequences, verbose=0)
    initial_metrics = pipeline.evaluate_predictions(split.test_labels, initial_probabilities)

    unlabeled_sequences = pipeline.vectorize_texts(initial_tokenizer, prepared.unlabeled_df["Comment"].values)
    pseudo_probabilities = initial_model.predict(unlabeled_sequences, verbose=0)
    pseudo_df = pipeline.build_pseudo_labeled_frame(prepared.unlabeled_df, pseudo_probabilities)

    final_train_df = pd.DataFrame({"text": split.train_texts, "label": split.train_labels})
    final_train_df = pd.concat(
        [final_train_df, pseudo_df[["text", "label"]]],
        ignore_index=True,
    )
    final_train_df = final_train_df.sample(frac=1.0, random_state=2020).reset_index(drop=True)

    final_tokenizer = pipeline.create_tokenizer(final_train_df["text"].values)
    final_embedding_matrix = pipeline.build_embedding_matrix(embeddings_path, final_tokenizer.word_index)

    final_train_texts_os, final_train_labels_os = pipeline.oversample_positive_class(
        final_train_df["text"].values,
        final_train_df["label"].values,
    )
    final_train_sequences = pipeline.vectorize_texts(final_tokenizer, final_train_texts_os)
    final_test_sequences = pipeline.vectorize_texts(final_tokenizer, split.test_texts)

    final_model = pipeline.create_cnn_model(len(final_tokenizer.word_index) + 1, final_embedding_matrix)
    final_model.fit(
        final_train_sequences,
        final_train_labels_os,
        validation_data=(final_test_sequences, split.test_labels),
        epochs=pipeline.EPOCHS,
        batch_size=pipeline.BATCH_SIZE,
        verbose=1,
        shuffle=True,
    )

    final_probabilities = final_model.predict(final_test_sequences, verbose=0)
    final_metrics = pipeline.evaluate_predictions(split.test_labels, final_probabilities)

    final_model.save(model_dir / "model.keras")
    pipeline.save_tokenizer(final_tokenizer, model_dir / "tokenizer.json")
    training_config = {
        "max_sequence_length": pipeline.MAX_SEQUENCE_LENGTH,
        "epochs": pipeline.EPOCHS,
        "batch_size": pipeline.BATCH_SIZE,
        "pseudo_negative_max": pipeline.PSEUDO_NEGATIVE_MAX,
        "pseudo_positive_min": pipeline.PSEUDO_POSITIVE_MIN,
    }
    pipeline.save_metadata(
        model_dir / "metadata.json",
        {
            "dataset": str(dataset_path),
            "embeddings": str(embeddings_path),
            **training_config,
            "positive_label": 1,
            "negative_label": 0,
        },
    )

    report_text = format_report(
        dataset_path=dataset_path,
        embeddings_path=embeddings_path,
        training_config=training_config,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        labeled_count=len(prepared.labeled_df),
        unlabeled_count=len(prepared.unlabeled_df),
        pseudo_count=len(pseudo_df),
        split_counts={
            "train": len(split.train_texts),
            "test": len(split.test_texts),
            "final_train": len(final_train_df),
        },
    )
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Saved model artifacts to: {model_dir}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
