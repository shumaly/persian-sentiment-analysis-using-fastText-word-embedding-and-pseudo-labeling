#!/usr/bin/env python3
"""Shared utilities for reproducing the Digikala sentiment experiments."""

from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "2020")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from preprocess import preprocess


DEFAULT_DATASET_CSV = Path("Digikala_3M.csv")
DEFAULT_EMBEDDING_PATH = Path("DigiKalaEmbeddingVectors.vec")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "artifacts"
EXPECTED_COLUMNS = {"Comment", "PositiveFeedback", "NegetiveFeedback"}
COLUMN_ALIASES = {
    "PositiveFeedback": ("PositiveFeedback", "Satisfied"),
    "NegetiveFeedback": ("NegetiveFeedback", "Unsatisfied"),
}


@dataclass
class FoldMetrics:
    fold: int
    auc: float
    f1: float
    positives: int
    negatives: int
    confusion_matrix: list[list[int]]


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_hardware() -> dict[str, object]:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        strategy_info = {
            "strategy": "mirrored",
            "num_replicas_in_sync": strategy.num_replicas_in_sync,
            "gpu_count": len(gpus),
        }
    else:
        strategy = tf.distribute.get_strategy()
        strategy_info = {
            "strategy": "default",
            "num_replicas_in_sync": strategy.num_replicas_in_sync,
            "gpu_count": len(gpus),
        }

    return {
        "strategy": strategy,
        "strategy_info": strategy_info,
        "hardware": {
            "gpu_devices": [gpu.name for gpu in gpus],
            "using_gpu": bool(gpus),
        },
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    for target, options in COLUMN_ALIASES.items():
        if target in renamed.columns:
            continue
        for option in options:
            if option in renamed.columns:
                renamed = renamed.rename(columns={option: target})
                break
    return renamed


def load_dataset(dataset_csv: Path, limit_rows: int | None = None) -> pd.DataFrame:
    if not dataset_csv.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_csv}. Download Digikala_3M.csv from Kaggle and place it in the project root."
        )
    df = pd.read_csv(dataset_csv, encoding="utf-8")
    df = normalize_columns(df)
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if limit_rows:
        df = df.head(limit_rows).copy()
    return df


def prepare_text_series(series: pd.Series, apply_preprocess: bool) -> pd.Series:
    series = series.astype(str)
    if apply_preprocess:
        cleaner = preprocess()
        series = series.apply(cleaner.clean)
    return series[series.str.len() > 0]


def prepare_labeled_data(df: pd.DataFrame, seed: int, apply_preprocess: bool = True) -> tuple[np.ndarray, np.ndarray]:
    labeled = df[(df["PositiveFeedback"] == 1) | (df["NegetiveFeedback"] == 1)].copy()
    labeled = labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    labeled["Label"] = 0
    labeled.loc[labeled["NegetiveFeedback"] == 1, "Label"] = 1
    prepared = prepare_text_series(labeled["Comment"], apply_preprocess=apply_preprocess)
    labeled = labeled.loc[prepared.index].copy()
    labeled["Comment"] = prepared
    return labeled["Comment"].to_numpy(), labeled["Label"].to_numpy(dtype=np.int32)


def prepare_unlabeled_text(df: pd.DataFrame, apply_preprocess: bool = True) -> np.ndarray:
    unlabeled = df[(df["PositiveFeedback"] == 0) & (df["NegetiveFeedback"] == 0)].copy()
    prepared = prepare_text_series(unlabeled["Comment"], apply_preprocess=apply_preprocess)
    return prepared.to_numpy()


def create_tokenizer(texts: np.ndarray, num_words: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./;<=>?@][\\]^{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    return tokenizer


def pad_texts(tokenizer: Tokenizer, texts: np.ndarray, max_seq_len: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_seq_len, padding="post")


def build_embed_matrix(embed_path: Path, word_index: dict[str, int], emb_size: int = 100) -> np.ndarray:
    if not embed_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embed_path}. Put DigiKalaEmbeddingVectors.vec in the project root."
        )
    with embed_path.open(encoding="utf-8") as handle:
        embed_index = {}
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) != emb_size + 1:
                continue
            embed_index[parts[0]] = np.asarray(parts[1:], dtype="float32")
    embed_matrix = np.zeros((len(word_index) + 1, emb_size), dtype="float32")
    for word, idx in word_index.items():
        embed_vector = embed_index.get(word)
        if embed_vector is not None:
            embed_matrix[idx] = embed_vector
    del embed_index
    gc.collect()
    return embed_matrix


def build_cnn_model(vocab_size: int, embedding_matrix: np.ndarray, max_seq_len: int) -> Sequential:
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_seq_len, weights=[embedding_matrix]))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, kernel_size=3, padding="same", activation="relu", strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_bilstm_model(vocab_size: int, embedding_matrix: np.ndarray, max_seq_len: int) -> Sequential:
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_seq_len, weights=[embedding_matrix]))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def oversample_like_notebook(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    negative_mask = y_train == 1
    return np.append(x_train, x_train[negative_mask]), np.append(y_train, y_train[negative_mask])


def evaluate_predictions(y_true: np.ndarray, pred_proba: np.ndarray, fold: int) -> FoldMetrics:
    pred_class = (pred_proba >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true, pred_class)
    return FoldMetrics(
        fold=fold,
        auc=float(roc_auc_score(y_true, pred_proba)),
        f1=float(f1_score(y_true, pred_class)),
        positives=int((y_true == 1).sum()),
        negatives=int((y_true == 0).sum()),
        confusion_matrix=cm.tolist(),
    )


def summarize(metrics: list[FoldMetrics]) -> dict[str, float]:
    aucs = np.asarray([metric.auc for metric in metrics], dtype=np.float64)
    f1s = np.asarray([metric.f1 for metric in metrics], dtype=np.float64)
    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=0)),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std(ddof=0)),
    }


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def run_neural_cv(
    texts: np.ndarray,
    labels: np.ndarray,
    embedding_path: Path,
    model_kind: str,
    hardware_bundle: dict[str, object],
    folds: int = 5,
    seed: int = 2020,
    epochs: int = 7,
    batch_size: int = 2048,
    max_seq_len: int = 400,
    num_words: int = 2_000_000,
) -> tuple[list[FoldMetrics], dict[str, object]]:
    tokenizer = create_tokenizer(texts, num_words)
    embedding_matrix = build_embed_matrix(embedding_path, tokenizer.word_index)
    build_model = build_cnn_model if model_kind == "cnn" else build_bilstm_model
    strategy = hardware_bundle["strategy"]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics: list[FoldMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"#Fold: {fold}")
        x_train, x_test = texts[train_idx], texts[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        x_train, y_train = oversample_like_notebook(x_train, y_train)

        x_train_pad = pad_texts(tokenizer, x_train, max_seq_len)
        x_test_pad = pad_texts(tokenizer, x_test, max_seq_len)

        with strategy.scope():
            model = build_model(len(tokenizer.word_index) + 1, embedding_matrix, max_seq_len)

        model.fit(
            x_train_pad,
            y_train,
            validation_data=(x_test_pad, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
        )
        pred_proba = model.predict(x_test_pad, verbose=0).ravel()
        fold_metrics.append(evaluate_predictions(y_test, pred_proba, fold))
        tf.keras.backend.clear_session()
        gc.collect()

    return fold_metrics, {
        "token_count": len(tokenizer.word_index),
        "embedding_shape": list(embedding_matrix.shape),
        "distribution": hardware_bundle["strategy_info"],
    }


def run_tfidf_cv(texts: np.ndarray, labels: np.ndarray, model_kind: str, folds: int = 5, seed: int = 2020):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100000)
    vectorizer.fit(texts)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics: list[FoldMetrics] = []
    final_model = None
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"#Fold: {fold}")
        x_train, x_test = texts[train_idx], texts[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        x_train, y_train = oversample_like_notebook(x_train, y_train)
        x_train_tf = vectorizer.transform(x_train)
        x_test_tf = vectorizer.transform(x_test)

        if model_kind == "tfidf-nb":
            model = MultinomialNB()
            model.fit(x_train_tf, y_train)
            pred_proba = model.predict_proba(x_test_tf)[:, 1]
        else:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(x_train_tf, y_train)
            pred_proba = model.predict_proba(x_test_tf)[:, 1]
        final_model = model
        fold_metrics.append(evaluate_predictions(y_test, pred_proba, fold))
    return fold_metrics, vectorizer, final_model


def pseudo_label_dataset(
    df: pd.DataFrame,
    embedding_path: Path,
    hardware_bundle: dict[str, object],
    seed: int = 2020,
    epochs: int = 7,
    batch_size: int = 2048,
    max_seq_len: int = 400,
    num_words: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    texts, labels = prepare_labeled_data(df, seed=seed, apply_preprocess=True)
    unlabeled_texts = prepare_unlabeled_text(df, apply_preprocess=True)
    tokenizer = create_tokenizer(texts, num_words)
    embedding_matrix = build_embed_matrix(embedding_path, tokenizer.word_index)
    x_pad = pad_texts(tokenizer, texts, max_seq_len)

    with hardware_bundle["strategy"].scope():
        model = build_cnn_model(len(tokenizer.word_index) + 1, embedding_matrix, max_seq_len)

    model.fit(x_pad, labels, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    unlabeled_pad = pad_texts(tokenizer, unlabeled_texts, max_seq_len)
    pseudo_proba = model.predict(unlabeled_pad, verbose=0).ravel()
    pseudo_mask = (pseudo_proba < 1e-7) | (pseudo_proba > 0.9)
    pseudo_labels = (pseudo_proba[pseudo_mask] > 0.5).astype(np.int32)

    tf.keras.backend.clear_session()
    gc.collect()

    augmented_texts = np.concatenate([texts, unlabeled_texts[pseudo_mask]])
    augmented_labels = np.concatenate([labels, pseudo_labels])
    return augmented_texts, augmented_labels, {
        "pseudo_label_count": int(pseudo_mask.sum()),
        "distribution": hardware_bundle["strategy_info"],
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics_to_rows(state: str, metrics: list[FoldMetrics]) -> list[dict[str, object]]:
    aucs = [metric.auc for metric in metrics]
    f1s = [metric.f1 for metric in metrics]
    return [
        {
            "States": state,
            "Index": "AUC:",
            "Fold1": aucs[0],
            "Fold2": aucs[1],
            "Fold3": aucs[2],
            "Fold4": aucs[3],
            "Fold5": aucs[4],
            "Mean": float(np.mean(aucs)),
            "Error (SEM)": sem(aucs),
        },
        {
            "States": state,
            "Index": "F-score:",
            "Fold1": f1s[0],
            "Fold2": f1s[1],
            "Fold3": f1s[2],
            "Fold4": f1s[3],
            "Fold5": f1s[4],
            "Mean": float(np.mean(f1s)),
            "Error (SEM)": sem(f1s),
        },
    ]


def save_keras_artifact(artifact_dir: Path, model: Sequential, tokenizer: Tokenizer, metadata: dict[str, object]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / "model.keras")
    (artifact_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    save_json(artifact_dir / "metadata.json", metadata)


def train_and_save_final_cnn(
    texts: np.ndarray,
    labels: np.ndarray,
    embedding_path: Path,
    hardware_bundle: dict[str, object],
    artifact_dir: Path,
    seed: int = 2020,
    epochs: int = 7,
    batch_size: int = 2048,
    max_seq_len: int = 400,
    num_words: int = 2_000_000,
) -> None:
    tokenizer = create_tokenizer(texts, num_words)
    embedding_matrix = build_embed_matrix(embedding_path, tokenizer.word_index)
    x_pad = pad_texts(tokenizer, texts, max_seq_len)

    with hardware_bundle["strategy"].scope():
        model = build_cnn_model(len(tokenizer.word_index) + 1, embedding_matrix, max_seq_len)

    model.fit(x_pad, labels, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    save_keras_artifact(
        artifact_dir,
        model,
        tokenizer,
        {
            "model_type": "cnn",
            "max_seq_len": max_seq_len,
            "apply_preprocess": True,
            "seed": seed,
            "embedding_path": str(embedding_path.resolve()),
            "distribution": hardware_bundle["strategy_info"],
        },
    )
    tf.keras.backend.clear_session()
    gc.collect()


def save_sklearn_artifact(artifact_dir: Path, model, vectorizer, metadata: dict[str, object]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    joblib.dump(vectorizer, artifact_dir / "vectorizer.joblib")
    save_json(artifact_dir / "metadata.json", metadata)


def load_cnn_artifact(artifact_dir: Path):
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json((artifact_dir / "tokenizer.json").read_text(encoding="utf-8"))
    model = load_model(artifact_dir / "model.keras")
    return model, tokenizer, metadata


def predict_with_cnn(artifact_dir: Path, sentences: list[str]) -> list[dict[str, object]]:
    model, tokenizer, metadata = load_cnn_artifact(artifact_dir)
    series = pd.Series(sentences, dtype="object")
    if metadata.get("apply_preprocess", True):
        cleaner = preprocess()
        series = series.apply(cleaner.clean)
    padded = pad_texts(tokenizer, series.to_numpy(), int(metadata["max_seq_len"]))
    pred = model.predict(padded, verbose=0).ravel()
    return [
        {
            "sentence": sentence,
            "cleaned_sentence": cleaned,
            "negative_probability": float(score),
            "positive_probability": float(1.0 - score),
            "predicted_label": "negative" if score >= 0.5 else "positive",
        }
        for sentence, cleaned, score in zip(sentences, series.tolist(), pred.tolist())
    ]
