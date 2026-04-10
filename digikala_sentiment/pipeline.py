import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from digikala_sentiment.keras_compat import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Sequential,
    Tokenizer,
    pad_sequences,
    tokenizer_from_json,
)
from digikala_sentiment.utils.preprocess import TextPreprocessor


SEED = 2020
EMBEDDING_SIZE = 100
MAX_SEQUENCE_LENGTH = 400
TOKENIZER_NUM_WORDS = 2_000_000
BATCH_SIZE = 2048
EPOCHS = 7
PSEUDO_NEGATIVE_MAX = 1e-7
PSEUDO_POSITIVE_MIN = 0.9


@dataclass
class PreparedDataset:
    labeled_df: pd.DataFrame
    unlabeled_df: pd.DataFrame


@dataclass
class DatasetSplit:
    train_texts: np.ndarray
    test_texts: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_and_prepare_dataset(dataset_path: Path) -> PreparedDataset:
    df = pd.read_csv(dataset_path, encoding="utf-8")

    labeled_df = df[(df["Satisfied"] == 1) | (df["Unsatisfied"] == 1)].copy()
    labeled_df = labeled_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    labeled_df["Label"] = 0
    labeled_df.loc[labeled_df["Unsatisfied"] == 1, "Label"] = 1

    cleaner = TextPreprocessor()
    labeled_df["Comment"] = labeled_df["Comment"].astype(str).apply(cleaner.clean)

    unlabeled_df = df[(df["Satisfied"] == 0) & (df["Unsatisfied"] == 0)].copy()
    unlabeled_df["Comment"] = unlabeled_df["Comment"].astype(str).apply(cleaner.clean)

    return PreparedDataset(labeled_df=labeled_df, unlabeled_df=unlabeled_df)


def split_labeled_data(labeled_df: pd.DataFrame, test_size: float = 0.2) -> DatasetSplit:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        labeled_df["Comment"].values,
        labeled_df["Label"].values,
        test_size=test_size,
        random_state=SEED,
        stratify=labeled_df["Label"].values,
    )
    return DatasetSplit(
        train_texts=train_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        test_labels=test_labels,
    )


def create_tokenizer(texts: np.ndarray) -> Tokenizer:
    tokenizer = Tokenizer(
        num_words=TOKENIZER_NUM_WORDS,
        filters='!"#$%&()*+,-./;<=>?@][\\]^{|}~\t\n',
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def get_coefs(word: str, *arr: str) -> tuple[str, np.ndarray]:
    return word, np.asarray(arr, dtype="float32")


def build_embedding_matrix(embed_path: Path, word_index: dict[str, int]) -> np.ndarray:
    embed_index = dict(get_coefs(*line.strip().split(" ")) for line in embed_path.open(encoding="utf-8"))
    embed_matrix = np.zeros((len(word_index) + 1, EMBEDDING_SIZE))

    for word, idx in word_index.items():
        embed_vector = embed_index.get(word)
        if embed_vector is not None:
            embed_matrix[idx] = embed_vector

    del embed_index
    gc.collect()
    return embed_matrix


def vectorize_texts(tokenizer: Tokenizer, texts: np.ndarray) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")


def oversample_positive_class(texts: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive_mask = labels == 1
    oversampled_texts = np.append(texts, texts[positive_mask])
    oversampled_labels = np.append(labels, labels[positive_mask])
    return oversampled_texts, oversampled_labels


def create_cnn_model(vocab_size: int, embedding_matrix: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(
        Embedding(
            vocab_size,
            EMBEDDING_SIZE,
            input_length=MAX_SEQUENCE_LENGTH,
            weights=[embedding_matrix],
        )
    )
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


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    flat_probabilities = probabilities.reshape(-1)
    predicted_labels = (flat_probabilities > 0.5).astype("int32")
    return {
        "auc": float(roc_auc_score(y_true, flat_probabilities)),
        "f1": float(f1_score(y_true, predicted_labels)),
        "confusion_matrix": confusion_matrix(y_true, predicted_labels).tolist(),
        "probabilities": flat_probabilities,
        "predicted_labels": predicted_labels,
    }


def build_pseudo_labeled_frame(unlabeled_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    flat_probabilities = probabilities.reshape(-1)
    pseudo_df = pd.DataFrame({"text": unlabeled_df["Comment"].values, "proba": flat_probabilities})
    pseudo_df = pseudo_df[
        (pseudo_df["proba"] < PSEUDO_NEGATIVE_MAX) | (pseudo_df["proba"] > PSEUDO_POSITIVE_MIN)
    ].copy()
    pseudo_df["label"] = 0
    pseudo_df.loc[pseudo_df["proba"] > 0.5, "label"] = 1
    return pseudo_df


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path.write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: Path) -> Tokenizer:
    return tokenizer_from_json(path.read_text(encoding="utf-8"))


def save_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
