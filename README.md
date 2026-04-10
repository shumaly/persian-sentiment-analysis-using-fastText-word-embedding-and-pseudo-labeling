# Persian Sentiment Analysis with fastText Embeddings and Pseudo-Labeling

![Project Diagram](e-commerce%20sentiment%20analysis.png)

## Introduction

This repository contains a training-oriented implementation of Persian sentiment analysis built on Digikala product comments, pretrained fastText word embeddings, and a CNN-based pseudo-labeling pipeline.

The goal of the project is to provide a clean and reproducible codebase for training the sentiment model and understanding the workflow behind it. The repository is intentionally focused on training, data preparation, and experiment structure. Local inference code has been removed from the project, and inference is instead hosted separately on Hugging Face.

At a high level, the project combines three practical ideas:

1. A relatively lightweight preprocessing pipeline for Persian review text
2. Pretrained fastText embeddings trained on Digikala comments
3. Semi-supervised learning through pseudo-labeling to benefit from unlabeled data

This makes the repository useful both as a reproducible project implementation and as a reference for Persian sentiment-analysis workflows built around large review datasets.

Reported results from the original work:

- AUC: `0.9944`
- F-score: `0.9288`

## Project Scope

This repository is designed for:

- training the sentiment model from the available dataset and embedding resources
- preparing the required dataset and embedding files automatically
- inspecting the preprocessing and modeling workflow in Python modules
- reviewing the original experimental flow in notebook form

This repository is not designed for:

- serving a local inference API
- providing a local command-line prediction script
- acting as a general-purpose framework for many datasets

Those boundaries are intentional. They keep the repository focused, easier to maintain, and more aligned with its strongest use case: training and reproducing the Digikala sentiment pipeline.

## Method Summary

The training flow in this repository works as follows:

1. The Digikala comments dataset is loaded from a CSV file.
2. Text comments are normalized and cleaned using a regex-based preprocessing pipeline.
3. Rows with explicit sentiment labels are separated from unlabeled rows.
4. The labeled data is split into training and test sets.
5. A tokenizer is fit on the training texts.
6. A fastText embedding matrix is constructed from the extracted `.vec` file.
7. A CNN model is trained on the labeled training set.
8. That initial model predicts probabilities for the unlabeled subset.
9. Only very confident predictions are selected as pseudo-labels.
10. The final model is retrained on the union of the original labeled training data and the selected pseudo-labeled samples.
11. Final artifacts and a report are written to the `artifacts/` directory.

This structure lets the project use both labeled and unlabeled data while keeping the implementation fairly understandable.

## Repository Layout

The most important files and folders are listed below.

### Project Tree

```text
.
├── README.md
├── e-commerce sentiment analysis.png
├── requirements.txt
├── train_sentiment_model.py
├── assets/
│   └── embeddings/
│       └── DigiKalaEmbeddingVectors.zip
├── artifacts/
│   ├── data/
│   │   └── raw/
│   │       └── Digikala_3M.csv
│   ├── models/
│   │   └── cnn_pseudolabel/
│   │       ├── metadata.json
│   │       ├── model.keras
│   │       └── tokenizer.json
│   ├── reports/
│   │   └── cnn_pseudolabel_report.txt
│   └── resources/
│       └── embeddings/
│           └── DigiKalaEmbeddingVectors.vec
├── digikala_sentiment/
│   ├── __init__.py
│   ├── keras_compat.py
│   ├── pipeline.py
│   └── utils/
│       ├── __init__.py
│       ├── download_dataset.py
│       └── preprocess.py
└── notebooks/
    └── digikala-sentiment-workflow-and-results.ipynb
```

This schematic shows both source files and the main generated artifact locations. Some paths under `artifacts/` appear only after data preparation or training has been run.

### Top-level Files

- `train_sentiment_model.py`
  Main training entry point. Most users should start here. It handles argument parsing, checks whether the required resources exist, prepares missing resources, starts the training pipeline, and saves the final outputs.
- `requirements.txt`
  Python package requirements needed for training and notebook use. It lists the core dependencies for model training, data handling, notebook execution, and automatic dataset download.
- `README.md`
  Project documentation. It explains the project purpose, training workflow, file structure, data sources, and usage.
- `e-commerce sentiment analysis.png`
  Project image used in the README to visually present the topic and context of the repository.

### Package Code

- `digikala_sentiment/pipeline.py`
  Core training logic. This module contains the main data preparation helpers, tokenizer creation, embedding matrix building, CNN model definition, pseudo-label selection, evaluation, and metadata saving. It is the central module for understanding how the current training pipeline works.
- `digikala_sentiment/keras_compat.py`
  Compatibility imports for different Keras and TensorFlow setups. It helps the project import model-related classes from different package layouts depending on the installed environment.
- `digikala_sentiment/utils/preprocess.py`
  Text preprocessing logic used before tokenization and model training. It defines the Persian text cleaning and normalization steps applied before the model processes the comments.
- `digikala_sentiment/utils/download_dataset.py`
  Dataset download and embedding extraction utilities. It downloads the Kaggle dataset when needed and extracts the zipped embedding archive into the local training resources folder.

### Assets and Generated Artifacts

- `assets/embeddings/DigiKalaEmbeddingVectors.zip`
  The zipped embedding archive stored in the repository. It is kept as a `.zip` so the repository can store the embedding resource in a more manageable form than a large raw `.vec` file.
- `artifacts/data/raw/`
  Default location for the downloaded dataset CSV. This folder is created or reused automatically by the training workflow.
- `artifacts/resources/embeddings/`
  Default location for the extracted `.vec` embedding file used during training. The file in this folder is generated from the zipped archive when necessary.
- `artifacts/models/cnn_pseudolabel/`
  Default location for the final model artifacts. This folder contains the trained Keras model, tokenizer, and metadata.
- `artifacts/reports/`
  Default location for the generated training report. The report summarizes configuration details, dataset sizes, and evaluation metrics.

### Notebook

- `notebooks/digikala-sentiment-workflow-and-results.ipynb`
  Notebook version of the workflow for readers who want to inspect the experiment in a more research-oriented format.

### Folder-by-Folder Explanation

- `assets/`
  Stores repository-owned static resources that are not generated during execution.
- `assets/embeddings/`
  Stores the zipped fastText embedding archive used by the project.
- `artifacts/`
  Stores outputs created by running the code, keeping generated files separate from the source tree.
- `artifacts/data/`
  Stores dataset-related files created or downloaded during project use.
- `artifacts/data/raw/`
  Stores the raw downloaded Digikala dataset CSV used as training input.
- `artifacts/resources/`
  Stores prepared resources that the training code consumes directly.
- `artifacts/resources/embeddings/`
  Stores the extracted `DigiKalaEmbeddingVectors.vec` file used to build the embedding matrix.
- `artifacts/models/`
  Stores trained model outputs.
- `artifacts/models/cnn_pseudolabel/`
  Stores the final model files for the CNN pseudo-labeling training run.
- `artifacts/reports/`
  Stores training summaries and generated reports.
- `digikala_sentiment/`
  Main Python package of the project. This is where the reusable project logic lives.
- `digikala_sentiment/utils/`
  Helper modules for preprocessing and resource preparation.
- `notebooks/`
  Notebook-based experimental and explanatory material.

## Data Sources and External Links

### Kaggle Dataset

The Digikala dataset used by this project is available on Kaggle:

- <https://www.kaggle.com/datasets/sajjdeus/digikala-sentimentanalysis-3milioncomments>

You do not need to download the dataset manually in the normal training workflow. When `python train_sentiment_model.py` starts, it checks whether the dataset already exists locally. If the dataset is missing, it is downloaded automatically through `kagglehub`.

The Kaggle page is still the right place to visit if you want:

- more information about the dataset
- metadata about the source data
- direct access to the dataset page
- a reference link for the data used by the project

By default, the training script creates or reuses this local dataset path:

- `artifacts/data/raw/Digikala_3M.csv`

### Hugging Face Inference

Inference is available separately on Hugging Face:

- <https://huggingface.co/spaces/sajjadly/persian_sentiment>

The local inference script was removed from this repository so that the codebase remains focused on model training, data preparation, and reproducibility.

## Installation

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Depending on the local environment, TensorFlow installation may require platform-specific setup. If TensorFlow is not available, the training code will not run.

## Training Workflow

### Basic Command

To run training with the default paths:

```bash
python train_sentiment_model.py
```

This is the recommended way to use the repository.

### What Happens Automatically

When the training script runs, it performs several setup steps automatically:

- it checks whether the dataset CSV already exists locally
- if the dataset is missing, it downloads it automatically from Kaggle using `kagglehub`
- it checks whether the extracted embedding file already exists locally
- if the embedding file is missing, it extracts it from `assets/embeddings/DigiKalaEmbeddingVectors.zip`
- it creates output directories for model artifacts and reports if they do not already exist

So for standard usage, starting training is enough. A separate manual dataset download is usually not necessary.

After setup, the actual training pipeline begins:

- the dataset is loaded into pandas
- labeled rows and unlabeled rows are separated
- comments are cleaned with the project preprocessor
- labeled data is split into train and test partitions
- an initial tokenizer and embedding matrix are built
- the first CNN model is trained
- the unlabeled partition is scored by that first model
- highly confident pseudo-labels are selected
- the final model is trained on the expanded dataset
- metrics are calculated on the holdout set
- the model, tokenizer, metadata, and report are saved

## Model and Training Details

The current implementation in `digikala_sentiment/pipeline.py` uses the following settings:

- random seed: `2020`
- embedding size: `100`
- maximum sequence length: `400`
- tokenizer vocabulary cap: `2,000,000`
- batch size: `2048`
- epochs: `7`

Pseudo-label filtering thresholds:

- pseudo negative max probability: `1e-7`
- pseudo positive min probability: `0.9`

### Label Construction

The code builds labels from the dataset columns as follows:

- rows where `Satisfied == 1` or `Unsatisfied == 1` are treated as labeled data
- rows where both `Satisfied == 0` and `Unsatisfied == 0` are treated as unlabeled data
- the final binary label is initialized as `0`
- rows with `Unsatisfied == 1` are reassigned to label `1`

This means the repository uses the dataset's existing sentiment indicators rather than introducing a new annotation scheme.

### Class Balancing

The training code also performs a simple oversampling step:

- all rows with label `1` are appended once again to the training set

This is a straightforward balancing strategy implemented directly in the current code.

### Model Architecture

The CNN model defined in the training pipeline consists of:

- an embedding layer initialized from the fastText embedding matrix
- dropout
- a 1D convolution layer with `128` filters and kernel size `3`
- global max pooling
- a dense layer with `64` units
- dropout
- a dense layer with `16` units
- dropout
- a final sigmoid output layer

The model is compiled with:

- loss: `binary_crossentropy`
- optimizer: `adam`
- metric: `accuracy`

## Preprocessing

The preprocessing logic lives in:

- `digikala_sentiment/utils/preprocess.py`

The preprocessor applies a lightweight set of transformations designed to normalize review text before tokenization. In the current implementation, it includes steps such as:

- converting Persian and Arabic digits to standard Latin digits
- removing some URLs and `@`-style patterns
- reducing repeated punctuation and formatting noise
- removing HTML-like tags
- removing digits
- cleaning repeated slashes, underscores, and extra spaces
- removing a few specific tokens such as `product`, `dkp`, `br`, and `mm`

The project intentionally uses relatively simple preprocessing rather than heavy linguistic normalization. That fits the spirit of the original approach and keeps the pipeline easier to reproduce.

## Embedding File Handling

The repository stores the embedding resource here:

- `assets/embeddings/DigiKalaEmbeddingVectors.zip`

The extracted file used during training is:

- `artifacts/resources/embeddings/DigiKalaEmbeddingVectors.vec`

This design has two practical advantages:

- the repository can keep the embedding resource in a GitHub-friendlier archive format
- the training script can reconstruct the exact local `.vec` path it needs when training begins

If the extracted `.vec` file is already present, the code reuses it. If it is not present, the archive is extracted automatically.

## Outputs

By default, training produces the following files:

- `artifacts/models/cnn_pseudolabel/model.keras`
- `artifacts/models/cnn_pseudolabel/tokenizer.json`
- `artifacts/models/cnn_pseudolabel/metadata.json`
- `artifacts/reports/cnn_pseudolabel_report.txt`

### Model File

`model.keras` contains the final trained Keras model.

### Tokenizer File

`tokenizer.json` stores the tokenizer vocabulary and configuration used during training.

### Metadata File

`metadata.json` stores useful training information such as:

- dataset path
- embedding path
- maximum sequence length
- epoch count
- batch size
- pseudo-label thresholds
- positive and negative label identifiers

### Training Report

The text report summarizes the training run, including:

- data and embedding file locations
- training configuration
- labeled and unlabeled row counts
- train/test split sizes
- selected pseudo-labeled sample count
- initial model holdout metrics
- final model holdout metrics

## Command-Line Options

You can inspect the supported arguments with:

```bash
python train_sentiment_model.py --help
```

The main optional arguments are:

- `--dataset`
  Path to the dataset CSV file. If the file does not exist, the code attempts to download it.
- `--embedding-archive`
  Path to the zip archive that contains the embedding vectors.
- `--embedding-dir`
  Directory where the `.vec` file should be extracted.
- `--model-dir`
  Directory where model artifacts should be saved.
- `--report-path`
  Path where the text report should be written.
- `--test-size`
  Fraction of labeled data reserved for evaluation.

Example:

```bash
python train_sentiment_model.py \
  --dataset artifacts/data/raw/Digikala_3M.csv \
  --embedding-archive assets/embeddings/DigiKalaEmbeddingVectors.zip \
  --embedding-dir artifacts/resources/embeddings \
  --model-dir artifacts/models/cnn_pseudolabel \
  --report-path artifacts/reports/cnn_pseudolabel_report.txt \
  --test-size 0.2
```

## Preparing Resources Without Full Training

If you want to download the dataset and extract the embeddings before starting a training run, use:

```bash
python -m digikala_sentiment.utils.download_dataset
```

This command prepares both of the required resources:

- `artifacts/data/raw/Digikala_3M.csv`
- `artifacts/resources/embeddings/DigiKalaEmbeddingVectors.vec`

That can be useful if you want to confirm the environment is set up correctly before launching a full training job.

## Notebook

The notebook included in `notebooks/` is intended to document the workflow in a more exploratory and research-friendly format:

- `notebooks/digikala-sentiment-workflow-and-results.ipynb`

It is useful for:

- reviewing the broader modeling process
- understanding the original experiment flow
- inspecting intermediate ideas in notebook form
- comparing the script-based implementation to the notebook-based workflow

More specifically, this notebook contains:

- an introduction to the Digikala sentiment-analysis experiments
- a data loading and preprocessing section
- dataset inspection and example rows from the CSV
- tokenizer and embedding-matrix preparation steps
- deep-learning modeling experiments on the Digikala comments
- TF-IDF-based baseline material in the same workflow
- intermediate outputs, metrics, and experiment-oriented analysis

So the `.ipynb` file is not just a short demo. It is the research-style experimental record of the project and explains the workflow in a broader way than the training script alone.

For actual project usage, the Python scripts are the recommended interface because they are more structured and reproducible than notebook execution.

## Practical Notes

### Kaggle Access

The automatic dataset download relies on `kagglehub`. That means the environment must have the necessary access and configuration to retrieve the dataset.

### TensorFlow Requirement

The training pipeline requires TensorFlow or a compatible Keras/TensorFlow stack. Without that dependency, the model cannot be trained.

### Existing Artifacts

If the dataset CSV or extracted embedding file already exists in the expected paths, the helper utilities will reuse them instead of downloading or extracting again.

### Memory and Runtime

Because the dataset and embedding resources are large, training may require a machine with enough RAM, storage space, and a suitable Python environment. Runtime will depend heavily on available hardware.

## Reproducibility and Maintenance Notes

This repository now emphasizes a cleaner and more professional structure than the original collection of scripts and notebooks. The current layout separates:

- training entry points
- reusable package code
- utility helpers
- notebook material
- generated artifacts
- static assets

That makes it easier to understand what belongs to the source tree and what belongs to generated output.

At the same time, the project remains intentionally simple. It does not introduce configuration frameworks, experiment tracking systems, or large abstractions that would make the core workflow harder to follow.

## Summary

In short, this repository provides:

- a reproducible training entry point for Persian sentiment analysis on Digikala comments
- automatic preparation of the dataset and embedding resources
- a clear implementation of a CNN plus pseudo-labeling workflow
- reusable preprocessing and training modules
- a notebook for deeper inspection of the experimental process
- a separate Hugging Face deployment for inference

If you want to train the model, start with:

```bash
python -m pip install -r requirements.txt
python train_sentiment_model.py
```
