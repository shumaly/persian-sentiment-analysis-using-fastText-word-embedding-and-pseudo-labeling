# Accurate and Lightweight Persian Sentiment Analysis for E-commerce Reviews ([Try the demo](https://huggingface.co/spaces/sajjadly/persian_sentiment) • [Cite this work](#article-information))

<p align="center">
  <img src="e-commerce%20sentiment%20analysis.png" alt="Project Diagram" width="520">
</p>

A practical Persian sentiment analysis project built on **e-commerce reviews**, **fastText embeddings**, and a **CNN-based model**.

This repository is mainly for people who want to **train**, **understand**, **reproduce**, or **use in practice** the workflow behind the model. It is designed to be simple to run, easy to inspect, and useful as a reference for Persian review classification.

## What this project offers

### Lightweight and practical

Uses a CNN-based approach that is simpler than many heavier NLP pipelines.

### Built for Persian text

Designed for Persian e-commerce comments, where sentiment analysis can be challenging.

### Trained with large-scale review data

Uses **3 million e-commerce reviews** together with pretrained embeddings.

## Highlights

- **Model type:** CNN
- **Embeddings:** pretrained fastText
- **Training data:** 3 million e-commerce reviews
- **Learning strategy:** supervised training + pseudo-labeling
- **Reported performance:**
  - **AUC:** `0.9944`
  - **F-score:** `0.9288`

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run training:

```bash
python train_sentiment_model.py
```

That is the main entry point for most users.

This repository is mainly for **training** and for **inspecting the pipeline and results**. For inference, check the Hugging Face demo running on a free CPU plan: [https://huggingface.co/spaces/sajjadly/persian\_sentiment](https://huggingface.co/spaces/sajjadly/persian_sentiment)

## Project structure

```text
.
├── README.md
├── requirements.txt
├── train_sentiment_model.py
├── assets/
│   └── embeddings/
│       └── DigiKalaEmbeddingVectors.zip
├── artifacts/
│   ├── data/
│   ├── models/
│   ├── reports/
│   └── resources/
├── digikala_sentiment/
│   ├── pipeline.py
│   ├── keras_compat.py
│   └── utils/
│       ├── download_dataset.py
│       └── preprocess.py
└── notebooks/
    └── digikala-sentiment-workflow-and-results.ipynb
```

## Main files

### `train_sentiment_model.py`

The main training script. This is where most users should start.

### `digikala_sentiment/pipeline.py`

Contains the core training logic, including preprocessing, tokenizer setup, embedding matrix creation, CNN training, pseudo-label selection, evaluation, and saving outputs.

### `digikala_sentiment/utils/preprocess.py`

Handles text preprocessing and normalization before tokenization.

### `digikala_sentiment/utils/download_dataset.py`

Helps download the dataset and prepare the embedding resources when needed.

### `notebooks/digikala-sentiment-workflow-and-results.ipynb`

A notebook version of the workflow for people who want a more exploratory, research-style view of the project. It is useful for inspecting the workflow step by step.

## Data and resources

### Dataset

The Digikala dataset used in this project is available on Kaggle.

In the standard workflow, you usually do not need to download it manually. The training script checks whether the dataset exists and downloads it automatically if needed.

If you want more information about the dataset, see the Kaggle page: [https://www.kaggle.com/datasets/sajjdeus/digikala-sentimentanalysis-3milioncomments](https://www.kaggle.com/datasets/sajjdeus/digikala-sentimentanalysis-3milioncomments)

## Outputs

After training, the main outputs are saved under `artifacts/`.

Typical files include:

- `artifacts/models/cnn_pseudolabel/model.keras`
- `artifacts/models/cnn_pseudolabel/tokenizer.json`
- `artifacts/models/cnn_pseudolabel/metadata.json`
- `artifacts/reports/cnn_pseudolabel_report.txt`

These files store the trained model, tokenizer, training metadata, and a summary report.

## Article information

If you use this repository in research or academic work, please cite the original article.

**Title:** Persian sentiment analysis of an online store independent of pre-processing using convolutional neural network with fastText embeddings\
**Authors:** Sajjad Shumaly, Mohsen YazdiNejad, Yanhui Guo\
**Journal:** PeerJ Computer Science\
**Year:** 2021\
**DOI:** 10.7717/peerj-cs.422

You can read the article here: [https://peerj.com/articles/cs-422/](https://peerj.com/articles/cs-422/)
