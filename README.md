# Persian Sentiment Analysis With fastText and Pseudo Labeling

## Overview
This repository reproduces the sentiment-analysis pipeline described in the original notebooks using standalone Python scripts. The workflow compares four models on the Digikala comments dataset:

- BiLSTM with fastText embeddings
- CNN with fastText embeddings
- Naive Bayes with TF-IDF features
- Logistic Regression with TF-IDF features

It also reproduces the CNN comparison across three situations:

- Before preprocessing
- After preprocessing
- After pseudo labeling

The best final model is a CNN artifact that can be used later for inference on new Persian sentences.

## Required Files
Put these files in the project root before training:

1. `Digikala_3M.csv`
Download the dataset manually from Kaggle:
`https://www.kaggle.com/datasets/sajjdeus/digikala-sentimentanalysis-3milioncomments`

2. `DigiKalaEmbeddingVectors.vec`
Place the fastText embedding file in the project root.

The scripts do not download the dataset automatically. They expect the dataset CSV to be available locally in the root folder.

## Project Files
Core files:

- `preprocess.py`: Persian text cleaning used before feature extraction.
- `sentiment_pipeline.py`: Shared training, evaluation, pseudo-labeling, artifact saving, and inference helpers.
- `reproduce_tables.py`: Runs the full paper-style pipeline and writes the comparison tables to CSV files under `results/`.
- `inference.py`: Loads the saved best CNN artifact and predicts sentiment for new Persian sentences.

Per-model training scripts:

- `train_cnn.py`
- `train_bilstm.py`
- `train_naive_bayes.py`
- `train_logistic_regression.py`

Outputs:

- `results/table4_model_comparison.csv`
- `results/table5_cnn_situations.csv`
- `results/summary.json`
- `results/artifacts/best_cnn_model/`

## Pipeline
The original notebook pipeline is now implemented in scripts with the same high-level flow:

1. Load `Digikala_3M.csv`
2. Normalize dataset column names
3. Build labeled data from positive and negative feedback rows
4. Clean comments with `preprocess.py` when preprocessing is enabled
5. Extract features
For BiLSTM and CNN: use `DigiKalaEmbeddingVectors.vec`
For Naive Bayes and Logistic Regression: use TF-IDF
6. Run 5-fold stratified cross-validation
7. Compute AUC and F-score for each fold
8. Reproduce Table 4 and Table 5 as CSV files
9. Train and save the final best CNN model for inference

Pseudo labeling flow:

1. Train a CNN on labeled data
2. Predict unlabeled comments
3. Keep only highly confident predictions
4. Add those pseudo-labeled rows back into the training set
5. Retrain the CNN on the augmented dataset

## Installation
Create or activate a Python environment, then install:

```bash
pip install -r requirements.txt
```

## Reproduce The Tables
Run the full comparison and artifact export:

```bash
python reproduce_tables.py --dataset-csv ./Digikala_3M.csv
```

For a lighter smoke test:

```bash
python reproduce_tables.py --dataset-csv ./Digikala_3M.csv --limit-rows 50000 --epochs 1
```

## Run A Single Model
Examples:

```bash
python train_cnn.py --dataset-csv ./Digikala_3M.csv --variant after-prep
python train_cnn.py --dataset-csv ./Digikala_3M.csv --variant pseudo --save-artifact
python train_bilstm.py --dataset-csv ./Digikala_3M.csv
python train_naive_bayes.py --dataset-csv ./Digikala_3M.csv
python train_logistic_regression.py --dataset-csv ./Digikala_3M.csv
```

## Inference
After `reproduce_tables.py` finishes, the best CNN model is saved under:

`results/artifacts/best_cnn_model/`

Run inference on new Persian sentences like this:

```bash
python inference.py --text "این گوشی خیلی خوبه" --text "اصلا از خریدش راضی نیستم"
```

You can also use an input file with one sentence per line:

```bash
python inference.py --input-file sentences.txt --output-json results/predictions.json
```

Each prediction includes:

- original sentence
- cleaned sentence
- negative probability
- positive probability
- predicted label

## Notes
- The notebook files were removed because the script pipeline now covers the reproducible workflow more clearly.
- The old split `DK.part01.rar` to `DK.part05.rar` archives were removed because the dataset should be downloaded manually from Kaggle instead.
- Multi-GPU training is supported through TensorFlow mirrored strategy when multiple GPUs are visible on one machine.
