#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentiment_pipeline import DEFAULT_ARTIFACTS_DIR, predict_with_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on new Persian sentences using the saved CNN artifact.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "best_cnn_model")
    parser.add_argument("--text", action="append", help="Sentence to classify. Pass multiple times for multiple sentences.")
    parser.add_argument("--input-file", type=Path, help="Text file with one sentence per line.")
    parser.add_argument("--output-json", type=Path, help="Optional output path for the predictions.")
    return parser.parse_args()


def collect_sentences(args: argparse.Namespace) -> list[str]:
    sentences: list[str] = []
    if args.text:
        sentences.extend(args.text)
    if args.input_file:
        sentences.extend([line.strip() for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()])
    if not sentences:
        raise SystemExit("Provide at least one --text or --input-file.")
    return sentences


def main() -> None:
    args = parse_args()
    sentences = collect_sentences(args)
    predictions = predict_with_cnn(args.artifact_dir, sentences)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved predictions to {args.output_json}")
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
