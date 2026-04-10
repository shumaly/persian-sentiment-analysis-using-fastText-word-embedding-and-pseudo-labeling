import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KAGGLE_DATASET = "sajjdeus/digikala-sentimentanalysis-3milioncomments"
DEFAULT_DATASET_FILENAME = "Digikala_3M.csv"
DEFAULT_EMBEDDING_FILENAME = "DigiKalaEmbeddingVectors.vec"
DEFAULT_DATA_DIR = PROJECT_ROOT / "artifacts" / "data" / "raw"
DEFAULT_EMBEDDING_ARCHIVE_PATH = PROJECT_ROOT / "assets" / "embeddings" / "DigiKalaEmbeddingVectors.zip"
DEFAULT_EMBEDDING_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "resources" / "embeddings"


def download_dataset(
    target_path: Path | None = None,
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    filename: str = DEFAULT_DATASET_FILENAME,
) -> Path:
    target = (target_path or (DEFAULT_DATA_DIR / filename)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return target

    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(dataset_slug))
    source = dataset_dir / filename

    if not source.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {source}")

    shutil.copy2(source, target)
    return target


def ensure_embedding_file(
    archive_path: Path | None = None,
    output_dir: Path | None = None,
    filename: str = DEFAULT_EMBEDDING_FILENAME,
) -> Path:
    archive = (archive_path or DEFAULT_EMBEDDING_ARCHIVE_PATH).resolve()
    destination_dir = (output_dir or DEFAULT_EMBEDDING_OUTPUT_DIR).resolve()
    destination = destination_dir / filename

    if destination.exists():
        return destination

    if not archive.exists():
        raise FileNotFoundError(f"Embedding archive not found: {archive}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extract(filename, path=destination_dir)

    if not destination.exists():
        raise FileNotFoundError(f"Expected embedding file not found after extraction: {destination}")

    return destination


def main() -> None:
    dataset_path = download_dataset()
    embedding_path = ensure_embedding_file()
    print(f"Dataset ready at: {dataset_path}")
    print(f"Embeddings ready at: {embedding_path}")


if __name__ == "__main__":
    main()
