import os
import zipfile

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


def load_files(train_dir, test_dir):
    """Load dataset file paths; download and extract if directories are empty."""
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if not any(train_dir.iterdir()):
        logger.info("Data directories are empty — starting download")
        zip_filename = "journey-springfield.zip"

        try:
            import gdown

            gdown.download(
                id="1RxBQiZgRAfio2tWhEE7lzZ6IaJzLheH1",
                output=zip_filename,
                quiet=False,
            )
        except ImportError:
            raise RuntimeError("gdown is required for downloading data")
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

        logger.info("Extracting data archive")
        try:
            with zipfile.ZipFile(zip_filename, "r") as zf:
                for member in tqdm(zf.infolist(), desc="Extracting", unit="file"):
                    zf.extract(member, path=train_dir)
                    zf.extract(member, path=test_dir)
            os.remove(zip_filename)
            logger.info("Extraction complete")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    else:
        logger.info(f"Data found in '{train_dir}' and '{test_dir}'")

    train_val_files = sorted(list(train_dir.rglob("*.jpg")))
    test_files = sorted(list(test_dir.rglob("*.jpg")))
    return train_val_files, test_files


def get_label_encoder(files):
    """Fit a LabelEncoder on parent directory names."""
    labels = [f.parent.name for f in files]
    le = LabelEncoder()
    le.fit(labels)
    return le, labels