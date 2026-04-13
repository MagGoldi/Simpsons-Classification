"""
utils.py — вспомогательные функции: загрузка данных и кодирование меток.
"""

import os
import zipfile
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


def load_files(train_dir: Path, test_dir: Path) -> tuple:
    """Загрузить пути к изображениям; при пустых директориях скачать и извлечь датасет.

    Args:
        train_dir: Директория с обучающими данными.
        test_dir:  Директория с тестовыми данными.

    Returns:
        Tuple (train_val_files, test_files) — отсортированные списки путей.

    Raises:
        RuntimeError: Если gdown не установлен или скачивание/распаковка завершились ошибкой.
    """
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
            raise RuntimeError("gdown is required to download the dataset: pip install gdown")
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
        logger.info(f"Data found: '{train_dir}' and '{test_dir}'")

    train_val_files = sorted(train_dir.rglob("*.jpg"))
    test_files      = sorted(test_dir.rglob("*.jpg"))
    return train_val_files, test_files


def get_label_encoder(files: list) -> tuple:
    """Обучить LabelEncoder на именах родительских директорий файлов.

    Args:
        files: Список путей к изображениям (имя папки = имя класса).

    Returns:
        Tuple (label_encoder, labels) — обученный encoder и список строковых меток.
    """
    labels = [f.parent.name for f in files]
    le = LabelEncoder()
    le.fit(labels)
    return le, labels