"""
dataset.py — torch Dataset и фабрика DataLoader'ов для датасета Симпсонов.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import torchvision.transforms.v2 as v2
from PIL import Image

from config import DATA_MODES, BATCH_SIZE, RESCALE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, MIN_SIZE_UPSAMPLE
from src.logger import setup_logger

logger = setup_logger(__name__)


class SimpsonsDataset(Dataset):
    """PyTorch Dataset для задачи классификации персонажей Симпсонов.

    Args:
        files:         Список путей к изображениям (Path objects).
        label_encoder: Обученный sklearn LabelEncoder.
        mode:          Один из DATA_MODES: "train", "val", "test".
    """

    def __init__(self, files, label_encoder, mode: str):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        if self.mode not in DATA_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Expected one of {DATA_MODES}")
        self.label_encoder = label_encoder
        self.len_ = len(self.files)

    def __len__(self) -> int:
        return self.len_

    def __getitem__(self, index):
        """Вернуть (tensor, label) для train/val или tensor для test."""
        x = self._apply_transforms(self._load_image(self.files[index]))
        if self.mode == "test":
            return x
        label = self.label_encoder.transform([self.files[index].parent.name]).item()
        return x, label

    def _load_image(self, file) -> Image.Image:
        """Загрузить изображение с диска в PIL."""
        image = Image.open(file)
        image.load()
        return image

    def _apply_transforms(self, image) -> torch.Tensor:
        """Применить аугментации (train) или только resize+normalize (val/test)."""
        if self.mode == "train":
            transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(RESCALE_SIZE),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=45),
                v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ])
        else:
            transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(RESCALE_SIZE),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ])
        return transform(image)


def _create_label_to_paths(files, label_encoder) -> dict:
    """Построить словарь {label_int: [Path, ...]} по списку файлов."""
    labels = [label_encoder.transform([f.parent.name]).item() for f in files]
    dct = {lbl: [] for lbl in np.unique(labels).tolist()}
    for path, lbl in zip(files, labels):
        dct[lbl].append(path)
    return dct


def _upsample_files(train_files, label_encoder, min_size: int = MIN_SIZE_UPSAMPLE) -> list:
    """Дублировать сэмплы миноритарных классов до min_size изображений на класс.

    Args:
        train_files:   Список путей к обучающим изображениям.
        label_encoder: Обученный LabelEncoder.
        min_size:      Минимальное количество сэмплов на класс.

    Returns:
        Расширенный список путей.
    """
    dct = _create_label_to_paths(train_files, label_encoder)
    upsampled = []
    for lbl, paths in dct.items():
        n = len(paths)
        if n < min_size:
            factor = min_size // n
            remainder = min_size - n * factor
            paths = paths * factor + paths[:remainder]
            logger.debug(f"Class {lbl}: upsampled {n} → {len(paths)} samples")
        upsampled.extend(paths)
    return upsampled


def create_dataloaders(
    train_files,
    val_files,
    label_encoder,
    balanced: bool = False,
    upsample: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """Создать DataLoader'ы для train и val разбивок.

    Args:
        train_files:   Пути к обучающим изображениям.
        val_files:     Пути к валидационным изображениям.
        label_encoder: Обученный LabelEncoder.
        balanced:      Использовать WeightedRandomSampler для балансировки.
        upsample:      Апсэмплировать миноритарные классы перед обучением.

    Returns:
        Tuple (loaders_dict, train_dataset, val_dataset), где
        loaders_dict — {"train": DataLoader, "val": DataLoader}.
    """
    if upsample:
        train_files = _upsample_files(train_files, label_encoder)
        logger.info(f"Upsampling done: {len(train_files)} total train samples")

    train_dataset = SimpsonsDataset(train_files, label_encoder, mode="train")
    val_dataset   = SimpsonsDataset(val_files,   label_encoder, mode="val")

    if balanced:
        train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )
        sample_weights = torch.from_numpy(
            np.array([class_weights[lbl] for lbl in train_labels])
        ).float()
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return {"train": train_loader, "val": val_loader}, train_dataset, val_dataset