"""
config.py — центральная конфигурация проекта.

Для запуска обучения достаточно отредактировать секцию
«USER SETTINGS» ниже и запустить:

    python scripts/train.py
"""

import torch
from pathlib import Path

# ==============================================================================
# USER SETTINGS — единственное место, которое нужно менять перед запуском
# ==============================================================================

# --- Модель -------------------------------------------------------------------
# Варианты:
#   "simple_cnn"          — лёгкая свёрточная сеть (baseline - быстрая, слабая)
#   "resnet18" / "resnet34" / "resnet50" / "resnet101" / "resnet152"
#   "efficientnet-b0" … "efficientnet-b7"
MODEL_NAME: str = "efficientnet-b4"

# Загружать предобученные веса ImageNet?
PRETRAINED: bool = True

# --- Обучение -----------------------------------------------------------------
MAX_EPOCHS: int = 25
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EARLY_STOPPING_PATIENCE: int = 7

# --- Fine-tuning --------------------------------------------------------------
# Если True — после базового обучения размораживает backbone и дообучает
# при пониженном LR (FINETUNE_LR).
FINE_TUNING: bool = True
FINETUNE_LR: float = 1e-5
FINETUNE_EPOCHS: int = 15  # эпохи для этапа fine-tuning

# --- Данные -------------------------------------------------------------------
VAL_SIZE: float = 0.2        # доля валидации (0.0–1.0)
UPSAMPLE: bool = True        # апсэмплинг миноритарных классов
MIN_SIZE_UPSAMPLE: int = 300 # минимальное кол-во сэмплов на класс

# ==============================================================================
# SYSTEM — не изменять
# ==============================================================================

# Пути
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
TRAIN_DIR    = DATA_DIR / "train"
TEST_DIR     = DATA_DIR / "testset"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# Препроцессинг изображений
RESCALE_SIZE    = [224, 224]
NORMALIZE_MEAN  = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD   = [0.229, 0.224, 0.225]  # ImageNet std

# Аугментации (применяются только на train)
AUGMENTATIONS_TRAIN = {
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomRotation":       {"degrees": 25},
    "RandomAffine":         {"degrees": 0, "translate": (0.1, 0.1)},
}

# Количество классов в датасете — не трогать
NUM_CLASSES: int = 42

# Режимы датасета
DATA_MODES = ["train", "val", "test"]

# Устройство определяется автоматически
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")