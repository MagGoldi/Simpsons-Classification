import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "testset"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_MODES = ["train", "val", "test"]

RESCALE_SIZE = [224, 224]
BATCH_SIZE = 64
NUM_CLASSES = 42
MAX_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUGMENTATIONS_TRAIN = {
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomRotation": {"degrees": 10},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2},
}

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]