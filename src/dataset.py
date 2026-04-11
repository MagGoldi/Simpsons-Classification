import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import torchvision.transforms.v2 as v2
from PIL import Image

from config import DATA_MODES, BATCH_SIZE, RESCALE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from src.logger import setup_logger

logger = setup_logger(__name__)


class SimpsonsDataset(Dataset):
    def __init__(self, files, label_encoder, mode):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        if self.mode not in DATA_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Expected one of {DATA_MODES}"
            )
        self.label_encoder = label_encoder
        self.len_ = len(self.files)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        x = self._load_image(self.files[index])
        x = self._apply_transforms(x)

        if self.mode == "test":
            return x

        label = self.label_encoder.transform([self.files[index].parent.name]).item()
        return x, label

    def _load_image(self, file):
        image = Image.open(file)
        image.load()
        return image

    def _apply_transforms(self, image):
        if self.mode == "train":
            transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(RESCALE_SIZE),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=25),
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


def create_dataloaders(train_files, val_files, label_encoder, balanced=False):
    train_ds = SimpsonsDataset(train_files, label_encoder, mode="train")
    val_ds = SimpsonsDataset(val_files, label_encoder, mode="val")

    if balanced:
        train_labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )
        sample_weights = torch.from_numpy(
            np.array([class_weights[label] for label in train_labels])
        ).float()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return {"train": train_loader, "val": val_loader}