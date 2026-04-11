import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import SimpsonsDataset
from src.models import SimpleCnn
from src.utils import load_files
from src.logger import setup_logger
from src.utils import get_label_encoder

logger = setup_logger(__name__)


def predict(model, loader, device):
    model.eval()
    all_predictions = []
    logger.info("Generating predictions on test set")

    with torch.no_grad():
        for inputs in tqdm(loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(-1)
            all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_predictions)


def main():
    le_path = config.DATA_DIR / "label_encoder.pkl"
    if not le_path.exists():
        logger.warning(f"{le_path} not found — recreating from train files")
        train_val_files, _ = load_files(config.TRAIN_DIR, config.TEST_DIR)
        label_encoder, _ = get_label_encoder(train_val_files)
    else:
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Loaded label encoder")

    _, test_files = load_files(config.TRAIN_DIR, config.TEST_DIR)
    test_dataset = SimpsonsDataset(test_files, label_encoder=label_encoder, mode="test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE)

    n_classes = len(label_encoder.classes_)
    model = SimpleCnn(n_classes=n_classes).to(config.DEVICE)

    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint {checkpoint_path} not found")
        return

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {checkpoint_path}")

    predicted_labels = predict(model, test_loader, config.DEVICE)
    predicted_names = label_encoder.inverse_transform(predicted_labels)
    file_names = [Path(p).name for p in test_files]

    submission = pd.DataFrame({"Id": file_names, "Expected": predicted_names})

    output_path = config.REPORTS_DIR / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission preview:\n{submission.head(10)}")


if __name__ == "__main__":
    main()