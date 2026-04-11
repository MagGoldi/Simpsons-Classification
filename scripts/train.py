import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

import config
from src.dataset import create_dataloaders, SimpsonsDataset
from src.models import SimpleCnn
from src.trainer import train_loop
from src.utils import load_files, get_label_encoder
from src.visualization import generate_eda_reports, generate_post_training_reports
from src.logger import setup_logger

logger = setup_logger(__name__)


def main():
    train_val_files, test_files = load_files(config.TRAIN_DIR, config.TEST_DIR)
    label_encoder, train_val_labels = get_label_encoder(train_val_files)

    le_path = config.DATA_DIR / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved to {le_path}")

    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, stratify=train_val_labels
    )

    loaders = create_dataloaders(train_files, val_files, label_encoder, balanced=False)

    generate_eda_reports(
        train_files, val_files, label_encoder,
        output_dir=str(config.REPORTS_DIR),
    )

    model = SimpleCnn(n_classes=len(label_encoder.classes_)).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    criterion = torch.nn.CrossEntropyLoss()

    result = train_loop(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=criterion,
        max_epochs=config.MAX_EPOCHS,
        device=config.DEVICE,
        experiment_name="simple_cnn_unbalanced",
    )

    logger.info("Generating post-training reports")
    val_dataset_vis = SimpsonsDataset(val_files, label_encoder=label_encoder, mode="val")
    generate_post_training_reports(
        result=result,
        val_dataset=val_dataset_vis,
        label_encoder=label_encoder,
        output_dir=str(config.REPORTS_DIR),
    )

    checkpoint_dir = config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), checkpoint_dir / "best_model.pth")
    logger.info(f"Best model checkpoint saved to {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()