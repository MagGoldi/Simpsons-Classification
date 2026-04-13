"""
train.py — точка входа для обучения модели.

Все настройки берутся из config.py. Запуск:

    python scripts/train.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

import config
from src.dataset import create_dataloaders, SimpsonsDataset
from src.models import build_model
from src.trainer import train_loop
from src.utils import load_files, get_label_encoder
from src.visualization import generate_eda_reports, generate_post_training_reports
from src.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Полный пайплайн обучения: загрузка данных → обучение → отчёты → чекпоинт."""

    # --- Данные ---------------------------------------------------------------
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME} | Fine-tuning: {config.FINE_TUNING}")

    train_val_files, test_files = load_files(config.TRAIN_DIR, config.TEST_DIR)
    label_encoder, train_val_labels = get_label_encoder(train_val_files)
    logger.info(
        f"Dataset: {len(train_val_files)} train samples, "
        f"{len(test_files)} test samples, "
        f"{len(label_encoder.classes_)} classes"
    )

    # Сохраняем label encoder для инференса и evaluate.py
    le_path = config.DATA_DIR / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved → {le_path}")

    train_files, val_files = train_test_split(
        train_val_files, test_size=config.VAL_SIZE, stratify=train_val_labels, random_state=42
    )

    loaders, train_dataset, val_dataset = create_dataloaders(
        train_files, val_files, label_encoder,
        balanced=False, upsample=config.UPSAMPLE,
    )

    generate_eda_reports(
        train_dataset, val_dataset, label_encoder,
        output_dir=str(config.REPORTS_DIR),
    )

    # --- Модель ---------------------------------------------------------------
    n_classes = len(label_encoder.classes_)
    model = build_model(
        model_name=config.MODEL_NAME,
        n_classes=n_classes,
        pretrained=config.PRETRAINED,
    ).to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    # --- Этап 1: обучение с замороженным backbone -----------------------------
    logger.info("Stage 1: training classifier head (backbone frozen)")
    model.freeze_backbone()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    result = train_loop(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=criterion,
        max_epochs=config.MAX_EPOCHS,
        device=config.DEVICE,
        patience=config.EARLY_STOPPING_PATIENCE,
        experiment_name=f"{config.MODEL_NAME}_frozen",
    )

    # --- Этап 2: fine-tuning (опционально) ------------------------------------
    if config.FINE_TUNING:
        logger.info("Stage 2: fine-tuning (backbone unfrozen)")
        model.unfreeze_backbone()

        optimizer_ft = optim.AdamW(
            model.parameters(),
            lr=config.FINETUNE_LR,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="min", patience=3)

        result = train_loop(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            optimizer=optimizer_ft,
            scheduler=scheduler_ft,
            loss_func=criterion,
            max_epochs=config.FINETUNE_EPOCHS,
            device=config.DEVICE,
            patience=config.EARLY_STOPPING_PATIENCE,
            experiment_name=f"{config.MODEL_NAME}_finetune",
        )

    # --- Отчёты и чекпоинт ---------------------------------------------------
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
    ckpt_path = checkpoint_dir / "best_model.pth"
    torch.save(result["model"].state_dict(), ckpt_path)
    logger.info(f"Checkpoint saved → {ckpt_path}")
    logger.info(
        f"Training complete. Best val F1-macro: {result['best_val_f1_macro']:.4f} "
        f"(epoch {result['best_epoch'] + 1})"
    )


if __name__ == "__main__":
    main()