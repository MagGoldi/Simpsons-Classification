import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import config
from src.dataset import SimpsonsDataset, create_dataloaders
from src.models import SimpleCnn
from src.trainer import evaluate
from src.metrics import classwise_error_analysis
from src.utils import load_files, get_label_encoder
from src.visualization import (
    plot_confusion_matrix,
    plot_error_analysis,
    analyze_predictions,
    show_misclassified_examples,
    show_model_predictions,
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_model(checkpoint_path, n_classes, device):
    model = SimpleCnn(n_classes=n_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def generate_classification_report(targets, preds, label_encoder, save_path):
    """Full sklearn classification report saved as both text and JSON."""
    class_names = list(label_encoder.classes_)
    report_text = classification_report(
        targets, preds, target_names=class_names, zero_division=0
    )
    logger.info(f"Classification Report:\n{report_text}")

    report_dict = classification_report(
        targets, preds, target_names=class_names, zero_division=0, output_dict=True
    )

    text_path = save_path.with_suffix(".txt")
    with open(text_path, "w") as f:
        f.write(report_text)
    logger.info(f"Classification report (text) saved to {text_path}")

    json_path = save_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Classification report (json) saved to {json_path}")

    return report_dict


def generate_summary(metrics, report_dict, save_path):
    """One-page evaluation summary with key metrics."""
    summary = {
        "val_loss": metrics["loss"],
        "val_accuracy": metrics["accuracy"],
        "val_f1_macro": float(metrics["f1_macro"]),
        "val_f1_weighted": float(metrics["f1_weighted"]),
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation summary saved to {save_path}")

    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on validation set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.REPORTS_DIR),
        help="Directory for evaluation reports",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Validation split ratio",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    le_path = config.DATA_DIR / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Loaded label encoder from cache")
    else:
        logger.info("Label encoder not found — fitting from train data")
        train_val_files, _ = load_files(config.TRAIN_DIR, config.TEST_DIR)
        label_encoder, _ = get_label_encoder(train_val_files)

    train_val_files, _ = load_files(config.TRAIN_DIR, config.TEST_DIR)
    _, train_val_labels = get_label_encoder(train_val_files)

    _, val_files = train_test_split(
        train_val_files, test_size=args.test_size, stratify=train_val_labels
    )

    val_dataset = SimpsonsDataset(val_files, label_encoder=label_encoder, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    n_classes = len(label_encoder.classes_)
    model = load_model(checkpoint_path, n_classes, config.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("Running evaluation on validation set")
    metrics, preds, targets, probs = evaluate(
        model, val_loader, criterion, config.DEVICE, n_classes
    )

    logger.info(
        f"Val Loss: {metrics['loss']:.4f} | "
        f"Val Acc: {metrics['accuracy']:.4f} | "
        f"Val F1m: {metrics['f1_macro']:.4f} | "
        f"Val F1w: {metrics['f1_weighted']:.4f}"
    )

    report_dict = generate_classification_report(
        targets, preds, label_encoder,
        save_path=output_dir / "eval_classification_report",
    )

    generate_summary(
        metrics, report_dict,
        save_path=output_dir / "eval_summary.json",
    )

    df_errors = analyze_predictions(preds, targets, probs, label_encoder)
    df_errors.to_csv(output_dir / "eval_error_statistics.csv", index=False)

    classwise_error_analysis(
        preds, targets, probs, label_encoder,
        save_path=str(output_dir / "eval_classwise_errors.csv"),
    )

    plot_confusion_matrix(
        preds, targets, label_encoder,
        save_path=str(output_dir / "eval_confusion_matrix.png"),
    )

    plot_error_analysis(
        df_errors,
        save_path=str(output_dir / "eval_error_analysis.png"),
    )

    show_model_predictions(
        preds, probs, val_dataset, label_encoder,
        save_path=str(output_dir / "eval_predictions_grid.png"),
    )

    show_misclassified_examples(
        preds, targets, probs, val_dataset, label_encoder,
        save_path=str(output_dir / "eval_misclassified.png"),
    )

    logger.info("Evaluation complete — all reports saved")


if __name__ == "__main__":
    main()
