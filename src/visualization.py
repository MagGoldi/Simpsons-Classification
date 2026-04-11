import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow

import config
from src.dataset import SimpsonsDataset
from src.logger import setup_logger

logger = setup_logger(__name__)


def _save_figure(save_path):
    """Persist current figure to disk, creating parent directories as needed."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()


def _denormalize(img_tensor):
    """Reverse ImageNet normalization for display."""
    img = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array(config.NORMALIZE_MEAN)
    std = np.array(config.NORMALIZE_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def imshow(inp, title=None, plt_ax=None):
    if plt_ax is None:
        _, plt_ax = plt.subplots()
    plt_ax.imshow(_denormalize(inp))
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def show_augmentations(dataset, num_images=5, save_path=None):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    for i in range(num_images):
        img, label = dataset[i]
        axes[i].imshow(_denormalize(img))
        axes[i].set_title(f"Label idx: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    _save_figure(save_path)


def show_images(dataset, label_encoder, n_rows=3, n_cols=6, save_path=None):
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        sharey=True, sharex=True,
    )
    for ax in axes.flatten():
        idx = int(np.random.uniform(0, len(dataset)))
        img, label = dataset[idx]
        try:
            text_label = label_encoder.inverse_transform([label])[0]
        except Exception:
            text_label = str(label)
        display_label = " ".join(w.capitalize() for w in text_label.split("_"))
        imshow(img.data.cpu(), title=display_label, plt_ax=ax)
        ax.set_axis_off()
    plt.tight_layout()
    _save_figure(save_path)


def plot_training_history(history, save_path=None):
    df = pd.DataFrame(history).set_index("epoch")
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    if "train_loss" in df.columns:
        ax1.plot(df.index, df["train_loss"], marker="o", label="Train Loss", linewidth=2)
    if "val_loss" in df.columns:
        ax1.plot(df.index, df["val_loss"], marker="s", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve", fontsize=14)
    ax1.legend()

    if "train_acc" in df.columns:
        ax2.plot(df.index, df["train_acc"], marker="o", label="Train Acc", linewidth=2)
    if "val_acc" in df.columns:
        ax2.plot(df.index, df["val_acc"], marker="s", label="Val Acc", linewidth=2)
    if "val_f1_macro" in df.columns:
        ax2.plot(df.index, df["val_f1_macro"], marker="^", label="Val F1 (Macro)", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Accuracy & F1", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    _save_figure(save_path)

    if len(df) > 1:
        diff_df = df.astype(float).diff().dropna()
        diff_df = diff_df.rename(columns={col: f"Δ{col}" for col in diff_df.columns})
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            diff_df.T, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, cbar_kws={"label": "Delta"},
        )
        plt.title("Per-Epoch Metric Changes", fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.tight_layout()
        heatmap_path = str(save_path).replace(".png", "_heatmap.png") if save_path else None
        _save_figure(heatmap_path)


@torch.no_grad()
def show_model_predictions(predictions, probabilities, dataset, label_encoder,
                           n_rows=3, n_cols=3, save_path=None):
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        sharey=True, sharex=True,
    )
    indices = np.random.choice(len(dataset), n_rows * n_cols, replace=False)

    for i, idx in enumerate(indices):
        ax = axs.flatten()[i]
        img, label = dataset[idx]

        true_name = label_encoder.inverse_transform([label])[0]
        true_label = " ".join(w.capitalize() for w in true_name.split("_"))

        pred_idx = predictions[idx]
        pred_prob = probabilities[idx][pred_idx] * 100
        pred_name = label_encoder.inverse_transform([pred_idx])[0]
        pred_label = " ".join(w.capitalize() for w in pred_name.split("_"))

        imshow(img.cpu(), title=f"Actual: {true_label}", plt_ax=ax)

        color = "green" if pred_idx == label else "red"
        ax.add_patch(patches.Rectangle((0, 190), 224, 34, color="white", alpha=0.7))
        ax.text(5, 205, f"{pred_label}\n{pred_prob:.1f}%", color=color, weight="bold", fontsize=9)
        ax.set_axis_off()

    plt.tight_layout()
    _save_figure(save_path)


@torch.no_grad()
def analyze_predictions(all_preds, all_labels, all_probs, label_encoder):
    """Aggregate prediction statistics per class without re-running inference."""
    class_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": []})

    for true, pred, probs in zip(all_labels, all_preds, all_probs):
        name = label_encoder.inverse_transform([true])[0].split("_")[-1]
        class_stats[name]["total"] += 1
        if true == pred:
            class_stats[name]["correct"] += 1
        else:
            pred_name = label_encoder.inverse_transform([pred])[0].split("_")[-1]
            class_stats[name]["errors"].append({"pred": pred_name, "conf": probs[pred]})

    results = []
    for name, stats in class_stats.items():
        err_count = stats["total"] - stats["correct"]
        results.append({
            "class": name,
            "total": stats["total"],
            "correct": stats["correct"],
            "errors": err_count,
            "error_rate": (err_count / stats["total"] * 100) if stats["total"] > 0 else 0,
        })
    return pd.DataFrame(results).sort_values("error_rate", ascending=False)


def plot_error_analysis(df_results, save_path=None, top_n=10):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    top_errors = df_results.nlargest(top_n, "error_rate")
    sns.barplot(x="error_rate", y="class", data=top_errors, ax=axes[0],
                hue="class", palette="Reds_r", legend=False)
    axes[0].set_title(f"Top {top_n} Class Error Rates (%)")

    sns.barplot(x="total", y="class", data=df_results.head(top_n), ax=axes[1],
                hue="class", palette="Blues_r", legend=False)
    axes[1].set_title("Samples per Class")

    df_melt = df_results.head(top_n).melt(id_vars="class", value_vars=["correct", "errors"])
    sns.barplot(x="value", y="class", hue="variable", data=df_melt, ax=axes[2])
    axes[2].set_title("Correct vs Errors")

    plt.tight_layout()
    _save_figure(save_path)


def plot_confusion_matrix(all_preds, all_labels, label_encoder, save_path=None, top_n=12):
    class_names = [
        label_encoder.inverse_transform([i])[0].split("_")[-1]
        for i in range(len(label_encoder.classes_))
    ]
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    error_counts = cm.sum(axis=1) - np.diag(cm)
    top_classes = np.argsort(error_counts)[-top_n:][::-1]

    cm_filtered = cm_norm[top_classes][:, top_classes]
    labels_filtered = [class_names[i] for i in top_classes]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_filtered, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=labels_filtered, yticklabels=labels_filtered,
    )
    plt.title(f"Confusion Matrix (Top {top_n} Error Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    _save_figure(save_path)


def show_misclassified_examples(predictions, labels, probabilities, dataset,
                                label_encoder, save_path=None, num_examples=9):
    mis_indices = np.where(predictions != labels)[0]
    if len(mis_indices) == 0:
        logger.info("No misclassified examples found")
        return

    plot_indices = np.random.choice(mis_indices, min(len(mis_indices), num_examples), replace=False)

    misclassified = []
    for idx in plot_indices:
        img, label = dataset[idx]
        pred = predictions[idx]
        true_name = label_encoder.inverse_transform([label])[0].split("_")[-1]
        pred_name = label_encoder.inverse_transform([pred])[0].split("_")[-1]
        conf = probabilities[idx][pred] * 100
        misclassified.append({"img": img, "true": true_name, "pred": pred_name, "conf": conf})

    n_cols = 3
    n_rows = (len(misclassified) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, ex in enumerate(misclassified):
        imshow(ex["img"], plt_ax=axes[i])
        axes[i].set_title(
            f"T: {ex['true']} | P: {ex['pred']}\nConf: {ex['conf']:.1f}%",
            color="red",
        )
        axes[i].axis("off")

    for j in range(len(misclassified), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Misclassified Examples", fontsize=16)
    plt.tight_layout()
    _save_figure(save_path)


def generate_eda_reports(train_files, val_files, label_encoder, output_dir=None):
    output_dir = output_dir or str(config.REPORTS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = SimpsonsDataset(train_files, label_encoder=label_encoder, mode="val")
    val_dataset = SimpsonsDataset(val_files, label_encoder=label_encoder, mode="val")

    show_augmentations(train_dataset, num_images=5,
                       save_path=os.path.join(output_dir, "01_augmentations.png"))
    show_images(val_dataset, label_encoder, n_rows=3, n_cols=6,
                save_path=os.path.join(output_dir, "02_sample_images.png"))


def generate_post_training_reports(result, val_dataset, label_encoder, output_dir=None):
    """Generate all post-training visualizations from cached predictions."""
    output_dir = output_dir or str(config.REPORTS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    history = result["history"]
    all_preds = result["val_predictions"]
    all_labels = result["val_targets"]
    all_probs = result["val_probabilities"]

    plot_training_history(
        history, save_path=os.path.join(output_dir, "03_training_history.png")
    )
    show_model_predictions(
        all_preds, all_probs, val_dataset, label_encoder,
        save_path=os.path.join(output_dir, "04_predictions_grid.png"),
    )

    df_errors = analyze_predictions(all_preds, all_labels, all_probs, label_encoder)
    df_errors.to_csv(os.path.join(output_dir, "05_error_statistics.csv"), index=False)

    plot_error_analysis(
        df_errors, save_path=os.path.join(output_dir, "06_error_analysis.png")
    )
    plot_confusion_matrix(
        all_preds, all_labels, label_encoder,
        save_path=os.path.join(output_dir, "07_confusion_matrix.png"),
    )
    show_misclassified_examples(
        all_preds, all_labels, all_probs, val_dataset, label_encoder,
        save_path=os.path.join(output_dir, "08_misclassified_examples.png"),
    )

    logger.info(f"Post-training reports saved to '{output_dir}'")


def log_confusion_matrix(y_true, y_pred, class_names, step, artifact_path="confusion_matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix (Step {step})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), f"{artifact_path}/step_{step}.png")
    plt.close()