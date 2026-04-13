"""
trainer.py — цикл обучения с early stopping, LR scheduler и MLflow логированием.
"""

import torch
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from tqdm import tqdm

import config
from src.metrics import calculate_f1_score
from src.visualization import log_confusion_matrix
from src.logger import setup_logger

warnings.filterwarnings("ignore", message="Found torch version.*contains a local version label")
warnings.filterwarnings("ignore", message="Saving pytorch model by Pickle.*requires exercising caution")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mlflow")

logger = setup_logger(__name__)


def train_one_epoch(model, loader, optimizer, loss_func, device, num_classes: int) -> dict:
    """Выполнить одну эпоху обучения.

    Returns:
        dict с ключами: loss, accuracy, f1_macro, f1_weighted.
    """
    model.train()
    total_loss, num_batches = 0.0, 0
    all_preds, all_targets = [], []

    for x_batch, y_batch in tqdm(loader, desc="Train", leave=False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outp = model(x_batch)
        loss = loss_func(outp, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(outp.argmax(-1).detach().cpu())
        all_targets.append(y_batch.detach().cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss":        total_loss / num_batches,
        "accuracy":    (all_preds == all_targets).float().mean().item(),
        "f1_macro":    calculate_f1_score(all_preds, all_targets, num_classes, "macro"),
        "f1_weighted": calculate_f1_score(all_preds, all_targets, num_classes, "weighted"),
    }


@torch.no_grad()
def evaluate(model, loader, loss_func, device, num_classes: int) -> tuple:
    """Оценить модель на переданном DataLoader'е без обновления весов.

    Returns:
        (metrics_dict, preds_np, targets_np, probs_np)
    """
    model.eval()
    total_loss, num_batches = 0.0, 0
    all_preds, all_targets, all_probs = [], [], []

    for x_batch, y_batch in tqdm(loader, desc="Eval ", leave=False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        outp = model(x_batch)
        total_loss += loss_func(outp, y_batch).item()
        num_batches += 1

        all_probs.append(torch.softmax(outp, dim=-1).detach().cpu())
        all_preds.append(outp.argmax(-1).detach().cpu())
        all_targets.append(y_batch.detach().cpu())

    all_probs   = torch.cat(all_probs)
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {
        "loss":        total_loss / num_batches,
        "accuracy":    (all_preds == all_targets).float().mean().item(),
        "f1_macro":    calculate_f1_score(all_preds, all_targets, num_classes, "macro"),
        "f1_weighted": calculate_f1_score(all_preds, all_targets, num_classes, "weighted"),
    }
    return metrics, all_preds.numpy(), all_targets.numpy(), all_probs.numpy()


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    max_epochs: int,
    device,
    num_classes: int = config.NUM_CLASSES,
    augments_used: str = "",
    scheduler=None,
    run_name: str = None,
    experiment_name: str = None,
    patience: int = 7,
    min_delta: float = 1e-4,
    class_names=None,
) -> dict:
    """Полный цикл обучения с early stopping и MLflow трекингом.

    Args:
        model:            Обучаемая модель.
        train_loader:     DataLoader для train.
        val_loader:       DataLoader для val.
        optimizer:        Оптимизатор.
        loss_func:        Функция потерь.
        max_epochs:       Максимальное количество эпох.
        device:           Устройство (cuda/cpu).
        num_classes:      Количество классов.
        augments_used:    Строка с описанием аугментаций (для MLflow).
        scheduler:        LR scheduler (опционально).
        run_name:         Имя MLflow run (опционально).
        experiment_name:  Имя MLflow эксперимента.
        patience:         Число эпох без улучшения до early stopping.
        min_delta:        Минимальное улучшение метрики.
        class_names:      Список имён классов для матрицы ошибок (опционально).

    Returns:
        dict с ключами: history, best_epoch, best_val_f1_macro,
        val_predictions, val_targets, val_probabilities, model.
    """
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    run_name = run_name or (f"aug_{augments_used}" if augments_used else "baseline")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model":        type(model).__name__,
            "max_epochs":   max_epochs,
            "batch_size":   train_loader.batch_size,
            "optimizer":    type(optimizer).__name__,
            "loss_func":    type(loss_func).__name__,
            "augmentations": augments_used,
            "num_classes":  num_classes,
            "patience":     patience,
            "min_delta":    min_delta,
        })
        for pg in optimizer.param_groups:
            mlflow.log_params({f"opt_{k}": v for k, v in pg.items() if k != "params"})

        model.to(device)
        best_val_f1      = -float("inf")
        best_epoch       = 0
        best_model_state = None
        epochs_no_improve = 0
        history = []

        for epoch in range(max_epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_func, device, num_classes)
            val_metrics, val_preds, val_targets, val_probs = evaluate(
                model, val_loader, loss_func, device, num_classes
            )

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            record = {
                "epoch":          epoch + 1,
                "train_loss":     train_metrics["loss"],
                "train_acc":      train_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss":       val_metrics["loss"],
                "val_acc":        val_metrics["accuracy"],
                "val_f1_macro":   val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
            }
            history.append(record)

            mlflow.log_metrics(
                {k: v for k, v in record.items() if k != "epoch"},
                step=epoch,
            )

            if class_names and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == max_epochs - 1):
                log_confusion_matrix(val_targets, val_preds, class_names, step=epoch)

            # --- early stopping & best model ----------------------------------
            current_val_f1 = val_metrics["f1_macro"]
            if current_val_f1 - best_val_f1 > min_delta:
                best_val_f1      = current_val_f1
                best_epoch       = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                mlflow.pytorch.log_model(model, artifact_path="best_model")
                logger.info(f"  ↑ New best at epoch {epoch + 1} | val_f1_macro={best_val_f1:.4f}")
            else:
                epochs_no_improve += 1

            logger.info(
                f"Epoch {epoch + 1:>3}/{max_epochs} | "
                f"loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"acc {val_metrics['accuracy']:.4f} | "
                f"f1w {val_metrics['f1_weighted']:.4f} | "
                f"f1m {val_metrics['f1_macro']:.4f}"
            )

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs (no improvement for {patience})")
                break

        # Восстанавливаем лучшие веса
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch + 1} (val_f1_macro={best_val_f1:.4f})")

        mlflow.pytorch.log_model(model, artifact_path="final_best_model")

        # Сохраняем историю обучения
        history_path = config.REPORTS_DIR / "training_history.csv"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        mlflow.log_artifact(str(history_path))

        # Финальная оценка на val
        final_val_metrics, final_preds, final_targets, final_probs = evaluate(
            model, val_loader, loss_func, device, num_classes
        )

        return {
            "history":           history,
            "best_epoch":        best_epoch,
            "best_val_f1_macro": best_val_f1,
            "val_predictions":   final_preds,
            "val_targets":       final_targets,
            "val_probabilities": final_probs,
            "model":             model,
        }