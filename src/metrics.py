"""
metrics.py — кастомные метрики классификации без sklearn.
"""

import torch
import pandas as pd
from collections import defaultdict


def calculate_f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> float:
    """Вычислить F1-score из тензоров без sklearn.

    Args:
        preds:       Тензор предсказанных классов.
        targets:     Тензор истинных меток.
        num_classes: Общее количество классов.
        average:     "macro" или "weighted".

    Returns:
        Скалярное значение F1 как float.

    Raises:
        ValueError: При неизвестном значении average.
    """
    confusion = torch.zeros(num_classes, num_classes, device=preds.device)
    confusion.index_put_(
        (targets, preds),
        torch.ones(len(targets), device=targets.device),
        accumulate=True,
    )

    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    precision    = tp / (tp + fp + 1e-10)
    recall       = tp / (tp + fn + 1e-10)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-10)

    if average == "macro":
        return f1_per_class.mean().item()
    if average == "weighted":
        weights = targets.bincount(minlength=num_classes).float()
        weights /= weights.sum()
        return (f1_per_class * weights).sum().item()
    raise ValueError(f"Unsupported average mode: '{average}'. Use 'macro' or 'weighted'.")


def classwise_error_analysis(preds, targets, probs, label_encoder, save_path=None) -> pd.DataFrame:
    """Детальный анализ ошибок по классам из кэшированных предсказаний.

    Args:
        preds:         Массив предсказанных индексов.
        targets:       Массив истинных индексов.
        probs:         Массив вероятностей.
        label_encoder: Обученный LabelEncoder.
        save_path:     Путь для сохранения CSV (опционально).

    Returns:
        DataFrame с колонками: class, total, error_count, error_rate, most_common_mistake.
    """
    error_stats = defaultdict(lambda: {"total": 0, "errors": 0, "misclassified_as": []})

    for true, pred in zip(targets, preds):
        cls = label_encoder.inverse_transform([true])[0]
        error_stats[cls]["total"] += 1
        if true != pred:
            error_stats[cls]["errors"] += 1
            error_stats[cls]["misclassified_as"].append(
                label_encoder.inverse_transform([pred])[0]
            )

    rows = []
    for cls, stats in error_stats.items():
        most_common = (
            pd.Series(stats["misclassified_as"]).mode().iloc[0]
            if stats["misclassified_as"]
            else None
        )
        rows.append({
            "class":               cls,
            "total":               stats["total"],
            "error_count":         stats["errors"],
            "error_rate":          stats["errors"] / max(stats["total"], 1),
            "most_common_mistake": most_common,
        })

    df = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    if save_path:
        df.to_csv(save_path, index=False)
    return df