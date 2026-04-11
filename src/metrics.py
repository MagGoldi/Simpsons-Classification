import torch
import pandas as pd
from collections import defaultdict

def calculate_f1_score(preds, targets, num_classes, average="macro"):
    """Compute F1-score directly from tensors without sklearn dependency."""
    confusion = torch.zeros(num_classes, num_classes, device=preds.device)
    confusion.index_put_(
        (targets, preds),
        torch.ones(len(targets), device=targets.device),
        accumulate=True,
    )

    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-10)

    if average == "macro":
        return f1_per_class.mean()
    elif average == "weighted":
        class_counts = targets.bincount(minlength=num_classes).float()
        weights = class_counts / class_counts.sum()
        return (f1_per_class * weights).sum()
    else:
        raise ValueError(f"Unsupported average mode: {average}")


def classwise_error_analysis(preds, targets, probs, label_encoder, save_path=None):
    """Per-class error breakdown from pre-computed predictions."""
    error_stats = defaultdict(lambda: {"total": 0, "errors": 0, "misclassified_as": []})

    for true, pred in zip(targets, preds):
        class_name = label_encoder.inverse_transform([true])[0]
        error_stats[class_name]["total"] += 1
        if true != pred:
            error_stats[class_name]["errors"] += 1
            error_stats[class_name]["misclassified_as"].append(
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
            "class": cls,
            "total": stats["total"],
            "error_count": stats["errors"],
            "error_rate": stats["errors"] / max(stats["total"], 1),
            "most_common_mistake": most_common,
        })

    df = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    if save_path:
        df.to_csv(save_path, index=False)
    return df