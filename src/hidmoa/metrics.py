"""
分类评估指标: Accuracy, Precision, Recall, F1 (macro / micro / per-class)
"""

from typing import Dict, List, Sequence, Tuple


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    从预测列表和标签列表计算全部指标。

    返回 dict:
        overall_acc, macro_acc, micro_acc,
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1,
        per_class: {cls_id: {acc, precision, recall, f1, tp, fp, fn, total}}
    """
    # 收集所有出现的类
    all_classes = sorted(set(all_labels) | set(all_preds))

    # 逐类统计 TP / FP / FN / total
    per_class = {}
    for c in all_classes:
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)
        total = sum(1 for l in all_labels if l == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = tp / total if total > 0 else 0.0

        per_class[c] = {
            "acc": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "total": total,
        }

    # Overall / Micro
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    overall_acc = correct / len(all_labels) if all_labels else 0.0
    total_tp = sum(per_class[c]["tp"] for c in all_classes)
    total_fp = sum(per_class[c]["fp"] for c in all_classes)
    total_fn = sum(per_class[c]["fn"] for c in all_classes)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )
    micro_acc = overall_acc

    # Macro average (只对 label 中出现的类取平均)
    label_classes = sorted(set(all_labels))
    n_cls = len(label_classes)
    macro_acc = sum(per_class[c]["acc"] for c in label_classes) / n_cls if n_cls else 0.0
    macro_precision = sum(per_class[c]["precision"] for c in label_classes) / n_cls if n_cls else 0.0
    macro_recall = sum(per_class[c]["recall"] for c in label_classes) / n_cls if n_cls else 0.0
    macro_f1 = sum(per_class[c]["f1"] for c in label_classes) / n_cls if n_cls else 0.0

    result = {
        "overall_acc": overall_acc,
        "macro_acc": macro_acc,
        "micro_acc": micro_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }

    if verbose:
        print(f"\n  Macro Acc / F1 : {macro_acc:.4f} / {macro_f1:.4f}")
        print(f"  Micro Acc / F1 : {micro_acc:.4f} / {micro_f1:.4f}")
        print(f"  Macro  P/R/F1 : {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")
        print(f"  {'Class':<22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'#':>5s}")
        print(f"  {'─' * 52}")
        for c in label_classes:
            m = per_class[c]
            name = class_names[c] if class_names and c < len(class_names) else str(c)
            print(f"  [{c}] {name:<18s} {m['acc']:6.4f} {m['precision']:6.4f} {m['recall']:6.4f} {m['f1']:6.4f} {m['total']:5d}")

    return result

