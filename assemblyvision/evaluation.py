from __future__ import annotations

from collections import defaultdict


def labels_from_intervals(intervals: list[dict], frame_count: int, fps: float) -> list[str]:
    labels = ["unknown"] * frame_count
    for interval in intervals:
        start = max(0, round(interval["start_s"] * fps))
        end = min(frame_count, round(interval["end_s"] * fps))
        labels[start:end] = [interval["label"]] * max(0, end - start)
    return labels


def classification_metrics(truth: list[str], predicted: list[str]) -> dict:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Truth and predicted labels must have the same non-zero length")
    labels = sorted(set(truth) | set(predicted))
    per_class, confusion = {}, defaultdict(lambda: defaultdict(int))
    for actual, guess in zip(truth, predicted):
        confusion[actual][guess] += 1
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
    accuracy = sum(a == b for a, b in zip(truth, predicted)) / len(truth)
    macro_f1 = sum(value["f1"] for value in per_class.values()) / len(per_class)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class,
            "confusion_matrix": {a: dict(row) for a, row in confusion.items()}}
