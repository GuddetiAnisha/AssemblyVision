import pytest

from assemblyvision.evaluation import classification_metrics, labels_from_intervals


def test_metrics_known_values():
    result = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert result["accuracy"] == 0.75
    assert result["per_class"]["a"]["recall"] == 0.5
    assert result["per_class"]["b"]["precision"] == pytest.approx(2 / 3)


def test_intervals_to_frames():
    labels = labels_from_intervals([{"label": "pick", "start_s": 0, "end_s": 1}], 20, 10)
    assert labels.count("pick") == 10
    assert labels.count("unknown") == 10
