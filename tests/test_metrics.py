import numpy as np

from concrete_crack_pipeline.metrics import calculate_metrics, expected_calibration_error, find_best_accuracy_threshold


def test_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.1, 0.9, 0.99])
    m = calculate_metrics(y, p, 0.5)
    assert m["accuracy"] == 1.0
    assert m["specificity"] == 1.0
    assert m["recall"] == 1.0
    assert expected_calibration_error(y, p) >= 0


def test_threshold_search_returns_valid_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    threshold, frame = find_best_accuracy_threshold(y, p, 0.1, 0.9, 9)
    assert 0.1 <= threshold <= 0.9
    assert not frame.empty
