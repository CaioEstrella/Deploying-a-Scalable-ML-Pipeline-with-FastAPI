import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def trained_model():
    """Fixture that returns a simple trained model and test data."""
    X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_train = np.array([1, 0, 1, 0])
    model = train_model(X_train, y_train)
    return model, X_train, y_train


def test_train_model(trained_model):
    """train_model should return a RandomForestClassifier."""
    model, _, _ = trained_model
    assert isinstance(model, RandomForestClassifier)


def test_inference(trained_model):
    """inference should return a numpy array with the same length as input."""
    model, X_train, _ = trained_model
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_train)


def test_compute_model_metrics():
    """compute_model_metrics should return 1.0 for all metrics on perfect predictions."""
    y = np.array([1, 0, 1, 0, 1])
    preds = np.array([1, 0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert fbeta == pytest.approx(1.0)
