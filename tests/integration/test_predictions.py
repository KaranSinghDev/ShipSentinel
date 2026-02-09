"""
Integration tests for prediction and training routes.
The Predictor singleton is mocked so no MLflow server or trained model is required.
"""
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from shipsentinel.db.models import Shipment


@pytest.fixture()
def seeded_shipment(db_session):
    s = Shipment(
        id="PRED-001",
        carrier="FedEx", origin="NYC", destination="LA",
        service_type="express", customer_tier="gold",
        distance_km=4500.0, weight_kg=15.0,
        shipment_date=datetime(2024, 3, 1, 9, 0),
        scheduled_delivery=datetime(2024, 3, 3, 9, 0),
    )
    db_session.add(s)
    db_session.flush()
    return s


def _mock_predictor(prob=0.72, predicted=True):
    """Return a mock Predictor that returns (prob, predicted) without loading MLflow."""
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.model_version = "v-test"
    mock.predict.return_value = (prob, predicted)
    return mock


def test_predict_returns_201(client, seeded_shipment):
    with patch("shipsentinel.api.routes.predictions.Predictor.get", return_value=_mock_predictor()):
        resp = client.post(f"/predictions/{seeded_shipment.id}")
    assert resp.status_code == 201
    body = resp.json()
    assert body["breach_probability"] == pytest.approx(0.72)
    assert body["breach_predicted"] is True
    assert body["model_version"] == "v-test"
    assert body["shipment_id"] == "PRED-001"


def test_predict_404_for_unknown_shipment(client):
    with patch("shipsentinel.api.routes.predictions.Predictor.get", return_value=_mock_predictor()):
        resp = client.post("/predictions/NONEXISTENT")
    assert resp.status_code == 404


def test_predict_503_when_model_not_loaded(client, seeded_shipment):
    from shipsentinel.ml.predictor import ModelNotLoadedError
    mock = MagicMock()
    mock.is_loaded.return_value = False
    mock.predict.side_effect = ModelNotLoadedError("no model")
    with patch("shipsentinel.api.routes.predictions.Predictor.get", return_value=mock):
        resp = client.post(f"/predictions/{seeded_shipment.id}")
    assert resp.status_code == 503


def test_training_start_returns_202(client):
    with patch("shipsentinel.api.routes.predictions.train_model") as mock_task:
        mock_task.delay = MagicMock()
        resp = client.post("/training/start")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "running"
    assert "model_version" in body
