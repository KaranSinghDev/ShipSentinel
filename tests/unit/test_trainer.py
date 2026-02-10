"""
Unit tests for ml/trainer.py.
MLflow is redirected to a temp directory so no server is needed.
LightGBM trains on small synthetic data.
"""
from datetime import datetime, timedelta
from unittest.mock import patch
import pytest

from shipsentinel.ml.data import MIN_TRAINING_SAMPLES
from shipsentinel.ml.trainer import _make_version
from tests.conftest import TEST_SETTINGS


@pytest.fixture()
def db_with_data(db):
    """Seed db with MIN_TRAINING_SAMPLES labelled shipments."""
    from shipsentinel.db.models import Shipment
    base = datetime(2024, 3, 1, 8, 0)
    for i in range(MIN_TRAINING_SAMPLES):
        db.add(Shipment(
            id=f"TR-{i:04d}",
            carrier=["FedEx", "UPS", "DHL"][i % 3],
            origin="NYC", destination="LA",
            service_type=["express", "standard"][i % 2],
            customer_tier=["gold", "silver", "bronze"][i % 3],
            distance_km=float(2000 + i * 10),
            weight_kg=float(5 + i % 20),
            shipment_date=base + timedelta(hours=i),
            scheduled_delivery=base + timedelta(hours=i + 48),
            sla_breached=bool(i % 3 == 0),
        ))
    db.flush()
    return db


def test_make_version_format():
    v = _make_version()
    assert v.startswith("v")
    assert len(v) > 10


def test_train_returns_expected_keys(db_with_data, tmp_path):
    from shipsentinel.config import Settings
    from shipsentinel.ml.trainer import train
    settings = Settings(
        database_url="sqlite:///:memory:",
        mlflow_tracking_uri=f"file://{tmp_path}/mlruns",
        model_registry_name="test-lgbm",
    )
    with patch("shipsentinel.ml.trainer.mlflow.sklearn.log_model"), \
         patch("shipsentinel.ml.trainer.mlflow.set_tag"):
        result = train(db_with_data, settings)

    assert {"auc", "mlflow_run_id", "n_samples", "model_version"} <= result.keys()
    assert result["n_samples"] == MIN_TRAINING_SAMPLES


def test_train_auc_in_valid_range(db_with_data, tmp_path):
    from shipsentinel.config import Settings
    from shipsentinel.ml.trainer import train
    settings = Settings(
        database_url="sqlite:///:memory:",
        mlflow_tracking_uri=f"file://{tmp_path}/mlruns",
        model_registry_name="test-lgbm",
    )
    with patch("shipsentinel.ml.trainer.mlflow.sklearn.log_model"), \
         patch("shipsentinel.ml.trainer.mlflow.set_tag"):
        result = train(db_with_data, settings)
    assert 0.0 <= result["auc"] <= 1.0


def test_train_insufficient_data_propagates(db):
    from shipsentinel.config import Settings
    from shipsentinel.ml.trainer import train
    from shipsentinel.ml.data import InsufficientDataError
    settings = Settings(
        database_url="sqlite:///:memory:",
        mlflow_tracking_uri="file:///tmp/test-mlruns",
        model_registry_name="test-lgbm",
    )
    with pytest.raises(InsufficientDataError):
        train(db, settings)
