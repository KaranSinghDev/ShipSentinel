"""
Unit tests for ml/trainer.py.
MLflow is redirected to a temp directory so no server is needed.
The LightGBM model is trained on small synthetic data.
"""
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from shipsentinel.ml.data import MIN_TRAINING_SAMPLES
from shipsentinel.ml.trainer import train, _make_version, LGBM_PARAMS


@pytest.fixture()
def settings_with_temp_mlflow(tmp_path):
    """Settings pointing to a file-based MLflow tracking URI (no server needed)."""
    from shipsentinel.config import Settings
    return Settings(
        database_url="sqlite:///:memory:",
        mlflow_tracking_uri=f"file://{tmp_path}/mlruns",
        model_registry_name="test-lgbm",
    )


@pytest.fixture()
def minimal_session_with_data(db_session):
    """Seed db_session with MIN_TRAINING_SAMPLES labelled shipments."""
    from shipsentinel.db.models import Shipment
    base = datetime(2024, 3, 1, 8, 0)
    for i in range(MIN_TRAINING_SAMPLES):
        db_session.add(Shipment(
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
    db_session.flush()
    return db_session


def test_make_version_format():
    v = _make_version()
    assert v.startswith("v")
    assert len(v) > 10


def test_train_returns_expected_keys(minimal_session_with_data, settings_with_temp_mlflow):
    """train() should return auc, mlflow_run_id, n_samples, model_version."""
    # Patch mlflow registration (no MLflow server)
    with patch("shipsentinel.ml.trainer.mlflow.sklearn.log_model"), \
         patch("shipsentinel.ml.trainer.mlflow.set_tag"):
        result = train(minimal_session_with_data, settings_with_temp_mlflow)

    assert "auc" in result
    assert "mlflow_run_id" in result
    assert "n_samples" in result
    assert "model_version" in result
    assert result["n_samples"] == MIN_TRAINING_SAMPLES


def test_train_auc_in_valid_range(minimal_session_with_data, settings_with_temp_mlflow):
    """Cross-validation AUC must be in [0.0, 1.0]."""
    with patch("shipsentinel.ml.trainer.mlflow.sklearn.log_model"), \
         patch("shipsentinel.ml.trainer.mlflow.set_tag"):
        result = train(minimal_session_with_data, settings_with_temp_mlflow)
    assert 0.0 <= result["auc"] <= 1.0


def test_train_insufficient_data_propagates(db_session, settings_with_temp_mlflow):
    """train() should propagate InsufficientDataError when < MIN_TRAINING_SAMPLES."""
    from shipsentinel.ml.data import InsufficientDataError
    with pytest.raises(InsufficientDataError):
        train(db_session, settings_with_temp_mlflow)
