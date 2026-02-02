"""
TDD unit tests for ShipSentinel SQLAlchemy models (no DB needed).
"""
from datetime import datetime, timedelta
from shipsentinel.db.models import Shipment, Prediction, TrainingRun


def test_shipment_defaults():
    now = datetime.utcnow()
    s = Shipment(
        id="SHIP-TEST-001",
        carrier="FedEx",
        origin="Mumbai",
        destination="Delhi",
        service_type="express",
        customer_tier="gold",
        distance_km=1400.0,
        weight_kg=5.5,
        shipment_date=now,
        scheduled_delivery=now + timedelta(days=2),
    )
    assert s.id == "SHIP-TEST-001"
    assert s.sla_breached is None
    assert s.actual_delivery is None


def test_prediction_breach_flag():
    p = Prediction(
        shipment_id="SHIP-TEST-001",
        breach_probability=0.73,
        breach_predicted=True,
        model_version="v1.0.0",
        feature_snapshot={"distance_km": 1400.0},
    )
    assert p.breach_predicted is True
    assert 0.0 <= p.breach_probability <= 1.0


def test_training_run_defaults():
    tr = TrainingRun(model_version="v1.0.0")
    assert tr.status == "running"
    assert tr.completed_at is None
    assert tr.metrics is None
