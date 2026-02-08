"""
Unit tests for ml/data.py — DB query layer for training data.
Uses the in-memory SQLite + transaction-rollback fixture from conftest.
"""
import pytest
from shipsentinel.ml.data import get_labelled_shipments, InsufficientDataError, MIN_TRAINING_SAMPLES


def _make_shipment(db_session, suffix: str, sla_breached: bool | None = True):
    from datetime import datetime
    from shipsentinel.db.models import Shipment
    s = Shipment(
        id=f"SHIP-{suffix}",
        carrier="FedEx", origin="NYC", destination="LA",
        service_type="standard", customer_tier="gold",
        distance_km=4500.0, weight_kg=10.0,
        shipment_date=datetime(2024, 1, 1, 9, 0),
        scheduled_delivery=datetime(2024, 1, 4, 9, 0),
        sla_breached=sla_breached,
    )
    db_session.add(s)
    db_session.flush()
    return s


def test_insufficient_data_raises(db_session):
    """Fewer than MIN_TRAINING_SAMPLES labelled shipments → InsufficientDataError."""
    for i in range(MIN_TRAINING_SAMPLES - 1):
        _make_shipment(db_session, str(i), sla_breached=bool(i % 2))
    with pytest.raises(InsufficientDataError):
        get_labelled_shipments(db_session)


def test_unlabelled_shipments_excluded(db_session):
    """Shipments without outcome (sla_breached=None) must not appear in training data."""
    for i in range(MIN_TRAINING_SAMPLES):
        _make_shipment(db_session, f"lab-{i}", sla_breached=bool(i % 2))
    _make_shipment(db_session, "unlabelled", sla_breached=None)

    rows = get_labelled_shipments(db_session)
    assert all(r["sla_breached"] is not None for r in rows)
    assert len(rows) == MIN_TRAINING_SAMPLES


def test_returns_required_feature_keys(db_session):
    """Every returned dict must have all feature keys + label."""
    for i in range(MIN_TRAINING_SAMPLES):
        _make_shipment(db_session, f"k-{i}", sla_breached=True)
    rows = get_labelled_shipments(db_session)
    required = {
        "carrier", "origin", "destination", "service_type", "customer_tier",
        "distance_km", "weight_kg", "shipment_date", "scheduled_delivery", "sla_breached",
    }
    for row in rows:
        assert required <= set(row.keys())
