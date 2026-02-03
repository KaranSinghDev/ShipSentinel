"""
TDD unit tests for ShipSentinel feature engineering.
Written before implementation — define the contract the code must satisfy.
"""
import pytest
from datetime import datetime, timedelta
from shipsentinel.ml.features import (
    compute_sla_window_hours,
    extract_temporal_features,
    build_feature_row,
    build_feature_dataframe,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_sla_window_48h():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    assert compute_sla_window_hours(t0, t0 + timedelta(days=2)) == pytest.approx(48.0)


def test_sla_window_fractional():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    assert compute_sla_window_hours(t0, t0 + timedelta(hours=36, minutes=30)) == pytest.approx(36.5)


def test_temporal_monday_morning():
    dt = datetime(2024, 1, 1, 9, 0, 0)  # Monday
    r = extract_temporal_features(dt)
    assert r["shipment_hour"] == 9
    assert r["shipment_dow"] == 0
    assert r["shipment_month"] == 1


def test_temporal_friday_evening():
    dt = datetime(2024, 6, 14, 18, 30, 0)  # Friday
    r = extract_temporal_features(dt)
    assert r["shipment_hour"] == 18
    assert r["shipment_dow"] == 4
    assert r["shipment_month"] == 6


def _make_shipment(now=None, days=2):
    if now is None:
        now = datetime(2024, 3, 15, 14, 0, 0)
    return {
        "carrier": "FedEx",
        "origin": "Mumbai",
        "destination": "Delhi",
        "service_type": "express",
        "customer_tier": "gold",
        "distance_km": 1400.0,
        "weight_kg": 5.5,
        "shipment_date": now,
        "scheduled_delivery": now + timedelta(days=days),
    }


def test_feature_row_has_all_features():
    row = build_feature_row(_make_shipment())
    for f in ALL_FEATURES:
        assert f in row, f"Missing feature: {f}"


def test_feature_row_sla_window():
    now = datetime(2024, 3, 15, 10, 0, 0)
    shipment = _make_shipment(now, days=3)
    row = build_feature_row(shipment)
    assert row["sla_window_hours"] == pytest.approx(72.0)


def test_feature_dataframe_shape():
    df = build_feature_dataframe([_make_shipment(), _make_shipment()])
    assert df.shape == (2, len(ALL_FEATURES))


def test_feature_dataframe_categorical_dtype():
    df = build_feature_dataframe([_make_shipment()])
    for col in CATEGORICAL_FEATURES:
        assert str(df[col].dtype) == "category", f"{col} should be category dtype"


def test_feature_dataframe_no_nulls():
    df = build_feature_dataframe([_make_shipment()])
    assert df.isnull().sum().sum() == 0
